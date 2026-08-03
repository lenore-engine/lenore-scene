// The participating medium a scene sits in: how dense it is, how that density
// falls off with height, how it scatters, and the optical depth that follows
// along a view ray.
//
// The model is exponential height density with a closed-form integral along the
// ray, plus one directional lobe shaped by Henyey-Greenstein. The integral lives
// here rather than only in a shader because it is the part that can be wrong:
// arithmetic in a fragment shader is checked by looking at it, and this way the
// same closed form is pinned by tests and a pass mirrors one function.
//
// Height is the world Y axis, matching the rest of the module.

const std = @import("std");
const resources = @import("lenore-resources");
const vec = @import("vec.zig");

const Vec3 = resources.Vec3;

// A density above this is opaque within a hundredth of a world unit, so nothing
// beyond it describes a medium anyone looks through.
const max_density: f32 = 1e4;

// exp(-88) is smaller than the smallest normal f32, so an optical depth past
// this is exactly opaque and carrying more of it only risks an infinity.
const max_optical_depth: f32 = 88;

// How far apart the two exponents have to be before the closed form is worth
// taking. It subtracts them, so it keeps a relative eps / spread of its digits,
// and below 1e-3 that is a part in 10^4 or worse. The plain length it falls back
// to is wrong by spread / 2, which at the same point is a part in 2000, so this
// is where the two are equally good and both are better than anywhere else.
const min_exponent_spread: f32 = 1e-3;

// The forward lobe at this anisotropy is already 780 times the isotropic value,
// so a scatter strength visible to the side blows out along the axis. Past it
// the cube in the denominator falls fast enough that the peak stops being a
// number a scene can be tuned against.
const max_phase_anisotropy: f32 = 0.95;

pub const FogError = error{
    // The in-scatter colour is negative or not finite. Linear and unbounded
    // above, since it is radiance.
    InvalidFogColour,

    // The density, the height falloff or the reference height cannot describe a
    // medium.
    InvalidFogDensity,

    // The anisotropy is outside the range the phase function can be tuned in.
    InvalidPhaseAnisotropy,

    // The sun scatter strength is negative or not finite.
    InvalidFogScatter,

    // The opacity clamp is outside [0, 1].
    InvalidFogOpacity,

    // The march length, or the range local media are gathered over, is not a
    // usable distance. The second must fit inside the first.
    InvalidMarchRange,

    // The step count is not a march.
    InvalidMarchSteps,

    // A noise or wind parameter is negative or not finite.
    InvalidMarchNoise,

    // A strength the march scales a term by is negative or not finite.
    InvalidMarchStrength,
};

pub const FogSettings = struct {
    // Which implementation owns the medium. One field rather than a switch and a
    // mode, because two of them is how a medium ends up applied twice: the
    // analytic path and a raymarch both believing they are on.
    //
    // `volumetric` below stays tuned while this is `.off` or `.analytic`, which
    // is why it is a field and not a payload on this enum. Toggling fog off in
    // an editor and back on must not lose what was set.
    mode: Mode = .off,

    // In-scatter colour away from the sun: the ambient tint of the medium.
    // Linear, and radiance, so it may exceed one.
    colour: Vec3 = .{ 0.30, 0.34, 0.40 },

    // Extinction per world unit at the reference height.
    density: f32 = 0.03,

    // Rate at which density falls with height, per world unit. Zero is a medium
    // of uniform density, which is plain distance fog. Positive pools it below
    // `height_ref`.
    height_falloff: f32 = 0,

    // The world height at which the density is exactly `density`.
    height_ref: f32 = 0,

    // Henyey-Greenstein anisotropy. Above zero scatters forward, which is what
    // makes mist glow toward the sun; below zero scatters back; zero is even in
    // every direction.
    phase_anisotropy: f32 = 0.76,

    // How much of the sun's radiance the medium scatters, at the isotropic
    // reference. The phase function is normalized so that this is a plain
    // multiplier: see `phase`.
    //
    // Zero by default. The lobe needs a light designated as the sun, and a scene
    // that has not designated one should not change appearance for having fog.
    sun_scatter: f32 = 0,

    // Upper bound on how opaque the medium is allowed to get, whatever the
    // optical depth says. An art control: distant geometry that fades to exactly
    // the fog colour cannot be read.
    max_opacity: f32 = 1,

    // Inert unless `mode` is `.volumetric`.
    volumetric: VolumetricSettings = .{},

    pub const Mode = enum { off, analytic, volumetric };

    pub fn validate(self: FogSettings) FogError!void {
        if (!vec.finite(self.colour) or !@reduce(.And, self.colour >= @as(Vec3, @splat(0))))
            return error.InvalidFogColour;

        if (!(self.density >= 0 and self.density <= max_density)) return error.InvalidFogDensity;
        if (!(self.height_falloff >= 0 and self.height_falloff < std.math.inf(f32)))
            return error.InvalidFogDensity;
        if (!(@abs(self.height_ref) < std.math.inf(f32))) return error.InvalidFogDensity;

        if (!(@abs(self.phase_anisotropy) <= max_phase_anisotropy))
            return error.InvalidPhaseAnisotropy;
        if (!(self.sun_scatter >= 0 and self.sun_scatter < std.math.inf(f32)))
            return error.InvalidFogScatter;
        if (!(self.max_opacity >= 0 and self.max_opacity <= 1)) return error.InvalidFogOpacity;

        try self.volumetric.validate();
    }

    // Optical depth along a ray, the integral of density over the distance
    // travelled. `direction_y` is the vertical component of a unit direction, so
    // the distance is in world units. An infinite distance is the sky, and needs
    // no case of its own: the profile is evaluated at the far end like any
    // other, and an endless ray that does not leave the medium comes back
    // opaque, which is what it is.
    //
    // The density profile is exp(-falloff * (y - ref)), so the integral is
    // density * (exp(start) - exp(end)) / rate, where the two exponents are the
    // profile at the ends of the ray and the rate is falloff * direction_y. The
    // limit as the rate goes to zero is the plain length.
    //
    // Written as the difference of the two ends rather than as a starting
    // density times a factor, which is the same algebra and not the same
    // arithmetic. Both exponents are then heights the ray actually reaches, so
    // neither is an extreme the other has to cancel: a camera far above a layer
    // has a start that underflows to zero and an end that overflows, and their
    // product form loses the answer entirely while their difference is exact.
    pub fn opticalDepth(self: FogSettings, from_height: f32, direction_y: f32, distance: f32) f32 {
        if (!(self.density > 0) or !(distance > 0)) return 0;

        const start = -self.height_falloff * (from_height - self.height_ref);
        const end = -self.height_falloff * (from_height + direction_y * distance - self.height_ref);

        const rate = self.height_falloff * direction_y;
        const spread = rate * distance;
        const depth = if (@abs(spread) > min_exponent_spread)
            self.density * (@exp(start) - @exp(end)) / rate
        else
            self.density * @exp(start) * distance;

        // The two exponents are ordered by the sign of the rate, and so is the
        // division by it, so the result is never negative and needs no floor.
        //
        // One comparison for every way this ends without a number: an infinite
        // depth, a difference of two infinities, and a rate of zero over an
        // endless ray. All of them mean the medium is opaque.
        if (!(depth < max_optical_depth)) return max_optical_depth;
        return depth;
    }

    // How much of the fog colour has replaced what is behind it, in [0, 1].
    pub fn opacity(self: FogSettings, from_height: f32, direction_y: f32, distance: f32) f32 {
        const depth = self.opticalDepth(from_height, direction_y, distance);
        return (1 - @exp(-depth)) * self.max_opacity;
    }

    // Henyey-Greenstein, normalized so that the isotropic case is one rather
    // than 1/(4 pi). The anisotropy then only redistributes the lobe and
    // `sun_scatter` is the whole of its scale, which is what lets it be tuned:
    // against the 1/(4 pi) convention the same setting carries a factor of
    // 0.0796 that has to be undone by eye.
    //
    // `cos_theta` is between the view direction and the direction toward the
    // sun. The denominator is at least (1 - |g|)^3, which the validated bound on
    // the anisotropy keeps clear of zero.
    pub fn phase(self: FogSettings, cos_theta: f32) f32 {
        const g = self.phase_anisotropy;
        const g2 = g * g;
        const denominator = 1 + g2 - 2 * g * cos_theta;
        return (1 - g2) / (denominator * @sqrt(denominator));
    }
};

// What the raymarch needs beyond the medium above. Inert unless the fog mode
// selects it.
pub const VolumetricSettings = struct {
    // How far along the view ray the march runs. Media past it are the analytic
    // model's to describe, so this bounds where a shaft can appear rather than
    // where the fog ends.
    max_distance: f32 = 200,

    // Samples along that distance. What a viewer sees is the spacing between
    // them, which is `sampleSpacing`.
    steps: u32 = 24,

    // Amplitude of the animated density modulation. Zero is a medium of purely
    // exponential density.
    noise_amplitude: f32 = 0.5,

    // Spatial frequency of that modulation, in reciprocal world units, so its
    // reciprocal is the size of one billow.
    noise_scale: f32 = 0.05,

    // Velocity the density pattern is carried at, world units per second.
    wind: Vec3 = .{ 1, 0, 0 },

    // Scales the part of the in-scatter that arrives from every direction, as
    // against the directional lobe. One matches the analytic model.
    ambient_strength: f32 = 1,

    // Extra filter radius when the march samples the sun's shadow map, in texels
    // of that map. Zero is a single tap.
    shadow_softness: f32 = 1,

    // How bounded media inside the march are evaluated.
    local_fog_mode: LocalFogMode = .analytic,

    // How far bounded media are gathered over, which is shorter than the march
    // in a scene with many of them. Cannot exceed `max_distance`: nothing
    // outside the march can contribute to it.
    local_fog_range: f32 = 48,

    pub const LocalFogMode = enum(u32) { off, analytic, froxel };

    // World units between two samples of the march. The number a viewer sees as
    // banding, and the one to compare against the size of a billow: a spacing
    // wider than the feature it is sampling cannot resolve it.
    pub fn sampleSpacing(self: VolumetricSettings) f32 {
        return self.max_distance / @as(f32, @floatFromInt(self.steps));
    }

    pub fn validate(self: VolumetricSettings) FogError!void {
        if (!(self.max_distance > 0 and self.max_distance < std.math.inf(f32)))
            return error.InvalidMarchRange;
        if (!(self.local_fog_range > 0 and self.local_fog_range <= self.max_distance))
            return error.InvalidMarchRange;

        if (self.steps == 0) return error.InvalidMarchSteps;

        if (!(self.noise_amplitude >= 0 and self.noise_amplitude < std.math.inf(f32)))
            return error.InvalidMarchNoise;
        if (!(self.noise_scale >= 0 and self.noise_scale < std.math.inf(f32)))
            return error.InvalidMarchNoise;
        if (!vec.finite(self.wind)) return error.InvalidMarchNoise;

        if (!(self.ambient_strength >= 0 and self.ambient_strength < std.math.inf(f32)))
            return error.InvalidMarchStrength;
        if (!(self.shadow_softness >= 0 and self.shadow_softness < std.math.inf(f32)))
            return error.InvalidMarchStrength;
    }
};
