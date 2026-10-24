// The orthographic view-projection a static sun shadow map is baked through, the
// world size of one of its texels, and the test for when the sun has moved far
// enough to need another bake.
//
// Conventions, shared with frustum.zig: zmath transforms a row vector, so clip
// space is `mul(position, view_proj)`, and the depth range is Vulkan's, z' in
// [0, 1]. The direction argument points TOWARD the sun, not along its travel.
//
// The matrix carries no Y flip, so the bake and the lookup both have to go
// through this one matrix for the vertical mirror to cancel.

const std = @import("std");
const zm = @import("zmath");
const resources = @import("lenore-resources");
const vec = @import("vec.zig");

const Sphere = resources.Sphere;
const Vec3 = resources.Vec3;

// Margin around the scene sphere, in shadow-map texels on each side, so that
// geometry lying exactly on the sphere still has whole texels around it to be
// filtered from. Two of them covers a 2x2 footprint, which reaches one texel,
// plus the one texel the normal offset moves the lookup by.
//
// A count of texels rather than a fraction of the radius, because what has to
// stay covered is a filter footprint and that is the same number of texels
// whatever the scene measures. Godot expands its cascade radius the same way,
// one texel per side: `radius *= texture_size / (texture_size - 2.0)`, in
// RendererSceneCull::_light_instance_setup_directional_shadow,
// servers/rendering/renderer_scene_cull.cpp:2294.
const fit_margin_texels: f32 = 2;

// How far the sun may turn from the direction a map was baked for before the
// map is baked again, in degrees.
//
// The lookup follows the sun between bakes by parallax (`parallax`), so a
// bake is no longer what moves a shadow's edge. What a bake does is redraw the
// map on a turned grid, and every edge then lands on that grid with a new
// rasterization error of up to half a texel. A sun turning slowly enough moves
// a ground shadow by less than that per bake, and an edge baked often walks
// backwards as often as forwards. Baking seldom is the cure, so the budget is
// an angle the correction stays good over and not a drift in texels.
//
// What bounds it is the lookup's reach. A caster at the map's near plane
// shifts a point's lookup by the tangent of the angle times the point's depth,
// at most `tan(angle)` of the map's width; at 0.25 degrees on a 2048 map that
// is 9 texels, inside the search the lookup does around every point. The
// correction is first order in the angle, and its error is in the estimate of
// how deep the caster is, not in the angle.
//
// Provisional: 0.25 degrees is two seconds of a sun crossing a 48-minute day's
// 360 degrees, chosen to be checked on screen and lengthened while complex
// edges hold.
pub const resample_degrees: f32 = 0.25;

// How long the lookup fades from the map before a bake to the one it made, in
// seconds. Each map is corrected to the current sun, so the two agree but for
// their rasterization, and the fade spreads that difference over time. Up to
// half a texel on an edge, but on a near-straight edge the new grid moves
// whole steps of its staircase, which a quarter of a second showed as a jump.
// Seconds and not the sun's angle, so a sun that stops does not stop a fade
// halfway.
//
// While it runs both maps are read, so it costs the second map's lookup for
// this share of every `resample_degrees`. The older map then serves the
// budget's angle plus this long's turn, which the lookup's reach covers.
pub const fade_seconds: f32 = 1.0;

// Above this, the sun is too close to the world up axis for `cross(up, dir)` to
// be well conditioned, and the fit uses the world X axis instead. The limit is
// 8.1 degrees from vertical, where the cross product still keeps 14 percent of
// its length; past it |dir[0]| is at most 0.141, so the X axis is at least as
// far from the sun direction as Y was.
//
// Which axis it is only rotates the map about the sun axis, and under the
// convention above that rotation cancels between the bake and the lookup.
const up_fallback_limit: f32 = 0.99;

// zmath 0.11.0-dev `orthographicRh` asserts that its width, its height and its
// depth range each differ from zero by more than 0.001, and ReleaseFast removes
// that assert rather than reporting it. All three are 2 * bounds.radius * slack
// here, and slack is above 1, so a radius above 0.001 clears every one of them.
const min_fit_radius: f32 = 1e-3;

// The square of a length of 1e-6, which is the shortest thing this treats as a
// direction. Squaring it stays far above where an f32 underflows, so the
// normalization below keeps its digits.
const min_sun_length_squared: f32 = 1e-12;

pub const FitError = error{
    // The scene has no extent to enclose, or bounds that are not finite. Both
    // arrive from asset data: `Aabb.compute` in lenore-resources reports a zero
    // box for zero vertices, and a vertex at infinity carries through its
    // minimum and maximum into the radius.
    DegenerateBounds,

    // The sun direction has no length, or is not finite. `zm.normalize3` divides
    // by the length with no guard, and `lookAtRh` normalizes eye minus focus
    // with none either, so the whole matrix would come back not-a-number.
    DegenerateSunDirection,
};

pub const SunShadowFit = struct {
    // The product the bake and the lookup both apply, `mul(view, proj)`.
    view_proj: zm.Mat,

    // World units across one texel of the map this fit was computed for. The
    // normal offset that keeps a surface from shadowing itself is authored in
    // texels and scaled by this, so it follows scene scale and map resolution
    // without being retuned.
    texel_world_size: f32,

    // World distance from near (zero) to far, for converting caster bias to depth.
    depth_span: f32,

    // World width of the square the map covers. Equal to `depth_span` for a
    // fit around the scene sphere; a cascade is narrower than it is deep, and
    // the lookup turns a penumbra sized in depth into map coordinates through
    // the ratio of the two.
    lateral_span: f32,

    // Unit, and the direction the fit was computed for.
    sun_dir: Vec3,

    // Squared chord between two unit directions at which the map goes stale.
    // See `stale` for why the threshold is a chord and not a cosine.
    resample_chord_squared: f32,

    // `bounds` encloses everything that casts into the map, and the whole fit is
    // derived from it, so the caller recomputes this when the bounds change as
    // well as when `stale` reports the sun has moved.
    //
    // The eye sits on the sun side at the slacked radius, so the sphere spans
    // the depth range with the same margin it gets laterally, and near is 0.
    //
    // No texel snapping. A fit that follows the camera has to snap its
    // projection to whole texels or the shadow edges crawl as it moves, which
    // Godot's cascades do in the same function cited above, at
    // renderer_scene_cull.cpp:2313, calling it the trick that stabilizes the
    // shadow. This fit is derived from the static bounds and the sun alone, so
    // between two bakes it does not move at all.
    pub fn compute(bounds: Sphere, sun_dir: Vec3, map_size: u32) FitError!SunShadowFit {
        const resolution: f32 = @floatFromInt(map_size);
        std.debug.assert(resolution > 2 * fit_margin_texels);

        if (!(bounds.radius > min_fit_radius and bounds.radius < std.math.inf(f32)))
            return error.DegenerateBounds;
        if (!vec.finite(bounds.centre)) return error.DegenerateBounds;

        if (!vec.finite(sun_dir)) return error.DegenerateSunDirection;
        const length_squared = @reduce(.Add, sun_dir * sun_dir);
        if (length_squared < min_sun_length_squared) return error.DegenerateSunDirection;

        // The margin is a count of texels on each side, so it is the projection
        // that grows to hold it and the scene sphere that keeps its size.
        const slack = resolution / (resolution - 2 * fit_margin_texels);
        const radius = bounds.radius * slack;

        const centre = zm.f32x4(bounds.centre[0], bounds.centre[1], bounds.centre[2], 1);
        const dir = zm.normalize3(zm.f32x4(sun_dir[0], sun_dir[1], sun_dir[2], 0));
        const eye = centre + dir * zm.f32x4s(radius);
        const up = if (@abs(dir[1]) > up_fallback_limit)
            zm.f32x4(1, 0, 0, 0)
        else
            zm.f32x4(0, 1, 0, 0);

        const view = zm.lookAtRh(eye, centre, up);
        const proj = zm.orthographicRh(2 * radius, 2 * radius, 0, 2 * radius);

        // The chord between two unit directions is 2 * sin(theta / 2).
        const chord = 2 * @sin(std.math.degreesToRadians(resample_degrees) / 2);

        return .{
            .view_proj = zm.mul(view, proj),
            .texel_world_size = 2 * radius / resolution,
            .depth_span = 2 * radius,
            .lateral_span = 2 * radius,
            .sun_dir = .{ dir[0], dir[1], dir[2] },
            .resample_chord_squared = chord * chord,
        };
    }

    // A cascade: a square `width` wide across the sun's rays around `at`, as
    // deep as the scene sphere along them, so whatever stands between the
    // square and the sun anywhere in the scene still casts into it.
    //
    // The square's centre is snapped to whole texels of its own map in the
    // sun's frame. Two cascades baked for the same sun around nearby points
    // then rasterize every caster onto the same grid, and moving the centre
    // moves no edge: Godot's cascades do the same, calling it the trick that
    // stabilizes the shadow (renderer_scene_cull.cpp:2313). The basis is
    // written out here rather than taken from `lookAtRh`, so the translation
    // is the snapped value itself and not a product that rounds it again.
    pub fn computeCascade(bounds: Sphere, sun_dir: Vec3, map_size: u32, width: f32, at: Vec3) FitError!SunShadowFit {
        const resolution: f32 = @floatFromInt(map_size);
        std.debug.assert(resolution > 2 * fit_margin_texels);

        if (!(bounds.radius > min_fit_radius and bounds.radius < std.math.inf(f32)))
            return error.DegenerateBounds;
        if (!vec.finite(bounds.centre)) return error.DegenerateBounds;
        if (!(width > min_fit_radius and width < std.math.inf(f32))) return error.DegenerateBounds;
        if (!vec.finite(at)) return error.DegenerateBounds;

        if (!vec.finite(sun_dir)) return error.DegenerateSunDirection;
        const length_squared = @reduce(.Add, sun_dir * sun_dir);
        if (length_squared < min_sun_length_squared) return error.DegenerateSunDirection;

        const slack = resolution / (resolution - 2 * fit_margin_texels);
        const half = width * slack / 2;
        const radius = bounds.radius * slack;
        const texel = 2 * half / resolution;

        // The axes `lookToLh` in zmath 0.11.0-dev builds for an eye looking
        // along -dir with this up, so the map is oriented as `compute`'s is.
        const dir = zm.normalize3(zm.f32x4(sun_dir[0], sun_dir[1], sun_dir[2], 0));
        const up = if (@abs(dir[1]) > up_fallback_limit)
            zm.f32x4(1, 0, 0, 0)
        else
            zm.f32x4(0, 1, 0, 0);
        const ax = zm.normalize3(zm.cross3(up, dir));
        const ay = zm.normalize3(zm.cross3(dir, ax));

        const point = zm.f32x4(at[0], at[1], at[2], 0);
        const x = @round(zm.dot3(ax, point)[0] / texel) * texel;
        const y = @round(zm.dot3(ay, point)[0] / texel) * texel;
        const centre = zm.f32x4(bounds.centre[0], bounds.centre[1], bounds.centre[2], 0);
        const eye_depth = zm.dot3(dir, centre)[0] + radius;

        const view: zm.Mat = .{
            zm.f32x4(ax[0], ay[0], dir[0], 0),
            zm.f32x4(ax[1], ay[1], dir[1], 0),
            zm.f32x4(ax[2], ay[2], dir[2], 0),
            zm.f32x4(-x, -y, -eye_depth, 1),
        };
        const proj = zm.orthographicRh(2 * half, 2 * half, 0, 2 * radius);
        const chord = 2 * @sin(std.math.degreesToRadians(resample_degrees) / 2);

        return .{
            .view_proj = zm.mul(view, proj),
            .texel_world_size = texel,
            .depth_span = 2 * radius,
            .lateral_span = 2 * half,
            .sun_dir = .{ dir[0], dir[1], dir[2] },
            .resample_chord_squared = chord * chord,
        };
    }

    // How far `point` stands from the middle of the map across the sun's
    // rays, in world units along whichever map axis is farther: half the
    // lateral span is the map's edge.
    pub fn lateralDistance(self: *const SunShadowFit, point: Vec3) f32 {
        const clip = zm.mul(zm.f32x4(point[0], point[1], point[2], 1), self.view_proj);
        return @max(@abs(clip[0]), @abs(clip[1])) * self.lateral_span / 2;
    }

    // True when the sun has moved far enough that the baked map no longer lines
    // up with it. `sun_dir` needs no particular length.
    //
    // The threshold is a squared chord and not the cosine a dot product would
    // give, because f32 has little room just below 1. At `resample_degrees` the
    // cosine is 1 - 9.5e-6, about 160 ulps below 1, and a tenth of the angle is
    // under two. The chord between two unit vectors is 2 * sin(theta / 2), which
    // is computed from differences that keep their digits at any angle.
    //
    // A direction that is zero or not finite makes the chord not a number, every
    // comparison against it false, and the answer "not stale". That keeps the
    // last good fit instead of baking one from a direction that has none.
    pub fn stale(self: *const SunShadowFit, sun_dir: Vec3) bool {
        const length = @sqrt(@reduce(.Add, sun_dir * sun_dir));
        const offset = sun_dir / @as(Vec3, @splat(length)) - self.sun_dir;
        return @reduce(.Add, offset * offset) > self.resample_chord_squared;
    }

    // How far a lookup in this fit's map moves per unit of depth between the
    // point and its caster, for a sun at `sun_dir` rather than the one the map
    // was baked for: the map coordinates (x, y) gain this times the depth the
    // caster stands in front of the point. Zero for the baked direction.
    //
    // The ray from a point toward the sun crosses the baked map's depths
    // obliquely once the sun has turned. Where it has climbed by a depth d it
    // stands over a texel displaced by d times this, and the caster found there
    // is the one that shades the point now (Turchyn, "Parallax-Corrected Cached
    // Shadow Maps", GPU Zen 2, 2019, chapter III-2, equation 2.4).
    //
    // Map coordinates are the lookup's: x and y are clip x and y mapped from
    // [-1, 1] to [0, 1], depth is clip z. A direction carried through the
    // matrix with no translation is how far those move per world unit along
    // it, so the shift per unit of depth is the lateral half-rate over the
    // depth rate, which falls as the ray climbs toward the sun.
    //
    // `sun_dir` needs no particular length. One that is not finite, or that has
    // turned so far that the ray no longer climbs, returns no shift: the lookup
    // then reads the map as baked, which is what it did before the correction.
    pub fn parallax(self: *const SunShadowFit, sun_dir: Vec3) [2]f32 {
        const along = zm.mul(zm.f32x4(sun_dir[0], sun_dir[1], sun_dir[2], 0), self.view_proj);
        const climb = -along[2];
        const length = @sqrt(@reduce(.Add, sun_dir * sun_dir));
        if (!(climb > 1e-3 * length) or !std.math.isFinite(climb)) return .{ 0, 0 };
        return .{ 0.5 * along[0] / climb, 0.5 * along[1] / climb };
    }
};

// Runtime look controls for sun shadows, authored per scene rather than baked
// into the fit: the bias clamp controls the bake, the others its lookup.
pub const SunShadowSettings = struct {
    // Off by default: a shadow map is a cost a scene opts into.
    enabled: bool = false,

    // How much of the sun's direct contribution a shadowed surface loses, in
    // [0, 1]. One is the physical answer when ambient light is accounted for
    // separately, and lower is an art control. Out of range is clamped rather
    // than refused: this is authored data, and a wrong number here is worth a
    // dimmer shadow, not a failed scene.
    strength: f32 = 1,

    // How far the shadow lookup is pushed off the surface along its normal, in
    // texels of the map. Depth quantized to one texel is the error that shows as
    // acne, so one texel of offset escapes it. Raising it trades that for light
    // leaking near contact points.
    normal_offset_texels: f32 = 1,

    // Maximum caster depth offset along the sun, in map texels, to keep steep
    // faces from detaching their shadows. Clamped to [0.05, 8]: Vulkan treats
    // zero as unclamped. Non-finite values use the default half texel.
    bias_clamp_texels: f32 = 0.5,

    // The sun's angular radius in degrees, which widens the shadow's edge with
    // the caster's distance from the surface. The sun seen from the Earth is
    // about 0.267. Zero leaves the edge the filter alone gives, a couple of
    // texels wide. Clamped to [0, 5]: wider is no longer a sun, and the search
    // for casters would spread its taps too thin.
    source_degrees: f32 = 0,

    // What a shadowed surface loses, with the switch above folded in.
    //
    // The switch is folded in here rather than tested by each caller, because a
    // field that reads as one and is honoured nowhere is worse than no field:
    // it looks like the shadow is off and the picture says otherwise. This is
    // the one accessor, so honouring it here is what makes the switch real.
    //
    // Off is a strength of zero rather than a separate state. Every consumer
    // already scales the shadow term by this, so zero is a lit surface exactly,
    // and nothing needs a branch for the disabled case.
    pub fn clampedStrength(self: SunShadowSettings) f32 {
        if (!self.enabled) return 0;
        return std.math.clamp(self.strength, 0, 1);
    }

    pub fn sourceTangent(self: SunShadowSettings) f32 {
        const degrees = if (std.math.isFinite(self.source_degrees)) std.math.clamp(self.source_degrees, 0, 5) else 0;
        return @tan(std.math.degreesToRadians(degrees));
    }

    pub fn clampedBiasTexels(self: SunShadowSettings) f32 {
        return if (std.math.isFinite(self.bias_clamp_texels))
            std.math.clamp(self.bias_clamp_texels, 0.05, 8)
        else
            (SunShadowSettings{}).bias_clamp_texels;
    }

    pub fn biasClampDepth(self: SunShadowSettings, fit: *const SunShadowFit) f32 {
        return self.clampedBiasTexels() * fit.texel_world_size / fit.depth_span;
    }

    pub fn normalOffsetWorld(self: SunShadowSettings, fit: *const SunShadowFit) f32 {
        return self.normal_offset_texels * fit.texel_world_size;
    }
};
