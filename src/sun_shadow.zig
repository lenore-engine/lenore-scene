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

// How far the projection of a point may drift, in texels, before the map is
// re-baked. One texel is the point below which the staleness cannot be
// represented in the image at all.
//
// DECIDE: a budget of one texel is the smallest that is not free, and the cost
// it buys is unmeasured. At N = 2048 it re-bakes every 0.028 degrees of sun
// movement, so a sun sweeping at one degree per second re-bakes 36 times a
// second, which is a full shadow pass per frame at 36 fps. What closes this:
// the bake's own cost against a sweep, with the drift budget raised until the
// re-bakes stop showing in the frame time, and then a look at whether the
// resulting lag is visible on a shadow edge.
const resample_drift_texels: f32 = 1;

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

        // One shadow texel is 2 * bounds.radius * slack / N world units, and
        // rotating the light by theta moves the projection of a point by up to
        // 2 * bounds.radius * theta. The radius cancels between them: the drift
        // in texels is theta * N / slack, which is why the budget can be stated
        // in texels and the angle falls out of the resolution alone.
        const drift_angle = resample_drift_texels * slack / resolution;

        return .{
            .view_proj = zm.mul(view, proj),
            .texel_world_size = 2 * radius / resolution,
            .sun_dir = .{ dir[0], dir[1], dir[2] },
            .resample_chord_squared = drift_angle * drift_angle,
        };
    }

    // True when the sun has moved far enough that the baked map no longer lines
    // up with it. `sun_dir` needs no particular length.
    //
    // The threshold is a squared chord and not the cosine a dot product would
    // give, because f32 runs out of room just below the threshold. Measured at a
    // one-texel budget and N = 2048, where the angle is 4.89e-4: its cosine is
    // 1 - 1.19e-7, two ulps below 1, and the cosine of a quarter of it rounds to
    // exactly 1. The chord between two unit vectors is 2 * sin(theta / 2), which
    // is theta to a part in 10^8 at this scale, and it is computed from
    // differences that keep their digits. The same quarter-texel movement is
    // 1.5e-8 as a squared chord, six digits clear of zero.
    //
    // A direction that is zero or not finite makes the chord not a number, every
    // comparison against it false, and the answer "not stale". That keeps the
    // last good fit instead of baking one from a direction that has none.
    pub fn stale(self: *const SunShadowFit, sun_dir: Vec3) bool {
        const length = @sqrt(@reduce(.Add, sun_dir * sun_dir));
        const offset = sun_dir / @as(Vec3, @splat(length)) - self.sun_dir;
        return @reduce(.Add, offset * offset) > self.resample_chord_squared;
    }
};

// Runtime look controls for sun shadows, authored per scene rather than baked
// into the fit: none of them changes the map, only how it is read.
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

    pub fn clampedStrength(self: SunShadowSettings) f32 {
        return std.math.clamp(self.strength, 0, 1);
    }

    pub fn normalOffsetWorld(self: SunShadowSettings, fit: *const SunShadowFit) f32 {
        return self.normal_offset_texels * fit.texel_world_size;
    }
};
