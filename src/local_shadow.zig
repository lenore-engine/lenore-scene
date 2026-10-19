// The perspective view-projections a point or spot light's shadow is baked
// through, one per layer of a layered depth image: one layer for a spot, six
// for a point light, a face per axis. And which layers each light gets when
// the image has room for fewer than were asked for.
//
// Conventions as in sun_shadow.zig: zmath transforms a row vector, so clip space
// is `mul(position, view_proj)`; the depth range is Vulkan's, z' in [0, 1]; and
// no matrix carries a Y flip, so the bake and the lookup both have to go
// through the same one.
//
// A point light's six faces are separate layers read through their own
// matrices, not a cube image sampled by direction. The lookup picks the face by
// the largest axis of the direction from the light, in `face_axes` order, and
// projects through that face's matrix, so which way each face is turned about
// its axis is this file's choice alone and never has to match the cube face
// table of the Vulkan specification.

const std = @import("std");
const zm = @import("zmath");
const resources = @import("lenore-resources");
const light_module = @import("light.zig");
const vec = @import("vec.zig");

const Light = light_module.Light;
const Vec3 = resources.Vec3;

// Texels kept clear on each side of a layer, as the sun's fit keeps them: a 2x2
// comparison footprint reaches one texel, and the normal offset moves the
// lookup by one more.
//
// A point light needs it most. A direction on the edge between two faces is
// read from whichever the largest-axis test picks, and with the margin it is
// still two texels inside that face rather than on the border, which the
// sampler reads as lit.
const fit_margin_texels: f32 = 2;

// The nearest a caster may be to the light and still be drawn into its map.
// Godot uses this distance for both kinds, `z_near = MIN(0.025f, radius)`, in
// RendererSceneCull::_light_instance_update_shadow,
// servers/rendering/renderer_scene_cull.cpp:2483 (point) and :2574 (spot), at
// godotengine/godot 81ab961f9106.
//
// It is also why a lamp's own shade has to stand clear of the light, or be
// left out of the casters: anything nearer than this is clipped from the map,
// and anything just beyond it shadows the whole room.
pub const near: f32 = 0.025;

// A range at or below this leaves no depth between the near plane and itself.
// zmath 0.11.0-dev `perspectiveFovRh` asserts that far and near differ by more
// than 0.001, and ReleaseFast removes that assert rather than reporting it; at
// twice the near distance they differ by 0.025.
const min_range: f32 = 2 * near;

// DECIDE: the widest spot cone that casts a shadow, from the axis. One
// perspective projection cannot reach 90 degrees, and towards it the texels at
// the rim stretch by the square of the secant: at 80 degrees a rim texel covers
// about 33 times the area of a central one. A wider spot is better lit as a
// point light with a cone, which costs six layers instead of one. Closed by
// looking at the widest lamp a game actually places.
const max_spot_outer: f32 = std.math.degreesToRadians(80);

// zmath 0.11.0-dev `perspectiveFovRh` asserts that the sine of half its field
// of view differs from zero by more than 0.001. A cone this narrow is a beam a
// millimetre wide at a metre and lights nothing worth shadowing.
const min_spot_outer: f32 = 0.002;

// Above this, a spot axis is too close to the world up axis for the view basis
// to be well conditioned, and the fit turns its up to the world X axis. The
// same limit and the same reasoning as the sun's fit; it only rotates the map
// about the axis, which cancels between the bake and the lookup.
const up_fallback_limit: f32 = 0.99;

// The square of the shortest axis treated as a direction, as in light.zig.
const min_direction_length_squared: f32 = 1e-12;

// The point-light faces in layer order. The lookup selects face `2 * axis +
// (component < 0)` for the direction's largest component, so this order is a
// contract with the shader.
pub const face_axes = [6]Vec3{
    .{ 1, 0, 0 }, .{ -1, 0, 0 },
    .{ 0, 1, 0 }, .{ 0, -1, 0 },
    .{ 0, 0, 1 }, .{ 0, 0, -1 },
};

// Any up that is not along the face's own axis serves; see the header.
const face_ups = [6]Vec3{
    .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 },
};

pub const LocalShadowError = error{
    // A directional light has no position to project from. Its shadow is the
    // sun's fit.
    NotLocal,

    // The range is not above `min_range`, or is not finite.
    RangeTooShort,

    // The spot's outer angle is outside what one projection covers.
    ConeOutOfRange,

    // The position, or a spot's axis, is not finite, or the axis has no
    // length. `Light`'s constructors refuse both, but a scene light can be
    // written as a literal, and `lookToRh` would turn either into a matrix of
    // not-a-numbers.
    DegenerateLight,
};

pub const LocalShadowFit = struct {
    // The first `layers` are used: the spot's one, or the point light's six in
    // `face_axes` order.
    view_proj: [6]zm.Mat,
    layers: u32,

    // World units across one texel at unit distance along the view axis. The
    // normal offset is authored in texels like the sun's, and a perspective
    // texel grows with distance, so the lookup multiplies this by the
    // fragment's distance from the light.
    texel_slope: f32,

    // `light` is in world space, as `Light.placed` leaves it.
    pub fn compute(light: Light, map_size: u32) LocalShadowError!LocalShadowFit {
        const resolution: f32 = @floatFromInt(map_size);
        std.debug.assert(resolution > 2 * fit_margin_texels);
        // The projection grows to keep the margin outside what it has to
        // cover, as the sun's sphere does.
        const slack = resolution / (resolution - 2 * fit_margin_texels);

        switch (light.kind) {
            .directional => return error.NotLocal,
            .point => |source| {
                try checkRange(source.range);
                if (!vec.finite(source.position)) return error.DegenerateLight;
                // A face covers a tangent of one either side of its axis,
                // which is the 90 degrees that tiles the sphere.
                const tan_half = slack;
                const proj = perspective(tan_half, source.range);
                const eye = point4(source.position);
                var fit: LocalShadowFit = .{
                    .view_proj = undefined,
                    .layers = 6,
                    .texel_slope = 2 * tan_half / resolution,
                };
                for (&fit.view_proj, face_axes, face_ups) |*matrix, axis, up| {
                    matrix.* = zm.mul(zm.lookToRh(eye, direction4(axis), direction4(up)), proj);
                }
                return fit;
            },
            .spot => |source| {
                try checkRange(source.range);
                if (!vec.finite(source.position) or !vec.finite(source.direction)) return error.DegenerateLight;
                if (@reduce(.Add, source.direction * source.direction) < min_direction_length_squared)
                    return error.DegenerateLight;
                // Light's constructor keeps the cosine in [0, 1], but a scene
                // light can be written as a literal, and `acos` past one is
                // not a number, which no comparison below would catch.
                const outer = std.math.acos(std.math.clamp(source.cos_outer, -1, 1));
                if (!(outer >= min_spot_outer and outer <= max_spot_outer))
                    return error.ConeOutOfRange;

                const tan_half = @tan(outer) * slack;
                const up = if (@abs(source.direction[1]) > up_fallback_limit)
                    Vec3{ 1, 0, 0 }
                else
                    Vec3{ 0, 1, 0 };
                const view = zm.lookToRh(point4(source.position), direction4(source.direction), direction4(up));
                var fit: LocalShadowFit = .{
                    .view_proj = @splat(zm.identity()),
                    .layers = 1,
                    .texel_slope = 2 * tan_half / resolution,
                };
                fit.view_proj[0] = zm.mul(view, perspective(tan_half, source.range));
                return fit;
            },
        }
    }
};

// Gives each fit its first layer in an image of `capacity` layers, in the
// order the fits come, and null to one that does not fit in what is left.
// Returns the layers used.
//
// Order is the priority: the caller puts first the lights whose shadows matter
// most. A point light that does not fit does not stop a spot after it from
// taking the one layer that is left, so a later light can cast while an
// earlier one does not; the alternative leaves layers empty to keep an order
// nobody sees.
pub fn assignLayers(fits: []const LocalShadowFit, capacity: u32, first_layer: []?u32) u32 {
    std.debug.assert(first_layer.len == fits.len);
    var used: u32 = 0;
    for (fits, first_layer) |fit, *first| {
        if (fit.layers > capacity - used) {
            first.* = null;
            continue;
        }
        first.* = used;
        used += fit.layers;
    }
    return used;
}

// Square, from `near` to the light's range.
fn perspective(tan_half: f32, range: f32) zm.Mat {
    return zm.perspectiveFovRh(2 * std.math.atan(tan_half), 1, near, range);
}

fn checkRange(range: f32) LocalShadowError!void {
    if (!(range > min_range and range < std.math.inf(f32))) return error.RangeTooShort;
}

fn point4(v: Vec3) zm.Vec {
    return zm.f32x4(v[0], v[1], v[2], 1);
}

fn direction4(v: Vec3) zm.Vec {
    return zm.f32x4(v[0], v[1], v[2], 0);
}
