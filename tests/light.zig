const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Light = scene.Light;

const tolerance = 1e-5;
const white = res.Vec3{ 1, 1, 1 };
const nan = std.math.nan(f32);
const inf = std.math.inf(f32);

fn expectVec(expected: [3]f32, actual: res.Vec3) !void {
    inline for (0..3) |axis|
        try testing.expectApproxEqAbs(expected[axis], actual[axis], tolerance);
}

test "a direction is stored unit whatever length it arrived with" {
    const light = try Light.directional(white, 3, .{ 0, -250, 0 });
    try expectVec(.{ 0, -1, 0 }, light.kind.directional);
    try testing.expectEqual(@as(f32, 3), light.intensity);
}

test "a spot keeps its cone as cosines, the inner one larger" {
    const light = try Light.spot(white, 1, .{
        .position = .{ 1, 2, 3 },
        .direction = .{ 0, 0, -1 },
        .range = 10,
        .inner_angle = std.math.pi / 6.0,
        .outer_angle = std.math.pi / 4.0,
    });
    // Cosine falls as the angle opens, so the inner angle carries the larger of
    // the two and a consumer blending between them can subtract in one order.
    try testing.expectApproxEqAbs(@cos(std.math.pi / 6.0), light.kind.spot.cos_inner, tolerance);
    try testing.expectApproxEqAbs(@cos(std.math.pi / 4.0), light.kind.spot.cos_outer, tolerance);
    try testing.expect(light.kind.spot.cos_inner > light.kind.spot.cos_outer);
}

test "a cone whose angles are all but equal keeps a gap between its cosines" {
    // Distinct angles whose cosines are within a whisker of each other. Whoever
    // blends across them divides by the difference, so a gap of zero is what
    // this exists to prevent, and the angles alone do not guarantee one.
    const light = try Light.spot(white, 1, .{
        .position = .{ 0, 0, 0 },
        .direction = .{ 0, 0, -1 },
        .range = 1,
        .inner_angle = 0.7853980,
        .outer_angle = 0.7853985,
    });
    try testing.expect(light.kind.spot.cos_inner - light.kind.spot.cos_outer >= 1e-4);
}

test "a cone outside the format's bounds is refused" {
    const base: Light.SpotParams = .{
        .position = .{ 0, 0, 0 },
        .direction = .{ 0, 0, -1 },
        .range = 1,
        .inner_angle = 0.3,
        .outer_angle = 0.6,
    };
    const cases = [_]struct { inner: f32, outer: f32 }{
        .{ .inner = -0.1, .outer = 0.6 }, // inner below zero
        .{ .inner = 0.6, .outer = 0.6 }, // not strictly ordered
        .{ .inner = 0.7, .outer = 0.6 }, // inverted
        .{ .inner = 0.3, .outer = 1.6 }, // outer past a right angle
        .{ .inner = nan, .outer = 0.6 },
        .{ .inner = 0.3, .outer = nan },
    };
    for (cases) |case| {
        var params = base;
        params.inner_angle = case.inner;
        params.outer_angle = case.outer;
        try testing.expectError(error.InvalidSpotCone, Light.spot(white, 1, params));
    }
    _ = try Light.spot(white, 1, base);
}

test "emission that is negative or not finite is refused" {
    for ([_]res.Vec3{ .{ -1, 0, 0 }, .{ 0, nan, 0 }, .{ 0, 0, inf } }) |colour|
        try testing.expectError(error.InvalidLightEmission, Light.directional(colour, 1, .{ 0, -1, 0 }));

    for ([_]f32{ -1, nan, inf }) |intensity|
        try testing.expectError(error.InvalidLightEmission, Light.directional(white, intensity, .{ 0, -1, 0 }));

    // Zero is not an error: an unlit light is a state a scene may hold.
    _ = try Light.directional(white, 0, .{ 0, -1, 0 });
}

test "a direction with no direction in it is refused" {
    for ([_]res.Vec3{ .{ 0, 0, 0 }, .{ 1e-8, 0, 0 }, .{ nan, 0, 1 }, .{ inf, 0, 0 } }) |direction|
        try testing.expectError(error.DegenerateLightDirection, Light.directional(white, 1, direction));
}

test "a range that is not a positive distance is refused" {
    for ([_]f32{ 0, -5, nan, inf }) |range|
        try testing.expectError(error.InvalidLightRange, Light.point(white, 1, .{ 0, 0, 0 }, range));
}

test "a position that is not finite is refused" {
    try testing.expectError(error.InvalidLightPosition, Light.point(white, 1, .{ 0, nan, 0 }, 5));
    try testing.expectError(error.InvalidLightPosition, Light.spot(white, 1, .{
        .position = .{ inf, 0, 0 },
        .direction = .{ 0, 0, -1 },
        .range = 5,
        .inner_angle = 0.3,
        .outer_angle = 0.6,
    }));
}

test "placing a point light moves it and leaves its range alone" {
    const light = try Light.point(white, 1, .{ 1, 0, 0 }, 10);
    const model = zm.mul(zm.scaling(2, 2, 2), zm.translation(0, 5, 0));
    const world = try light.placed(model);
    // The scale reaches the position, which is a point in the node's space, and
    // stops there: the range is a world distance the transform does not touch.
    try expectVec(.{ 2, 5, 0 }, world.kind.point.position);
    try testing.expectEqual(@as(f32, 10), world.kind.point.range);
}

test "placing a directional light turns it and leaves everything else" {
    // A quarter turn about Z sends -Y to +X.
    const turn = zm.matFromQuat(zm.quatFromNormAxisAngle(zm.f32x4(0, 0, 1, 0), std.math.pi / 2.0));
    const light = try Light.directional(white, 4, .{ 0, -1, 0 });
    const world = try light.placed(turn);
    try expectVec(.{ 1, 0, 0 }, world.kind.directional);
    try testing.expectEqual(@as(f32, 4), world.intensity);

    // Translation does not reach a direction: it is carried with w = 0.
    const moved = try light.placed(zm.translation(100, 200, 300));
    try expectVec(.{ 0, -1, 0 }, moved.kind.directional);
}

test "placing a spot light turns its axis as well as moving it" {
    // The case the offset-and-scale placement it replaces could not express: a
    // rotated node leaves the cone pointing where it was authored.
    const turn = zm.matFromQuat(zm.quatFromNormAxisAngle(zm.f32x4(1, 0, 0, 0), std.math.pi / 2.0));
    const model = zm.mul(turn, zm.translation(0, 3, 0));
    const light = try Light.spot(white, 1, .{
        .position = .{ 0, 0, 2 },
        .direction = .{ 0, 0, -1 },
        .range = 8,
        .inner_angle = 0.3,
        .outer_angle = 0.6,
    });
    const world = try light.placed(model);

    // A quarter turn about X sends +Z to -Y and -Z to +Y. The position is two
    // units along +Z, so the turn brings it to -2 on Y before the translation
    // lifts it to 1: a placement that only translated would leave it at 3.
    try expectVec(.{ 0, 1, 0 }, world.kind.spot.direction);
    try expectVec(.{ 0, 1, 0 }, world.kind.spot.position);
    try testing.expectApproxEqAbs(8.0, world.kind.spot.range, tolerance);

    // The cone itself is unchanged: a rotation does not open or close it.
    try testing.expectEqual(light.kind.spot.cos_inner, world.kind.spot.cos_inner);
    try testing.expectEqual(light.kind.spot.cos_outer, world.kind.spot.cos_outer);
}

test "no scale reaches a range, uniform or not" {
    // KHR_lights_punctual, "Light Shared Properties": a node transform does not
    // change `range`. Exact equality, because the range is not arithmetic on.
    const light = try Light.point(white, 1, .{ 0, 0, 0 }, 4);
    for ([_]zm.Mat{
        zm.scaling(1, 3, 2),
        zm.scaling(0.5, 0.5, 0.5),
        zm.scaling(1000, 1000, 1000),
        zm.scaling(0, 0, 0),
    }) |model| {
        const placed = try light.placed(model);
        try testing.expectEqual(@as(f32, 4), placed.kind.point.range);
    }

    const beam = try Light.spot(white, 1, .{
        .position = .{ 0, 0, 0 },
        .direction = .{ 0, 0, -1 },
        .range = 7,
        .inner_angle = 0.2,
        .outer_angle = 0.4,
    });
    const stretched = try beam.placed(zm.scaling(1, 3, 2));
    try testing.expectEqual(@as(f32, 7), stretched.kind.spot.range);
}

test "placing by a matrix that destroys the light is refused" {
    // A scale of zero no longer names a failure: it puts the light at one point
    // and the range it keeps is the authored one. What is still refused is a
    // placement that produces no position and no direction at all.
    const light = try Light.point(white, 1, .{ 1, 0, 0 }, 4);
    try testing.expectError(error.InvalidLightPosition, light.placed(zm.translation(inf, 0, 0)));
    try testing.expectError(error.InvalidLightPosition, light.placed(zm.scaling(nan, 1, 1)));

    const beam = try Light.directional(white, 1, .{ 0, -1, 0 });
    try testing.expectError(error.DegenerateLightDirection, beam.placed(zm.scaling(1, 0, 1)));
}

test "a derived range is where the falloff reaches the visible threshold" {
    // intensity / r^2 = 0.005, so an intensity of 5 reaches 1000 and one of 20
    // reaches four times as far, not twice.
    try testing.expectApproxEqAbs(@sqrt(1000.0), Light.rangeFor(5), 1e-3);
    try testing.expectApproxEqAbs(2 * @sqrt(1000.0), Light.rangeFor(20), 1e-3);
    try testing.expectEqual(@as(f32, 0), Light.rangeFor(-3));
}

test "the sun's disk defaults to the one in the sky and refuses nonsense" {
    const default: scene.SunAppearance = .{};
    try default.validate();
    try testing.expectApproxEqAbs(0.00464, default.angular_radius, 1e-5);

    for ([_]f32{ 0, -1, nan, std.math.pi }) |radius| {
        const appearance: scene.SunAppearance = .{ .angular_radius = radius };
        try testing.expectError(error.InvalidSunAppearance, appearance.validate());
    }
    for ([_]f32{ -0.5, nan, inf }) |scale| {
        const appearance: scene.SunAppearance = .{ .disk_radiance_scale = scale };
        try testing.expectError(error.InvalidSunAppearance, appearance.validate());
    }
}
