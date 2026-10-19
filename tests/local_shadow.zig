const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Light = scene.Light;
const LocalShadowFit = scene.LocalShadowFit;

const tolerance = 1e-4;
const map_size: u32 = 1024;
const margin_texels = 2.0;
const near = 0.025;

// The largest a projected coordinate may be and still leave the margin clear.
const resolution: f32 = @floatFromInt(map_size);
const inside = (resolution - 2 * margin_texels) / resolution;

// Off the origin, so a fit that builds its view at the origin is wrong
// everywhere below.
const source: res.Vec3 = .{ 2, 2.6, -3 };
const range: f32 = 6;

fn unit(v: [3]f32) res.Vec3 {
    const as_vector: res.Vec3 = v;
    return as_vector / @as(res.Vec3, @splat(@sqrt(@reduce(.Add, as_vector * as_vector))));
}

fn cross(a: res.Vec3, b: res.Vec3) res.Vec3 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

fn at(direction: [3]f32, distance: f32) res.Vec3 {
    return source + unit(direction) * @as(res.Vec3, @splat(distance));
}

// Device coordinates of `point` through `matrix`, after the divide.
fn project(matrix: zm.Mat, point: res.Vec3) res.Vec3 {
    const clip = zm.mul(zm.f32x4(point[0], point[1], point[2], 1), matrix);
    std.debug.assert(clip[3] > 0);
    return .{ clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3] };
}

// The face the shader picks: the largest component, negative second.
fn faceOf(direction: res.Vec3) usize {
    const magnitude: [3]f32 = @abs(direction);
    const component: [3]f32 = direction;
    var axis: usize = 0;
    if (magnitude[1] > magnitude[axis]) axis = 1;
    if (magnitude[2] > magnitude[axis]) axis = 2;
    return 2 * axis + @intFromBool(component[axis] < 0);
}

fn spotLight(direction: [3]f32, outer: f32) !Light {
    return Light.spot(.{ 1, 1, 1 }, 10, .{
        .position = source,
        .direction = direction,
        .range = range,
        .inner_angle = outer / 2,
        .outer_angle = outer,
    });
}

const cone = std.math.degreesToRadians(50);

test "a spot's axis lands in the middle, its near and far on the depth range's ends" {
    const fit = try LocalShadowFit.compute(try spotLight(.{ 0.3, -1, 0.2 }, cone), map_size);
    try testing.expectEqual(@as(u32, 1), fit.layers);
    const axis: [3]f32 = .{ 0.3, -1, 0.2 };
    const middle = project(fit.view_proj[0], at(axis, 3));
    try testing.expectApproxEqAbs(@as(f32, 0), middle[0], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 0), middle[1], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 0), project(fit.view_proj[0], at(axis, near))[2], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 1), project(fit.view_proj[0], at(axis, range))[2], tolerance);
}

test "the rim of a spot's cone stays the margin inside its layer" {
    const axis = unit(.{ 0.3, -1, 0.2 });
    const fit = try LocalShadowFit.compute(try spotLight(.{ axis[0], axis[1], axis[2] }, cone), map_size);
    // Four directions perpendicular to the axis. The rim is a circle in the
    // layer, so it is its radius that the margin bounds.
    const side = unit(cross(axis, .{ 1, 0, 0 }));
    const other = cross(axis, side);
    for ([_]res.Vec3{ side, -side, other, -other }) |across| {
        const rim = axis * @as(res.Vec3, @splat(@cos(cone))) + across * @as(res.Vec3, @splat(@sin(cone)));
        const projected = project(fit.view_proj[0], source + rim * @as(res.Vec3, @splat(2)));
        const radius = @sqrt(projected[0] * projected[0] + projected[1] * projected[1]);
        try testing.expectApproxEqAbs(inside, radius, tolerance);
    }
}

test "a ceiling lamp pointing straight down gets a finite matrix" {
    const fit = try LocalShadowFit.compute(try spotLight(.{ 0, -1, 0 }, cone), map_size);
    for (fit.view_proj[0]) |row| {
        const lanes: [4]f32 = row;
        for (lanes) |lane| try testing.expect(std.math.isFinite(lane));
    }
    const below = project(fit.view_proj[0], at(.{ 0, -1, 0 }, 2));
    try testing.expectApproxEqAbs(@as(f32, 0), below[0], tolerance);
    try testing.expectApproxEqAbs(@as(f32, 0), below[1], tolerance);
}

test "every direction from a point light lands inside the face the lookup picks" {
    const fit = try LocalShadowFit.compute(try Light.point(.{ 1, 1, 1 }, 10, source, range), map_size);
    try testing.expectEqual(@as(u32, 6), fit.layers);
    // The axes, the diagonals where three faces meet, the edges where two do,
    // and some in between.
    const directions = [_][3]f32{
        .{ 1, 0, 0 },        .{ -1, 0, 0 }, .{ 0, 1, 0 },    .{ 0, -1, 0 },       .{ 0, 0, 1 },
        .{ 0, 0, -1 },       .{ 1, 1, 1 },  .{ -1, -1, -1 }, .{ 1, -1, 1 },       .{ -1, 1, -1 },
        .{ 1, 1, 0 },        .{ 0, -1, 1 }, .{ -1, 0, -1 },  .{ 0.3, -0.9, 0.4 }, .{ -0.7, 0.2, 0.69 },
        .{ 0.01, 0.02, -1 },
    };
    for (directions) |direction| {
        const face = faceOf(unit(direction));
        for ([_]f32{ 0.1, 1, range * 0.99 }) |distance| {
            const projected = project(fit.view_proj[face], at(direction, distance));
            try testing.expect(@abs(projected[0]) <= inside + tolerance);
            try testing.expect(@abs(projected[1]) <= inside + tolerance);
            try testing.expect(projected[2] > 0 and projected[2] < 1);
        }
    }
}

test "the faces are in the order the lookup indexes them" {
    for (scene.shadow_face_axes, 0..) |axis, index| try testing.expectEqual(index, faceOf(axis));
}

test "one texel at unit distance is the layer's width over its resolution" {
    const fit = try LocalShadowFit.compute(try Light.point(.{ 1, 1, 1 }, 10, source, range), map_size);
    const slack = resolution / (resolution - 2 * margin_texels);
    try testing.expectApproxEqAbs(2 * slack / resolution, fit.texel_slope, 1e-7);
}

test "a fit refuses what it cannot project" {
    try testing.expectError(error.NotLocal, LocalShadowFit.compute(try Light.directional(.{ 1, 1, 1 }, 1, .{ 0, -1, 0 }), map_size));
    try testing.expectError(error.RangeTooShort, LocalShadowFit.compute(try Light.point(.{ 1, 1, 1 }, 1, source, 0.04), map_size));
    try testing.expectError(error.ConeOutOfRange, LocalShadowFit.compute(try spotLight(.{ 0, -1, 0 }, std.math.degreesToRadians(85)), map_size));

    var nowhere = try Light.point(.{ 1, 1, 1 }, 1, source, range);
    nowhere.kind.point.position[0] = std.math.nan(f32);
    try testing.expectError(error.DegenerateLight, LocalShadowFit.compute(nowhere, map_size));
    var aimless = try spotLight(.{ 0, -1, 0 }, cone);
    aimless.kind.spot.direction = .{ 0, 0, 0 };
    try testing.expectError(error.DegenerateLight, LocalShadowFit.compute(aimless, map_size));
}

test "layers go in order, and a light that does not fit leaves room for one that does" {
    const point = try LocalShadowFit.compute(try Light.point(.{ 1, 1, 1 }, 10, source, range), map_size);
    const spot = try LocalShadowFit.compute(try spotLight(.{ 0, -1, 0 }, cone), map_size);
    const fits = [_]LocalShadowFit{ point, spot, point, spot };
    var first: [fits.len]?u32 = undefined;
    try testing.expectEqual(@as(u32, 8), scene.assignShadowLayers(&fits, 8, &first));
    try testing.expectEqualSlices(?u32, &.{ 0, 6, null, 7 }, &first);
}
