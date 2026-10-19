const std = @import("std");
const scene = @import("lenore-scene");
const testing = std.testing;

test "split sum mirror limit and invalid inputs" {
    const extreme = try scene.split_sum.integrate(0x1p-126, 0);
    try testing.expectApproxEqAbs(@as(f32, 1), extreme[0] + extreme[1], 0.001);
    const mirror = try scene.split_sum.integrate(1, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1), mirror[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0), mirror[1], 0.0001);
    try testing.expectError(error.InvalidBrdfParameters, scene.split_sum.integrate(0, 0.5));
    try testing.expectError(error.InvalidBrdfParameters, scene.split_sum.integrate(0.5, std.math.nan(f32)));
}

test "split sum is passive and rough normal incidence loses single scatter energy" {
    var previous: f32 = 1.001;
    for ([_]f32{ 0.01, 0.2, 0.4, 0.6, 0.8, 1 }) |roughness| {
        const normal = try scene.split_sum.integrate(1, roughness);
        try testing.expect(normal[0] + normal[1] <= previous);
        previous = normal[0] + normal[1];
        for ([_]f32{ 0.05, 0.2, 0.5, 0.9, 1 }) |no_v| {
            const value = try scene.split_sum.integrate(no_v, roughness);
            try testing.expect(value[0] >= 0 and value[1] >= 0);
            // Finite deterministic quadrature can overshoot the unit bound.
            try testing.expect(value[0] + value[1] <= 1.005);
        }
    }
    const normal = try scene.split_sum.integrate(1, 0.01);
    const grazing = try scene.split_sum.integrate(0.1, 0.01);
    try testing.expect(grazing[1] > normal[1]);
}

test "built table stores scale then bias at texel centres" {
    const table = try testing.allocator.create(scene.split_sum.Table);
    defer testing.allocator.destroy(table);
    scene.split_sum.build(table);
    for ([_]usize{ 0, 32, 64, 127 }) |y| for ([_]usize{ 0, 32, 64, 127 }) |x| {
        const expected = try scene.split_sum.integrate((@as(f32, @floatFromInt(x)) + 0.5) / 128, (@as(f32, @floatFromInt(y)) + 0.5) / 128);
        for (0..2) |c| try testing.expectEqual(@as(f16, @floatCast(expected[c])), table[y][x][c]);
    };
}
