const std = @import("std");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const probes = scene.probe_grid;
const ProbeGrid = probes.ProbeGrid;

fn bounds(min: [3]f32, max: [3]f32) res.Aabb {
    return .{ .min = min, .max = max };
}

test "a grid covers its bounds, centred, with a probe more than cells per axis" {
    const grid = try ProbeGrid.fit(bounds(.{ -1, 0, 2 }, .{ 1, 3, 2.9 }), 0.5);
    try testing.expectEqual([3]u32{ 5, 7, 3 }, grid.counts);
    // X: 2 m is exactly four cells; Z: 0.9 m rounds up to two, centred, so the
    // lattice overhangs the bounds by 0.05 m at each side.
    try testing.expectApproxEqAbs(@as(f32, -1), grid.origin[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.95), grid.origin[2], 1e-6);
    const last = grid.position(grid.count() - 1);
    try testing.expectApproxEqAbs(@as(f32, 1), last[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3), last[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.95), last[2], 1e-5);
}

test "an index and a cell are the same probe both ways" {
    const grid = try ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ 3, 2, 1 }), 0.5);
    for (0..grid.count()) |probe| {
        const cell = grid.cellOf(@intCast(probe));
        try testing.expectEqual(@as(u32, @intCast(probe)), grid.index(cell));
    }
}

test "a point is located in the cell around it, and clamped outside the grid" {
    const grid = try ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ 2, 2, 2 }), 0.5);
    const inside = grid.locate(.{ 0.6, 1.25, 1.999 });
    try testing.expectEqual([3]u32{ 1, 2, 3 }, inside.base);
    try testing.expectApproxEqAbs(@as(f32, 0.2), inside.fraction[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), inside.fraction[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.998), inside.fraction[2], 1e-5);
    // The far face is the last cell at fraction one, not a cell past the grid.
    const far = grid.locate(.{ 5, -3, 2 });
    try testing.expectEqual([3]u32{ 3, 0, 3 }, far.base);
    try testing.expectEqual(@as(f32, 1), far.fraction[0]);
    try testing.expectEqual(@as(f32, 0), far.fraction[1]);
}

test "a grid refuses what it cannot be" {
    try testing.expectError(error.InvalidSpacing, ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ 1, 1, 1 }), 0));
    try testing.expectError(error.InvalidSpacing, ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ 1, 1, 1 }), std.math.nan(f32)));
    try testing.expectError(error.DegenerateBounds, ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ 1, 0, 1 }), 0.5));
    try testing.expectError(error.DegenerateBounds, ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ std.math.inf(f32), 1, 1 }), 0.5));
    try testing.expectError(error.TooManyProbes, ProbeGrid.fit(bounds(.{ 0, 0, 0 }, .{ 100, 100, 100 }), 0.5));
}

test "Fibonacci directions are unit and balanced over the sphere" {
    const n = 256;
    var sum: res.Vec3 = @splat(0);
    for (0..n) |i| {
        const direction = probes.fibonacciDirection(@intCast(i), n);
        try testing.expectApproxEqAbs(@as(f32, 1), @sqrt(@reduce(.Add, direction * direction)), 1e-5);
        sum += direction;
    }
    inline for (0..3) |axis| try testing.expectApproxEqAbs(@as(f32, 0), sum[axis] / n, 1e-2);
}

// Integrated by the same Fibonacci points the rays use, so this also says the
// rays are dense enough to project with.
fn project(comptime n: u32, radiance: fn (res.Vec3) f32) [probes.sh_count]f32 {
    var coefficients: [probes.sh_count]f32 = @splat(0);
    for (0..n) |i| {
        const direction = probes.fibonacciDirection(@intCast(i), n);
        const basis = probes.shBasis(direction);
        const value = radiance(direction);
        for (&coefficients, basis) |*coefficient, b| coefficient.* += value * b;
    }
    for (&coefficients) |*coefficient| coefficient.* *= 4 * std.math.pi / @as(f32, n);
    return coefficients;
}

test "the nine harmonics are orthonormal over the sphere" {
    const n = 4096;
    for (0..probes.sh_count) |i| {
        for (0..probes.sh_count) |j| {
            var sum: f32 = 0;
            for (0..n) |k| {
                const basis = probes.shBasis(probes.fibonacciDirection(@intCast(k), n));
                sum += basis[i] * basis[j];
            }
            const integral = sum * 4 * std.math.pi / n;
            try testing.expectApproxEqAbs(@as(f32, if (i == j) 1 else 0), integral, 1e-2);
        }
    }
}

fn uniform(_: res.Vec3) f32 {
    return 1;
}

fn upward(direction: res.Vec3) f32 {
    return direction[1];
}

test "irradiance from projected radiance is the cosine integral" {
    // Unit radiance from everywhere: the cosine over a hemisphere integrates
    // to pi whichever way the surface faces.
    const flat = project(1024, uniform);
    for ([_]res.Vec3{ .{ 0, 1, 0 }, .{ 1, 0, 0 }, .{ 0, -0.6, 0.8 } }) |normal| {
        try testing.expectApproxEqAbs(@as(f32, std.math.pi), probes.irradiance(flat, normal), 1e-2);
    }
    // Radiance linear in the height: the first band convolves to 2 pi / 3 of
    // the normal's height, the factor ambient lighting in scene.slang uses
    // over pi.
    const linear = project(1024, upward);
    for ([_]res.Vec3{ .{ 0, 1, 0 }, .{ 0.6, 0.8, 0 }, .{ 0, -1, 0 } }) |normal| {
        try testing.expectApproxEqAbs(2 * std.math.pi / 3.0 * normal[1], probes.irradiance(linear, normal), 1e-2);
    }
}

test "octahedral coordinates come back to the direction they encode" {
    for (0..200) |i| {
        const direction = probes.fibonacciDirection(@intCast(i), 200);
        const coordinate = probes.octEncode(direction);
        try testing.expect(@abs(coordinate[0]) <= 1 and @abs(coordinate[1]) <= 1);
        const back = probes.octDecode(coordinate);
        inline for (0..3) |axis| try testing.expectApproxEqAbs(direction[axis], back[axis], 1e-5);
    }
    // The poles and the equator's seam.
    for ([_]res.Vec3{ .{ 0, 1, 0 }, .{ 0, -1, 0 }, .{ 1, 0, 0 }, .{ 0, 0, -1 } }) |direction| {
        const back = probes.octDecode(probes.octEncode(direction));
        inline for (0..3) |axis| try testing.expectApproxEqAbs(direction[axis], back[axis], 1e-5);
    }
}
