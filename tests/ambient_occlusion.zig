const std = @import("std");
const zm = @import("zmath");
const scene = @import("lenore-scene");

const testing = std.testing;
const Reconstruction = scene.AmbientOcclusionReconstruction;
const Settings = scene.AmbientOcclusionSettings;

// A camera's own projection at an aspect that is not one, so swapped axes fail.
const aspect: f32 = 16.0 / 10.0;
const camera: scene.Camera = .{ .projection = .{ .perspective = .{ .fov_y = 1.2, .near = 0.05, .far = 300 } } };

fn project(matrix: zm.Mat, point: [3]f32) [3]f32 {
    const clip = zm.mul(zm.f32x4(point[0], point[1], point[2], 1), matrix);
    return .{ clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3] };
}

test "a point sent through the projection comes back from its depth and position" {
    const matrix = try camera.projectionMatrix(aspect);
    const reconstruction = try Reconstruction.compute(matrix);
    const points = [_][3]f32{
        .{ 0, 0, -1 },    .{ 0.3, -0.2, -0.7 },   .{ -4, 2.5, -9 },
        .{ 12, -6, -40 }, .{ 0.01, 0.02, -0.06 },
    };
    for (points) |point| {
        const ndc = project(matrix, point);
        try testing.expect(ndc[2] >= 0 and ndc[2] <= 1);
        const back = reconstruction.viewPoint(ndc[0], ndc[1], ndc[2]);
        // Relative to the distance: depth precision falls with it.
        const scale = -point[2];
        for (back, point) |got, want| try testing.expectApproxEqAbs(want, got, 1e-4 * scale);
    }
}

test "the near and far planes are the ends of the depth range" {
    const reconstruction = try Reconstruction.compute(try camera.projectionMatrix(aspect));
    try testing.expectApproxEqRel(@as(f32, 0.05), reconstruction.distance(0), 1e-5);
    try testing.expectApproxEqRel(@as(f32, 300), reconstruction.distance(1), 1e-3);
}

test "a world unit at unit distance spans the pixels the field of view gives it" {
    const reconstruction = try Reconstruction.compute(try camera.projectionMatrix(aspect));
    // The full height is 2 tan(fov / 2) world units at unit distance.
    const expected = 800 / (2 * @tan(@as(f32, 1.2) / 2));
    try testing.expectApproxEqRel(expected, reconstruction.pixelsPerUnit(800), 1e-5);
}

test "an orthographic or degenerate matrix is refused" {
    try testing.expectError(error.NotPerspective, Reconstruction.compute(zm.orthographicRh(4, 4, 0.1, 10)));
    try testing.expectError(error.NotPerspective, Reconstruction.compute(zm.identity()));
}

test "off, unusable or excessive settings read as off or are held" {
    try testing.expectEqual(@as(f32, 0), (Settings{}).clampedIntensity());
    try testing.expectEqual(@as(f32, 1), (Settings{ .enabled = true }).clampedIntensity());
    try testing.expectEqual(@as(f32, 0), (Settings{ .enabled = true, .intensity = std.math.nan(f32) }).clampedIntensity());
    try testing.expectEqual(@as(f32, 16), (Settings{ .enabled = true, .intensity = 1000 }).clampedIntensity());
    try testing.expectEqual(@as(?f32, null), (Settings{ .radius = 0 }).usableRadius());
    try testing.expectEqual(@as(?f32, null), (Settings{ .radius = std.math.inf(f32) }).usableRadius());
    try testing.expectEqual(@as(?f32, 0.5), (Settings{}).usableRadius());
}
