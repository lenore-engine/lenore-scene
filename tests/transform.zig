const std = @import("std");
const zm = @import("zmath");
const scene = @import("lenore-scene");

const testing = std.testing;
const Transform = scene.Transform;

const tolerance = 1e-5;

// zmath composes for a row vector: a point is transformed as v * M, per its
// module header. Applying the matrices that way here is what makes these tests
// describe the convention the engine actually composes in, rather than the one a
// reader assumes.
fn apply(matrix: zm.Mat, point: [3]f32) [3]f32 {
    const result = zm.mul(zm.f32x4(point[0], point[1], point[2], 1.0), matrix);
    return .{ result[0], result[1], result[2] };
}

fn expectPoint(expected: [3]f32, actual: [3]f32) !void {
    for (expected, actual) |want, got| try testing.expectApproxEqAbs(want, got, tolerance);
}

fn quarterTurn(axis: zm.Vec) zm.Quat {
    return zm.quatFromNormAxisAngle(axis, 0.5 * std.math.pi);
}

const axis_x = zm.f32x4(1, 0, 0, 0);
const axis_y = zm.f32x4(0, 1, 0, 0);
const axis_z = zm.f32x4(0, 0, 1, 0);

test "identity leaves a point where it is" {
    try expectPoint(.{ 3, -4, 5 }, apply(Transform.identity.modelMatrix(), .{ 3, -4, 5 }));
}

test "the model matrix scales, then rotates, then translates" {
    // Every factor is asymmetric under the swaps this is meant to catch: the
    // scale is non-uniform, so scaling after the rotation would move the point
    // along a different axis, and the translation is not on the rotation axis.
    const transform: Transform = .{
        .translation = zm.f32x4(10, 0, 0, 0),
        .rotation = quarterTurn(axis_z),
        .scale = zm.f32x4(2, 1, 1, 0),
    };

    // (1, 0, 0) scales to (2, 0, 0), turns onto +Y as (0, 2, 0), then shifts.
    // Rotating before scaling would land on (10, 1, 0) instead.
    try expectPoint(.{ 10, 2, 0 }, apply(transform.modelMatrix(), .{ 1, 0, 0 }));
}

test "rotate applies the current orientation first and the delta after it" {
    var transform: Transform = .identity;
    transform.rotation = quarterTurn(axis_x);
    transform.rotate(quarterTurn(axis_y));

    // +Z turns onto -Y about X, and -Y lies on the second axis, so the turn
    // about Y leaves it there. Taking the delta first would send +Z to +X.
    try expectPoint(.{ 0, -1, 0 }, apply(transform.modelMatrix(), .{ 0, 0, 1 }));
}

test "euler angles are applied about Z, then Y, then X" {
    const quarter = 0.5 * std.math.pi;

    var transform: Transform = .identity;
    transform.rotation = scene.rotationFromEulerZyx(.{ quarter, 0, quarter });
    // +X turns onto +Y about Z, then onto +Z about X. Taking X first would
    // leave +X untouched and stop at +Y.
    try expectPoint(.{ 0, 0, 1 }, apply(transform.modelMatrix(), .{ 1, 0, 0 }));

    transform.rotation = scene.rotationFromEulerZyx(.{ quarter, quarter, quarter });
    // +Z is on the rotation axis of the first turn, so it survives it, moves to
    // +X about Y, and stays there about X. Turning about Y before Z would end
    // on +Z instead.
    try expectPoint(.{ 1, 0, 0 }, apply(transform.modelMatrix(), .{ 0, 0, 1 }));
}
