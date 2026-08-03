const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Frustum = scene.Frustum;

// An eye ten units up the +Z axis looking back at the origin, with a 90 degree
// vertical field of view and a 2:1 aspect. Verified against zmath 0.11.0-dev:
// the projection carries 0.5 in [0][0] and 1 in [1][1], so at the origin, ten
// units away, the frustum reaches ten up and down and twenty left and right.
//
// Two asymmetries are deliberate and both were found by mutation. The aspect
// tells the horizontal planes from the vertical ones. The eye being off the
// origin gives the lateral planes a non-zero distance term, without which the
// normalization can be written over four lanes instead of three and no test
// notices.
const eye_z = 10.0;
const near = 0.1;
const far = 100.0;

fn testFrustum() Frustum {
    const view = zm.lookAtRh(zm.f32x4(0, 0, eye_z, 1), zm.f32x4(0, 0, 0, 1), zm.f32x4(0, 1, 0, 0));
    const projection = zm.perspectiveFovRh(0.5 * std.math.pi, 2.0, near, far);
    return .fromViewProj(zm.mul(view, projection));
}

// A small box around a point, for the cases that are about where a thing is
// rather than about how big it is.
fn speck(centre: [3]f32) res.Aabb {
    const half: res.Vec3 = @splat(0.5);
    const at: res.Vec3 = centre;
    return .{ .min = at - half, .max = at + half };
}

fn box(min: [3]f32, max: [3]f32) res.Aabb {
    return .{ .min = min, .max = max };
}

test "a box in front of the eye is visible and one behind it is not" {
    const frustum = testFrustum();
    try testing.expect(frustum.intersectsAabb(speck(.{ 0, 0, 0 })));
    try testing.expect(!frustum.intersectsAabb(speck(.{ 0, 0, eye_z + 10 })));
}

test "the horizontal planes are wider than the vertical ones" {
    const frustum = testFrustum();
    // Ten units from the eye the bounds are ten vertically and twenty
    // horizontally, so the same offset falls outside on one axis and inside on
    // the other. Extracting the planes from the wrong column swaps both answers.
    try testing.expect(!frustum.intersectsAabb(speck(.{ 0, 15, 0 })));
    try testing.expect(frustum.intersectsAabb(speck(.{ 15, 0, 0 })));
}

test "all four lateral planes reject" {
    const frustum = testFrustum();
    // One case per plane, because a plane extracted as a duplicate of another
    // still rejects everything the one it duplicates rejects. Only the side it
    // alone is responsible for tells them apart.
    try testing.expect(!frustum.intersectsAabb(speck(.{ -25, 0, 0 })));
    try testing.expect(!frustum.intersectsAabb(speck(.{ 25, 0, 0 })));
    try testing.expect(!frustum.intersectsAabb(speck(.{ 0, -15, 0 })));
    try testing.expect(!frustum.intersectsAabb(speck(.{ 0, 15, 0 })));
}

test "the near plane sits at the Vulkan depth range, not the OpenGL one" {
    const frustum = testFrustum();
    // Between the eye and the near plane. Reading the near plane as column 3
    // plus column 2, which is the form the [-w, w] depth range wants, puts this
    // box inside instead.
    try testing.expect(!frustum.intersectsAabb(box(
        .{ -0.001, -0.001, eye_z - 0.061 },
        .{ 0.001, 0.001, eye_z - 0.059 },
    )));
    try testing.expect(frustum.intersectsAabb(box(
        .{ -0.001, -0.001, eye_z - 0.201 },
        .{ 0.001, 0.001, eye_z - 0.199 },
    )));
}

test "the far plane bounds the range" {
    const frustum = testFrustum();
    try testing.expect(!frustum.intersectsAabb(speck(.{ 0, 0, eye_z - 200 })));
    try testing.expect(frustum.intersectsAabb(speck(.{ 0, 0, eye_z - 99 })));
}

test "a box straddling a plane is visible on both sides of the frustum" {
    const frustum = testFrustum();
    // The bound is twenty and each box runs from eighteen to twenty-five, so
    // part of it is inside and it must be drawn. This is what one corner per
    // plane buys: the corner furthest along the inward normal is the near end on
    // the right and the far end on the left, so testing a fixed corner, or the
    // centre, rejects one of the two.
    try testing.expect(frustum.intersectsAabb(box(.{ 18, -1, -1 }, .{ 25, 1, 1 })));
    try testing.expect(frustum.intersectsAabb(box(.{ -25, -1, -1 }, .{ -18, 1, 1 })));
    // Vertically as well, where the bound is ten.
    try testing.expect(frustum.intersectsAabb(box(.{ -1, 8, -1 }, .{ 1, 15, 1 })));
    try testing.expect(frustum.intersectsAabb(box(.{ -1, -15, -1 }, .{ 1, -8, 1 })));
}

test "a box swallowing the whole frustum is visible" {
    const frustum = testFrustum();
    // No corner of it is inside and no corner of the frustum is inside a face
    // of it in any useful sense; the plane test still admits it, because no
    // plane has the box entirely behind it.
    try testing.expect(frustum.intersectsAabb(box(.{ -1000, -1000, -1000 }, .{ 1000, 1000, 1000 })));
}
