const zm = @import("zmath");
const resources = @import("lenore-resources");

const Aabb = resources.Aabb;
const Sphere = resources.Sphere;

// The axis-aligned box a mesh's object-space box occupies once placed.
//
// Culling, picking and the sun-shadow fit all start here. The box rather than
// the sphere, because this is exact for every transform and a scaled sphere
// radius is not: the largest stretch a matrix applies to a direction is its
// largest singular value, and the longest row of the linear part bounds that
// from below. A sheared chain, which two non-uniform scales with a rotation
// between them produce and which glTF section 5.25.4 also lets a node state
// outright, would then get a sphere too small and cull geometry that is on
// screen.
//
// Minimising a linear function over a box is done one output axis at a time:
// each input axis contributes its low or its high face depending on the sign of
// the coefficient, which is what taking the elementwise minimum and maximum of
// the two products does. Exact for any linear map, shear included. Godot culls
// with the same construction, `Transform3D::xform(const AABB &)` in
// `core/math/transform_3d.h`, written per component rather than per row.
//
// DECIDE: correctness picked the box, cost has not been measured. This runs per
// object per frame, and against the scaled-radius path it trades three products
// and two elementwise selects for one length, then hands the frustum six planes
// to test against a box rather than a point. The box should also reject more,
// because it is tighter than a sphere for anything longer than it is wide, so
// the two effects push opposite ways and neither is a paper question. What
// closes this: frame times over a scene with tens of thousands of instances,
// box against sphere, with the draw count recorded beside the time. Until then
// this stands on being the one that cannot be wrong.
pub fn worldAabb(local: Aabb, model: zm.Mat) Aabb {
    // The translation row seeds both bounds. Its fourth lane is 1 and takes no
    // part in a position, so it is dropped here rather than masked later.
    var low = zm.f32x4(model[3][0], model[3][1], model[3][2], 0);
    var high = low;

    // Row j carries where the j-th basis vector lands, so it is the coefficient
    // of the j-th input axis in all three outputs at once.
    inline for (0..3) |axis| {
        const row = model[axis];
        const from_low = row * zm.f32x4s(local.min[axis]);
        const from_high = row * zm.f32x4s(local.max[axis]);
        low += @min(from_low, from_high);
        high += @max(from_low, from_high);
    }

    return .{
        .min = .{ low[0], low[1], low[2] },
        .max = .{ high[0], high[1], high[2] },
    };
}

// The smallest box containing both. The sun-shadow fit folds this over the
// static scene to get one volume to enclose.
pub fn unionAabb(a: Aabb, b: Aabb) Aabb {
    return .{ .min = @min(a.min, b.min), .max = @max(a.max, b.max) };
}

// The sphere through a box's corners, for the one consumer that needs a sphere
// rather than a box: an orthographic shadow fit, which wants a volume with no
// preferred axis so that the projection does not change size as the sun moves.
//
// Circumscribed rather than inscribed, so it contains the box.
pub fn sphereAroundAabb(box: Aabb) Sphere {
    const half = (box.max - box.min) * @as(resources.Vec3, @splat(0.5));
    return .{
        .centre = box.min + half,
        .radius = @sqrt(@reduce(.Add, half * half)),
    };
}
