// A pointer position becomes a world ray, and a ray meets a placed box.
//
// The ray is built on the CPU rather than read back from a depth attachment. A
// readback needs transfer usage on the depth image, a staging copy and a fence
// wait, it answers for the frame that has already been presented rather than the
// one under the cursor, and it still cannot name which object a pixel belongs to
// without a second target rendering identifiers. This path runs on a gesture and
// costs nothing per frame.
//
// Precision is the mesh's object-space box, and for a skinned mesh that box is
// the bind pose, the same caveat culling carries.
//
// Conventions, shared with frustum.zig: zmath transforms a row vector. Pointer
// coordinates are in the same pixels the viewport is measured in, with the
// origin at its top left corner and Y growing downward.

const std = @import("std");
const zm = @import("zmath");
const resources = @import("lenore-resources");

const Camera = @import("camera.zig").Camera;
const ProjectionError = @import("camera.zig").ProjectionError;

const Aabb = resources.Aabb;
const Vec3 = resources.Vec3;

pub const Error = ProjectionError || error{
    // The viewport has no area, so no pixel in it names a direction.
    EmptyViewport,

    // The pointer is outside the region the scene is drawn into. Ordinary: a
    // click lands on the panel beside the viewport, not on the scene.
    PointerOutsideViewport,
};

// The pixel region the scene is drawn into. The ray is only as correct as this
// rectangle: it is what the aspect ratio is taken from, so a viewport that is
// not the one rendered gives a ray that misses by the difference between them.
pub const Viewport = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
};

pub const Ray = struct {
    origin: Vec3,

    // Unit length, which is what makes every distance below a world distance.
    direction: Vec3,

    pub fn at(self: Ray, t: f32) Vec3 {
        return self.origin + self.direction * @as(Vec3, @splat(t));
    }
};

// Where a ray meets one box.
pub const Intersection = struct {
    // Distance along the ray to the entry face, in world units.
    t: f32,

    // The ray began inside the box, and `t` is 0 because there is no entry face
    // in front of it. A caller that wants a point on a surface, an orbit pivot
    // for instance, rejects these. A caller that wants an object keeps them and
    // ranks them last, or the box around a room swallows every click inside it.
    inside: bool,
};

// The ray through a pointer position, in world space.
//
// The aspect ratio comes from the viewport rather than from the camera, so the
// ray cannot disagree with the projection about the shape of the target. What
// the caller still owns is passing the rectangle the scene was actually drawn
// into.
pub fn cameraRay(camera: Camera, viewport: Viewport, pointer: [2]f32) Error!Ray {
    if (!(viewport.width > 0) or !(viewport.height > 0)) return error.EmptyViewport;

    const u = (pointer[0] - viewport.x) / viewport.width;
    const v = (pointer[1] - viewport.y) / viewport.height;
    if (!(u >= 0) or u >= 1 or !(v >= 0) or v >= 1) return error.PointerOutsideViewport;

    const projection = try camera.projectionMatrix(viewport.width / viewport.height);

    // The two lanes are the RECIPROCALS of the view-space half extents at unit
    // distance, not the half extents: zmath's `perspectiveFovRh` writes
    // cot(fov/2)/aspect and cot(fov/2) there. So the normalized coordinate is
    // divided by them.
    const half_width_reciprocal = projection[0][0];
    const half_height_reciprocal = projection[1][1];

    const ndc_x = 2 * u - 1;
    const ndc_y = 1 - 2 * v;

    const placement = camera.placement();
    const direction = placement.front +
        placement.right * @as(Vec3, @splat(ndc_x / half_width_reciprocal)) +
        placement.up * @as(Vec3, @splat(ndc_y / half_height_reciprocal));

    // The front component is unit and orthogonal to the other two, so the sum is
    // never shorter than one and never needs a guard before the division.
    const length = @sqrt(@reduce(.Add, direction * direction));
    return .{
        .origin = placement.position,
        .direction = direction / @as(Vec3, @splat(length)),
    };
}

// Ray against one placed box. `model` is the row-vector object-to-world
// transform, and the ray is taken into object space, so rotation and non-uniform
// scale are exact rather than inflated into something that encloses them.
pub fn intersectInstance(ray: Ray, model: zm.Mat, box: Aabb) ?Intersection {
    const inverse = zm.inverse(model);

    // A position carries w = 1 under a row-vector transform, a direction w = 0.
    // The object-space direction is deliberately left unnormalized: its length is
    // the inverse scale, which is exactly what keeps `t` a world distance.
    const origin = zm.mul(zm.f32x4(ray.origin[0], ray.origin[1], ray.origin[2], 1), inverse);
    const direction = zm.mul(zm.f32x4(ray.direction[0], ray.direction[1], ray.direction[2], 0), inverse);
    const object_direction = Vec3{ direction[0], direction[1], direction[2] };

    // zmath returns an all-zero matrix from `inverseDet` when the determinant is
    // zero, which a model matrix with a zero scale has. The direction is then
    // zero as well, and a zero direction slab-tests as lying inside every box it
    // starts within, so an object scaled to nothing would answer every ray.
    if (@reduce(.And, object_direction == @as(Vec3, @splat(0)))) return null;

    return intersectAabb(.{ origin[0], origin[1], origin[2] }, object_direction, box);
}

// The slab test. `direction` need not be unit: `t` comes back in whatever units
// its length is, which is what lets the object-space test above report a world
// distance.
pub fn intersectAabb(origin: Vec3, direction: Vec3, box: Aabb) ?Intersection {
    var enter: f32 = 0;
    var exit: f32 = std.math.floatMax(f32);

    inline for (0..3) |axis| {
        const from = origin[axis];
        const along = direction[axis];
        const low = box.min[axis];
        const high = box.max[axis];

        if (along == 0) {
            // Parallel to this pair of planes, so the ray is between them for
            // its whole length or for none of it. Taken as a branch rather than
            // left to the arithmetic below, where a zero divides to an infinity
            // and a ray starting exactly on the plane multiplies it by a zero.
            if (from < low or from > high) return null;
        } else {
            const reciprocal = 1 / along;
            const first = (low - from) * reciprocal;
            const second = (high - from) * reciprocal;
            enter = @max(enter, @min(first, second));
            exit = @min(exit, @max(first, second));
            if (enter > exit) return null;
        }
    }

    // `enter` starts at zero, so a box entirely behind the origin has already
    // left through `enter > exit`, and what reaches here begins either at an
    // entry face ahead or at the origin itself. The clamp makes `enter == 0`
    // equivalent to the containment below for any finite ray, which is worth
    // knowing and not worth writing: the test states the property, where reading
    // it off the clamp derives it.
    return .{ .t = enter, .inside = contains(origin, box) };
}

fn contains(point: Vec3, box: Aabb) bool {
    return @reduce(.And, point >= box.min) and @reduce(.And, point <= box.max);
}
