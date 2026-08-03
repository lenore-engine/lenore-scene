// The six planes of a view frustum, and the box test culling runs against
// them. Pure math, so what decides whether an object is drawn is checkable
// without a device.
//
// Conventions, all three of which the arithmetic below depends on: zmath
// transforms a row vector, so clip space is `mul(position, view_proj)`; the
// depth range is Vulkan's, z' in [0, w] rather than [-w, w]; and a plane faces
// inward, so a point is on the inside of one when `dot3(normal, point) + d` is
// not negative.

const zm = @import("zmath");
const resources = @import("lenore-resources");

pub const Frustum = struct {
    // Inward normal in xyz and the distance term in w, at whatever scale the
    // extraction produced. The scale is positive and equal across the four
    // lanes of one plane, so it cancels out of every sign the test below takes.
    // Nothing here needs a metric distance; a consumer that does normalizes by
    // the length of the normal, where the need is visible.
    planes: [6]zm.Vec,

    // `view_proj` is the same product the vertex path applies. Whether it
    // carries the Vulkan Y flip does not matter: the flip negates the second
    // clip coordinate, which exchanges the bottom plane with the top one, and
    // nothing reads either individually.
    pub fn fromViewProj(view_proj: zm.Mat) Frustum {
        const m = view_proj;
        // Clip coordinate j of a point p is dot(p, column j), so each clip-space
        // inequality is a plane once it is written as a dot product against p.
        // This is the standard extraction from the combined matrix (Gribb and
        // Hartmann).
        const col0 = zm.f32x4(m[0][0], m[1][0], m[2][0], m[3][0]);
        const col1 = zm.f32x4(m[0][1], m[1][1], m[2][1], m[3][1]);
        const col2 = zm.f32x4(m[0][2], m[1][2], m[2][2], m[3][2]);
        const col3 = zm.f32x4(m[0][3], m[1][3], m[2][3], m[3][3]);

        return .{
            .planes = .{
                col3 + col0, // x' >= -w
                col3 - col0, // x' <=  w
                col3 + col1, // y' >= -w
                col3 - col1, // y' <=  w
                col2, // z' >= 0, the Vulkan depth range
                col3 - col2, // z' <=  w
            },
        };
    }

    // Conservative: false only when the box lies entirely outside one plane. A
    // box straddling two planes near a corner can pass while touching no part
    // of the frustum, which costs a draw and never a missing object.
    //
    // Only one corner of the eight has to be looked at per plane, the one
    // furthest along that plane's inward normal: if even that corner is outside,
    // all eight are. Which corner it is follows from the signs of the normal,
    // one axis at a time, which is what the select does.
    //
    // A degenerate view-projection makes every plane not-a-number, and a
    // comparison against one is false, so culling then admits everything rather
    // than hiding the scene. That is the failure to want, but it is the whole
    // of the protection: producing a usable frustum is the camera's job.
    pub fn intersectsAabb(self: *const Frustum, box: resources.Aabb) bool {
        const origin: resources.Vec3 = @splat(0);
        for (self.planes) |plane| {
            const normal = resources.Vec3{ plane[0], plane[1], plane[2] };
            const corner = @select(f32, normal >= origin, box.max, box.min);
            if (@reduce(.Add, normal * corner) + plane[3] < 0) return false;
        }
        return true;
    }
};
