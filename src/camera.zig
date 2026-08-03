// Where the view is taken from and what it can see: an authored pose, a
// projection, and the matrices derived from them.
//
// Nothing here is cached. The pose is four numbers and the derivation is two
// sine-cosine pairs and a matrix product, so recomputing it once a frame costs
// less than the bookkeeping that would keep a cache honest, and there is no
// state that can be stale.
//
// Conventions, shared with frustum.zig: zmath transforms a row vector, so clip
// space is `mul(position, view_proj)`, and the depth range is Vulkan's, z' in
// [0, w]. Clip Y points up. A backend whose framebuffer counts Y downward
// applies that flip on its own side; this module names no graphics API.

const std = @import("std");
const zm = @import("zmath");
const resources = @import("lenore-resources");

const Vec3 = resources.Vec3;

// zmath 0.11.0-dev `perspectiveFovRh` asserts that near and far are both above
// zero, that the sine of half the field of view differs from zero by more than
// 0.001, that far and near differ by more than 0.001, and that the aspect ratio
// differs from zero by more than 0.01. ReleaseFast removes all four, and the
// camera is authored data, so the bounds below stand in for them and each one is
// stricter than the assert it replaces.
const min_fov: f32 = 0.01;
const max_fov: f32 = 3.13;
const min_near: f32 = 1e-4;
const min_depth_range: f32 = 0.01;
const min_aspect: f32 = 0.02;

pub const ProjectionError = error{
    // The field of view, the near plane or the far plane cannot describe a
    // volume. Authored, so this is checked rather than asserted.
    DegenerateProjection,

    // The target has no usable shape. A window collapsed to zero on one axis
    // reaches here, which is an ordinary runtime state on a tiling compositor
    // and not a broken scene: the frame it happens on has nothing to draw.
    DegenerateAspect,
};

// Only the projection that exists. Orthographic is a variant to be added when
// something needs one, and until then it cannot be selected, so no consumer has
// to answer for a mode that silently falls back to another.
pub const Projection = union(enum) {
    perspective: Perspective,

    pub const Perspective = struct {
        // Vertical, in radians, so the horizontal extent follows the aspect
        // ratio. 45 degrees.
        fov_y: f32 = std.math.pi / 4.0,

        // The near plane dominates depth precision far more than the far plane
        // does, so raising it is the first thing to try against depth fighting.
        near: f32 = 0.01,
        far: f32 = 1000,
    };
};

// What the pose is measured from. The angles below are the same in both cases;
// only the point they are anchored to differs, so switching between them is a
// conversion of one field and cannot leave a second one behind.
//
// An orbit derives the eye from the pivot, so the pivot always lies on the view
// axis and a camera facing away from it is not expressible. That is the whole
// contract: a controller wanting a rotation centre off the axis, a picked point
// under the cursor, keeps that point itself and drives the `eye` case, moving
// the eye and the angles together.
pub const Anchor = union(enum) {
    // The eye is placed directly.
    eye: Vec3,

    // The eye is placed at `distance` back along the view direction from
    // `target`, so the angles swing it around that point.
    orbit: struct { target: Vec3, distance: f32 },
};

// The pose resolved into world space. A controller reading input wants the whole
// of it, and the view matrix is built from it, so it is produced in one pass
// rather than through three accessors that each redo the trigonometry.
pub const Placement = struct {
    position: Vec3,
    front: Vec3,
    right: Vec3,
    up: Vec3,

    pub fn view(self: Placement) zm.Mat {
        return zm.lookToRh(
            zm.f32x4(self.position[0], self.position[1], self.position[2], 1),
            zm.f32x4(self.front[0], self.front[1], self.front[2], 0),
            zm.f32x4(self.up[0], self.up[1], self.up[2], 0),
        );
    }
};

pub const Camera = struct {
    anchor: Anchor = .{ .eye = .{ 0, 0, 0 } },

    // Azimuth in radians, measured in the XZ plane from +X toward +Z, so -pi/2
    // looks down -Z. Unbounded: the trigonometry below is periodic and a
    // controller that accumulates turns has nothing to wrap.
    yaw: f32 = -std.math.pi / 2.0,

    // Elevation in radians, positive upward. The derivation stays orthonormal at
    // the poles, so nothing here needs the usual clamp just short of vertical. A
    // controller that wants one imposes it for its own reasons.
    pitch: f32 = 0,

    projection: Projection = .{ .perspective = .{} },

    // The basis is written out rather than taken from cross products, which is
    // what removes the singularity: `right` is horizontal by construction and
    // depends on the azimuth alone, so it survives a view straight down where
    // `cross(front, world_up)` collapses to zero.
    pub fn placement(self: Camera) Placement {
        const cos_yaw = @cos(self.yaw);
        const sin_yaw = @sin(self.yaw);
        const cos_pitch = @cos(self.pitch);
        const sin_pitch = @sin(self.pitch);

        const front: Vec3 = .{ cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch };
        const right: Vec3 = .{ -sin_yaw, 0, cos_yaw };
        // right x front, expanded. The yaw terms cancel to one on the middle
        // lane, so this is unit length wherever the other two are.
        const up: Vec3 = .{ -cos_yaw * sin_pitch, cos_pitch, -sin_yaw * sin_pitch };

        return .{
            .position = switch (self.anchor) {
                .eye => |eye| eye,
                .orbit => |orbit| orbit.target - front * @as(Vec3, @splat(orbit.distance)),
            },
            .front = front,
            .right = right,
            .up = up,
        };
    }

    pub fn projectionMatrix(self: Camera, aspect: f32) ProjectionError!zm.Mat {
        if (!(aspect >= min_aspect and aspect < std.math.inf(f32))) return error.DegenerateAspect;

        switch (self.projection) {
            .perspective => |perspective| {
                if (!(perspective.fov_y >= min_fov and perspective.fov_y <= max_fov))
                    return error.DegenerateProjection;
                if (!(perspective.near >= min_near)) return error.DegenerateProjection;
                if (!(perspective.far >= perspective.near + min_depth_range))
                    return error.DegenerateProjection;

                return zm.perspectiveFovRh(perspective.fov_y, aspect, perspective.near, perspective.far);
            },
        }
    }

    // The product the vertex path and the frustum extraction both take.
    pub fn viewProjection(self: Camera, aspect: f32) ProjectionError!zm.Mat {
        return zm.mul(self.placement().view(), try self.projectionMatrix(aspect));
    }

    // Aim at a point without moving the eye. A point directly above or below
    // leaves the azimuth alone rather than resetting it, because that azimuth is
    // still the direction the camera faces as soon as it tilts back down, and a
    // point coincident with the eye names no direction and changes nothing.
    pub fn lookAt(self: *Camera, point: Vec3) void {
        const eye = self.placement().position;
        const to_point = point - eye;
        const distance = @sqrt(@reduce(.Add, to_point * to_point));
        if (!(distance > 0)) return;

        self.pitch = std.math.asin(std.math.clamp(to_point[1] / distance, -1, 1));
        if (to_point[0] != 0 or to_point[2] != 0)
            self.yaw = std.math.atan2(to_point[2], to_point[0]);

        // Under an orbit anchor the eye is derived from the angles that just
        // changed, so holding it still means the pivot moves to the point being
        // aimed at. That is also the only pivot consistent with the new angles.
        switch (self.anchor) {
            .eye => {},
            .orbit => self.anchor = .{ .orbit = .{ .target = point, .distance = distance } },
        }
    }

    // Pivot around `target` from where the camera stands. The eye keeps its
    // place and the angles turn to face the pivot, which is the only pair an
    // orbit anchor can hold. The distance follows from the two points, so no
    // caller carries one and the pivot cannot be left over from an earlier
    // position.
    pub fn orbitAround(self: *Camera, target: Vec3) void {
        self.lookAt(target);
        const to_eye = self.placement().position - target;
        self.anchor = .{ .orbit = .{
            .target = target,
            .distance = @sqrt(@reduce(.Add, to_eye * to_eye)),
        } };
    }

    // Drop the pivot and stand where the camera already is.
    //
    // The eye is read into a local first. A union literal assigned to a field is
    // built in place, so writing `self.anchor = .{ .eye = self.placement()... }`
    // lets the new tag land before the operand that still has to read the old
    // one, and the position comes back from a payload that is half overwritten.
    pub fn detach(self: *Camera) void {
        const eye = self.placement().position;
        self.anchor = .{ .eye = eye };
    }
};
