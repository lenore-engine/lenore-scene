const zm = @import("zmath");
const resources = @import("lenore-resources");

// Where one thing sits in the world: the three factors an asset authors, an
// animation drives and the renderer composes into a model matrix.
//
// The factors are stored wide because that is the form composeTransform and the
// animation types consume, so nothing repacks on the way in or out. Measured on
// 0.16, @sizeOf(@Vector(3, f32)) is 16 with 16-byte alignment: the narrow form
// occupies the same 48 bytes here and only adds the repack. Lanes 0..2 of the
// translation and the scale carry the value and lane 3 is read by nothing, which
// is the same shape SkeletonTemplate stores its bind pose in.
pub const Transform = struct {
    translation: zm.Vec,
    rotation: zm.Quat,
    scale: zm.Vec,

    pub const identity: Transform = .{
        .translation = zm.f32x4(0, 0, 0, 0),
        .rotation = zm.qidentity(),
        .scale = zm.f32x4(1, 1, 1, 0),
    };

    pub fn modelMatrix(self: *const Transform) zm.Mat {
        return resources.composeTransform(self.translation, self.rotation, self.scale);
    }

    // The current orientation happens first and `delta` after it, which is what
    // makes a repeated call spin an object in world space rather than about its
    // own drifting axes.
    //
    // Measured against zmath 0.11.0-dev rather than taken from its naming:
    // quatToMat(qmul(a, b)) transforms a row vector exactly as mul(mul(v, Ma),
    // Mb) does, so the left operand is the rotation applied first.
    pub fn rotate(self: *Transform, delta: zm.Quat) void {
        self.rotation = zm.qmul(self.rotation, delta);
    }
};

// Euler angles in radians about the world X, Y and Z axes, applied Z first,
// then Y, then X.
//
// An authoring convenience with no asset behind it: glTF stores a quaternion,
// so nothing that is loaded arrives in this form. It exists because a hand
// placed prop is easier to write as three angles, and it is one function rather
// than a method so that a camera or an editor can reach the same convention
// without holding a Transform.
pub fn rotationFromEulerZyx(angles: [3]f32) zm.Quat {
    const x = zm.quatFromNormAxisAngle(zm.f32x4(1, 0, 0, 0), angles[0]);
    const y = zm.quatFromNormAxisAngle(zm.f32x4(0, 1, 0, 0), angles[1]);
    const z = zm.quatFromNormAxisAngle(zm.f32x4(0, 0, 1, 0), angles[2]);
    return zm.qmul(zm.qmul(z, y), x);
}
