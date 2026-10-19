// Screen-space ambient occlusion, its host half: what an application asks for,
// and the constants a pass needs to turn a depth sample back into a point in
// view space.
//
// The constants are read off the projection matrix the frame draws with rather
// than recomputed from the camera's field of view and planes, so the
// reconstruction cannot describe a different projection from the one the depth
// was written through.
//
// Conventions as in camera.zig: zmath's `perspectiveFovRh`, row vectors, view
// space looking down -Z, Vulkan depth in [0, 1] from the near plane to the far
// one. Depth is not reversed.

const std = @import("std");
const zm = @import("zmath");

pub const AmbientOcclusionSettings = struct {
    // Off by default: the pass is a cost a scene opts into.
    enabled: bool = false,

    // How far from a point another surface can still occlude it, in world
    // units. It is also the scale of the effect: a radius near the size of the
    // creases that should darken. Half a metre suits rooms and furniture.
    radius: f32 = 0.5,

    // How dark full occlusion gets, as a multiplier on the estimate before it
    // is clamped. One is the estimator's own scale.
    intensity: f32 = 1,

    // Subtracted from each sample's facing term, in world units at unit
    // distance, so a flat surface does not occlude itself through the
    // quantization of its own depth. Scaled by the point's distance in the
    // shader, since depth error grows with it.
    bias: f32 = 0.01,

    // The strength the shader reads, with the switch folded in for the same
    // reason `SunShadowSettings.clampedStrength` folds it: one accessor
    // honours it, so off is a value and not a branch on each side.
    pub fn clampedIntensity(self: AmbientOcclusionSettings) f32 {
        if (!self.enabled) return 0;
        if (!(self.intensity > 0)) return 0;
        return @min(self.intensity, max_intensity);
    }

    // Authored data, so a radius that is not a positive finite distance
    // switches the effect off rather than reaching a shader that divides by
    // its sixth power.
    pub fn usableRadius(self: AmbientOcclusionSettings) ?f32 {
        if (!(self.radius > min_radius and self.radius < std.math.inf(f32))) return null;
        return self.radius;
    }
};

// Past this the estimate saturates to black over most of a scene; an
// intensity this large is a mistake rather than a look.
const max_intensity: f32 = 16;

// A millimetre. The estimator divides by the radius to the sixth power, and
// below this that leaves f32's range.
const min_radius: f32 = 1e-3;

pub const ReconstructionError = error{
    // The matrix is not one `perspectiveFovRh` builds: no perspective divide,
    // or a zero or non-finite scale on an axis.
    NotPerspective,
};

// What turns a pixel's device coordinates and stored depth into a point in
// view space.
//
// zmath's `perspectiveFovRh` (0.11.0-dev) writes clip x = x * m00, clip y =
// y * m11, clip z = z * m22 + m32 and clip w = -z, where z is the view-space
// depth, negative in front of the eye. So for stored depth d and device
// coordinates (u, v):
//
//   distance = -z = m32 / (d + m22)        (positive in front of the eye)
//   x = u * distance / m00,  y = v * distance / m11
//
// with v the device coordinate before the Vulkan flip, so up is positive; the
// shader undoes the flip it reads through. Checked by test against points sent
// through the same matrix.
pub const Reconstruction = struct {
    // 1 / m00 and 1 / m11: the half extents of the view at unit distance.
    half_width: f32,
    half_height: f32,
    // m22 and m32.
    depth_scale: f32,
    depth_offset: f32,

    pub fn compute(projection: zm.Mat) ReconstructionError!Reconstruction {
        const m00 = projection[0][0];
        const m11 = projection[1][1];
        // The divide by -z: without it this is not the projection the
        // formulas above invert.
        if (projection[2][3] != -1 or projection[3][3] != 0) return error.NotPerspective;
        if (!(@abs(m00) > 0 and @abs(m00) < std.math.inf(f32))) return error.NotPerspective;
        if (!(@abs(m11) > 0 and @abs(m11) < std.math.inf(f32))) return error.NotPerspective;
        return .{
            .half_width = 1 / m00,
            .half_height = 1 / m11,
            .depth_scale = projection[2][2],
            .depth_offset = projection[3][2],
        };
    }

    // The distance in front of the eye a stored depth stands for. The shader
    // computes the same expression; this one is what the tests hold it to.
    pub fn distance(self: Reconstruction, depth: f32) f32 {
        return self.depth_offset / (depth + self.depth_scale);
    }

    // The point in view space at device coordinates `u`, `v` (up positive) and
    // stored depth `depth`.
    pub fn viewPoint(self: Reconstruction, u: f32, v: f32, depth: f32) [3]f32 {
        const d = self.distance(depth);
        return .{ u * d * self.half_width, v * d * self.half_height, -d };
    }

    // Pixels across one world unit at unit distance, for a target `height`
    // pixels tall. The shader divides a world radius by the point's distance
    // and scales by this to know how far to spread its samples on screen.
    pub fn pixelsPerUnit(self: Reconstruction, height: u32) f32 {
        return @as(f32, @floatFromInt(height)) / (2 * self.half_height);
    }
};
