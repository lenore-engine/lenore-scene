// Predicates over the vector type the module shares with lenore-resources.
// Internal: nothing here is part of the module's surface, it is the arithmetic
// several files would otherwise each write out.

const std = @import("std");
const resources = @import("lenore-resources");

const Vec3 = resources.Vec3;

// Not-a-number fails this as well, since every comparison against one is false.
pub fn finite(v: Vec3) bool {
    return @reduce(.And, @abs(v) < @as(Vec3, @splat(std.math.inf(f32))));
}
