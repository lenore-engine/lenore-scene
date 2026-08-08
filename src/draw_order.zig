// The order visible draws have to be recorded in for alpha compositing to come
// out right. It produces the `order` that draw_batches.zig coalesces, and it is
// the step that file's header names as happening before it.
//
// glTF 2.0 section 3.9.4 requires no sorting for OPAQUE or MASK, and leaves
// depth writes and ordering to the implementation for BLEND. What blending does
// require is that a blended surface is composited over what is already behind
// it, so every blended draw follows every solid one and the blended run goes
// back to front.
//
// No graphics API and no allocation: it writes indices into a buffer the caller
// owns, which is what lets it run every frame.

const std = @import("std");
const resources = @import("lenore-resources");

const Vec3 = resources.Vec3;

// Which half of the frame a draw belongs to. OPAQUE and MASK share one layer,
// because a masked fragment either survives the cutoff and writes depth and
// colour like an opaque one or is discarded and writes nothing.
pub const Layer = enum { solid, blended };

// One draw's ordering input, separate from the draw itself so this step reads
// two numbers per draw instead of walking transforms and materials.
pub const Key = struct {
    layer: Layer,
    // Any measure that grows with distance from the eye. Only the comparison
    // between two of them is read, so a squared distance serves and costs no
    // square root.
    depth: f32,
};

pub const OrderError = error{
    // The indices written are u32, matching the draw contract.
    TooManyDraws,

    // No output is written on an error, so the caller's buffer is untouched.
    // One slot per draw is always enough: the order is a permutation.
    OrderCapacityExceeded,

    // A depth that is not finite makes every comparison against it false, which
    // leaves the blended run in an arbitrary order rather than a wrong one that
    // can be read off the screen. It reaches here from a transform, so it is
    // asset data and gets an error rather than an assert the shipping build
    // removes.
    DepthNotFinite,
};

// The squared distance from the eye to a point, which is what a key carries.
// The centre of a draw's world bounds is the usual argument.
pub fn depthOf(eye: Vec3, centre: Vec3) f32 {
    const offset = centre - eye;
    return @reduce(.Add, offset * offset);
}

// Writes the draw indices in recording order and returns the written prefix.
// The complete input is validated before the first index is written.
//
// Solid draws keep the order they arrived in. Sorting them front to back would
// reject some overdraw through early depth rejection, and it would also scatter
// draws that share a mesh and a material, which is the adjacency draw_batches.zig
// coalesces on. That trade has not been measured on this renderer, and the
// specification asks for neither, so the cheaper half is not spent here.
pub fn order(keys: []const Key, destination: []u32) OrderError![]u32 {
    if (keys.len > std.math.maxInt(u32)) return error.TooManyDraws;
    if (destination.len < keys.len) return error.OrderCapacityExceeded;

    var solid_count: usize = 0;
    for (keys) |key| {
        if (!std.math.isFinite(key.depth)) return error.DepthNotFinite;
        if (key.layer == .solid) solid_count += 1;
    }

    var solid_index: usize = 0;
    var blended_index: usize = solid_count;
    for (keys, 0..) |key, index| switch (key.layer) {
        .solid => {
            destination[solid_index] = @intCast(index);
            solid_index += 1;
        },
        .blended => {
            destination[blended_index] = @intCast(index);
            blended_index += 1;
        },
    };

    // std/sort/block.zig: stable, in place, and O(1) memory with no allocator,
    // which is what a per-frame sort needs. Stability is not a detail here: two
    // draws at the same depth keep the order they were submitted in, so a frame
    // that did not move records the same batches as the one before it.
    std.sort.block(u32, destination[solid_count..keys.len], keys, farthestFirst);
    return destination[0..keys.len];
}

fn farthestFirst(keys: []const Key, a: u32, b: u32) bool {
    return keys[a].depth > keys[b].depth;
}
