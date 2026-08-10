// The order draws have to be recorded in for alpha compositing to come out
// right. It produces the `order` that draw_batches.zig coalesces, and it is the
// step that file's header names as happening before it.
//
// glTF 2.0 section 3.9.4 requires no sorting for OPAQUE or MASK, and leaves
// depth writes and ordering to the implementation for BLEND. What blending does
// require is that a blended surface is composited over what is already behind
// it, so every blended draw follows every solid one and the blended run goes
// back to front.
//
// Culled draws are moved behind the visible ones rather than dropped, so the
// output stays a permutation of the input and the caller gets one list with a
// boundary in it. A shadow bake needs the draws a camera cannot see: a caster
// outside the view still casts into it, and an orthographic sun fit built around
// the whole scene contains every one of them. Dropping one here would also drop
// whatever a caller packs per draw in this order, leaving a later pass with a
// batch it can name and no data behind it.
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
// three values per draw instead of walking transforms and materials.
pub const Key = struct {
    layer: Layer,
    // Any measure that grows with distance from the eye. Only the comparison
    // between two of them is read, so a squared distance serves and costs no
    // square root.
    depth: f32,
    // Whether the camera can see this draw. It sits here rather than in a
    // parallel array because it decides which side of the boundary a draw is
    // written to, which makes it an ordering input like the other two. A second
    // slice would have to correspond to this one by position and nothing would
    // check that it did.
    //
    // True is always safe. Anything whose bounds cannot be trusted to contain it
    // this frame is passed as visible: a skinned or morphed mesh is bounded by
    // its bind pose, which is not where its vertices are.
    visible: bool = true,
};

// What `order` wrote: every draw, with the visible ones first.
pub const Order = struct {
    draws: []u32,
    // How many of `draws` the camera can see. The rest are behind it, in no
    // particular order, and exist for the passes that do not use the camera.
    visible: usize,

    // The prefix a camera pass records. Solid draws come before blended ones
    // inside it, which is the invariant the recorder validates.
    pub fn visibleDraws(self: Order) []u32 {
        return self.draws[0..self.visible];
    }
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

// Writes every draw index in recording order and reports where the visible ones
// end. The complete input is validated before the first index is written.
//
// Four regions, in this order: visible solid, visible blended, culled solid,
// culled blended. Visible before culled so that a camera pass reads a prefix
// rather than a filtered walk, and solid before blended inside each half so the
// compositing rule holds in the half that composites. The culled half keeps the
// same split for no reason a pass depends on; it costs one cursor and keeps the
// two halves the same shape.
//
// Solid draws keep the order they arrived in. Sorting them front to back would
// reject some overdraw through early depth rejection, and it would also scatter
// draws that share a mesh and a material, which is the adjacency draw_batches.zig
// coalesces on. That trade has not been measured on this renderer, and the
// specification asks for neither, so the cheaper half is not spent here.
//
// Only the visible blended run is sorted. The culled one is recorded by no pass
// that blends: a shadow bake reads the depth it writes and composites nothing.
pub fn order(keys: []const Key, destination: []u32) OrderError!Order {
    if (keys.len > std.math.maxInt(u32)) return error.TooManyDraws;
    if (destination.len < keys.len) return error.OrderCapacityExceeded;

    var visible_solid: usize = 0;
    var visible_count: usize = 0;
    var culled_solid: usize = 0;
    for (keys) |key| {
        if (!std.math.isFinite(key.depth)) return error.DepthNotFinite;
        if (key.visible) {
            visible_count += 1;
            if (key.layer == .solid) visible_solid += 1;
        } else if (key.layer == .solid) culled_solid += 1;
    }

    // The four cursors, each starting where the region before it ends.
    var solid_index: usize = 0;
    var blended_index: usize = visible_solid;
    var culled_index: usize = visible_count;
    var culled_blended_index: usize = visible_count + culled_solid;
    for (keys, 0..) |key, index| {
        const cursor = if (key.visible)
            switch (key.layer) {
                .solid => &solid_index,
                .blended => &blended_index,
            }
        else switch (key.layer) {
            .solid => &culled_index,
            .blended => &culled_blended_index,
        };
        destination[cursor.*] = @intCast(index);
        cursor.* += 1;
    }

    // std/sort/block.zig: stable, in place, and O(1) memory with no allocator,
    // which is what a per-frame sort needs. Stability is not a detail here: two
    // draws at the same depth keep the order they were submitted in, so a frame
    // that did not move records the same batches as the one before it.
    std.sort.block(u32, destination[visible_solid..visible_count], keys, farthestFirst);
    return .{ .draws = destination[0..keys.len], .visible = visible_count };
}

fn farthestFirst(keys: []const Key, a: u32, b: u32) bool {
    return keys[a].depth > keys[b].depth;
}
