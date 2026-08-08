const std = @import("std");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Vec3 = res.Vec3;

fn solid(depth: f32) scene.DrawKey {
    return .{ .layer = .solid, .depth = depth };
}

fn blended(depth: f32) scene.DrawKey {
    return .{ .layer = .blended, .depth = depth };
}

// Filled into the destination so a test can tell an index that was written from
// one that happened to hold the right value already.
const untouched: u32 = 0xDEAD_BEEF;

test "solid draws come first and keep the order they arrived in" {
    // Interleaved on input, and the solid run must read 0, 2, 4 rather than any
    // depth-sorted permutation of them.
    const keys = [_]scene.DrawKey{
        solid(9),
        blended(1),
        solid(1),
        blended(9),
        solid(5),
    };
    var storage: [keys.len]u32 = @splat(untouched);

    const ordered = try scene.orderDraws(&keys, &storage);

    try testing.expectEqualSlices(u32, &.{ 0, 2, 4, 3, 1 }, ordered);
}

test "the blended run is recorded back to front" {
    const keys = [_]scene.DrawKey{
        blended(1),
        blended(100),
        blended(10),
    };
    var storage: [keys.len]u32 = @splat(untouched);

    const ordered = try scene.orderDraws(&keys, &storage);

    // Farthest first, so what is nearest is composited over what is behind it.
    try testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, ordered);
}

test "equal depths keep their submission order" {
    // The stability the sort was chosen for. Without it a frame that did not
    // move can record a different batch list than the one before it.
    //
    // Long enough that an unstable sort cannot pass by accident. std/sort/pdq.zig
    // sends a slice of 24 or fewer to insertion sort, which leaves an already
    // ordered run alone, so a shorter case here would agree with either sort.
    const count = 64;
    const keys: [count]scene.DrawKey = @splat(blended(4));
    var storage: [count]u32 = @splat(untouched);

    const ordered = try scene.orderDraws(&keys, &storage);

    for (ordered, 0..) |index, submitted|
        try testing.expectEqual(@as(u32, @intCast(submitted)), index);
}

test "a stable order survives depths that repeat in runs" {
    // Equal keys spread through distinct ones, which is where an unstable sort
    // reorders even when the fully equal case happens to survive.
    const count = 96;
    var keys: [count]scene.DrawKey = undefined;
    for (&keys, 0..) |*key, index| key.* = blended(@floatFromInt(index / 8));
    var storage: [count]u32 = @splat(untouched);

    const ordered = try scene.orderDraws(&keys, &storage);

    // Depths descend, and inside one run the submission order is kept.
    for (ordered[1..], ordered[0 .. ordered.len - 1]) |current, previous| {
        try testing.expect(keys[previous].depth >= keys[current].depth);
        if (keys[previous].depth == keys[current].depth)
            try testing.expect(previous < current);
    }
}

test "a list of one layer is a permutation of every index" {
    const solids = [_]scene.DrawKey{ solid(3), solid(1), solid(2) };
    var storage: [3]u32 = @splat(untouched);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, try scene.orderDraws(&solids, &storage));

    const blends = [_]scene.DrawKey{ blended(3), blended(1), blended(2) };
    try testing.expectEqualSlices(u32, &.{ 0, 2, 1 }, try scene.orderDraws(&blends, &storage));
}

test "an empty list orders into an empty prefix" {
    var storage: [2]u32 = @splat(untouched);
    const ordered = try scene.orderDraws(&.{}, &storage);
    try testing.expectEqual(@as(usize, 0), ordered.len);
    // And nothing was written past it.
    try testing.expectEqual(untouched, storage[0]);
}

test "a destination shorter than the draw list is refused before anything is written" {
    const keys = [_]scene.DrawKey{ solid(1), blended(2), solid(3) };
    var storage: [2]u32 = @splat(untouched);

    try testing.expectError(
        error.OrderCapacityExceeded,
        scene.orderDraws(&keys, &storage),
    );
    for (storage) |slot| try testing.expectEqual(untouched, slot);
}

test "a depth that is not finite is refused before anything is written" {
    // The check has to reach a key in either layer and either position, since a
    // partition that ran first would already have written indices.
    for ([_]f32{
        std.math.nan(f32),
        std.math.inf(f32),
        -std.math.inf(f32),
    }) |bad| {
        const trailing = [_]scene.DrawKey{ solid(1), blended(2), blended(bad) };
        var storage: [3]u32 = @splat(untouched);
        try testing.expectError(
            error.DepthNotFinite,
            scene.orderDraws(&trailing, &storage),
        );
        for (storage) |slot| try testing.expectEqual(untouched, slot);

        const leading = [_]scene.DrawKey{ solid(bad), blended(2) };
        try testing.expectError(
            error.DepthNotFinite,
            scene.orderDraws(&leading, &storage),
        );
    }
}

test "the depth is the squared distance from the eye" {
    const eye: Vec3 = .{ 1, 2, 3 };
    // A 3-4-5 offset, so the squared distance is exactly 25 and the test does
    // not rest on a tolerance.
    try testing.expectEqual(@as(f32, 25), scene.depthOf(eye, .{ 4, 6, 3 }));
    try testing.expectEqual(@as(f32, 0), scene.depthOf(eye, eye));
}

test "a farther centre orders behind a nearer one whichever side of the eye it is" {
    // Squaring loses the sign, which is what makes this worth pinning: two
    // draws on opposite sides of the eye must still compare by distance.
    const eye: Vec3 = .{ 0, 0, 0 };
    const near = scene.depthOf(eye, .{ 0, 0, -2 });
    const far = scene.depthOf(eye, .{ 0, 0, 5 });
    try testing.expect(near < far);

    const keys = [_]scene.DrawKey{ blended(near), blended(far) };
    var storage: [2]u32 = @splat(untouched);
    try testing.expectEqualSlices(u32, &.{ 1, 0 }, try scene.orderDraws(&keys, &storage));
}
