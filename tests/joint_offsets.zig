const std = @import("std");
const scene = @import("lenore-scene");

const testing = std.testing;

// Only joint counts reach the planner, so a test needs no skeleton, no pose and
// no allocator. That is the point of the signature: an entity's count is not its
// skeleton's, because several skins may share one.

test "skinned entities pack in draw order and unskinned ones take no capacity" {
    // The unskinned entity sits between the two skinned ones, so an
    // implementation that advanced the running offset for it, or that assigned
    // in some order other than the draw list's, disagrees on the second base.
    const poses = [_]?u32{ 5, null, 3 };
    var offsets: [3]u32 = undefined;

    const total = try scene.assignJointOffsets(&poses, &offsets, 100);

    try testing.expectEqual(@as(u32, 8), total);
    try testing.expectEqual(@as(u32, 0), offsets[0]);
    try testing.expectEqual(scene.no_joint_base, offsets[1]);
    try testing.expectEqual(@as(u32, 5), offsets[2]);
}

test "an unskinned entity is marked, not given the first base" {
    var offsets: [1]u32 = undefined;
    const total = try scene.assignJointOffsets(&.{null}, &offsets, 0);

    // Zero is where the first skinned entity would sit, so the two cases have to
    // be distinguishable in the plan alone.
    try testing.expectEqual(@as(u32, 0), total);
    try testing.expect(offsets[0] != 0);
    try testing.expectEqual(scene.no_joint_base, offsets[0]);
}

test "an entity with no joints is planned, not treated as unskinned" {
    const poses = [_]?u32{ 2, 0, 2 };
    var offsets: [3]u32 = undefined;

    const total = try scene.assignJointOffsets(&poses, &offsets, 100);

    try testing.expectEqual(@as(u32, 4), total);
    // A real base, at the point the empty run reached, and not the marker.
    try testing.expectEqual(@as(u32, 2), offsets[1]);
    try testing.expectEqual(@as(u32, 2), offsets[2]);
}

test "planning the same draw list twice gives the same plan" {
    const poses = [_]?u32{ 4, null, 1 };
    var first: [3]u32 = undefined;
    var second: [3]u32 = undefined;

    const total_first = try scene.assignJointOffsets(&poses, &first, 100);
    const total_second = try scene.assignJointOffsets(&poses, &second, 100);

    try testing.expectEqual(total_first, total_second);
    try testing.expectEqualSlices(u32, &first, &second);
}

test "the plan does not depend on how much capacity is spare" {
    const poses = [_]?u32{ 6, 6 };
    var tight: [2]u32 = undefined;
    var loose: [2]u32 = undefined;

    _ = try scene.assignJointOffsets(&poses, &tight, 12);
    _ = try scene.assignJointOffsets(&poses, &loose, 4096);

    try testing.expectEqualSlices(u32, &tight, &loose);
}

test "a frame that exactly fills the capacity is planned" {
    const poses = [_]?u32{ 7, 7 };
    var offsets: [2]u32 = undefined;

    // The boundary between the last slot that fits and the first that does not.
    // An off-by-one in the capacity test refuses this frame.
    const total = try scene.assignJointOffsets(&poses, &offsets, 14);

    try testing.expectEqual(@as(u32, 14), total);
    try testing.expectEqual(@as(u32, 7), offsets[1]);
}

test "one joint past the capacity is refused" {
    const poses = [_]?u32{ 7, 7 };
    var offsets: [2]u32 = undefined;

    try testing.expectError(
        error.JointCapacityExceeded,
        scene.assignJointOffsets(&poses, &offsets, 13),
    );
}

test "a destination of the wrong length is refused before anything is planned" {
    const poses = [_]?u32{ 2, 2 };
    var short: [1]u32 = undefined;
    var long: [3]u32 = undefined;

    try testing.expectError(
        error.OffsetCountMismatch,
        scene.assignJointOffsets(&poses, &short, 100),
    );
    try testing.expectError(
        error.OffsetCountMismatch,
        scene.assignJointOffsets(&poses, &long, 100),
    );
}

test "an empty draw list uses no capacity" {
    const empty: [0]?u32 = .{};
    var offsets: [0]u32 = .{};

    try testing.expectEqual(@as(u32, 0), try scene.assignJointOffsets(&empty, &offsets, 0));
}
