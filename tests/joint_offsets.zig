const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Template = res.SkeletonTemplate;
const Pose = res.SkeletonPose;

// A skeleton of one slot with `joint_count` joints bound to it. Only the joint
// count reaches the planner, so the hierarchy is the smallest one a template
// accepts and the bind values are never read.
//
// Heap allocated as a unit because the pose borrows the template by pointer and
// the pair has to keep one address for the pose's lifetime.
const Skeleton = struct {
    template: Template,
    pose: Pose,

    fn make(allocator: std.mem.Allocator, joint_count: usize) !*Skeleton {
        const joint_slot = try allocator.alloc(u16, joint_count);
        defer allocator.free(joint_slot);
        @memset(joint_slot, 0);
        const inverse_bind = try allocator.alloc(zm.Mat, joint_count);
        defer allocator.free(inverse_bind);
        @memset(inverse_bind, zm.identity());

        const self = try allocator.create(Skeleton);
        errdefer allocator.destroy(self);

        self.template = try Template.init(allocator, .{
            .slot_parent = &.{res.no_parent},
            .slot_prefix = &.{zm.identity()},
            .bind_translations = &.{zm.f32x4(0, 0, 0, 1)},
            .bind_rotations = &.{zm.qidentity()},
            .bind_scales = &.{zm.f32x4(1, 1, 1, 1)},
            .inverse_bind = inverse_bind,
            .joint_slot = joint_slot,
        });
        errdefer self.template.deinit(allocator);

        self.pose = try Pose.init(allocator, &self.template);
        return self;
    }

    fn destroy(self: *Skeleton, allocator: std.mem.Allocator) void {
        self.pose.deinit(allocator);
        self.template.deinit(allocator);
        allocator.destroy(self);
    }
};

test "posed entities pack in draw order and unposed ones take no capacity" {
    const allocator = testing.allocator;
    const five = try Skeleton.make(allocator, 5);
    defer five.destroy(allocator);
    const three = try Skeleton.make(allocator, 3);
    defer three.destroy(allocator);

    // The unposed entity sits between the two posed ones, so an implementation
    // that advanced the running offset for it, or that assigned in some order
    // other than the draw list's, disagrees on the second base.
    const poses = [_]?*const Pose{ &five.pose, null, &three.pose };
    var offsets: [3]u32 = undefined;

    const total = try scene.assignJointOffsets(&poses, &offsets, 100);

    try testing.expectEqual(@as(u32, 8), total);
    try testing.expectEqual(@as(u32, 0), offsets[0]);
    try testing.expectEqual(scene.no_joint_base, offsets[1]);
    try testing.expectEqual(@as(u32, 5), offsets[2]);
}

test "an unposed entity is marked, not given the first base" {
    var offsets: [1]u32 = undefined;
    const total = try scene.assignJointOffsets(&.{null}, &offsets, 0);

    // Zero is where the first posed entity would sit, so the two cases have to
    // be distinguishable in the plan alone.
    try testing.expectEqual(@as(u32, 0), total);
    try testing.expect(offsets[0] != 0);
    try testing.expectEqual(scene.no_joint_base, offsets[0]);
}

test "a pose with no joints is planned, not treated as unposed" {
    const allocator = testing.allocator;
    const empty = try Skeleton.make(allocator, 0);
    defer empty.destroy(allocator);
    const two = try Skeleton.make(allocator, 2);
    defer two.destroy(allocator);

    const poses = [_]?*const Pose{ &two.pose, &empty.pose, &two.pose };
    var offsets: [3]u32 = undefined;

    const total = try scene.assignJointOffsets(&poses, &offsets, 100);

    try testing.expectEqual(@as(u32, 4), total);
    // A real base, at the point the empty pose reached, and not the marker.
    try testing.expectEqual(@as(u32, 2), offsets[1]);
    try testing.expectEqual(@as(u32, 2), offsets[2]);
}

test "planning the same draw list twice gives the same plan" {
    const allocator = testing.allocator;
    const four = try Skeleton.make(allocator, 4);
    defer four.destroy(allocator);
    const one = try Skeleton.make(allocator, 1);
    defer one.destroy(allocator);

    const poses = [_]?*const Pose{ &four.pose, null, &one.pose };
    var first: [3]u32 = undefined;
    var second: [3]u32 = undefined;

    const total_first = try scene.assignJointOffsets(&poses, &first, 100);
    const total_second = try scene.assignJointOffsets(&poses, &second, 100);

    try testing.expectEqual(total_first, total_second);
    try testing.expectEqualSlices(u32, &first, &second);
}

test "the plan does not depend on how much capacity is spare" {
    const allocator = testing.allocator;
    const six = try Skeleton.make(allocator, 6);
    defer six.destroy(allocator);

    const poses = [_]?*const Pose{ &six.pose, &six.pose };
    var tight: [2]u32 = undefined;
    var loose: [2]u32 = undefined;

    _ = try scene.assignJointOffsets(&poses, &tight, 12);
    _ = try scene.assignJointOffsets(&poses, &loose, 4096);

    try testing.expectEqualSlices(u32, &tight, &loose);
}

test "a frame that exactly fills the capacity is planned" {
    const allocator = testing.allocator;
    const seven = try Skeleton.make(allocator, 7);
    defer seven.destroy(allocator);

    const poses = [_]?*const Pose{ &seven.pose, &seven.pose };
    var offsets: [2]u32 = undefined;

    // The boundary between the last slot that fits and the first that does not.
    // An off-by-one in the capacity test refuses this frame.
    const total = try scene.assignJointOffsets(&poses, &offsets, 14);

    try testing.expectEqual(@as(u32, 14), total);
    try testing.expectEqual(@as(u32, 7), offsets[1]);
}

test "one joint past the capacity is refused" {
    const allocator = testing.allocator;
    const seven = try Skeleton.make(allocator, 7);
    defer seven.destroy(allocator);

    const poses = [_]?*const Pose{ &seven.pose, &seven.pose };
    var offsets: [2]u32 = undefined;

    try testing.expectError(
        error.JointCapacityExceeded,
        scene.assignJointOffsets(&poses, &offsets, 13),
    );
}

test "a destination of the wrong length is refused before anything is planned" {
    const allocator = testing.allocator;
    const two = try Skeleton.make(allocator, 2);
    defer two.destroy(allocator);

    const poses = [_]?*const Pose{ &two.pose, &two.pose };
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
    const empty: [0]?*const Pose = .{};
    var offsets: [0]u32 = .{};

    try testing.expectEqual(@as(u32, 0), try scene.assignJointOffsets(&empty, &offsets, 0));
}
