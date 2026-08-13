// Where each drawn entity's joint matrices sit in the one joint array a frame
// uploads.
//
// The frame is packed from zero every time, so an entity that stops being drawn
// cannot leave a hole behind: the next frame closes it. That costs one pass over
// the draw list and buys a layout with no free list, no fragmentation and no
// state carried between frames.
//
// The plan is returned rather than written back into the entities it describes.
// An offset is frame state and not entity state: two views of one world produce
// two different plans for the same entities in the same frame, and a field on
// the entity can only hold one of them.

const std = @import("std");

// The joint base of an entity that has no pose.
//
// Zero cannot serve here, because zero is where the first skinned entity sits.
// An unskinned entity sharing it would make a skinned draw reached by mistake
// read another entity's joints and come out plausible, which is the expensive
// kind of wrong. This is a marker within the plan, not a base any draw records.
pub const no_joint_base: u32 = std.math.maxInt(u32);

pub const JointOffsetError = error{
    // The destination does not have one slot per entity. Both slices are
    // indexed by the same draw-list position, so a length mismatch means the
    // caller is describing two different draw lists.
    OffsetCountMismatch,

    // The frame's joints do not fit the array reserved for them. The caller
    // decides what to do with the frame; nothing has been uploaded yet.
    JointCapacityExceeded,
};

// Assigns each skinned entity a contiguous run of joint slots, in draw order,
// within a frame array of `capacity` slots, and returns how many slots the frame
// uses. Entities with no joints take no capacity and receive `no_joint_base`.
//
// Counts rather than poses, because an entity's joint count is not its
// skeleton's: several skins may share one skeleton, and each reads its own run
// of it. Taking the pose here would have made every such entity claim the whole
// skeleton's joints.
//
// The result is a pure function of those counts, so the same draw list assigns
// the same offsets every time it is planned.
pub fn assignJointOffsets(
    joint_counts: []const ?u32,
    offsets: []u32,
    capacity: u32,
) JointOffsetError!u32 {
    if (offsets.len != joint_counts.len) return error.OffsetCountMismatch;

    var used: u32 = 0;
    for (joint_counts, offsets) |entry, *offset| {
        const count = entry orelse {
            offset.* = no_joint_base;
            continue;
        };

        // The count is tested before it is narrowed, and the test subtracts
        // rather than adds. `used` never exceeds `capacity`, so the right side
        // cannot wrap. Narrowing the count to the offset's width first and then
        // adding is the form that can, and a wrapped sum passes the very test it
        // is here to fail.
        if (count > capacity - used) return error.JointCapacityExceeded;

        offset.* = used;
        used += count;
    }
    return used;
}
