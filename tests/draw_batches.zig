const std = @import("std");
const scene = @import("lenore-scene");

const testing = std.testing;

const MeshId = enum(u32) { body, head, prop };
const MaterialId = enum(u32) { cloth, skin, metal };
const Plan = scene.DrawBatches(MeshId, MaterialId);

fn draw(mesh: MeshId, material: MaterialId, face_culling: scene.FaceCulling) Plan.Draw {
    return .{
        .mesh = mesh,
        .material = material,
        .face_culling = face_culling,
    };
}

const sentinel: Plan.Batch = .{
    .mesh = .prop,
    .material = .metal,
    .face_culling = .none,
    .first_instance = 99,
    .instance_count = 99,
};

test "adjacent draws coalesce only when every batch key agrees" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.body, .cloth, .back),
        draw(.body, .cloth, .front),
        draw(.body, .skin, .front),
        draw(.head, .skin, .front),
        draw(.head, .skin, .none),
        draw(.body, .cloth, .back),
    };
    var storage: [draws.len]Plan.Batch = undefined;

    const batches = try Plan.build(&draws, &.{ 0, 1, 2, 3, 4, 5, 6 }, &storage);

    try testing.expectEqual(@as(usize, 6), batches.len);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .first_instance = 0,
        .instance_count = 2,
    }, batches[0]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .front,
        .first_instance = 2,
        .instance_count = 1,
    }, batches[1]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .skin,
        .face_culling = .front,
        .first_instance = 3,
        .instance_count = 1,
    }, batches[2]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .head,
        .material = .skin,
        .face_culling = .front,
        .first_instance = 4,
        .instance_count = 1,
    }, batches[3]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .head,
        .material = .skin,
        .face_culling = .none,
        .first_instance = 5,
        .instance_count = 1,
    }, batches[4]);
    // Equal to the first key but separated from it. Joining the two would move
    // this draw across the intervening order and invalidate depth sorting.
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .first_instance = 6,
        .instance_count = 1,
    }, batches[5]);
}

test "a reordered culled subset defines both batch order and instance offsets" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.prop, .metal, .back),
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage: [draws.len]Plan.Batch = @splat(sentinel);

    const batches = try Plan.build(&draws, &.{ 2, 0, 3 }, &storage);

    try testing.expectEqual(@as(usize, 2), batches.len);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .first_instance = 0,
        .instance_count = 2,
    }, batches[0]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .head,
        .material = .skin,
        .face_culling = .front,
        .first_instance = 2,
        .instance_count = 1,
    }, batches[1]);
    // The returned slice does not consume or clear spare caller-owned storage.
    try testing.expectEqualDeep(sentinel, storage[2]);
}

test "an empty order produces an empty plan without touching storage" {
    var storage = [_]Plan.Batch{sentinel};

    const batches = try Plan.build(&.{}, &.{}, &storage);

    try testing.expectEqual(@as(usize, 0), batches.len);
    try testing.expectEqualDeep(sentinel, storage[0]);
}

test "an invalid draw index is refused before any batch is written" {
    const draws = [_]Plan.Draw{draw(.body, .cloth, .back)};
    var storage: [2]Plan.Batch = @splat(sentinel);

    // The invalid index follows a valid one. A single-pass writer would have
    // changed the first destination entry before discovering the error.
    try testing.expectError(
        error.DrawIndexOutOfRange,
        Plan.build(&draws, &.{ 0, 1 }, &storage),
    );
    try testing.expectEqualDeep(sentinel, storage[0]);
    try testing.expectEqualDeep(sentinel, storage[1]);
}

test "insufficient batch capacity is refused without a partial plan" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage = [_]Plan.Batch{sentinel};

    try testing.expectError(
        error.BatchCapacityExceeded,
        Plan.build(&draws, &.{ 0, 1 }, &storage),
    );
    try testing.expectEqualDeep(sentinel, storage[0]);
}

test "exact batch capacity is accepted" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage: [2]Plan.Batch = undefined;

    const batches = try Plan.build(&draws, &.{ 0, 1 }, &storage);

    try testing.expectEqual(@as(usize, storage.len), batches.len);
}
