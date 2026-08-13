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
        .front_face = .counter_clockwise,
    };
}

// Every draw visible. What a test about coalescing wants: the boundary is
// exercised by the tests that name it, and passing it here would only repeat
// the order's own length at six call sites.
fn buildAllVisible(
    draws: []const Plan.Draw,
    order: []const u32,
    destination: []Plan.Batch,
) !Plan.Batches {
    return Plan.build(draws, order, order.len, destination);
}

const sentinel: Plan.Batch = .{
    .mesh = .prop,
    .material = .metal,
    .face_culling = .none,
    .front_face = .counter_clockwise,
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

    const batches = (try buildAllVisible(&draws, &.{ 0, 1, 2, 3, 4, 5, 6 }, &storage)).batches;

    try testing.expectEqual(@as(usize, 6), batches.len);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .front_face = .counter_clockwise,
        .first_instance = 0,
        .instance_count = 2,
    }, batches[0]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .front,
        .front_face = .counter_clockwise,
        .first_instance = 2,
        .instance_count = 1,
    }, batches[1]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .skin,
        .face_culling = .front,
        .front_face = .counter_clockwise,
        .first_instance = 3,
        .instance_count = 1,
    }, batches[2]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .head,
        .material = .skin,
        .face_culling = .front,
        .front_face = .counter_clockwise,
        .first_instance = 4,
        .instance_count = 1,
    }, batches[3]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .head,
        .material = .skin,
        .face_culling = .none,
        .front_face = .counter_clockwise,
        .first_instance = 5,
        .instance_count = 1,
    }, batches[4]);
    // Equal to the first key but separated from it. Joining the two would move
    // this draw across the intervening order and invalidate depth sorting.
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .front_face = .counter_clockwise,
        .first_instance = 6,
        .instance_count = 1,
    }, batches[5]);
}

test "the given order defines both batch order and instance offsets" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.prop, .metal, .back),
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage: [draws.len]Plan.Batch = @splat(sentinel);

    const batches = (try buildAllVisible(&draws, &.{ 2, 0, 3 }, &storage)).batches;

    try testing.expectEqual(@as(usize, 2), batches.len);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .front_face = .counter_clockwise,
        .first_instance = 0,
        .instance_count = 2,
    }, batches[0]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .head,
        .material = .skin,
        .face_culling = .front,
        .front_face = .counter_clockwise,
        .first_instance = 2,
        .instance_count = 1,
    }, batches[1]);
    // The returned slice does not consume or clear spare caller-owned storage.
    try testing.expectEqualDeep(sentinel, storage[2]);
}

test "an empty order produces an empty plan without touching storage" {
    var storage = [_]Plan.Batch{sentinel};

    const batches = (try buildAllVisible(&.{}, &.{}, &storage)).batches;

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
        buildAllVisible(&draws, &.{ 0, 1 }, &storage),
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
        buildAllVisible(&draws, &.{ 0, 1 }, &storage),
    );
    try testing.expectEqualDeep(sentinel, storage[0]);
}

test "exact batch capacity is accepted" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage: [2]Plan.Batch = undefined;

    const batches = (try buildAllVisible(&draws, &.{ 0, 1 }, &storage)).batches;

    try testing.expectEqual(@as(usize, storage.len), batches.len);
}

test "a run of one state is cut at the visibility boundary" {
    // The whole reason the boundary is passed in. Every draw here shares a key,
    // so without the cut they coalesce into one batch whose instance range spans
    // both sides and which neither pass can record: a camera pass would draw the
    // culled half and a bake reading the same range would be right by accident.
    const draws: [4]Plan.Draw = @splat(draw(.body, .cloth, .back));
    var storage: [draws.len]Plan.Batch = @splat(sentinel);

    const built = try Plan.build(&draws, &.{ 0, 1, 2, 3 }, 2, &storage);

    try testing.expectEqual(@as(usize, 2), built.batches.len);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .front_face = .counter_clockwise,
        .first_instance = 0,
        .instance_count = 2,
    }, built.batches[0]);
    try testing.expectEqualDeep(Plan.Batch{
        .mesh = .body,
        .material = .cloth,
        .face_culling = .back,
        .front_face = .counter_clockwise,
        .first_instance = 2,
        .instance_count = 2,
    }, built.batches[1]);
    // Batches, not draws: two draws are visible and they are one batch.
    try testing.expectEqual(@as(usize, 1), built.visible);
    try testing.expectEqual(@as(usize, 1), built.visibleBatches().len);
}

test "the visible count is a count of batches" {
    // Three visible draws over two states, then two culled draws over one. A
    // count that reported draws would say three here.
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
        draw(.prop, .metal, .none),
        draw(.prop, .metal, .none),
    };
    var storage: [draws.len]Plan.Batch = @splat(sentinel);

    const built = try Plan.build(&draws, &.{ 0, 1, 2, 3, 4 }, 3, &storage);

    try testing.expectEqual(@as(usize, 3), built.batches.len);
    try testing.expectEqual(@as(usize, 2), built.visible);
    for (built.visibleBatches(), built.batches[0..2]) |a, b|
        try testing.expectEqualDeep(a, b);
}

test "a boundary at either end leaves one side empty" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage: [draws.len]Plan.Batch = @splat(sentinel);

    // Nothing visible: the camera faces away and every batch is still built,
    // because the bake reads them.
    const none = try Plan.build(&draws, &.{ 0, 1 }, 0, &storage);
    try testing.expectEqual(@as(usize, 2), none.batches.len);
    try testing.expectEqual(@as(usize, 0), none.visible);
    try testing.expectEqual(@as(usize, 0), none.visibleBatches().len);

    // Everything visible, which is what the boundary at the end means.
    const all = try Plan.build(&draws, &.{ 0, 1 }, 2, &storage);
    try testing.expectEqual(@as(usize, 2), all.batches.len);
    try testing.expectEqual(@as(usize, 2), all.visible);
}

test "a boundary past the order is refused before anything is written" {
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        draw(.head, .skin, .front),
    };
    var storage: [draws.len]Plan.Batch = @splat(sentinel);

    try testing.expectError(
        error.VisibleCountOutOfRange,
        Plan.build(&draws, &.{ 0, 1 }, 3, &storage),
    );
    for (storage) |slot| try testing.expectEqualDeep(sentinel, slot);
}

test "a mirrored instance is its own batch even where every other key agrees" {
    // Same mesh, same material, same culling. The winding is what differs, and
    // it is a state of the draw rather than of the geometry: one instance of a
    // mesh can be mirrored while another is not.
    const draws = [_]Plan.Draw{
        draw(.body, .cloth, .back),
        .{ .mesh = .body, .material = .cloth, .face_culling = .back, .front_face = .clockwise },
        draw(.body, .cloth, .back),
    };
    var storage: [draws.len]Plan.Batch = undefined;

    const batches = (try buildAllVisible(&draws, &.{ 0, 1, 2 }, &storage)).batches;

    try testing.expectEqual(@as(usize, 3), batches.len);
    try testing.expectEqual(scene.FrontFace.counter_clockwise, batches[0].front_face);
    try testing.expectEqual(scene.FrontFace.clockwise, batches[1].front_face);
    try testing.expectEqual(scene.FrontFace.counter_clockwise, batches[2].front_face);

    // Two mirrored instances beside each other still coalesce: the key is
    // compared, not the geometry.
    const paired = [_]Plan.Draw{ draws[1], draws[1] };
    var pair_storage: [paired.len]Plan.Batch = undefined;
    const coalesced = (try buildAllVisible(&paired, &.{ 0, 1 }, &pair_storage)).batches;
    try testing.expectEqual(@as(usize, 1), coalesced.len);
    try testing.expectEqual(@as(u32, 2), coalesced[0].instance_count);
}
