const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Aabb = res.Aabb;

const tolerance = 1e-5;

fn box(min: [3]f32, max: [3]f32) Aabb {
    return .{ .min = min, .max = max };
}

fn expectBox(min: [3]f32, max: [3]f32, actual: Aabb) !void {
    // Indexing a vector needs a comptime index, so the loops are unrolled.
    inline for (0..3) |axis| {
        try testing.expectApproxEqAbs(min[axis], actual.min[axis], tolerance);
        try testing.expectApproxEqAbs(max[axis], actual.max[axis], tolerance);
    }
}

const unit = box(.{ -1, -1, -1 }, .{ 1, 1, 1 });

// v * M sends x to x + y and leaves the other axes alone. A shear is what the
// scaled-radius approach cannot bound, and a composed chain reaches it from two
// non-uniform scales with a rotation between them.
const shear_x_by_y: zm.Mat = .{
    zm.f32x4(1, 0, 0, 0),
    zm.f32x4(1, 1, 0, 0),
    zm.f32x4(0, 0, 1, 0),
    zm.f32x4(0, 0, 0, 1),
};

test "the identity leaves a box where it is" {
    try expectBox(.{ -1, -1, -1 }, .{ 1, 1, 1 }, scene.worldAabb(unit, zm.identity()));
}

test "a translated box moves and keeps its size" {
    try expectBox(.{ 9, 19, 29 }, .{ 11, 21, 31 }, scene.worldAabb(unit, zm.translation(10, 20, 30)));
}

test "a scaled box scales per axis" {
    try expectBox(.{ -2, -5, -3 }, .{ 2, 5, 3 }, scene.worldAabb(unit, zm.scaling(2, 5, 3)));
}

test "a negative scale keeps the bounds ordered" {
    // The low face of the box lands above the high one on X. Taking the two
    // products in the order they arrive, without the elementwise minimum and
    // maximum, leaves min above max and every later test on the box is wrong.
    try expectBox(.{ -4, 0, 0 }, .{ -2, 1, 1 }, scene.worldAabb(
        box(.{ 1, 0, 0 }, .{ 2, 1, 1 }),
        zm.scaling(-2, 1, 1),
    ));
}

test "a rotated box grows to the axis-aligned bound of its corners" {
    // A cube turned an eighth of a turn about Z spans the diagonal on X and Y,
    // and is untouched on Z.
    const turn = zm.matFromQuat(zm.quatFromNormAxisAngle(zm.f32x4(0, 0, 1, 0), 0.25 * std.math.pi));
    const root_two = @sqrt(2.0);
    try expectBox(
        .{ -root_two, -root_two, -1 },
        .{ root_two, root_two, 1 },
        scene.worldAabb(unit, turn),
    );
}

test "a sheared box is bounded exactly" {
    // x + y ranges over [-2, 2] and the other axes are untouched. This is the
    // case the construction exists for: no scalar radius bounds it.
    try expectBox(.{ -2, -1, -1 }, .{ 2, 1, 1 }, scene.worldAabb(unit, shear_x_by_y));
}

test "the translation applies after the linear part" {
    const model = zm.mul(zm.scaling(2, 5, 3), zm.translation(10, 20, 30));
    try expectBox(.{ 8, 15, 27 }, .{ 12, 25, 33 }, scene.worldAabb(unit, model));
}

test "a union spans both boxes on every axis" {
    // Each axis takes its low from one box and its high from the other, so an
    // implementation that returned either input, or that mixed up which end it
    // was folding, disagrees on at least one lane.
    try expectBox(.{ -1, -4, 0 }, .{ 3, 2, 7 }, scene.unionAabb(
        box(.{ -1, 1, 0 }, .{ 2, 2, 5 }),
        box(.{ 0, -4, 3 }, .{ 3, 1, 7 }),
    ));
}

test "a union with a box already inside is unchanged" {
    try expectBox(.{ -1, -1, -1 }, .{ 1, 1, 1 }, scene.unionAabb(unit, box(.{ 0, 0, 0 }, .{ 0.5, 0.5, 0.5 })));
}

test "the sphere around a box passes through its corners" {
    const around_unit = scene.sphereAroundAabb(unit);
    try testing.expectApproxEqAbs(@sqrt(3.0), around_unit.radius, tolerance);
    inline for (0..3) |axis|
        try testing.expectApproxEqAbs(0.0, around_unit.centre[axis], tolerance);

    // Off centre and with unequal sides, so a radius taken from the full extent
    // rather than the half, or a centre left at the low corner, is visible.
    const oblong = scene.sphereAroundAabb(box(.{ 0, 0, 0 }, .{ 2, 4, 4 }));
    try testing.expectApproxEqAbs(3.0, oblong.radius, tolerance);
    try testing.expectApproxEqAbs(1.0, oblong.centre[0], tolerance);
    try testing.expectApproxEqAbs(2.0, oblong.centre[1], tolerance);
    try testing.expectApproxEqAbs(2.0, oblong.centre[2], tolerance);
}
