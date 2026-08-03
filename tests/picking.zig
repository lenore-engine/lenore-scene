const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Camera = scene.Camera;
const Ray = scene.Ray;
const Viewport = scene.Viewport;

const tolerance = 1e-5;

// Square, so a ray through the centre of an edge has the same angle on both
// axes and a swapped pair of extents is visible.
const square: Viewport = .{ .x = 0, .y = 0, .width = 512, .height = 512 };

// A 90 degree vertical field of view reaches exactly one unit at unit distance,
// which turns every expectation below into a whole number.
const quarter_turn: Camera = .{
    .anchor = .{ .eye = .{ 0, 0, 0 } },
    .projection = .{ .perspective = .{ .fov_y = std.math.pi / 2.0, .near = 1, .far = 100 } },
};

fn box(min: [3]f32, max: [3]f32) res.Aabb {
    return .{ .min = min, .max = max };
}

fn expectVec(expected: [3]f32, actual: res.Vec3) !void {
    inline for (0..3) |axis|
        try testing.expectApproxEqAbs(expected[axis], actual[axis], tolerance);
}

fn ray(origin: [3]f32, direction: [3]f32) Ray {
    const as_vector: res.Vec3 = direction;
    return .{
        .origin = origin,
        .direction = as_vector / @as(res.Vec3, @splat(@sqrt(@reduce(.Add, as_vector * as_vector)))),
    };
}

test "the ray through the middle of the viewport is the view direction" {
    const centre = try scene.cameraRay(quarter_turn, square, .{ 256, 256 });
    try expectVec(.{ 0, 0, 0 }, centre.origin);
    try expectVec(.{ 0, 0, -1 }, centre.direction);
}

test "the ray starts at the eye wherever the camera stands" {
    const placed: Camera = .{
        .anchor = .{ .eye = .{ 3, -1, 7 } },
        .yaw = 0.6,
        .pitch = -0.2,
        .projection = quarter_turn.projection,
    };
    const shot = try scene.cameraRay(placed, square, .{ 256, 256 });
    try expectVec(.{ 3, -1, 7 }, shot.origin);
    try expectVec(
        .{ placed.placement().front[0], placed.placement().front[1], placed.placement().front[2] },
        shot.direction,
    );
}

test "the pointer's Y axis grows downward and the ray's does not" {
    // The corner of the viewport, one pixel inside it. Up on the screen is up in
    // the world: a projection read as though NDC Y pointed down sends this ray
    // below the horizon instead.
    const top_left = try scene.cameraRay(quarter_turn, square, .{ 0, 0 });
    try testing.expect(top_left.direction[0] < 0);
    try testing.expect(top_left.direction[1] > 0);

    const bottom_right = try scene.cameraRay(quarter_turn, square, .{ 511, 511 });
    try testing.expect(bottom_right.direction[0] > 0);
    try testing.expect(bottom_right.direction[1] < 0);
}

test "the ray through the viewport edge is the field of view's own edge" {
    // Halfway up the right edge. A 90 degree vertical field of view over a
    // square target reaches one unit sideways at one unit ahead, so the
    // direction is the diagonal.
    const edge = try scene.cameraRay(quarter_turn, square, .{ 511.999, 256 });
    const root_half = @sqrt(0.5);
    try testing.expectApproxEqAbs(root_half, edge.direction[0], 1e-3);
    try testing.expectApproxEqAbs(0.0, edge.direction[1], 1e-3);
    try testing.expectApproxEqAbs(-root_half, edge.direction[2], 1e-3);
}

test "a wide viewport widens the ray sideways and leaves it alone vertically" {
    // The same field of view over a target twice as wide reaches twice as far
    // sideways, so the horizontal edge ray is at 45 degrees for the square and
    // steeper here. The aspect ratio comes from the viewport, so this needs no
    // second setting to agree with.
    const wide: Viewport = .{ .x = 0, .y = 0, .width = 1024, .height = 512 };
    const edge = try scene.cameraRay(quarter_turn, wide, .{ 1023.999, 256 });
    const root_fifth = @sqrt(0.2);
    try testing.expectApproxEqAbs(2 * root_fifth, edge.direction[0], 1e-3);
    try testing.expectApproxEqAbs(-root_fifth, edge.direction[2], 1e-3);

    // The vertical half of the frustum is untouched by the width.
    const top = try scene.cameraRay(quarter_turn, wide, .{ 512, 0 });
    const square_top = try scene.cameraRay(quarter_turn, square, .{ 256, 0 });
    try testing.expectApproxEqAbs(square_top.direction[1], top.direction[1], 1e-3);
}

test "an offset viewport measures the pointer from its own corner" {
    // The panel the scene is drawn into does not start at the window's origin.
    const inset: Viewport = .{ .x = 100, .y = 40, .width = 512, .height = 512 };
    const centre = try scene.cameraRay(quarter_turn, inset, .{ 356, 296 });
    try expectVec(.{ 0, 0, -1 }, centre.direction);

    // And a pointer at the window origin is outside it.
    try testing.expectError(error.PointerOutsideViewport, scene.cameraRay(quarter_turn, inset, .{ 0, 0 }));
}

test "a pointer outside the viewport is refused on every side" {
    for ([_][2]f32{ .{ -1, 256 }, .{ 512, 256 }, .{ 256, -1 }, .{ 256, 512 } }) |pointer|
        try testing.expectError(error.PointerOutsideViewport, scene.cameraRay(quarter_turn, square, pointer));

    // Not a number is outside too, which the comparisons only manage because
    // they are written to fail rather than to pass.
    const nan = std.math.nan(f32);
    try testing.expectError(error.PointerOutsideViewport, scene.cameraRay(quarter_turn, square, .{ nan, 256 }));
    try testing.expectError(error.PointerOutsideViewport, scene.cameraRay(quarter_turn, square, .{ 256, nan }));
}

test "a viewport with no area is refused" {
    for ([_]Viewport{
        .{ .x = 0, .y = 0, .width = 0, .height = 512 },
        .{ .x = 0, .y = 0, .width = 512, .height = 0 },
        .{ .x = 0, .y = 0, .width = -512, .height = 512 },
    }) |viewport|
        try testing.expectError(error.EmptyViewport, scene.cameraRay(quarter_turn, viewport, .{ 0, 0 }));
}

test "a camera that describes no volume carries its own error out" {
    const broken: Camera = .{ .projection = .{ .perspective = .{ .near = 10, .far = 1 } } };
    try testing.expectError(error.DegenerateProjection, scene.cameraRay(broken, square, .{ 256, 256 }));
}

const unit = box(.{ -1, -1, -1 }, .{ 1, 1, 1 });

test "a ray meets the near face of a box ahead of it" {
    const hit = scene.intersectAabb(.{ 0, 0, 10 }, .{ 0, 0, -1 }, unit).?;
    try testing.expectApproxEqAbs(9.0, hit.t, tolerance);
    try testing.expect(!hit.inside);
}

test "a ray starting inside a box reports zero and says so" {
    const hit = scene.intersectAabb(.{ 0, 0, 0 }, .{ 0, 0, -1 }, unit).?;
    try testing.expectEqual(@as(f32, 0), hit.t);
    try testing.expect(hit.inside);

    // On the face is inside: the box is closed, and a caller ranking surface
    // hits ahead of interior ones needs the boundary to fall on one side.
    const on_face = scene.intersectAabb(.{ 0, 0, 1 }, .{ 0, 0, -1 }, unit).?;
    try testing.expect(on_face.inside);
}

test "a box behind the ray is missed" {
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectAabb(.{ 0, 0, 10 }, .{ 0, 0, 1 }, unit));
}

test "a ray passing beside a box on each axis is missed" {
    // One case per axis, because a slab test that drops an axis still rejects
    // everything the axes it keeps reject.
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectAabb(.{ 5, 0, 10 }, .{ 0, 0, -1 }, unit));
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectAabb(.{ 0, 5, 10 }, .{ 0, 0, -1 }, unit));
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectAabb(.{ 10, 0, 5 }, .{ -1, 0, 0 }, unit));
}

test "a ray parallel to a slab is inside it or outside it for its whole length" {
    // Exactly zero on two components, so the branch that avoids dividing by it
    // is what decides. Along the box, level with it: a hit.
    const along = scene.intersectAabb(.{ -10, 0.5, 0.5 }, .{ 1, 0, 0 }, unit).?;
    try testing.expectApproxEqAbs(9.0, along.t, tolerance);

    // Along the box but above it: no distance along the ray brings it back.
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectAabb(.{ -10, 5, 0 }, .{ 1, 0, 0 }, unit));

    // Starting exactly on the plane of a face while running parallel to it. The
    // division would give an infinity here and multiply it by a zero.
    const grazing = scene.intersectAabb(.{ -10, 1, 0 }, .{ 1, 0, 0 }, unit).?;
    try testing.expectApproxEqAbs(9.0, grazing.t, tolerance);
}

test "a diagonal ray enters through the face it actually reaches" {
    // The X slab is entered at t = 3 and the Z slab at t = 2, so the box begins
    // at 3. Taking the nearest of the two instead names a point at (2, 0, 1),
    // which is outside the box on X.
    const hit = scene.intersectAabb(.{ 4, 0, 2 }, .{ -1, 0, -0.5 }, unit).?;
    try testing.expectApproxEqAbs(3.0, hit.t, tolerance);
    try expectVec(.{ 1, 0, 0.5 }, res.Vec3{ 4, 0, 2 } + res.Vec3{ -1, 0, -0.5 } * @as(res.Vec3, @splat(hit.t)));
}

test "a placed box is tested in its own space" {
    const shot = ray(.{ 0, 0, 10 }, .{ 0, 0, -1 });
    const model = zm.mul(zm.scaling(1, 1, 2), zm.translation(0, 0, 0));

    // The box is twice as deep in world space, so the near face is at z = 2.
    const hit = scene.intersectInstance(shot, model, unit).?;
    try testing.expectApproxEqAbs(8.0, hit.t, tolerance);

    // And `t` stays a world distance: the point it names is on the world face.
    try expectVec(.{ 0, 0, 2 }, shot.at(hit.t));
}

test "a rotated box is met exactly rather than through something enclosing it" {
    // An eighth of a turn about Z, aimed at a corner that the object-space box
    // does not cover but a sphere around it would.
    const model = zm.matFromQuat(zm.quatFromNormAxisAngle(zm.f32x4(0, 0, 1, 0), 0.25 * std.math.pi));
    const corner = ray(.{ 1.2, 1.2, 10 }, .{ 0, 0, -1 });
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectInstance(corner, model, unit));

    // The same turn puts the diagonal on the X axis, so a ray there does hit.
    const diagonal = ray(.{ 1.2, 0, 10 }, .{ 0, 0, -1 });
    try testing.expect(scene.intersectInstance(diagonal, model, unit) != null);
}

test "a translated box moves with its transform" {
    const shot = ray(.{ 0, 0, 10 }, .{ 0, 0, -1 });
    const hit = scene.intersectInstance(shot, zm.translation(0, 0, -5), unit).?;
    try testing.expectApproxEqAbs(14.0, hit.t, tolerance);
}

test "a box scaled to nothing answers no ray" {
    // zmath inverts a singular matrix to zeros, and a zero direction lies within
    // every slab it starts in, so without the guard this object would be hit by
    // every ray whose origin transformed into its box.
    const shot = ray(.{ 0, 0, 10 }, .{ 0, 0, -1 });
    try testing.expectEqual(
        @as(?scene.Intersection, null),
        scene.intersectInstance(shot, zm.scaling(1, 1, 0), unit),
    );
}

test "the ray from a pixel finds the box under it and misses the one beside it" {
    // The whole path, from a pointer position to a placed box: the camera at the
    // origin looking down -Z, a unit box five ahead and one to the right.
    const model = zm.translation(1.2, 0, -5);
    const centre = try scene.cameraRay(quarter_turn, square, .{ 256, 256 });
    try testing.expectEqual(@as(?scene.Intersection, null), scene.intersectInstance(centre, model, unit));

    // Two thirds across is about 12 degrees off axis, which reaches it.
    const aside = try scene.cameraRay(quarter_turn, square, .{ 340, 256 });
    const hit = scene.intersectInstance(aside, model, unit).?;
    try testing.expect(!hit.inside);
    const point = aside.at(hit.t);
    try testing.expect(point[0] >= 0.2 and point[0] <= 2.2);
    try testing.expect(point[2] >= -6 and point[2] <= -4);
}
