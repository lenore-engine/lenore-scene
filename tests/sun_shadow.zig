const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Sphere = res.Sphere;
const SunShadowFit = scene.SunShadowFit;

const tolerance = 1e-4;

// The reference map resolution, and the margin the fit is built with, so that a
// clearance measured in texels below can be checked against the number the
// source states rather than against a fraction nobody can read.
const map_size: u32 = 2048;
const margin_texels = 2.0;

// Off the origin and away from unit radius, so a fit that forgets the centre, or
// that places the eye at `dir * radius` rather than at `centre + dir * radius`,
// disagrees with every expectation here.
const scene_bounds: Sphere = .{ .centre = .{ 5, -2, 3 }, .radius = 4 };

fn unit(v: [3]f32) res.Vec3 {
    const as_vector: res.Vec3 = v;
    return as_vector / @as(res.Vec3, @splat(@sqrt(@reduce(.Add, as_vector * as_vector))));
}

fn clipOf(fit: SunShadowFit, point: res.Vec3) zm.Vec {
    return zm.mul(zm.f32x4(point[0], point[1], point[2], 1), fit.view_proj);
}

// A point on the surface of `scene_bounds`, in the given direction.
fn onSphere(direction: [3]f32) res.Vec3 {
    return scene_bounds.centre + unit(direction) * @as(res.Vec3, @splat(scene_bounds.radius));
}

// Enough directions to reach every octant, so a fit that encloses the sphere on
// some axes and clips it on others cannot pass.
const surface_directions = [_][3]f32{
    .{ 1, 0, 0 },   .{ -1, 0, 0 },   .{ 0, 1, 0 },   .{ 0, -1, 0 },
    .{ 0, 0, 1 },   .{ 0, 0, -1 },   .{ 1, 1, 1 },   .{ -1, 1, 1 },
    .{ 1, -1, 1 },  .{ 1, 1, -1 },   .{ -1, -1, 1 }, .{ -1, 1, -1 },
    .{ 1, -1, -1 }, .{ -1, -1, -1 }, .{ 2, -1, 3 },  .{ -3, 2, -1 },
};

fn expectEncloses(fit: SunShadowFit) !void {
    for (surface_directions) |direction| {
        const clip = clipOf(fit, onSphere(direction));
        // Orthographic, so the fourth lane stays 1 and nothing is divided by it.
        try testing.expectApproxEqAbs(1.0, clip[3], tolerance);
        try testing.expect(@abs(clip[0]) <= 1);
        try testing.expect(@abs(clip[1]) <= 1);
        try testing.expect(clip[2] >= 0 and clip[2] <= 1);
    }
}

test "the fit encloses the whole scene sphere" {
    const fit = try SunShadowFit.compute(scene_bounds, unit(.{ 0.3, 0.8, -0.5 }), map_size);
    try expectEncloses(fit);
}

test "a sun straight overhead is fitted rather than degenerating" {
    // The up vector the view is built from is parallel to the sun here, and the
    // cross product that makes the basis is zero unless the fallback axis is
    // taken. Both poles, because a fallback that triggers on the sign rather
    // than the magnitude passes one of them.
    try expectEncloses(try SunShadowFit.compute(scene_bounds, .{ 0, 1, 0 }, map_size));
    try expectEncloses(try SunShadowFit.compute(scene_bounds, .{ 0, -1, 0 }, map_size));

    // Either side of the limit, so the branch itself is exercised and neither
    // arm produces a matrix that is not a number.
    try expectEncloses(try SunShadowFit.compute(scene_bounds, unit(.{ 0.1, 0.98, 0 }), map_size));
    try expectEncloses(try SunShadowFit.compute(scene_bounds, unit(.{ 0.1, 0.995, 0 }), map_size));
}

test "the eye is on the sun side, so depth grows away from the sun" {
    // The one asymmetry that distinguishes the sun direction from its negation.
    // The fit is otherwise symmetric under a sign flip of the direction, and
    // every enclosure test above passes with the sun behind the scene.
    const fit = try SunShadowFit.compute(scene_bounds, .{ 0, 0, 1 }, map_size);
    const near = clipOf(fit, onSphere(.{ 0, 0, 1 }));
    const far = clipOf(fit, onSphere(.{ 0, 0, -1 }));
    try testing.expect(near[2] < far[2]);
}

test "the fit carries no Y flip" {
    // Sun along +Z with the world up vector available, so the map's vertical
    // axis is the world's. A point above the centre has to land above it in
    // clip space; a Y flip in the projection sends it below.
    const fit = try SunShadowFit.compute(scene_bounds, .{ 0, 0, 1 }, map_size);
    const above = clipOf(fit, onSphere(.{ 0, 1, 0 }));
    try testing.expect(above[1] > 0);
    // And the horizontal axis is not mirrored either, which one signed check
    // alone would not tell from a rotation by half a turn.
    const right = clipOf(fit, onSphere(.{ 1, 0, 0 }));
    try testing.expect(right[0] > 0);
}

test "the margin around the sphere is the stated number of texels" {
    // The clearance works out to the margin exactly, on every axis and whatever
    // the scene measures: laterally the sphere reaches 1 / slack of the half
    // extent, and 1 - 1 / slack over two, in texels, is the margin. The depth
    // range gives the same figure because the eye is pushed back by the same
    // slack. A margin taken as a fraction of the radius instead fails here as
    // soon as the resolution moves.
    const fit = try SunShadowFit.compute(scene_bounds, .{ 0, 0, 1 }, map_size);
    const resolution: f32 = @floatFromInt(map_size);

    const right = clipOf(fit, onSphere(.{ 1, 0, 0 }));
    try testing.expectApproxEqAbs(margin_texels, (1 - right[0]) * 0.5 * resolution, 1e-2);

    const top = clipOf(fit, onSphere(.{ 0, 1, 0 }));
    try testing.expectApproxEqAbs(margin_texels, (1 - top[1]) * 0.5 * resolution, 1e-2);

    const near = clipOf(fit, onSphere(.{ 0, 0, 1 }));
    const far = clipOf(fit, onSphere(.{ 0, 0, -1 }));
    try testing.expectApproxEqAbs(margin_texels, near[2] * resolution, 1e-2);
    try testing.expectApproxEqAbs(margin_texels, (1 - far[2]) * resolution, 1e-2);
}

test "a texel measures the projection width over the resolution" {
    const fit = try SunShadowFit.compute(scene_bounds, .{ 0, 0, 1 }, map_size);
    const resolution: f32 = @floatFromInt(map_size);
    const slack = resolution / (resolution - 2 * margin_texels);
    try testing.expectApproxEqAbs(
        2 * scene_bounds.radius * slack / resolution,
        fit.texel_world_size,
        1e-7,
    );

    // Half the resolution over the same scene is twice the texel, and a shade
    // over twice because the margin is a texel count and its two texels are then
    // wider. A size taken from the radius alone, or from the resolution alone,
    // misses this by far more than the shade.
    const coarse = try SunShadowFit.compute(scene_bounds, .{ 0, 0, 1 }, map_size / 2);
    try testing.expectApproxEqAbs(2.0, coarse.texel_world_size / fit.texel_world_size, 0.01);
}

test "the stored sun direction is normalized whatever length it arrived with" {
    const fit = try SunShadowFit.compute(scene_bounds, .{ 0, 0, 250 }, map_size);
    try testing.expectApproxEqAbs(1.0, @reduce(.Add, fit.sun_dir * fit.sun_dir), tolerance);
    try testing.expectApproxEqAbs(1.0, fit.sun_dir[2], tolerance);
}

// A rotation of the sun about Y by `angle`, starting from +Z.
fn sunAt(angle: f32) res.Vec3 {
    return .{ @sin(angle), 0, @cos(angle) };
}

// The angle at which a fit goes stale, in radians.
const stale_angle = std.math.degreesToRadians(scene.sun_resample_degrees);

test "the map goes stale once the sun has turned by the budget" {
    const fit = try SunShadowFit.compute(scene_bounds, sunAt(0), map_size);

    try testing.expect(!fit.stale(sunAt(0)));
    try testing.expect(!fit.stale(sunAt(0.99 * stale_angle)));
    try testing.expect(fit.stale(sunAt(1.01 * stale_angle)));
    // Both directions of rotation, since a test written on a squared chord
    // passes trivially in one and a signed comparison would not.
    try testing.expect(fit.stale(sunAt(-1.01 * stale_angle)));
    try testing.expect(!fit.stale(sunAt(-0.99 * stale_angle)));
}

test "staleness is an angle, the same at any resolution and scene size" {
    // The lookup follows the sun between bakes, so what bounds a map's life is
    // how far the correction reaches, an angle; neither the texel size nor the
    // scene enters it.
    const coarse = try SunShadowFit.compute(scene_bounds, sunAt(0), map_size);
    const fine = try SunShadowFit.compute(scene_bounds, sunAt(0), 4 * map_size);
    const huge = try SunShadowFit.compute(.{ .centre = .{ 0, 0, 0 }, .radius = 4000 }, sunAt(0), map_size);
    try testing.expectEqual(coarse.resample_chord_squared, fine.resample_chord_squared);
    try testing.expectEqual(coarse.resample_chord_squared, huge.resample_chord_squared);
}

test "an unusable sun direction leaves the fit standing" {
    const fit = try SunShadowFit.compute(scene_bounds, sunAt(0), map_size);
    // Not a number in, not stale out: keeping the last good fit beats baking one
    // through a matrix that has no directions in it.
    try testing.expect(!fit.stale(.{ 0, 0, 0 }));
    try testing.expect(!fit.stale(@splat(std.math.nan(f32))));
    // Length is not part of the question.
    try testing.expect(!fit.stale(.{ 0, 0, 1000 }));
    try testing.expect(fit.stale(.{ 1000, 0, 0 }));
}

test "bounds with no extent are refused" {
    const dir = unit(.{ 0, 1, 1 });
    // The empty scene, which is what Aabb.compute reports for zero vertices.
    try testing.expectError(error.DegenerateBounds, SunShadowFit.compute(
        .{ .centre = .{ 0, 0, 0 }, .radius = 0 },
        dir,
        map_size,
    ));
    // Below what zmath's own guard on the orthographic extent tolerates, and
    // that guard is compiled out of the shipping build.
    try testing.expectError(error.DegenerateBounds, SunShadowFit.compute(
        .{ .centre = .{ 0, 0, 0 }, .radius = 1e-4 },
        dir,
        map_size,
    ));
    inline for (.{ std.math.nan(f32), std.math.inf(f32) }) |bad| {
        try testing.expectError(error.DegenerateBounds, SunShadowFit.compute(
            .{ .centre = .{ 0, 0, 0 }, .radius = bad },
            dir,
            map_size,
        ));
        try testing.expectError(error.DegenerateBounds, SunShadowFit.compute(
            .{ .centre = .{ 1, bad, 3 }, .radius = 4 },
            dir,
            map_size,
        ));
    }
}

test "a sun direction with no direction in it is refused" {
    try testing.expectError(error.DegenerateSunDirection, SunShadowFit.compute(
        scene_bounds,
        .{ 0, 0, 0 },
        map_size,
    ));
    // Short enough that squaring it would lose the digits the normalization
    // needs.
    try testing.expectError(error.DegenerateSunDirection, SunShadowFit.compute(
        scene_bounds,
        .{ 1e-8, 0, 0 },
        map_size,
    ));
    inline for (.{ std.math.nan(f32), std.math.inf(f32) }) |bad| {
        try testing.expectError(error.DegenerateSunDirection, SunShadowFit.compute(
            scene_bounds,
            .{ 0, bad, 1 },
            map_size,
        ));
    }
}

test "the look settings convert texels and bound the strength" {
    const fit = try SunShadowFit.compute(scene_bounds, sunAt(0), map_size);
    const settings: scene.SunShadowSettings = .{
        .enabled = true,
        .normal_offset_texels = 3,
        .strength = 1.5,
    };
    try testing.expectApproxEqAbs(3 * fit.texel_world_size, settings.normalOffsetWorld(&fit), 1e-7);
    try testing.expectEqual(@as(f32, 1), settings.clampedStrength());

    const negative: scene.SunShadowSettings = .{ .enabled = true, .strength = -0.5 };
    try testing.expectEqual(@as(f32, 0), negative.clampedStrength());

    // A setting nobody filled in is switched off, so it takes nothing from a
    // surface. The offset is unaffected: it is a distance in the map's texels
    // and means the same whether or not the lookup is read.
    const default: scene.SunShadowSettings = .{};
    try testing.expectEqual(@as(f32, 0), default.clampedStrength());
    try testing.expectApproxEqAbs(fit.texel_world_size, default.normalOffsetWorld(&fit), 1e-7);
}

// Where a world point lands in the map, as the lookup addresses it: clip x
// and y mapped from [-1, 1] to [0, 1], and clip z as the depth.
fn mapPoint(fit: SunShadowFit, point: res.Vec3) [3]f32 {
    const clip = clipOf(fit, point);
    return .{ clip[0] * 0.5 + 0.5, clip[1] * 0.5 + 0.5, clip[2] };
}

test "the parallax is zero for the sun the map was baked for" {
    const sun = unit(.{ 0.3, 0.8, -0.5 });
    const fit = try SunShadowFit.compute(scene_bounds, sun, map_size);
    const shift = fit.parallax(sun);
    try testing.expectApproxEqAbs(@as(f32, 0), shift[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), shift[1], 1e-6);
    // The length of the direction does not enter.
    const long = fit.parallax(sun * @as(res.Vec3, @splat(40)));
    try testing.expectApproxEqAbs(@as(f32, 0), long[0], 1e-6);
}

test "the parallax finds the texel a ray toward the turned sun climbs over" {
    // Checked against the projection itself rather than the formula: a point
    // and a point further along the ray toward the turned sun, both carried
    // through the baked matrix. The lateral distance between their texels has
    // to be the shift times the depth between them, for either turn, for a
    // ray that climbs a little or a lot, and for a turn about any axis.
    const baked = unit(.{ 0.3, 0.8, -0.5 });
    const fit = try SunShadowFit.compute(scene_bounds, baked, map_size);
    const receiver: res.Vec3 = scene_bounds.centre + res.Vec3{ 0.7, -1.1, 0.4 };
    const turns = [_][3]f32{ .{ 0.004, 0, 0 }, .{ -0.004, 0, 0 }, .{ 0, 0.003, 0.002 }, .{ 0.001, -0.002, -0.003 } };
    for (turns) |turn| {
        const current = unit(baked + @as(res.Vec3, turn));
        const shift = fit.parallax(current);
        for ([_]f32{ 0.2, 1.5, 4 }) |distance| {
            const caster = receiver + current * @as(res.Vec3, @splat(distance));
            const from = mapPoint(fit, receiver);
            const to = mapPoint(fit, caster);
            const climbed = from[2] - to[2];
            try testing.expect(climbed > 0);
            try testing.expectApproxEqAbs(to[0] - from[0], shift[0] * climbed, 1e-6);
            try testing.expectApproxEqAbs(to[1] - from[1], shift[1] * climbed, 1e-6);
        }
    }
}

test "the parallax is the tangent of the turn" {
    // Width and depth share one scale in this map, so a ray turned by an angle
    // from the baked direction crosses the map at that angle's tangent. At the
    // budget this is what sizes the lookup's reach.
    const fit = try SunShadowFit.compute(scene_bounds, sunAt(0), map_size);
    const shift = fit.parallax(sunAt(stale_angle));
    const length = @sqrt(shift[0] * shift[0] + shift[1] * shift[1]);
    try testing.expectApproxEqRel(@tan(stale_angle), length, 1e-3);
}

test "a direction with no use in it makes no shift" {
    const fit = try SunShadowFit.compute(scene_bounds, sunAt(0), map_size);
    const nan = std.math.nan(f32);
    try testing.expectEqual([2]f32{ 0, 0 }, fit.parallax(.{ nan, 0, 1 }));
    // Turned past the horizon of the map: the ray no longer climbs.
    try testing.expectEqual([2]f32{ 0, 0 }, fit.parallax(sunAt(std.math.pi * 0.75)));
    try testing.expectEqual([2]f32{ 0, 0 }, fit.parallax(.{ 0, 0, 0 }));
}

test "the sun's size becomes a tangent, held to a sun and refused when not a number" {
    const sun: scene.SunShadowSettings = .{ .source_degrees = 0.267 };
    try testing.expectApproxEqAbs(@tan(std.math.degreesToRadians(@as(f32, 0.267))), sun.sourceTangent(), 1e-9);
    try testing.expectEqual(@as(f32, 0), (scene.SunShadowSettings{}).sourceTangent());
    try testing.expectEqual(@as(f32, 0), (scene.SunShadowSettings{ .source_degrees = -1 }).sourceTangent());
    try testing.expectEqual(@as(f32, 0), (scene.SunShadowSettings{ .source_degrees = std.math.nan(f32) }).sourceTangent());
    try testing.expectApproxEqAbs(@tan(std.math.degreesToRadians(@as(f32, 5))), (scene.SunShadowSettings{ .source_degrees = 40 }).sourceTangent(), 1e-7);
}

test "the switch decides whether a surface loses anything at all" {
    // A strength that is neither zero nor one, so a disabled setting returning
    // the strength, or an enabled one returning a constant, both show.
    const off: scene.SunShadowSettings = .{ .enabled = false, .strength = 0.75 };
    const on: scene.SunShadowSettings = .{ .enabled = true, .strength = 0.75 };

    try testing.expectEqual(@as(f32, 0), off.clampedStrength());
    try testing.expectEqual(@as(f32, 0.75), on.clampedStrength());

    // Off by default, so a setting nobody filled in draws no shadow.
    const untouched: scene.SunShadowSettings = .{};
    try testing.expectEqual(@as(f32, 0), untouched.clampedStrength());
}

test "an enabled setting still clamps a strength an asset got wrong" {
    const over: scene.SunShadowSettings = .{ .enabled = true, .strength = 4 };
    const under: scene.SunShadowSettings = .{ .enabled = true, .strength = -2 };

    try testing.expectEqual(@as(f32, 1), over.clampedStrength());
    try testing.expectEqual(@as(f32, 0), under.clampedStrength());
}

test "the normal offset follows the fit and not the switch" {
    // The offset is a distance in the map's own texels, so it means the same
    // thing whether or not the lookup is being read. Folding the switch into it
    // as well would make a disabled shadow change what a re-enabled one looks
    // like.
    const fit: scene.SunShadowFit = .{
        .view_proj = zm.identity(),
        .sun_dir = .{ 0, 1, 0 },
        .texel_world_size = 0.25,
        .resample_chord_squared = 1,
    };
    const off: scene.SunShadowSettings = .{ .enabled = false, .normal_offset_texels = 2 };
    const on: scene.SunShadowSettings = .{ .enabled = true, .normal_offset_texels = 2 };

    try testing.expectEqual(@as(f32, 0.5), off.normalOffsetWorld(&fit));
    try testing.expectEqual(@as(f32, 0.5), on.normalOffsetWorld(&fit));
}
