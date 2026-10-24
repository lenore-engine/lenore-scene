const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Sphere = res.Sphere;
const SunShadowFit = scene.SunShadowFit;
const SunCascades = scene.SunCascades;

// A scene as large as the test stand's middle tier, off the origin.
const world: Sphere = .{ .centre = .{ 3, 1, -2 }, .radius = 360 };
const settings: scene.SunCascadeSettings = .{ .fine_size = 4096, .coarse_size = 2048 };

fn unit(v: [3]f32) res.Vec3 {
    const as_vector: res.Vec3 = v;
    return as_vector / @as(res.Vec3, @splat(@sqrt(@reduce(.Add, as_vector * as_vector))));
}

const sun = unit(.{ 0.4, 0.7, -0.3 });

fn clipOf(fit: SunShadowFit, point: res.Vec3) zm.Vec {
    return zm.mul(zm.f32x4(point[0], point[1], point[2], 1), fit.view_proj);
}

fn sameMatrix(a: zm.Mat, b: zm.Mat) bool {
    for (0..4) |row| {
        if (@reduce(.Or, a[row] != b[row])) return false;
    }
    return true;
}

test "a cascade covers its square across the rays and the whole scene along them" {
    const at: res.Vec3 = .{ 40, 2, 15 };
    const fit = try SunShadowFit.computeCascade(world, sun, 4096, 24, at);
    const at_clip = clipOf(fit, at);
    // Snapped, so the camera is within half a texel of the middle.
    try testing.expect(@abs(at_clip[0]) * fit.lateral_span / 2 <= fit.texel_world_size);
    try testing.expect(@abs(at_clip[1]) * fit.lateral_span / 2 <= fit.texel_world_size);
    // A point 11.9 m across the rays stays inside; the margin is two texels.
    const across = unit(zm.vecToArr3(zm.cross3(zm.f32x4(sun[0], sun[1], sun[2], 0), zm.f32x4(0, 1, 0, 0))));
    const inside = clipOf(fit, at + across * @as(res.Vec3, @splat(11.9)));
    try testing.expect(@abs(inside[0]) < 1 and @abs(inside[1]) < 1);
    const outside = clipOf(fit, at + across * @as(res.Vec3, @splat(12.5)));
    try testing.expect(@abs(outside[0]) > 1 or @abs(outside[1]) > 1);
    // Toward the sun, to the scene sphere's edge and no farther.
    const toward = world.radius - 1;
    const high = clipOf(fit, at + sun * @as(res.Vec3, @splat(toward - @reduce(.Add, (at - world.centre) * sun))));
    try testing.expect(high[2] >= 0 and high[2] < 0.01);
    try testing.expectApproxEqRel(2 * world.radius, fit.depth_span, 0.01);
    try testing.expectApproxEqRel(@as(f32, 24), fit.lateral_span, 0.01);
    try testing.expectApproxEqRel(fit.lateral_span / 4096, fit.texel_world_size, 1e-6);
}

test "a camera moving less than a texel keeps the cascade bitwise" {
    const a = try SunShadowFit.computeCascade(world, sun, 4096, 24, .{ 40, 2, 15 });
    const b = try SunShadowFit.computeCascade(world, sun, 4096, 24, .{ 40.0005, 2, 15.0005 });
    try testing.expect(sameMatrix(a.view_proj, b.view_proj));
}

test "a moved centre moves the map by whole texels" {
    const a = try SunShadowFit.computeCascade(world, sun, 4096, 24, .{ 40, 2, 15 });
    const b = try SunShadowFit.computeCascade(world, sun, 4096, 24, .{ 47.3, 2, 9.1 });
    const point: res.Vec3 = .{ 43, 0, 12 };
    const pa = clipOf(a, point);
    const pb = clipOf(b, point);
    const texels_x = (pb[0] - pa[0]) * 4096 / 2;
    const texels_y = (pb[1] - pa[1]) * 4096 / 2;
    try testing.expect(@abs(texels_x) + @abs(texels_y) > 100);
    try testing.expectApproxEqAbs(@round(texels_x), texels_x, 0.02);
    try testing.expectApproxEqAbs(@round(texels_y), texels_y, 0.02);
}

test "parallax and staleness hold for a cascade as for the single fit" {
    const fit = try SunShadowFit.computeCascade(world, sun, 4096, 24, .{ 40, 2, 15 });
    try testing.expect(!fit.stale(sun));
    const zero = fit.parallax(sun);
    try testing.expectApproxEqAbs(@as(f32, 0), zero[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), zero[1], 1e-5);
    // A point and a caster above it toward the turned sun land on one texel
    // once the shift is applied, as tests/sun_shadow.zig checks the single fit.
    const turned = unit(.{ 0.41, 0.7, -0.3 });
    const shift = fit.parallax(turned);
    const receiver: res.Vec3 = .{ 41, 0, 14 };
    const caster = receiver + turned * @as(res.Vec3, @splat(3));
    const r = clipOf(fit, receiver);
    const c = clipOf(fit, caster);
    const depth = r[2] - c[2];
    const expected_x = (r[0] * 0.5 + 0.5) + shift[0] * depth;
    const expected_y = (r[1] * 0.5 + 0.5) + shift[1] * depth;
    try testing.expectApproxEqAbs(c[0] * 0.5 + 0.5, expected_x, 1e-4);
    try testing.expectApproxEqAbs(c[1] * 0.5 + 0.5, expected_y, 1e-4);
}

test "degenerate cascade input is an error" {
    try testing.expectError(error.DegenerateBounds, SunShadowFit.computeCascade(world, sun, 4096, 0, .{ 0, 0, 0 }));
    try testing.expectError(error.DegenerateBounds, SunShadowFit.computeCascade(world, sun, 4096, 24, .{ std.math.nan(f32), 0, 0 }));
    try testing.expectError(error.DegenerateSunDirection, SunShadowFit.computeCascade(world, .{ 0, 0, 0 }, 4096, 24, .{ 0, 0, 0 }));
}

test "a small scene keeps one map around its sphere" {
    var cascades: SunCascades = .{};
    const room: Sphere = .{ .centre = .{ 0, 1, 0 }, .radius = 10 };
    try cascades.refit(room, sun, .{ 0, 1, 0 }, settings);
    try testing.expectEqual(@as(u32, 1), cascades.count);
    const only = cascades.cascades[0];
    try testing.expectEqual(scene.SunCascadeImage.fine, only.image);
    const single = try SunShadowFit.compute(room, sun, 4096);
    try testing.expect(sameMatrix(single.view_proj, only.target.?.view_proj));
}

test "a scene between the two widths has a near cascade and the scene map" {
    var cascades: SunCascades = .{};
    try cascades.refit(.{ .centre = .{ 0, 0, 0 }, .radius = 40 }, sun, .{ 0, 0, 0 }, settings);
    try testing.expectEqual(@as(u32, 2), cascades.count);
    try testing.expectEqual(scene.SunCascadeImage.coarse, cascades.cascades[1].image);
    try testing.expectEqual(@as(f32, 0), cascades.cascades[1].width);
}

fn bakeAll(cascades: *SunCascades) !void {
    var frames: u32 = 0;
    while (cascades.next()) |index| : (frames += 1) {
        cascades.baked(index);
        try testing.expect(frames < 8);
    }
}

test "one cascade bakes per frame, the finer first, and a passed one is served next" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try testing.expectEqual(@as(u32, 3), cascades.count);
    try testing.expectEqual(@as(?u32, 0), cascades.next());
    cascades.baked(0);
    try testing.expectEqual(@as(?u32, 1), cascades.next());
    cascades.baked(1);
    // The near cascade asks again before the far one has had its turn: the
    // far one has waited two frames, so it goes first.
    cascades.invalidate();
    cascades.cascades[2].waiting = 2;
    try testing.expectEqual(@as(?u32, 2), cascades.next());
}

test "casters that move every frame still bake every cascade in turn" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try bakeAll(&cascades);
    var seen = [_]u32{ 0, 0, 0 };
    for (0..9) |_| {
        cascades.invalidate();
        const index = cascades.next().?;
        seen[index] += 1;
        cascades.baked(index);
    }
    for (seen) |count| try testing.expect(count >= 2);
}

test "walking moves the near square after a quarter of its width and the far map never" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try bakeAll(&cascades);
    try cascades.refit(world, sun, .{ 0, 0, 4 }, settings);
    try testing.expect(cascades.next() == null);
    try cascades.refit(world, sun, .{ 0, 0, 14 }, settings);
    try testing.expect(cascades.cascades[0].target != null);
    try testing.expect(cascades.cascades[1].target == null);
    try testing.expect(cascades.cascades[2].target == null);
    try bakeAll(&cascades);
    try cascades.refit(world, sun, .{ 0, 0, 300 }, settings);
    try testing.expect(cascades.cascades[1].target != null);
    try testing.expect(cascades.cascades[2].target == null);
}

test "a moved square fades from the old one; changed casters fade from nothing" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try bakeAll(&cascades);
    try testing.expect(cascades.cascades[0].previous == null);
    const first_layer = cascades.cascades[0].shownLayer();

    try cascades.refit(world, sun, .{ 0, 0, 14 }, settings);
    try testing.expectEqual(1 - first_layer, cascades.cascades[0].bakeLayer());
    try bakeAll(&cascades);
    try testing.expect(cascades.cascades[0].previous != null);
    try testing.expectEqual(1 - first_layer, cascades.cascades[0].shownLayer());
    try testing.expectEqual(first_layer, cascades.cascades[0].previousLayer());
    try testing.expectEqual(@as(f32, 0), cascades.cascades[0].share());
    cascades.advance(0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), cascades.cascades[0].share(), 1e-6);
    cascades.advance(0.6);
    try testing.expect(cascades.cascades[0].previous == null);

    cascades.invalidate();
    try bakeAll(&cascades);
    for (cascades.active()) |cascade| try testing.expect(cascade.previous == null);
}

test "a turned sun re-aims every cascade" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try bakeAll(&cascades);
    try cascades.refit(world, unit(.{ 0.4, 0.7005, -0.3 }), .{ 0, 0, 0 }, settings);
    try testing.expect(cascades.next() == null);
    try cascades.refit(world, unit(.{ 0.45, 0.7, -0.3 }), .{ 0, 0, 0 }, settings);
    for (cascades.active()) |cascade| try testing.expect(cascade.target != null);
}

test "the coarse cascades take two layers each of the small image" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try testing.expectEqual(@as(u32, 0), cascades.cascades[0].base_layer);
    try testing.expectEqual(@as(u32, 0), cascades.cascades[1].base_layer);
    try testing.expectEqual(@as(u32, 2), cascades.cascades[2].base_layer);
    try testing.expectEqual(@as(u32, 2048), @as(u32, @intFromFloat(cascades.cascades[1].target.?.lateral_span / cascades.cascades[1].target.?.texel_world_size + 0.5)));
}

test "another scene sphere starts the cascades over" {
    var cascades: SunCascades = .{};
    try cascades.refit(world, sun, .{ 0, 0, 0 }, settings);
    try bakeAll(&cascades);
    var larger = world;
    larger.radius += 10;
    try cascades.refit(larger, sun, .{ 0, 0, 0 }, settings);
    for (cascades.active()) |cascade| {
        try testing.expect(cascade.shown == null);
        try testing.expect(cascade.target != null);
    }
}
