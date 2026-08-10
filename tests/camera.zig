const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const Camera = scene.Camera;
const Placement = scene.Placement;

const tolerance = 1e-5;
const aspect = 16.0 / 9.0;

fn expectVec(expected: [3]f32, actual: res.Vec3) !void {
    inline for (0..3) |axis|
        try testing.expectApproxEqAbs(expected[axis], actual[axis], tolerance);
}

fn dot(a: res.Vec3, b: res.Vec3) f32 {
    return @reduce(.Add, a * b);
}

fn clipOf(view_proj: zm.Mat, point: [3]f32) zm.Vec {
    return zm.mul(zm.f32x4(point[0], point[1], point[2], 1), view_proj);
}

fn expectOrthonormal(placement: Placement) !void {
    for ([_]res.Vec3{ placement.front, placement.right, placement.up }) |axis|
        try testing.expectApproxEqAbs(1.0, dot(axis, axis), tolerance);

    try testing.expectApproxEqAbs(0.0, dot(placement.front, placement.right), tolerance);
    try testing.expectApproxEqAbs(0.0, dot(placement.front, placement.up), tolerance);
    try testing.expectApproxEqAbs(0.0, dot(placement.right, placement.up), tolerance);

    // Right-handed, in the sense the view matrix depends on: right cross front
    // is up rather than its negation, which is what tells this basis from the
    // mirror image that satisfies every check above.
    const cross: res.Vec3 = .{
        placement.right[1] * placement.front[2] - placement.right[2] * placement.front[1],
        placement.right[2] * placement.front[0] - placement.right[0] * placement.front[2],
        placement.right[0] * placement.front[1] - placement.right[1] * placement.front[0],
    };
    try expectVec(.{ placement.up[0], placement.up[1], placement.up[2] }, cross);
}

test "the default pose looks down -Z with Y up" {
    const placement = (Camera{}).placement();
    try expectVec(.{ 0, 0, 0 }, placement.position);
    try expectVec(.{ 0, 0, -1 }, placement.front);
    try expectVec(.{ 1, 0, 0 }, placement.right);
    try expectVec(.{ 0, 1, 0 }, placement.up);
}

test "the basis stays orthonormal everywhere, the poles included" {
    const angles = [_]f32{ -3.0, -1.6, -std.math.pi / 2.0, -0.4, 0, 0.7, std.math.pi / 2.0, 2.9 };
    for (angles) |yaw| {
        for (angles) |pitch| {
            const camera: Camera = .{ .yaw = yaw, .pitch = pitch };
            try expectOrthonormal(camera.placement());
        }
    }

    // Straight down, where a basis taken from cross(front, world up) divides by
    // zero. `right` is horizontal by construction here, so it survives.
    const down: Camera = .{ .pitch = -std.math.pi / 2.0, .yaw = 0 };
    const placement = down.placement();
    try expectVec(.{ 0, -1, 0 }, placement.front);
    try expectVec(.{ 0, 0, 1 }, placement.right);
    try expectVec(.{ 1, 0, 0 }, placement.up);
}

test "yaw sweeps the horizon from +X toward +Z" {
    const east: Camera = .{ .yaw = 0 };
    try expectVec(.{ 1, 0, 0 }, east.placement().front);

    const quarter: Camera = .{ .yaw = std.math.pi / 2.0 };
    try expectVec(.{ 0, 0, 1 }, quarter.placement().front);

    // Pitch lifts the front vector and shortens its horizontal part by the
    // cosine, rather than leaving it unit in the plane.
    const lifted: Camera = .{ .yaw = 0, .pitch = std.math.pi / 3.0 };
    try expectVec(.{ 0.5, @sqrt(3.0) / 2.0, 0 }, lifted.placement().front);
}

test "the view matrix puts the eye at the origin looking down -Z" {
    const camera: Camera = .{ .anchor = .{ .eye = .{ 4, 5, 6 } }, .yaw = -0.9, .pitch = 0.3 };
    const placement = camera.placement();
    const view = placement.view();

    const eye = zm.mul(zm.f32x4(4, 5, 6, 1), view);
    try testing.expectApproxEqAbs(0.0, eye[0], tolerance);
    try testing.expectApproxEqAbs(0.0, eye[1], tolerance);
    try testing.expectApproxEqAbs(0.0, eye[2], tolerance);

    // A point one unit ahead sits on the negative Z axis of view space, which is
    // what right-handed means here and what the projection then expects.
    const ahead = placement.position + placement.front;
    const seen = zm.mul(zm.f32x4(ahead[0], ahead[1], ahead[2], 1), view);
    try testing.expectApproxEqAbs(0.0, seen[0], tolerance);
    try testing.expectApproxEqAbs(0.0, seen[1], tolerance);
    try testing.expectApproxEqAbs(-1.0, seen[2], tolerance);

    // And a point off to the right lands on positive X, not negative.
    const beside = placement.position + placement.right;
    const aside = zm.mul(zm.f32x4(beside[0], beside[1], beside[2], 1), view);
    try testing.expectApproxEqAbs(1.0, aside[0], tolerance);
}

test "an orbit anchor holds the eye at a distance behind the target" {
    const camera: Camera = .{
        .anchor = .{ .orbit = .{ .target = .{ 1, 2, 3 }, .distance = 10 } },
        .yaw = -std.math.pi / 2.0,
        .pitch = 0,
    };
    // Looking down -Z from ten units along +Z of the target.
    try expectVec(.{ 1, 2, 13 }, camera.placement().position);

    // Swinging the angles keeps the distance to the target exactly, which is the
    // property the reference's free-mode translate lost by leaving the target
    // where it was.
    var swung = camera;
    swung.yaw = 1.1;
    swung.pitch = -0.6;
    const offset = swung.placement().position - res.Vec3{ 1, 2, 3 };
    try testing.expectApproxEqAbs(10.0, @sqrt(dot(offset, offset)), 1e-4);

    // And the camera still looks at the target: the eye is behind it along the
    // view direction by construction.
    const placement = swung.placement();
    const to_target = res.Vec3{ 1, 2, 3 } - placement.position;
    try expectVec(.{ placement.front[0], placement.front[1], placement.front[2] }, to_target / @as(res.Vec3, @splat(10)));
}

test "aiming keeps the eye where it is" {
    var camera: Camera = .{ .anchor = .{ .eye = .{ 0, 0, 10 } } };
    camera.lookAt(.{ 0, 0, 0 });
    const placement = camera.placement();
    try expectVec(.{ 0, 0, 10 }, placement.position);
    try expectVec(.{ 0, 0, -1 }, placement.front);

    // From a corner, so both angles have to move and neither can be read off the
    // other.
    var oblique: Camera = .{ .anchor = .{ .eye = .{ 3, 4, 5 } } };
    oblique.lookAt(.{ 0, 0, 0 });
    const aimed = oblique.placement();
    const expected = res.Vec3{ -3, -4, -5 } / @as(res.Vec3, @splat(@sqrt(50.0)));
    try expectVec(.{ expected[0], expected[1], expected[2] }, aimed.front);
}

test "aiming straight up keeps the azimuth it had" {
    // The reference discarded the azimuth here, so tilting back down afterwards
    // faced a direction the camera had never been turned to.
    var camera: Camera = .{ .anchor = .{ .eye = .{ 0, 0, 0 } }, .yaw = 1.2, .pitch = 0 };
    camera.lookAt(.{ 0, 5, 0 });
    try testing.expectApproxEqAbs(1.2, camera.yaw, tolerance);
    try testing.expectApproxEqAbs(std.math.pi / 2.0, camera.pitch, tolerance);

    // A point on the eye names no direction, so nothing moves.
    var still: Camera = .{ .anchor = .{ .eye = .{ 7, 8, 9 } }, .yaw = 0.4, .pitch = -0.2 };
    still.lookAt(.{ 7, 8, 9 });
    try testing.expectEqual(@as(f32, 0.4), still.yaw);
    try testing.expectEqual(@as(f32, -0.2), still.pitch);
}

test "aiming an orbiting camera moves the pivot rather than the eye" {
    var camera: Camera = .{
        .anchor = .{ .orbit = .{ .target = .{ 0, 0, 0 }, .distance = 10 } },
        .yaw = -std.math.pi / 2.0,
    };
    try expectVec(.{ 0, 0, 10 }, camera.placement().position);

    camera.lookAt(.{ 4, 0, 10 });
    // The eye has not moved, the pivot is the point aimed at, and the distance
    // is the one between them. An implementation that only set the angles would
    // have swung the eye instead.
    try expectVec(.{ 0, 0, 10 }, camera.placement().position);
    try expectVec(.{ 1, 0, 0 }, camera.placement().front);
    try testing.expectApproxEqAbs(4.0, camera.anchor.orbit.distance, 1e-4);
}

test "taking a pivot turns the camera and leaves the eye alone" {
    var camera: Camera = .{ .anchor = .{ .eye = .{ 2, 3, 4 } }, .yaw = 0.8, .pitch = -0.3 };

    camera.orbitAround(.{ 0, 0, 0 });
    // The eye stays and the angles turn to the pivot. They have to: an orbit
    // places the eye behind its target along the view direction, so keeping the
    // angles instead would have moved the camera to wherever they pointed.
    try expectVec(.{ 2, 3, 4 }, camera.placement().position);
    try testing.expectApproxEqAbs(@sqrt(29.0), camera.anchor.orbit.distance, 1e-4);
    const to_pivot = res.Vec3{ -2, -3, -4 } / @as(res.Vec3, @splat(@sqrt(29.0)));
    try expectVec(.{ to_pivot[0], to_pivot[1], to_pivot[2] }, camera.placement().front);

    // Detaching changes nothing that is visible; it only stops the angles from
    // carrying the eye with them.
    const orbiting = camera.placement();
    camera.detach();
    try expectVec(.{ orbiting.position[0], orbiting.position[1], orbiting.position[2] }, camera.placement().position);
    try expectVec(.{ orbiting.front[0], orbiting.front[1], orbiting.front[2] }, camera.placement().front);
}

test "taking a pivot the camera already faces leaves the angles untouched" {
    var camera: Camera = .{ .anchor = .{ .eye = .{ 0, 0, 10 } }, .yaw = -std.math.pi / 2.0, .pitch = 0 };
    camera.orbitAround(.{ 0, 0, 0 });
    try testing.expectApproxEqAbs(-std.math.pi / 2.0, camera.yaw, tolerance);
    try testing.expectApproxEqAbs(0.0, camera.pitch, tolerance);
    try testing.expectApproxEqAbs(10.0, camera.anchor.orbit.distance, 1e-4);
}

test "the projection maps the near plane to zero depth and the far plane to one" {
    const camera: Camera = .{ .projection = .{ .perspective = .{ .near = 0.5, .far = 100 } } };
    const projection = try camera.projectionMatrix(aspect);

    // In view space, which is where the projection starts: right-handed, so what
    // is in front sits on negative Z.
    const near = zm.mul(zm.f32x4(0, 0, -0.5, 1), projection);
    try testing.expectApproxEqAbs(0.0, near[2] / near[3], tolerance);

    const far = zm.mul(zm.f32x4(0, 0, -100, 1), projection);
    try testing.expectApproxEqAbs(1.0, far[2] / far[3], tolerance);

    // Vulkan's range, not OpenGL's: the near plane is at 0 and not at -1.
    try testing.expect(near[2] / near[3] > -0.5);
}

test "the aspect ratio widens the horizontal extent and leaves the vertical one" {
    const camera: Camera = .{ .projection = .{ .perspective = .{ .fov_y = std.math.pi / 2.0, .near = 1, .far = 100 } } };
    const square = try camera.projectionMatrix(1);
    const wide = try camera.projectionMatrix(2);

    // A 90 degree vertical field of view reaches one unit up at one unit ahead.
    const top = zm.mul(zm.f32x4(0, 1, -1, 1), square);
    try testing.expectApproxEqAbs(1.0, top[1] / top[3], tolerance);

    // Twice as wide a target sees twice as far sideways for the same vertical
    // field of view, so the same point falls to half the clip coordinate.
    const side_square = zm.mul(zm.f32x4(1, 0, -1, 1), square);
    const side_wide = zm.mul(zm.f32x4(1, 0, -1, 1), wide);
    try testing.expectApproxEqAbs(1.0, side_square[0] / side_square[3], tolerance);
    try testing.expectApproxEqAbs(0.5, side_wide[0] / side_wide[3], tolerance);

    // The vertical extent is untouched by the aspect ratio.
    const top_wide = zm.mul(zm.f32x4(0, 1, -1, 1), wide);
    try testing.expectApproxEqAbs(1.0, top_wide[1] / top_wide[3], tolerance);
}

test "the camera applies no Y flip" {
    // The flip between a Y-up clip space and a framebuffer that counts Y
    // downward belongs to whoever owns that framebuffer. A point above the eye
    // line therefore has to land above the centre here.
    const camera: Camera = .{ .anchor = .{ .eye = .{ 0, 0, 0 } } };
    const view_proj = try camera.viewProjection(aspect);
    const above = clipOf(view_proj, .{ 0, 1, -5 });
    try testing.expect(above[1] > 0);

    const right = clipOf(view_proj, .{ 1, 0, -5 });
    try testing.expect(right[0] > 0);
}

test "the view-projection is the view followed by the projection" {
    const camera: Camera = .{ .anchor = .{ .eye = .{ -2, 1, 6 } }, .yaw = -1.4, .pitch = 0.2 };
    const composed = try camera.viewProjection(aspect);
    const expected = zm.mul(camera.placement().view(), try camera.projectionMatrix(aspect));

    // The order is the one the frustum extraction reads columns out of, and the
    // reverse product is a different matrix entirely.
    inline for (0..4) |row| {
        inline for (0..4) |column|
            try testing.expectApproxEqAbs(expected[row][column], composed[row][column], tolerance);
    }
}

test "a target with no shape is refused" {
    const camera: Camera = .{};
    for ([_]f32{ 0, -1, 0.001, std.math.nan(f32), std.math.inf(f32) }) |bad|
        try testing.expectError(error.DegenerateAspect, camera.projectionMatrix(bad));
}

test "a projection that describes no volume is refused" {
    // Each of these clears one of the guards that zmath asserts and ReleaseFast
    // then removes.
    const cases = [_]scene.Projection.Perspective{
        .{ .fov_y = 0 },
        .{ .fov_y = std.math.nan(f32) },
        .{ .fov_y = std.math.pi },
        .{ .near = 0 },
        .{ .near = -1 },
        .{ .near = 10, .far = 10 },
        .{ .near = 10, .far = 5 },
        .{ .far = std.math.nan(f32) },
    };
    for (cases) |perspective| {
        const camera: Camera = .{ .projection = .{ .perspective = perspective } };
        try testing.expectError(error.DegenerateProjection, camera.projectionMatrix(aspect));
    }

    // The defaults are not among them.
    _ = try (Camera{}).projectionMatrix(aspect);
}

test "a ray from the basis lands on the device coordinate it was built for" {
    const camera: Camera = .{ .anchor = .{ .eye = .{ -2, 1, 6 } }, .yaw = -1.4, .pitch = 0.2 };
    const basis = try camera.rayBasis(aspect);
    const view_proj = try camera.viewProjection(aspect);
    const eye = camera.placement().position;

    // The corners and the centre. This is the whole contract of the basis, and
    // it is stated against the matrix the geometry path uses rather than against
    // a second derivation of the same trigonometry: the two agree or a
    // background drawn from one does not line up with a scene drawn from the
    // other.
    const corners = [_][2]f32{
        .{ 0, 0 },
        .{ 1, 1 },
        .{ -1, 1 },
        .{ 1, -1 },
        .{ -1, -1 },
        .{ 0.3, -0.7 },
    };
    for (corners) |device| {
        const ray = basis.front +
            basis.right * @as(res.Vec3, @splat(device[0])) +
            basis.up * @as(res.Vec3, @splat(device[1]));
        const point = eye + ray;
        const clip = clipOf(view_proj, .{ point[0], point[1], point[2] });

        try testing.expect(clip[3] > 0);
        try testing.expectApproxEqAbs(device[0], clip[0] / clip[3], tolerance);
        try testing.expectApproxEqAbs(device[1], clip[1] / clip[3], tolerance);
    }
}

test "the ray basis does not depend on where the eye stands" {
    const angles: Camera = .{ .yaw = 0.9, .pitch = -0.35 };
    var here = angles;
    here.anchor = .{ .eye = .{ 0, 0, 0 } };
    var far_away = angles;
    far_away.anchor = .{ .eye = .{ 1200, -400, 2500 } };

    // What pins a background at infinity: translation moves the eye and not the
    // ray, so the picture parallaxes only under rotation.
    const from_here = try here.rayBasis(aspect);
    const from_far = try far_away.rayBasis(aspect);
    try expectVec(.{ from_here.right[0], from_here.right[1], from_here.right[2] }, from_far.right);
    try expectVec(.{ from_here.up[0], from_here.up[1], from_here.up[2] }, from_far.up);
    try expectVec(.{ from_here.front[0], from_here.front[1], from_here.front[2] }, from_far.front);
}

test "a ray basis is refused wherever a projection is" {
    const camera: Camera = .{};
    for ([_]f32{ 0, -1, 0.001, std.math.nan(f32), std.math.inf(f32) }) |bad|
        try testing.expectError(error.DegenerateAspect, camera.rayBasis(bad));

    const cases = [_]scene.Projection.Perspective{
        .{ .fov_y = 0 },
        .{ .fov_y = std.math.pi },
        .{ .near = 0 },
        .{ .near = 10, .far = 5 },
    };
    for (cases) |perspective| {
        const degenerate: Camera = .{ .projection = .{ .perspective = perspective } };
        try testing.expectError(error.DegenerateProjection, degenerate.rayBasis(aspect));
    }
}
