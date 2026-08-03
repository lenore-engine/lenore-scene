const std = @import("std");
const zm = @import("zmath");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const LocalFogVolume = scene.LocalFogVolume;

const tolerance = 1e-5;
const nan = std.math.nan(f32);
const inf = std.math.inf(f32);

fn expectVec(expected: [3]f32, actual: res.Vec3) !void {
    inline for (0..3) |axis|
        try testing.expectApproxEqAbs(expected[axis], actual[axis], tolerance);
}

// A unit sphere at the origin with a quarter of its radius as shell.
const unit_ball: LocalFogVolume = .{ .density = 1, .edge_falloff = 0.25 };

test "the shell profile is flat in the core and zero outside" {
    const placed = try unit_ball.place();
    try testing.expectEqual(@as(f32, 1), placed.shellDensity(0));
    try testing.expectEqual(@as(f32, 1), placed.shellDensity(0.7));
    try testing.expectEqual(@as(f32, 1), placed.shellDensity(0.75));
    try testing.expectEqual(@as(f32, 0), placed.shellDensity(1));
    try testing.expectEqual(@as(f32, 0), placed.shellDensity(2));

    // Halfway through the shell a smoothstep is exactly half, whatever its
    // width, which a linear ramp also satisfies at that one point.
    try testing.expectApproxEqAbs(0.5, placed.shellDensity(0.875), tolerance);

    // A quarter of the way in it is not: smoothstep(0.25) is 0.15625, so the
    // density is 1 - that. This is what tells the curve from the ramp.
    try testing.expectApproxEqAbs(1 - 0.15625, placed.shellDensity(0.8125), tolerance);
}

test "a volume that fades from its centre has no core at all" {
    const soft: LocalFogVolume = .{ .density = 2, .edge_falloff = 1 };
    const placed = try soft.place();
    try testing.expectApproxEqAbs(2.0, placed.shellDensity(0), tolerance);
    try testing.expectApproxEqAbs(1.0, placed.shellDensity(0.5), tolerance);
    try testing.expectApproxEqAbs(0.0, placed.shellDensity(1), tolerance);
}

test "density is read in the volume's own space" {
    // Three different radii and an offset centre, so a placement that forgets
    // either lands somewhere else.
    const stretched: LocalFogVolume = .{
        .centre = .{ 10, 0, 0 },
        .radii = .{ 4, 1, 2 },
        .density = 1,
        .edge_falloff = 0.25,
    };
    const placed = try stretched.place();

    try testing.expectApproxEqAbs(1.0, placed.densityAt(.{ 10, 0, 0 }), tolerance);
    // Ends of each semi-axis are the boundary, whatever the axis measures.
    try testing.expectEqual(@as(f32, 0), placed.densityAt(.{ 14, 0, 0 }));
    try testing.expectEqual(@as(f32, 0), placed.densityAt(.{ 10, 1, 0 }));
    try testing.expectEqual(@as(f32, 0), placed.densityAt(.{ 10, 0, 2 }));
    // Half way out along the long axis is still core.
    try testing.expectApproxEqAbs(1.0, placed.densityAt(.{ 12, 0, 0 }), tolerance);
    // The same world offset along the short axis is already in the shell.
    try testing.expect(placed.densityAt(.{ 10, 0.8, 0 }) < 1);
}

test "a rotation turns the volume and not only its bounds" {
    // A long thin volume along X, turned a quarter turn about Z so it lies
    // along Y. A point three units up is then inside, and one three units out
    // along X is not.
    const turned: LocalFogVolume = .{
        .radii = .{ 4, 1, 1 },
        .rotation = zm.quatFromNormAxisAngle(zm.f32x4(0, 0, 1, 0), std.math.pi / 2.0),
        .density = 1,
        .edge_falloff = 0.25,
    };
    const placed = try turned.place();
    try testing.expectApproxEqAbs(1.0, placed.densityAt(.{ 0, 2, 0 }), tolerance);
    try testing.expectEqual(@as(f32, 0), placed.densityAt(.{ 2, 0, 0 }));
}

test "the world-to-unit transform undoes the rotation rather than repeating it" {
    // A turn about a principal axis cannot tell these apart: reflecting an
    // angle leaves an axis-aligned ellipsoid where it was, so the inverse
    // rotation and the rotation itself agree on every point. A general axis
    // with three distinct radii does not have that symmetry.
    const volume: LocalFogVolume = .{
        .radii = .{ 3, 1, 0.5 },
        .rotation = zm.quatFromNormAxisAngle(zm.normalize3(zm.f32x4(1, 2, -1, 0)), 0.9),
        .density = 1,
        .edge_falloff = 0.25,
    };
    const placed = try volume.place();

    // Where the tip of the long semi-axis lands, computed forwards.
    const linear = zm.mul(zm.scaling(3, 1, 0.5), zm.quatToMat(zm.normalize4(volume.rotation)));
    const tip = zm.mul(zm.f32x4(1, 0, 0, 0), linear);
    const along = res.Vec3{ tip[0], tip[1], tip[2] };

    // Half way out is core, just past the tip is outside, and the tip itself is
    // the boundary. Composing the rotation the wrong way round sends all three
    // somewhere else on the ellipsoid.
    try testing.expectApproxEqAbs(1.0, placed.densityAt(along * @as(res.Vec3, @splat(0.5))), tolerance);
    try testing.expectApproxEqAbs(0.0, placed.densityAt(along * @as(res.Vec3, @splat(1.01))), tolerance);
    try testing.expectApproxEqAbs(0.0, placed.densityAt(along), tolerance);
}

test "a query that is not a number reads as no density" {
    // The set sums the volumes that cover a point, so one volume answering with
    // something that is not a number would poison the total for all of them.
    const placed = try unit_ball.place();
    try testing.expectEqual(@as(f32, 0), placed.densityAt(.{ nan, 0, 0 }));
    try testing.expectEqual(@as(f32, 0), placed.shellDensity(nan));
}

test "an unnormalized rotation is accepted and normalized" {
    var scaled = unit_ball;
    scaled.rotation = zm.quatFromNormAxisAngle(zm.f32x4(0, 1, 0, 0), 0.7) * @as(zm.Quat, @splat(9));
    const placed = try scaled.place();
    // A sphere is unaffected by any rotation, so what this checks is that the
    // transform stayed a rotation rather than picking up the factor of nine.
    try testing.expectApproxEqAbs(0.0, placed.densityAt(.{ 1.01, 0, 0 }), tolerance);
    try testing.expectApproxEqAbs(1.0, placed.densityAt(.{ 0.7, 0, 0 }), tolerance);
}

test "the bounds are the tight box of the ellipsoid, not of a sphere around it" {
    const stretched: LocalFogVolume = .{ .centre = .{ 1, 2, 3 }, .radii = .{ 4, 1, 2 } };
    const axis_aligned = try stretched.place();
    try expectVec(.{ -3, 1, 1 }, axis_aligned.bounds.min);
    try expectVec(.{ 5, 3, 5 }, axis_aligned.bounds.max);

    // Turned an eighth of a turn about Z, the long axis projects onto both X
    // and Y by cos(45) times its length. A box taken from the largest radius
    // would give four on every axis instead.
    var turned = stretched;
    turned.rotation = zm.quatFromNormAxisAngle(zm.f32x4(0, 0, 1, 0), 0.25 * std.math.pi);
    const diagonal = try turned.place();
    const reach = @sqrt(0.5 * (16.0 + 1.0));
    try testing.expectApproxEqAbs(reach, diagonal.bounds.max[0] - 1, 1e-4);
    try testing.expectApproxEqAbs(reach, diagonal.bounds.max[1] - 2, 1e-4);
    try testing.expectApproxEqAbs(2.0, diagonal.bounds.max[2] - 3, 1e-4);
}

test "the bounds enclose the surface and touch it" {
    // Sampled rather than reasoned: every point of the ellipsoid's surface has
    // to be inside the box, and the box has to be reached on each axis.
    const volume: LocalFogVolume = .{
        .centre = .{ -2, 1, 0.5 },
        .radii = .{ 3, 0.5, 1.5 },
        .rotation = zm.quatFromNormAxisAngle(zm.normalize3(zm.f32x4(1, 2, -1, 0)), 0.9),
    };
    const placed = try volume.place();
    const linear = zm.mul(zm.scaling(3, 0.5, 1.5), zm.quatToMat(zm.normalize4(volume.rotation)));

    var reached: res.Vec3 = @splat(0);
    var seed: u32 = 12345;
    for (0..2000) |_| {
        // A crude but deterministic direction, normalized onto the sphere.
        var direction: res.Vec3 = undefined;
        inline for (0..3) |axis| {
            seed = seed *% 1664525 +% 1013904223;
            direction[axis] = @as(f32, @floatFromInt(seed >> 8)) / 8388608.0 - 1;
        }
        const length = @sqrt(@reduce(.Add, direction * direction));
        if (!(length > 0.1)) continue;
        const unit = direction / @as(res.Vec3, @splat(length));

        const world = zm.mul(zm.f32x4(unit[0], unit[1], unit[2], 0), linear);
        const point = volume.centre + res.Vec3{ world[0], world[1], world[2] };
        try testing.expect(@reduce(.And, point >= placed.bounds.min - @as(res.Vec3, @splat(1e-4))));
        try testing.expect(@reduce(.And, point <= placed.bounds.max + @as(res.Vec3, @splat(1e-4))));
        reached = @max(reached, @abs(point - volume.centre));
    }

    // And the box is not loose: two thousand directions come within a percent
    // of every face.
    const half = (placed.bounds.max - placed.bounds.min) * @as(res.Vec3, @splat(0.5));
    inline for (0..3) |axis|
        try testing.expect(reached[axis] > 0.99 * half[axis]);
}

test "a chord through the centre spans the diameter" {
    const placed = try unit_ball.place();
    const span = placed.chordSpan(.{ 0, 0, -10 }, .{ 0, 0, 1 }, inf).?;
    try testing.expectApproxEqAbs(9.0, span[0], tolerance);
    try testing.expectApproxEqAbs(11.0, span[1], tolerance);
}

test "a chord in a stretched volume is measured in world units" {
    // The direction is not renormalized in unit space, which is what keeps the
    // parameter a world distance: along the four-unit axis the chord is eight.
    const stretched: LocalFogVolume = .{ .radii = .{ 4, 1, 1 } };
    const placed = try stretched.place();
    const span = placed.chordSpan(.{ -10, 0, 0 }, .{ 1, 0, 0 }, inf).?;
    try testing.expectApproxEqAbs(6.0, span[0], tolerance);
    try testing.expectApproxEqAbs(14.0, span[1], tolerance);
}

test "a chord through an off-centre volume is placed by its centre" {
    // Every other chord case sits at the origin, where the translation part of
    // the transform is zero and a direction carried as a position looks the
    // same as one carried as a direction.
    const moved: LocalFogVolume = .{ .centre = .{ 5, -3, 2 } };
    const placed = try moved.place();
    const span = placed.chordSpan(.{ 5, -3, -8 }, .{ 0, 0, 1 }, inf).?;
    try testing.expectApproxEqAbs(9.0, span[0], tolerance);
    try testing.expectApproxEqAbs(11.0, span[1], tolerance);

    // And a ray along the same line through the origin misses it entirely.
    try testing.expectEqual(@as(?[2]f32, null), placed.chordSpan(.{ 0, 0, -8 }, .{ 0, 0, 1 }, inf));
}

test "a ray that misses, grazes or points away has no chord" {
    const placed = try unit_ball.place();
    try testing.expectEqual(@as(?[2]f32, null), placed.chordSpan(.{ 0, 2, -10 }, .{ 0, 0, 1 }, inf));
    // Tangent: one root, no length inside, nothing to integrate.
    try testing.expectEqual(@as(?[2]f32, null), placed.chordSpan(.{ 0, 1, -10 }, .{ 0, 0, 1 }, inf));
    // Behind the origin.
    try testing.expectEqual(@as(?[2]f32, null), placed.chordSpan(.{ 0, 0, -10 }, .{ 0, 0, -1 }, inf));
}

test "a chord is clipped to the ray it is asked about" {
    const placed = try unit_ball.place();
    // The far end is cut by the limit.
    const clipped = placed.chordSpan(.{ 0, 0, -10 }, .{ 0, 0, 1 }, 10).?;
    try testing.expectApproxEqAbs(9.0, clipped[0], tolerance);
    try testing.expectApproxEqAbs(10.0, clipped[1], tolerance);

    // A limit before the volume leaves nothing.
    try testing.expectEqual(@as(?[2]f32, null), placed.chordSpan(.{ 0, 0, -10 }, .{ 0, 0, 1 }, 5));

    // An origin inside starts at zero rather than behind the viewer.
    const inside = placed.chordSpan(.{ 0, 0, 0 }, .{ 0, 0, 1 }, inf).?;
    try testing.expectEqual(@as(f32, 0), inside[0]);
    try testing.expectApproxEqAbs(1.0, inside[1], tolerance);
}

fn pointOn(origin: res.Vec3, direction: res.Vec3, t: f32) res.Vec3 {
    return origin + direction * @as(res.Vec3, @splat(t));
}

// Dense midpoint integration of the same profile, as the oracle the quadrature
// is checked against.
fn denseTau(placed: LocalFogVolume.Placed, origin: res.Vec3, direction: res.Vec3, from: f32, to: f32) f32 {
    const steps = 20000;
    const step = (to - from) / steps;
    var total: f32 = 0;
    for (0..steps) |i| {
        const t = from + step * (@as(f32, @floatFromInt(i)) + 0.5);
        total += placed.densityAt(origin + direction * @as(res.Vec3, @splat(t))) * step;
    }
    return total;
}

test "the quadrature agrees with dense integration over a whole chord" {
    const placed = try unit_ball.place();
    const origin = res.Vec3{ 0, 0, -10 };
    const direction = res.Vec3{ 0, 0, 1 };

    // The exact answer is 1.75 for this profile: a core of 1.5 diameters at full
    // density, plus a shell whose smoothstep integrates to half of its width at
    // each end. Dense midpoint integration finds it, and the four-point rule
    // comes within 2.3 per cent, measured.
    const quadrature = placed.segmentTau(origin, direction, 0, 100);
    const dense = denseTau(placed, origin, direction, 9, 11);
    try testing.expectApproxEqAbs(1.75, dense, 1e-3);
    try testing.expectApproxEqAbs(dense, quadrature, 0.03 * dense);

    // Two points would be cheaper and are not enough, which is the reason four
    // are used. Both of its abscissae land at 0.577 of the chord, inside the
    // flat core, so it reports the core density over the whole span and misses
    // every part of the shell: 2.0 against 1.75, over by 14 per cent.
    const two_point = placed.densityAt(pointOn(origin, direction, 10 - 0.5773502692)) +
        placed.densityAt(pointOn(origin, direction, 10 + 0.5773502692));
    try testing.expectApproxEqAbs(2.0, two_point, 1e-4);
    try testing.expect(@abs(two_point - dense) > 0.1 * dense);

    // Off centre, where the chord is shorter and the shell is a larger part of
    // it, so a rule that only samples the core has nothing to hide behind.
    const grazing = res.Vec3{ 0, 0.8, -10 };
    const grazing_quadrature = placed.segmentTau(grazing, direction, 0, 100);
    const grazing_dense = denseTau(placed, grazing, direction, 9.3, 10.7);
    try testing.expectApproxEqAbs(grazing_dense, grazing_quadrature, 0.05 * grazing_dense);
}

test "the quadrature holds when a whole volume falls inside one segment" {
    // The case four points exist for. A small volume entirely within one march
    // segment puts both abscissae of a two-point rule in the flat core, and the
    // shell it misses is most of the volume.
    const small: LocalFogVolume = .{ .radii = @splat(0.1), .density = 1, .edge_falloff = 0.5 };
    const placed = try small.place();
    const origin = res.Vec3{ 0, 0, -5 };
    const direction = res.Vec3{ 0, 0, 1 };

    const quadrature = placed.segmentTau(origin, direction, 0, 20);
    const dense = denseTau(placed, origin, direction, 4.9, 5.1);
    try testing.expectApproxEqAbs(dense, quadrature, 0.05 * dense);
}

test "a segment shorter than the chord integrates only its own part" {
    const placed = try unit_ball.place();
    const origin = res.Vec3{ 0, 0, -10 };
    const direction = res.Vec3{ 0, 0, 1 };

    const half = placed.segmentTau(origin, direction, 0, 10);
    const whole = placed.segmentTau(origin, direction, 0, 100);
    try testing.expectApproxEqAbs(0.5 * whole, half, 0.02 * whole);

    // The two halves add up to the whole, which a rule that ignored the segment
    // bounds would fail twice over.
    const far = placed.segmentTau(origin, direction, 10, 100);
    try testing.expectApproxEqAbs(whole, half + far, 0.02 * whole);

    // A segment that misses the volume contributes nothing.
    try testing.expectEqual(@as(f32, 0), placed.segmentTau(origin, direction, 0, 8));
    try testing.expectEqual(@as(f32, 0), placed.segmentTau(origin, direction, 12, 100));

    // A segment of no length, and one whose ends arrive the wrong way round,
    // contribute nothing rather than a negative depth. Both land inside the
    // chord, where the density is not zero and the width is, so the arithmetic
    // alone would hand back the width's sign.
    try testing.expectEqual(@as(f32, 0), placed.segmentTau(origin, direction, 10, 10));
    try testing.expectEqual(@as(f32, 0), placed.segmentTau(origin, direction, 10.5, 10.2));
}

test "the quadrature rule is exact where a four-point rule has to be" {
    // The property that pins the abscissae and the weights: a Gauss-Legendre
    // rule of n points integrates polynomials of degree below 2n exactly, so
    // this one is exact through degree seven and wrong at degree eight.
    //
    // Run through the same rule the fog uses, by giving it a volume whose
    // density is one everywhere it is sampled and reading the arithmetic on the
    // abscissae directly.
    const abscissae = [_]f64{ 0.3399810435848563, 0.8611363115940526 };
    const weights = [_]f64{ 0.6521451548625461, 0.3478548451374538 };

    inline for (.{ 0, 1, 2, 3, 4, 5, 6, 7 }) |degree| {
        var sum: f64 = 0;
        inline for (abscissae, weights) |x, w| {
            sum += w * std.math.pow(f64, -x, degree);
            sum += w * std.math.pow(f64, x, degree);
        }
        // The integral of x^n over [-1, 1] is zero for odd n and 2/(n+1) for
        // even n.
        const expected: f64 = if (degree % 2 == 1) 0 else 2.0 / @as(f64, degree + 1);
        try testing.expectApproxEqAbs(expected, sum, 1e-12);
    }
}

test "a volume that cannot be placed is refused" {
    const cases = [_]struct { name: scene.LocalFogError, volume: LocalFogVolume }{
        .{ .name = error.InvalidVolumePlacement, .volume = .{ .centre = .{ nan, 0, 0 } } },
        .{ .name = error.InvalidVolumePlacement, .volume = .{ .radii = .{ 0, 1, 1 } } },
        .{ .name = error.InvalidVolumePlacement, .volume = .{ .radii = .{ 1, -1, 1 } } },
        .{ .name = error.InvalidVolumePlacement, .volume = .{ .radii = .{ 1, 1, inf } } },
        .{ .name = error.InvalidVolumePlacement, .volume = .{ .rotation = .{ 0, 0, 0, 0 } } },
        .{ .name = error.InvalidVolumePlacement, .volume = .{ .rotation = .{ nan, 0, 0, 1 } } },
        .{ .name = error.InvalidVolumeMedium, .volume = .{ .density = 0 } },
        .{ .name = error.InvalidVolumeMedium, .volume = .{ .density = nan } },
        .{ .name = error.InvalidVolumeMedium, .volume = .{ .edge_falloff = 0 } },
        .{ .name = error.InvalidVolumeMedium, .volume = .{ .edge_falloff = 1.5 } },
        .{ .name = error.InvalidVolumeMedium, .volume = .{ .scattering_albedo = .{ 1, 1.5, 1 } } },
        .{ .name = error.InvalidVolumeMedium, .volume = .{ .scattering_albedo = .{ -0.1, 1, 1 } } },
    };
    for (cases) |case| {
        try testing.expectError(case.name, case.volume.validate());
        try testing.expectError(case.name, case.volume.place());
    }

    try (LocalFogVolume{}).validate();
}

const Volumes = scene.LocalFogVolumes(4);

test "the set places what it is given and refuses what it cannot" {
    var volumes: Volumes = .{};
    try testing.expectEqual(@as(u32, 0), try volumes.add(unit_ball));
    try testing.expectEqual(@as(u32, 1), try volumes.add(.{ .centre = .{ 5, 0, 0 } }));
    try testing.expectEqual(@as(usize, 2), volumes.slice().len);

    // An invalid volume is rejected without taking a slot, so the set is what
    // it was before the call.
    try testing.expectError(error.InvalidVolumeMedium, volumes.add(.{ .density = -1 }));
    try testing.expectEqual(@as(usize, 2), volumes.slice().len);

    try testing.expectEqual(@as(?*const LocalFogVolume.Placed, null), volumes.get(2));
    try testing.expect(volumes.get(0) != null);

    _ = try volumes.add(.{});
    _ = try volumes.add(.{});
    try testing.expectError(error.TooManyLocalFogVolumes, volumes.add(.{}));

    volumes.clear();
    try testing.expectEqual(@as(usize, 0), volumes.slice().len);
    try testing.expectEqual(@as(?*const LocalFogVolume.Placed, null), volumes.get(0));
}

test "replacing a volume needs one that is already there" {
    var volumes: Volumes = .{};
    _ = try volumes.add(unit_ball);
    try volumes.set(0, .{ .centre = .{ 9, 0, 0 }, .density = 1 });
    try testing.expectApproxEqAbs(1.0, volumes.get(0).?.densityAt(.{ 9, 0, 0 }), tolerance);
    try testing.expectError(error.NoSuchLocalFogVolume, volumes.set(1, unit_ball));
    try testing.expectError(error.InvalidVolumeMedium, volumes.set(0, .{ .density = 0 }));
}

test "overlapping volumes add their density" {
    var volumes: Volumes = .{};
    _ = try volumes.add(.{ .density = 0.2, .edge_falloff = 0.25 });
    _ = try volumes.add(.{ .centre = .{ 0.5, 0, 0 }, .density = 0.3, .edge_falloff = 0.25 });

    // Inside both cores.
    try testing.expectApproxEqAbs(0.5, volumes.densityAt(.{ 0.25, 0, 0 }), tolerance);
    // Inside the first only.
    try testing.expectApproxEqAbs(0.2, volumes.densityAt(.{ -0.5, 0, 0 }), tolerance);
    // Outside both.
    try testing.expectEqual(@as(f32, 0), volumes.densityAt(.{ 0, 0, 5 }));
}

test "a volume outside the view is not visible" {
    const view = zm.lookAtRh(zm.f32x4(0, 0, 10, 1), zm.f32x4(0, 0, 0, 1), zm.f32x4(0, 1, 0, 0));
    const projection = zm.perspectiveFovRh(0.5 * std.math.pi, 1, 0.1, 100);
    const frustum: scene.Frustum = .fromViewProj(zm.mul(view, projection));

    const ahead = try (LocalFogVolume{}).place();
    try testing.expect(ahead.visible(&frustum));

    const behind = try (LocalFogVolume{ .centre = .{ 0, 0, 30 } }).place();
    try testing.expect(!behind.visible(&frustum));

    // Just outside on the side, and then wide enough to reach back in. The
    // bounds are what decides, so a volume whose extent matters is the case
    // that tells a box from a point.
    const aside = try (LocalFogVolume{ .centre = .{ 40, 0, 0 } }).place();
    try testing.expect(!aside.visible(&frustum));
    const wide = try (LocalFogVolume{ .centre = .{ 40, 0, 0 }, .radii = .{ 35, 1, 1 } }).place();
    try testing.expect(wide.visible(&frustum));
}

test "froxel slices are spread exponentially and round trip" {
    const depth = try scene.FroxelDepth.init(0.5, 200, 64);

    try testing.expectApproxEqAbs(0.5, depth.distanceAt(0), tolerance);
    try testing.expectApproxEqAbs(200.0, depth.distanceAt(63), 1e-3);

    // Exponential, not linear: the ratio between neighbouring slices is
    // constant, so the near field gets the slices and the far field the space.
    const ratio = depth.distanceAt(1) / depth.distanceAt(0);
    try testing.expectApproxEqAbs(ratio, depth.distanceAt(41) / depth.distanceAt(40), 1e-4);
    try testing.expect(depth.distanceAt(1) - depth.distanceAt(0) < depth.distanceAt(63) - depth.distanceAt(62));

    // The coordinate lands in the middle of a texel, not on its edge.
    const slices: f32 = 64;
    try testing.expectApproxEqAbs(0.5 / slices, depth.textureZ(0.5), 1e-6);
    try testing.expectApproxEqAbs((63 + 0.5) / slices, depth.textureZ(200), 1e-5);

    // And the two conversions are inverses over the whole range.
    for ([_]f32{ 0.5, 0.9, 3, 17, 60, 199, 200 }) |distance| {
        const z = depth.textureZ(distance);
        try testing.expectApproxEqAbs(distance, depth.distanceAtTextureZ(z), 1e-2 * distance);
    }
}

test "a froxel distribution with nothing to distribute is refused" {
    try testing.expectError(error.InvalidFroxelDepth, scene.FroxelDepth.init(0, 200, 64));
    try testing.expectError(error.InvalidFroxelDepth, scene.FroxelDepth.init(-1, 200, 64));
    try testing.expectError(error.InvalidFroxelDepth, scene.FroxelDepth.init(200, 200, 64));
    try testing.expectError(error.InvalidFroxelDepth, scene.FroxelDepth.init(0.5, inf, 64));
    try testing.expectError(error.InvalidFroxelDepth, scene.FroxelDepth.init(0.5, 200, 1));
}
