const std = @import("std");
const res = @import("lenore-resources");
const scene = @import("lenore-scene");

const testing = std.testing;
const FogSettings = scene.FogSettings;
const VolumetricSettings = scene.VolumetricSettings;

const tolerance = 1e-5;
const nan = std.math.nan(f32);
const inf = std.math.inf(f32);

// Uniform density, which is the case the closed form has to reduce to.
const uniform: FogSettings = .{ .mode = .analytic, .density = 0.1, .height_falloff = 0 };

// A layer pooled below y = 0, dense enough to matter over tens of units.
const layered: FogSettings = .{
    .mode = .analytic,
    .density = 0.1,
    .height_falloff = 0.2,
    .height_ref = 0,
};

test "uniform density integrates to density times distance" {
    // No height term, so the ray direction cannot matter.
    for ([_]f32{ -1, -0.3, 0, 0.5, 1 }) |direction_y|
        try testing.expectApproxEqAbs(2.0, uniform.opticalDepth(0, direction_y, 20), tolerance);

    // And the height of the ray cannot matter either.
    try testing.expectApproxEqAbs(2.0, uniform.opticalDepth(500, 0, 20), tolerance);
}

test "a horizontal ray in a layer stays at the density of its own height" {
    // Nothing changes along the ray, so the integral is again a product, but the
    // density is the profile's value at that height rather than the reference.
    const at_reference = layered.opticalDepth(0, 0, 10);
    try testing.expectApproxEqAbs(1.0, at_reference, tolerance);

    // Ten units up, the profile has fallen by exp(-2).
    const above = layered.opticalDepth(10, 0, 10);
    try testing.expectApproxEqAbs(@exp(-2.0), above, 1e-4);

    // And below the reference it has risen by the same factor.
    const below = layered.opticalDepth(-10, 0, 10);
    try testing.expectApproxEqAbs(@exp(2.0), below, 1e-3);
}

test "a climbing ray leaves the layer and its depth converges" {
    // Straight up from the reference height: the integral is base * (1 - e^-kd)/k
    // with k the falloff, so it approaches base/k however far the ray runs.
    const limit = 0.1 / 0.2;
    try testing.expectApproxEqAbs(limit * (1 - @exp(-2.0)), layered.opticalDepth(0, 1, 10), 1e-4);
    try testing.expectApproxEqAbs(limit * (1 - @exp(-20.0)), layered.opticalDepth(0, 1, 100), 1e-4);

    // An infinite ray is the sky, and it has to land on the limit rather than on
    // an infinity or on a number that is not one.
    const sky = layered.opticalDepth(0, 1, inf);
    try testing.expectApproxEqAbs(limit, sky, 1e-4);
}

test "a descending ray into a layer is opaque rather than infinite" {
    // The profile grows without bound downward, so the true integral over an
    // endless ray diverges. What comes back is the depth at which the medium is
    // already opaque, not an infinity that a later subtraction turns into a
    // value that is not a number.
    const down = layered.opticalDepth(0, -1, inf);
    try testing.expect(std.math.isFinite(down));
    try testing.expectApproxEqAbs(1.0, layered.opacity(0, -1, inf), tolerance);

    // The same from far below the reference plane, where the base density alone
    // would already have overflowed.
    const deep = layered.opticalDepth(-1000, -1, 1000);
    try testing.expect(std.math.isFinite(deep));
    try testing.expectApproxEqAbs(1.0, layered.opacity(-1000, -1, 1000), tolerance);
}

test "a ray descending from above a layer is opaque, not clear" {
    // The start of the ray is so far above the layer that the profile there
    // underflows to zero in f32, and the far end is so far below it that the
    // profile overflows. The true depth is enormous. Computing a starting
    // density and multiplying it by a path factor loses this entirely: zero
    // times a large number is zero, and the fog disappears exactly where it
    // should be thickest.
    const above = layered.opticalDepth(1000, -1, 1e6);
    try testing.expectEqual(@as(f32, 88), above);
    try testing.expectApproxEqAbs(1.0, layered.opacity(1000, -1, 1e6), tolerance);

    // The same ray stopped before it reaches the layer is still clear, so this
    // is not a blanket answer for every ray that starts high.
    try testing.expect(layered.opticalDepth(1000, -1, 10) < 1e-6);
}

test "an endless ray through a medium that does not thin out is opaque" {
    // Uniform density has nothing to converge to, however thin it is, and the
    // sky is an endless ray. Integrating it over a large but finite span
    // instead would make a thin medium clear.
    const thin_uniform: FogSettings = .{ .density = 1e-6, .height_falloff = 0 };
    try testing.expectEqual(@as(f32, 88), thin_uniform.opticalDepth(0, 1, inf));
    try testing.expectEqual(@as(f32, 88), thin_uniform.opticalDepth(0, 0, inf));

    // A horizontal ray in a layer has the same property: it never leaves it.
    try testing.expectEqual(@as(f32, 88), layered.opticalDepth(0, 0, inf));
}

test "the closed form is continuous across the near-zero rate branch" {
    // The integral divides by the rate, and below a threshold it is a plain
    // length instead. The two forms have to agree where they meet, or a ray
    // tilting through horizontal shows a seam.
    //
    // Thin enough that nothing here reaches the opaque clamp: at the clamp every
    // value is 88 and the comparison would hold whatever the branch did.
    const thin: FogSettings = .{ .density = 0.001, .height_falloff = 0.2 };
    const distance = 1000.0;

    // The branch is on how far apart the two exponents are, which is the rate
    // times the distance. With this falloff and length it crosses at a vertical
    // component of 5e-6. Either side of that, and then away from it.
    var previous: f32 = thin.opticalDepth(0, 1e-5, distance);
    try testing.expect(previous < 88);
    for ([_]f32{ 5.1e-6, 4.9e-6, 1e-6, 0 }) |direction_y| {
        const current = thin.opticalDepth(0, direction_y, distance);
        try testing.expectApproxEqAbs(previous, current, 1e-3);
        previous = current;
    }

    // And the limit is the uniform answer at that height.
    try testing.expectApproxEqAbs(0.001 * distance, previous, 1e-3);

    // Well below the threshold the closed form has nothing left: it subtracts
    // two numbers that agree to seven digits and divides by the difference. The
    // fallback is what keeps this accurate rather than noise.
    try testing.expectApproxEqAbs(0.001 * distance, thin.opticalDepth(0, 1e-9, distance), 1e-4);
    try testing.expectApproxEqAbs(0.001 * distance, thin.opticalDepth(0, -1e-9, distance), 1e-4);
}

test "a shallow gradient over a long ray is still integrated as a curve" {
    // What decides between the closed form and the plain length is how far apart
    // the two ends of the profile are, which is the rate times the distance and
    // not the rate on its own. A thin atmosphere climbed for a hundred thousand
    // units has a tiny rate and a wide spread, and treating it as uniform
    // overstates the depth tenfold.
    const atmosphere: FogSettings = .{ .density = 1e-6, .height_falloff = 1e-4 };
    const climbing = atmosphere.opticalDepth(0, 1, 1e5);
    try testing.expectApproxEqAbs(1e-6 * (1 - @exp(-10.0)) / 1e-4, climbing, 1e-7);
    try testing.expect(climbing < 0.5 * 1e-6 * 1e5);
}

test "a depth past opacity is held at the point the medium is already opaque" {
    // exp(-88) is under the smallest normal f32, so nothing beyond it is
    // distinguishable, and carrying it further is what produces an infinity.
    const thick: FogSettings = .{ .density = 1, .height_falloff = 0 };
    try testing.expectEqual(@as(f32, 88), thick.opticalDepth(0, 0, 1000));
    try testing.expectEqual(@as(f32, 88), thick.opticalDepth(0, 0, 1e9));
    try testing.expectApproxEqAbs(1.0, thick.opacity(0, 0, 1000), tolerance);

    // Just under it the depth is still the true product.
    try testing.expectApproxEqAbs(80.0, thick.opticalDepth(0, 0, 80), 1e-3);
}

test "a ray of no length and a medium of no density carry no depth" {
    try testing.expectEqual(@as(f32, 0), uniform.opticalDepth(0, 0, 0));
    try testing.expectEqual(@as(f32, 0), uniform.opticalDepth(0, 0, -5));

    const clear: FogSettings = .{ .density = 0 };
    try testing.expectEqual(@as(f32, 0), clear.opticalDepth(0, 0, 1000));
    try testing.expectEqual(@as(f32, 0), clear.opticalDepth(0, 1, inf));
}

test "opacity approaches one and the clamp holds it back" {
    // Thin fog is nearly transparent, thick fog nearly opaque.
    try testing.expectApproxEqAbs(1 - @exp(-2.0), uniform.opacity(0, 0, 20), tolerance);
    try testing.expectApproxEqAbs(1.0, uniform.opacity(0, 0, 10000), tolerance);

    var capped = uniform;
    capped.max_opacity = 0.8;
    try testing.expectApproxEqAbs(0.8, capped.opacity(0, 0, 10000), tolerance);
    // The clamp scales the whole curve rather than clipping its top, so a thin
    // fog is affected too.
    try testing.expectApproxEqAbs(0.8 * (1 - @exp(-2.0)), capped.opacity(0, 0, 20), tolerance);
}

test "the phase function is one in every direction when it is isotropic" {
    const even: FogSettings = .{ .phase_anisotropy = 0 };
    for ([_]f32{ -1, -0.5, 0, 0.5, 1 }) |cos_theta|
        try testing.expectApproxEqAbs(1.0, even.phase(cos_theta), tolerance);
}

test "a forward lobe peaks toward the sun and dips away from it" {
    const forward: FogSettings = .{ .phase_anisotropy = 0.76 };
    const toward = forward.phase(1);
    const across = forward.phase(0);
    const away = forward.phase(-1);

    // The closed form at the poles is (1 - g^2) / (1 -+ g)^3.
    try testing.expectApproxEqAbs((1 - 0.5776) / std.math.pow(f32, 1 - 0.76, 3), toward, 1e-2);
    try testing.expectApproxEqAbs((1 - 0.5776) / std.math.pow(f32, 1 + 0.76, 3), away, 1e-4);
    try testing.expect(toward > across and across > away);

    // Backward anisotropy is the mirror image, which a sign dropped from the
    // cosine term would hide.
    const backward: FogSettings = .{ .phase_anisotropy = -0.76 };
    try testing.expectApproxEqAbs(toward, backward.phase(-1), 1e-2);
    try testing.expectApproxEqAbs(away, backward.phase(1), 1e-4);
}

test "the anisotropy bound keeps the lobe finite" {
    // At the bound the peak is a large but usable number. Past it the cube in
    // the denominator runs away, which is what the bound exists to stop.
    const extreme: FogSettings = .{ .phase_anisotropy = 0.95 };
    try extreme.validate();
    try testing.expect(extreme.phase(1) < 1000);
    try testing.expect(extreme.phase(1) > 500);

    const past: FogSettings = .{ .phase_anisotropy = 0.99 };
    try testing.expectError(error.InvalidPhaseAnisotropy, past.validate());
}

test "the defaults are a valid medium" {
    try (FogSettings{}).validate();
    try uniform.validate();
    try layered.validate();
}

test "a medium that cannot be described is refused" {
    const cases = [_]struct { name: FogError, settings: FogSettings }{
        .{ .name = error.InvalidFogColour, .settings = .{ .colour = .{ -1, 0, 0 } } },
        .{ .name = error.InvalidFogColour, .settings = .{ .colour = .{ 0, nan, 0 } } },
        .{ .name = error.InvalidFogDensity, .settings = .{ .density = -1 } },
        .{ .name = error.InvalidFogDensity, .settings = .{ .density = nan } },
        .{ .name = error.InvalidFogDensity, .settings = .{ .density = 1e6 } },
        .{ .name = error.InvalidFogDensity, .settings = .{ .height_falloff = -0.1 } },
        .{ .name = error.InvalidFogDensity, .settings = .{ .height_falloff = inf } },
        .{ .name = error.InvalidFogDensity, .settings = .{ .height_ref = nan } },
        .{ .name = error.InvalidPhaseAnisotropy, .settings = .{ .phase_anisotropy = 1 } },
        .{ .name = error.InvalidPhaseAnisotropy, .settings = .{ .phase_anisotropy = -1 } },
        .{ .name = error.InvalidPhaseAnisotropy, .settings = .{ .phase_anisotropy = nan } },
        .{ .name = error.InvalidFogScatter, .settings = .{ .sun_scatter = -1 } },
        .{ .name = error.InvalidFogOpacity, .settings = .{ .max_opacity = 1.5 } },
        .{ .name = error.InvalidFogOpacity, .settings = .{ .max_opacity = -0.1 } },
    };
    for (cases) |case|
        try testing.expectError(case.name, case.settings.validate());
}

const FogError = scene.FogError;

test "a march that cannot be run is refused" {
    const cases = [_]struct { name: FogError, settings: VolumetricSettings }{
        .{ .name = error.InvalidMarchRange, .settings = .{ .max_distance = 0 } },
        .{ .name = error.InvalidMarchRange, .settings = .{ .max_distance = nan } },
        .{ .name = error.InvalidMarchRange, .settings = .{ .max_distance = inf } },
        // Local media are gathered inside the march, so a longer range is a
        // range over media the march never reaches.
        .{ .name = error.InvalidMarchRange, .settings = .{ .max_distance = 40, .local_fog_range = 48 } },
        .{ .name = error.InvalidMarchRange, .settings = .{ .local_fog_range = 0 } },
        .{ .name = error.InvalidMarchSteps, .settings = .{ .steps = 0 } },
        .{ .name = error.InvalidMarchNoise, .settings = .{ .noise_amplitude = -1 } },
        .{ .name = error.InvalidMarchNoise, .settings = .{ .noise_scale = nan } },
        .{ .name = error.InvalidMarchNoise, .settings = .{ .wind = .{ 0, inf, 0 } } },
        .{ .name = error.InvalidMarchStrength, .settings = .{ .ambient_strength = -1 } },
        .{ .name = error.InvalidMarchStrength, .settings = .{ .shadow_softness = nan } },
    };
    for (cases) |case|
        try testing.expectError(case.name, case.settings.validate());

    // And the march is validated through the medium that carries it.
    const carried: FogSettings = .{ .volumetric = .{ .steps = 0 } };
    try testing.expectError(error.InvalidMarchSteps, carried.validate());
}

test "the sample spacing is the march length over its steps" {
    const settings: VolumetricSettings = .{ .max_distance = 200, .steps = 25 };
    try testing.expectApproxEqAbs(8.0, settings.sampleSpacing(), tolerance);

    // The number to compare against the size of a billow: at the default noise
    // scale one billow is twenty units, so the default march resolves it.
    const defaults: VolumetricSettings = .{};
    try testing.expect(defaults.sampleSpacing() < 1 / defaults.noise_scale);
}
