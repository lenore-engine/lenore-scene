// Host checks for Hillaire's atmosphere transport and the Earth parameter set.
const std = @import("std");
const scene = @import("lenore-scene");
const testing = std.testing;
const Atmosphere = scene.Atmosphere;
const Tables = scene.Tables;
const ground = scene.planet_radius_offset;
const up: [3]f32 = .{ 0, 1, 0 };
const down: [3]f32 = .{ 0, -1, 0 };

fn sun(elevation_degrees: f32) [3]f32 {
    const elevation = elevation_degrees * std.math.pi / 180;
    return .{ @cos(elevation), @sin(elevation), 0 };
}
fn finitePositive(rgb: [3]f32) !void {
    for (rgb) |v| try testing.expect(std.math.isFinite(v) and v > 0);
}

test "Earth atmosphere transport and quadrature convergence" {
    const a: Atmosphere = .{};
    var tables: Tables = undefined;
    try tables.build(a);

    const zenith = try scene.transmittance(&tables, a, ground, up);
    // SetupEarthAtmosphere's saturated ozone tent is zero below 10 and above
    // 40 km, with unit peak at 25. Its column is (40-10)*1/2 = 15 km.
    // Integrate the exponentials from the viewer to the finite atmosphere top.
    // RenderTransmittanceLutPS uses only 40 fixed steps, with the first dt=0.3
    // steps and densities at segment ends. This underestimates vertical optical
    // depth, particularly Mie. Allow 4.5% in transmittance to retain that exact
    // reference discretisation rather than silently replacing its quadrature.
    for (0..3) |c| {
        const ray_column = a.rayleigh_scale_height * (@exp(-ground / a.rayleigh_scale_height) - @exp(-(a.top_radius - a.bottom_radius) / a.rayleigh_scale_height));
        const mie_column = a.mie_scale_height * (@exp(-ground / a.mie_scale_height) - @exp(-(a.top_radius - a.bottom_radius) / a.mie_scale_height));
        const analytic = @exp(-(a.rayleigh_scattering[c] * ray_column + a.mie_extinction[c] * mie_column + a.ozone_absorption[c] * 15));
        try testing.expectApproxEqRel(analytic, zenith[c], 0.045);
    }
    const horizon = try scene.transmittance(&tables, a, ground, .{ 1, 0, 0 });
    try testing.expect(horizon[0] > horizon[1] and horizon[1] > horizon[2]);
    for (0..3) |c| try testing.expect(horizon[c] < 0.25 * zenith[c]);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, try scene.sunAtGround(&tables, a, ground, sun(-2)));
    try testing.expectEqual(zenith, try scene.sunAtGround(&tables, a, ground, up));

    const blue = try scene.skyRadiance(&tables, a, ground, up, sun(55));
    try testing.expect(blue[2] > blue[1] and blue[1] > blue[0]);
    const warm = try scene.skyRadiance(&tables, a, ground, sun(2), sun(2));
    try testing.expect(warm[0] > warm[2]);
    for (tables.multiple_scattering_lut) |row| {
        for (row) |rgb| {
            for (rgb) |v| try testing.expect(std.math.isFinite(v) and v >= 0);
        }
    }

    const irradiance = try scene.skyIrradiance(&tables, a, ground, up, sun(55));
    try finitePositive(irradiance);
    // Noon and low sun exercise both the blue sky and narrow forward lobe.
    // Downward and tilted normals also include rays ending at the ground.
    for ([_][3]f32{ up, down, .{ 1, 1, 0 } }) |normal| {
        for ([_]f32{ 55, 2 }) |elevation| {
            const coarse = try scene.skyIrradiance(&tables, a, ground, normal, sun(elevation));
            const fine = try scene.skyIrradianceResolution(&tables, a, ground, normal, sun(elevation), 32, 128);
            try finitePositive(coarse);
            for (0..3) |c| {
                try testing.expectApproxEqRel(fine[c], coarse[c], 0.01);
            }
        }
    }

    var changed: Tables = undefined;
    var no_mie = a;
    no_mie.aerosol = 0;
    try changed.build(no_mie);
    const clear = try scene.transmittance(&changed, no_mie, ground, up);
    for (0..3) |c| try testing.expect(clear[c] > zenith[c]);
    const forward = try scene.skyRadiance(&tables, a, ground, sun(55), sun(55));
    const clear_forward = try scene.skyRadiance(&changed, no_mie, ground, sun(55), sun(55));
    const side = try scene.skyRadiance(&tables, a, ground, .{ 0, 0, 1 }, sun(55));
    const clear_side = try scene.skyRadiance(&changed, no_mie, ground, .{ 0, 0, 1 }, sun(55));
    for (0..3) |c| try testing.expect(forward[c] / side[c] > clear_forward[c] / clear_side[c]);

    var no_ozone = a;
    no_ozone.ozone = 0;
    try changed.build(no_ozone);
    const ozone_free = try scene.transmittance(&changed, no_ozone, ground, up);
    for (0..3) |c| try testing.expect(ozone_free[c] > zenith[c]);
    try testing.expect(ozone_free[1] - zenith[1] > ozone_free[0] - zenith[0]);
    try testing.expect(ozone_free[1] - zenith[1] > ozone_free[2] - zenith[2]);

    var reflective = a;
    reflective.ground_albedo = .{ 0.5, 0.5, 0.5 };
    try changed.build(reflective);
    for (0..3) |c| try testing.expect(changed.multiple_scattering_lut[0][24][c] > tables.multiple_scattering_lut[0][24][c]);

    // Vacuum is a valid model. Analytic segment integration must avoid 0/0.
    var vacuum = a;
    vacuum.rayleigh_scattering = .{ 0, 0, 0 };
    vacuum.aerosol = 0;
    vacuum.ozone = 0;
    try changed.build(vacuum);
    try testing.expectEqual([3]f32{ 1, 1, 1 }, try scene.transmittance(&changed, vacuum, ground, up));
    try testing.expectEqual([3]f32{ 0, 0, 0 }, try scene.skyRadiance(&changed, vacuum, ground, up, sun(55)));
    try testing.expectError(error.InvalidAltitude, scene.transmittance(&tables, a, -1, up));
    try testing.expectError(error.InvalidAltitude, scene.skyRadiance(&tables, a, std.math.nan(f32), up, up));
    try testing.expectError(error.InvalidDirection, scene.sunAtGround(&tables, a, ground, .{ 0, 0, 0 }));
    try testing.expectError(error.InvalidDirection, scene.skyRadiance(&tables, a, ground, up, .{ std.math.inf(f32), 0, 0 }));
    try testing.expectError(error.InvalidQuadrature, scene.skyIrradianceResolution(&tables, a, ground, up, up, 0, 64));

    // The reference power series requires feedback strictly below one. An
    // optically thick, purely scattering medium rounds to one in f32.
    var thick = vacuum;
    thick.rayleigh_scattering = .{ 10000, 10000, 10000 };
    try testing.expectError(error.NonConvergentScattering, changed.build(thick));
}

test "atmosphere rejects invalid caller parameters" {
    try (Atmosphere{}).validate();
    inline for (.{ "bottom_radius", "top_radius", "rayleigh_scale_height", "mie_scale_height", "ozone_layer_width", "aerosol", "ozone", "sky_samples_min", "sky_samples_max" }) |field| {
        for ([_]f32{ -1, std.math.inf(f32), std.math.nan(f32) }) |value| {
            var a: Atmosphere = .{};
            @field(a, field) = value;
            try testing.expectError(error.InvalidAtmosphere, a.validate());
        }
    }
    inline for (.{ "rayleigh_scattering", "mie_scattering", "mie_extinction", "ozone_absorption", "ground_albedo" }) |field| {
        for ([_]f32{ -1, std.math.inf(f32), std.math.nan(f32) }) |value| {
            var a: Atmosphere = .{};
            @field(a, field)[1] = value;
            try testing.expectError(error.InvalidAtmosphere, a.validate());
        }
    }
    inline for (.{ "ozone_linear_0", "ozone_constant_0", "ozone_linear_1", "ozone_constant_1" }) |field| {
        var a: Atmosphere = .{};
        @field(a, field) = std.math.nan(f32);
        try testing.expectError(error.InvalidAtmosphere, a.validate());
    }
    for ([_]f32{ -1, 1, std.math.nan(f32) }) |g| {
        var a: Atmosphere = .{};
        a.mie_g = g;
        try testing.expectError(error.InvalidAtmosphere, a.validate());
    }
    var a: Atmosphere = .{};
    a.top_radius = a.bottom_radius;
    try testing.expectError(error.InvalidAtmosphere, a.validate());
    a = .{};
    a.mie_scattering[0] = 0.1;
    try testing.expectError(error.InvalidAtmosphere, a.validate());
    a = .{};
    a.ground_albedo[0] = 1.1;
    try testing.expectError(error.InvalidAtmosphere, a.validate());
    a = .{};
    a.rayleigh_scale_height = 0;
    try testing.expectError(error.InvalidAtmosphere, a.validate());
    var tables: Tables = undefined;
    try testing.expectError(error.InvalidAtmosphere, tables.build(a));
}
