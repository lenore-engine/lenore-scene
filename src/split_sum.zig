// The split-sum table an environment's specular term reads: for a view at
// cosine NoV to a surface of a given roughness, the scale and bias Schlick's
// Fresnel is applied with. It depends on the BRDF alone, not on any
// environment, so an engine whose environment is drawn at run time builds it
// here once instead of loading the one that ships beside prefiltered maps.

const std = @import("std");

// Karis, Real Shading in Unreal Engine 4 (2013), split-sum integration.
// Visibility is height-correlated Smith GGX, as in KhronosGroup/glTF-IBL-Sampler,
// lib/source/shaders/filter.frag: V_SmithGGXCorrelated and LUT. Red is the
// Schlick scale and green the bias; roughness is perceptual roughness.
pub fn integrate(no_v: f32, roughness: f32) error{InvalidBrdfParameters}![2]f32 {
    if (!std.math.isFinite(no_v) or no_v <= 0 or no_v > 1 or
        !std.math.isFinite(roughness) or roughness < 0 or roughness > 1)
        return error.InvalidBrdfParameters;
    const count = 1024;
    const a2 = std.math.pow(f32, roughness, 4);
    const vx = @sqrt(@max(0, 1 - no_v * no_v));
    var scale: f32 = 0;
    var bias: f32 = 0;
    // A Hammersley set, the second coordinate the radical inverse of the
    // index, importance-sampling the GGX distribution of half vectors.
    for (0..count) |i| {
        const u = @as(f32, @floatFromInt(i)) / count;
        const v = @as(f32, @floatFromInt(@bitReverse(@as(u32, @intCast(i))))) * 0x1p-32;
        const phi = 2 * std.math.pi * u;
        const hz = @sqrt((1 - v) / (1 + (a2 - 1) * v));
        const hr = @sqrt(@max(0, 1 - hz * hz));
        const hx = hr * @cos(phi);
        const vh = @max(0, vx * hx + no_v * hz);
        const nl = 2 * vh * hz - no_v;
        if (nl <= 0 or vh <= 0) continue;
        // Widen products so a finite grazing NoV cannot underflow the
        // visibility denominator to zero in the smooth-surface limit.
        const nv64: f64 = no_v;
        const nl64: f64 = nl;
        const gv = nl64 * @sqrt(nv64 * nv64 * (1 - a2) + a2);
        const gl = nv64 * @sqrt(nl64 * nl64 * (1 - a2) + a2);
        const weight: f32 = @floatCast(2 * @as(f64, vh) * nl64 / (@as(f64, hz) * (gv + gl)));
        const fc = std.math.pow(f32, 1 - vh, 5);
        scale += (1 - fc) * weight;
        bias += fc * weight;
    }
    return .{ scale / count, bias / count };
}

pub const size = 128;
pub const Table = [size][size][2]f16;

// Texel centres match a clamped linear sampler over (NoV, roughness).
// The caller owns the table; generation allocates nothing.
pub fn build(table: *Table) void {
    for (table, 0..) |*row, y| for (row, 0..) |*texel, x| {
        const value = integrate((@as(f32, @floatFromInt(x)) + 0.5) / size, (@as(f32, @floatFromInt(y)) + 0.5) / size) catch unreachable;
        texel.* = .{ @floatCast(value[0]), @floatCast(value[1]) };
    };
}
