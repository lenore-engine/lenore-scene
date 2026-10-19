// The atmosphere, host side: Hillaire, "A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique", EGSR 2020, ported from Epic's MIT
// reference, UnrealEngineSkyAtmosphere, whose licence follows. Each function
// names the reference function it follows.
//
// What it is for: once per time-of-day step an application asks it for the
// sun's colour and share at the ground (`sunAtGround`) and for the sky's
// irradiance on a normal (`skyIrradiance`), which become the directional
// light and the ambient. Its tables are what the device's sky is drawn from,
// uploaded as they are, and `assets/shaders/include/atmosphere.slang` is the
// same march step for step; the atmosphere check example compares the two.
//
// Kilometres and linear RGB throughout, for a sun of unit irradiance above
// the atmosphere: a caller scales by its own. No sun disc in the radiance.
// Positions put the planet's centre at the origin with +Y up; the march's own
// geometry is in f64 (see `integrate`).

// MIT License
//
// Copyright (c) 2020 Epic Games, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const std = @import("std");

const V = @Vector(3, f32);
const W = @Vector(3, f64);
const pi = std.math.pi;

// PLANET_RADIUS_OFFSET in RenderSkyCommon.hlsl, in kilometres: where a viewer
// on the ground stands, and how far the sphere the earth's shadow is tested
// against is raised.
pub const planet_radius_offset: f32 = 0.01;

pub const AtmosphereError = error{
    InvalidAtmosphere,
    InvalidAltitude,
    InvalidDirection,
    InvalidQuadrature,
    NonConvergentScattering,
};

pub const Atmosphere = struct {
    // SetupEarthAtmosphere in SkyAtmosphereCommon.cpp. Coefficients are /km.
    bottom_radius: f32 = 6360,
    top_radius: f32 = 6460,
    rayleigh_scattering: [3]f32 = .{ 0.005802, 0.013558, 0.033100 },
    rayleigh_scale_height: f32 = 8,
    mie_scattering: [3]f32 = .{ 0.003996, 0.003996, 0.003996 },
    mie_extinction: [3]f32 = .{ 0.004440, 0.004440, 0.004440 },
    mie_scale_height: f32 = 1.2,
    mie_g: f32 = 0.8,
    ozone_absorption: [3]f32 = .{ 0.000650, 0.001881, 0.000085 },
    // The ozone's density as a tent: below `ozone_layer_width` kilometres
    // `ozone_linear_0 * h + ozone_constant_0`, above it the other pair, held
    // to [0, 1]. Zero below 10 km and above 40, one at 25.
    ozone_layer_width: f32 = 25,
    ozone_linear_0: f32 = 1.0 / 15.0,
    ozone_constant_0: f32 = -2.0 / 3.0,
    ozone_linear_1: f32 = -1.0 / 15.0,
    ozone_constant_1: f32 = 8.0 / 3.0,
    // The knobs: multipliers on the Mie scattering and extinction, and on the
    // ozone's absorption. More aerosol is a hazier, warmer sky.
    aerosol: f32 = 1,
    ozone: f32 = 1,
    // What the ground reflects, for multiple scattering and for the ground
    // a view ray ends on.
    ground_albedo: [3]f32 = .{ 0, 0, 0 },
    // SkyViewLutPS marches with a variable sample count between these, by the
    // ray's length; RayMarchMinMaxSPP in the reference, whose application sets
    // it from uiViewRayMarchMinSPP and uiViewRayMarchMaxSPP (Game.h: 4, 14).
    sky_samples_min: f32 = 4,
    sky_samples_max: f32 = 14,

    pub fn validate(self: Atmosphere) AtmosphereError!void {
        const lengths = [_]f32{
            self.bottom_radius,
            self.top_radius,
            self.rayleigh_scale_height,
            self.mie_scale_height,
            self.ozone_layer_width,
            self.aerosol,
            self.ozone,
        };
        for (lengths) |v| {
            if (!std.math.isFinite(v) or v < 0) return error.InvalidAtmosphere;
        }
        if (self.bottom_radius <= 0 or self.top_radius <= self.bottom_radius or
            self.rayleigh_scale_height <= 0 or self.mie_scale_height <= 0 or
            !std.math.isFinite(self.top_radius * self.top_radius) or
            !std.math.isFinite(self.mie_g) or @abs(self.mie_g) >= 1) return error.InvalidAtmosphere;
        // The tent's slopes and intercepts are signed, unlike densities.
        for ([_]f32{ self.ozone_linear_0, self.ozone_constant_0, self.ozone_linear_1, self.ozone_constant_1 }) |v| {
            if (!std.math.isFinite(v)) return error.InvalidAtmosphere;
        }
        for (0..3) |c| {
            const coefficients = [_]f32{
                self.rayleigh_scattering[c],
                self.mie_scattering[c],
                self.mie_extinction[c],
                self.ozone_absorption[c],
                self.ground_albedo[c],
            };
            for (coefficients) |v| {
                if (!std.math.isFinite(v) or v < 0) return error.InvalidAtmosphere;
            }
            if (self.mie_scattering[c] > self.mie_extinction[c] or self.ground_albedo[c] > 1 or
                !std.math.isFinite(self.mie_extinction[c] * self.aerosol) or
                !std.math.isFinite(self.ozone_absorption[c] * self.ozone)) return error.InvalidAtmosphere;
        }
        if (!std.math.isFinite(self.sky_samples_min) or !std.math.isFinite(self.sky_samples_max) or
            self.sky_samples_min < 1 or self.sky_samples_max < self.sky_samples_min or self.sky_samples_max > 1024)
            return error.InvalidAtmosphere;
    }
};

// The transmittance table, 256 by 64 over (view zenith cosine, altitude) in
// the reference's mapping, and the multiple-scattering table, 32 by 32 over
// (sun zenith cosine, altitude). About 208 KB: put it on the heap.
pub const Tables = struct {
    pub const transmittance_width = 256;
    pub const transmittance_height = 64;
    pub const multiple_scattering_size = 32;
    transmittance_lut: [64][256][3]f32 = undefined,
    multiple_scattering_lut: [32][32][3]f32 = undefined,

    // RenderTransmittanceLutPS then NewMultiScattCS. Rebuild after changing any
    // atmosphere parameter. The caller owns the storage and its lifetime.
    pub fn build(self: *Tables, a: Atmosphere) AtmosphereError!void {
        try a.validate();
        for (0..64) |y| {
            for (0..256) |x| {
                const params = uvToTransmittance(a, .{ (float(x) + 0.5) / 256, (float(y) + 0.5) / 64 });
                const p: V = .{ 0, params[0], 0 };
                const d: V = .{ @sqrt(@max(0, 1 - params[1] * params[1])), params[1], 0 };
                const end = sphere(p, d, a.top_radius);
                var depth: V = @splat(0);
                var previous: f32 = 0;
                for (0..40) |s| {
                    // IntegrateScatteredLuminance's fixed path uses a first
                    // segment of 0.3 steps and omits the final 0.7 steps.
                    const t = end * (float(s) + 0.3) / 40;
                    const m = medium(a, p + d * splat(t));
                    depth += m.extinction * splat(t - previous);
                    previous = t;
                }
                self.transmittance_lut[y][x] = @as(V, @exp(-depth));
            }
        }
        for (0..32) |y| {
            for (0..32) |x| {
                const u = subToUnit((float(x) + 0.5) / 32, 32);
                const v = subToUnit((float(y) + 0.5) / 32, 32);
                const mu = u * 2 - 1;
                const sun: V = .{ 0, mu, @sqrt(@max(0, 1 - mu * mu)) };
                const height = a.bottom_radius + clamp(v + planet_radius_offset) * (a.top_radius - a.bottom_radius - planet_radius_offset);
                var luminance: V = @splat(0);
                var feedback: V = @splat(0);
                for (0..8) |i| {
                    for (0..8) |j| {
                        // NewMultiScattCS SQRTSAMPLECOUNT grid, rotated from
                        // reference +Z up into this module's +Y up.
                        const theta = 2 * pi * (float(i) + 0.5) / 8;
                        const cos_phi = 1 - 2 * (float(j) + 0.5) / 8;
                        const sin_phi = @sqrt(1 - cos_phi * cos_phi);
                        const d: V = .{ @cos(theta) * sin_phi, cos_phi, @sin(theta) * sin_phi };
                        const r = integrate(self, a, .{ 0, height, 0 }, d, sun, false);
                        luminance += r.luminance / splat(64);
                        feedback += r.feedback / splat(64);
                    }
                }
                // Hillaire section 5.5, equation 10: the infinite power series.
                // IntegrateScatteredLuminance requires f_ms < 1. Pure,
                // optically thick scattering can round it to one in f32.
                // Report failure rather than store an infinite power series.
                if (!@reduce(.And, feedback < splat(1))) return error.NonConvergentScattering;
                const value = luminance / (splat(1) - feedback);
                if (!@reduce(.And, @abs(value) < splat(std.math.inf(f32)))) return error.NonConvergentScattering;
                self.multiple_scattering_lut[y][x] = value;
            }
        }
    }
};

// Directions are normalised here, so ordinary arrays need not already be unit.
// Altitudes must be within the atmosphere; the ground default is 0.01 km.
// Atmosphere.validate and Tables.build must precede these queries, using the
// same atmosphere. Validation of the model belongs at load, not in each ray.
pub fn transmittance(tables: *const Tables, a: Atmosphere, altitude_km: f32, direction: [3]f32) AtmosphereError![3]f32 {
    const p = try position(a, altitude_km);
    const d = try unit(direction);
    if (sphere(p, d, a.bottom_radius) >= 0) return .{ 0, 0, 0 };
    return sampleTransmittance(tables, a, p[1], d[1]);
}

pub fn sunAtGround(tables: *const Tables, a: Atmosphere, altitude_km: f32, toward_sun: [3]f32) AtmosphereError![3]f32 {
    return transmittance(tables, a, altitude_km, toward_sun);
}

// SkyViewLutPS: MieRayPhase=true, ground=false, variable sample count.
// Rays intersecting ground retain the scattering in front of it, with no
// surface radiance. Multiple scattering is enabled, solar irradiance is one.
pub fn skyRadiance(tables: *const Tables, a: Atmosphere, altitude_km: f32, view_direction: [3]f32, toward_sun: [3]f32) AtmosphereError![3]f32 {
    return integrate(tables, a, try position(a, altitude_km), try unit(view_direction), try unit(toward_sun), true).luminance;
}

pub fn skyIrradiance(tables: *const Tables, a: Atmosphere, altitude_km: f32, normal: [3]f32, toward_sun: [3]f32) AtmosphereError![3]f32 {
    return skyIrradianceResolution(tables, a, altitude_km, normal, toward_sun, 16, 64);
}

// Deterministic Gauss-Legendre quadrature in world zenith cosine, with
// midpoint azimuths over the portion above the surface normal. Split the
// domain at the planet horizon and the surface hemisphere's polar limits.
// This resolves the thin sky ring seen by a downward normal near the ground.
// The default 16 by 64 per interval is checked against 32 by 128 in tests.
// This host integration surrounds the reference sky march, rather than being
// a reference LUT pass. The resolution variant permits convergence tests.
pub fn skyIrradianceResolution(
    tables: *const Tables,
    a: Atmosphere,
    altitude_km: f32,
    normal: [3]f32,
    toward_sun: [3]f32,
    radial: usize,
    azimuthal: usize,
) AtmosphereError![3]f32 {
    if (radial == 0 or azimuthal == 0 or radial > 4096 or azimuthal > 4096) return error.InvalidQuadrature;
    const p = try position(a, altitude_km);
    const n = try unit(normal);
    const sun = try unit(toward_sun);
    const horizontal = @min(1, @sqrt(n[0] * n[0] + n[2] * n[2]));
    const centre = if (horizontal > 0) std.math.atan2(n[2], n[0]) else std.math.atan2(sun[2], sun[0]);
    const horizon = -@sqrt(@max(0, (p[1] - a.bottom_radius) * (p[1] + a.bottom_radius))) / p[1];
    var boundaries = [_]f32{ -1, -horizontal, horizon, 0, horizontal, 1 };
    std.mem.sort(f32, &boundaries, {}, std.sort.asc(f32));
    var sum: @Vector(3, f64) = @splat(0);
    for (0..boundaries.len - 1) |interval| {
        const lower = boundaries[interval];
        const width = boundaries[interval + 1] - lower;
        if (width == 0) continue;
        for (0..radial) |i| {
            const node = gaussLegendre(radial, i);
            const mu = lower + width * @as(f32, @floatCast(node[0]));
            const r = @sqrt(@max(0, 1 - mu * mu));
            const projected = horizontal * r;
            const vertical = n[1] * mu;
            if (vertical + projected <= 0) continue;
            const half_angle = if (vertical >= projected) pi else std.math.acos(std.math.clamp(-vertical / projected, -1, 1));
            for (0..azimuthal) |j| {
                const phi = centre + half_angle * (2 * (float(j) + 0.5) / float(azimuthal) - 1);
                const d: V = .{ r * @cos(phi), mu, r * @sin(phi) };
                const cosine = @max(0, dot(n, d));
                const radiance: @Vector(3, f64) = @floatCast(integrate(tables, a, p, d, sun, true).luminance);
                const weight = node[1] * width * cosine * 2 * half_angle / @as(f64, @floatFromInt(azimuthal));
                sum += radiance * @as(@Vector(3, f64), @splat(weight));
            }
        }
    }
    return @as(V, @floatCast(sum));
}

// Gauss-Legendre roots and weights mapped from [-1,1] to [0,1]. The
// recurrence and Newton iteration use f64 so quadrature error is transport
// error rather than root rounding. No table or allocation is needed.
fn gaussLegendre(count: usize, index: usize) [2]f64 {
    const n: f64 = @floatFromInt(count);
    const i: f64 = @floatFromInt(index);
    var x = @cos(pi * (i + 0.75) / (n + 0.5));
    var derivative: f64 = 0;
    for (0..32) |_| {
        var previous: f64 = 1;
        var current = x;
        for (2..count + 1) |k| {
            const degree: f64 = @floatFromInt(k);
            const next = ((2 * degree - 1) * x * current - (degree - 1) * previous) / degree;
            previous = current;
            current = next;
        }
        derivative = n * (x * current - previous) / (x * x - 1);
        const step = current / derivative;
        x -= step;
        if (@abs(step) < 1e-14) break;
    }
    return .{ (x + 1) * 0.5, 1 / ((1 - x * x) * derivative * derivative) };
}

const Medium = struct { ray: V, mie: V, scattering: V, extinction: V };

// sampleMediumRGB, RenderSkyCommon.hlsl. The ozone layer is saturated, not
// extended past its tent. Mie absorption is extinction minus scattering.
fn medium(a: Atmosphere, p: V) Medium {
    return mediumAt(a, length(p) - a.bottom_radius);
}

// At altitude `h` above the bottom sphere.
fn mediumAt(a: Atmosphere, h: f32) Medium {
    const ray = @as(V, a.rayleigh_scattering) * splat(@exp(-h / a.rayleigh_scale_height));
    const mie_density = a.aerosol * @exp(-h / a.mie_scale_height);
    const mie = @as(V, a.mie_scattering) * splat(mie_density);
    const ozone_density = clamp(if (h < a.ozone_layer_width)
        a.ozone_linear_0 * h + a.ozone_constant_0
    else
        a.ozone_linear_1 * h + a.ozone_constant_1);
    const extinction = ray +
        @as(V, a.mie_extinction) * splat(mie_density) +
        @as(V, a.ozone_absorption) * splat(a.ozone * ozone_density);
    return .{ .ray = ray, .mie = mie, .scattering = ray + mie, .extinction = extinction };
}

const Result = struct { luminance: V, feedback: V };

// IntegrateScatteredLuminance, RenderSkyRayMarching.hlsl. The false mode is
// NewMultiScattCS: 20 fixed steps, isotropic phase, no LUT feedback. The true
// mode is SkyViewLutPS with multiple scattering enabled. Both add the sunlit
// ground where the ray ends on it.
fn integrate(tables: *const Tables, a: Atmosphere, p: V, d: V, sun: V, comptime sky: bool) Result {
    const bottom = sphere(p, d, a.bottom_radius);
    const top = sphere(p, d, a.top_radius);
    const end = if (bottom >= 0) @min(bottom, top) else top;
    var result: Result = .{ .luminance = @splat(0), .feedback = @splat(0) };
    if (end <= 0) return result;
    const count: f32 = if (sky) a.sky_samples_min + (a.sky_samples_max - a.sky_samples_min) * clamp(end * 0.01) else 20;
    const count_floor = @floor(count);
    const end_floor = end * count_floor / count;
    const cosine = std.math.clamp(dot(d, sun), -1, 1);
    // hgPhase(g,-cosine), with USE_CornetteShanks defined. Its internal
    // negation restores a forward lobe towards the sun.
    const g = a.mie_g;
    // Equivalent to 1+g*g-2*g*cosine, avoiding cancellation near |g|=1.
    const denom = if (g >= 0) (1 - g) * (1 - g) + 2 * g * (1 - cosine) else (1 + g) * (1 + g) - 2 * g * (1 + cosine);
    const mie_phase = 3.0 / (8 * pi) * (1 - g * g) / (2 + g * g) * (1 + cosine * cosine) / (denom * @sqrt(denom));
    const ray_phase = 3.0 / (16 * pi) * (1 + cosine * cosine);
    var throughput: V = @splat(1);
    var previous: f32 = 0;
    var s: f32 = 0;
    while (s < count) : (s += 1) {
        var dt: f32 = undefined;
        var t: f32 = undefined;
        if (sky) {
            const f0 = s / count_floor;
            const f1 = (s + 1) / count_floor;
            const t0 = end_floor * f0 * f0;
            const t1 = if (f1 > 1) end else end_floor * f1 * f1;
            dt = t1 - t0;
            t = t0 + dt * 0.3;
        } else {
            t = end * (s + 0.3) / count;
            dt = t - previous;
            previous = t;
        }
        // In f64: at 6360 km an f32 position steps about half a metre, which
        // is coarse against the 10 m offset below, and rays that end on the
        // ground spend their whole march inside those 10 m.
        const point: W = @as(W, @floatCast(p)) + @as(W, @floatCast(d)) * @as(W, @splat(@as(f64, t)));
        const radius = @sqrt(@reduce(.Add, point * point));
        const up = point / @as(W, @splat(radius));
        const height: f32 = @floatCast(radius);
        const mu: f32 = @floatCast(@reduce(.Add, @as(W, @floatCast(sun)) * up));
        const m = mediumAt(a, height - a.bottom_radius);
        const segment = @exp(-m.extinction * splat(dt));
        const trans_sun = sampleTransmittance(tables, a, height, mu);
        // Earth shadow uses the offset sphere centre in the reference.
        const lowered = point - up * @as(W, @splat(@as(f64, planet_radius_offset)));
        const shadow: f32 = if (sphere64(lowered, @floatCast(sun), a.bottom_radius) >= 0) 0 else 1;
        const phase = if (sky) m.mie * splat(mie_phase) + m.ray * splat(ray_phase) else m.scattering * splat(1.0 / (4 * pi));
        const multi = if (sky) sampleMultiple(tables, a, height, mu) else splat(0);
        const source = splat(shadow) * trans_sun * phase + multi * m.scattering;
        var weight: V = undefined;
        inline for (0..3) |c| {
            // Analytic segment integral has limit dt in a vacuum. expm1's
            // short series avoids cancellation in the thin upper atmosphere.
            const tau = m.extinction[c] * dt;
            weight[c] = if (tau < 0.001) dt * (1 - tau * 0.5 + tau * tau / 6) else (1 - segment[c]) / m.extinction[c];
        }
        result.luminance += throughput * source * weight;
        result.feedback += throughput * m.scattering * weight;
        throughput *= segment;
    }
    // The reference adds the ground only when marching for the multiple
    // scattering table; the sky has it as well here. A view ray that ends on
    // the ground otherwise sees black below the horizon, which a scene whose
    // own ground stops short of the horizon shows as a dark band along it.
    if (bottom > 0 and end == bottom) {
        const point = p + d * splat(bottom);
        const height = length(point);
        const mu = dot(point / splat(height), sun);
        result.luminance += sampleTransmittance(tables, a, height, mu) * throughput * splat(clamp(mu) / pi) * @as(V, a.ground_albedo);
    }
    return result;
}

// UvToLutTransmittanceParams and LutTransmittanceParamsToUv. The reference
// deliberately leaves sub-UV conversion disabled for the transmittance LUT.
fn uvToTransmittance(a: Atmosphere, uv: [2]f32) [2]f32 {
    const h = @sqrt((a.top_radius - a.bottom_radius) * (a.top_radius + a.bottom_radius));
    const rho = h * uv[1];
    const radius = @sqrt(rho * rho + a.bottom_radius * a.bottom_radius);
    const d_min = a.top_radius - radius;
    const d = d_min + uv[0] * (rho + h - d_min);
    return .{ radius, if (d == 0) 1 else std.math.clamp((h * h - rho * rho - d * d) / (2 * radius * d), -1, 1) };
}

fn sampleTransmittance(tables: *const Tables, a: Atmosphere, radius: f32, mu: f32) V {
    const h = @sqrt((a.top_radius - a.bottom_radius) * (a.top_radius + a.bottom_radius));
    const rho = @sqrt(@max(0, (radius - a.bottom_radius) * (radius + a.bottom_radius)));
    const discriminant = radius * radius * (mu * mu - 1) + a.top_radius * a.top_radius;
    const d = @max(0, -radius * mu + @sqrt(@max(0, discriminant)));
    const d_min = a.top_radius - radius;
    return bilinear(256, 64, &tables.transmittance_lut, .{ (d - d_min) / (rho + h - d_min), rho / h });
}

// GetMultipleScattering and fromUnitToSubUvs/fromSubUvsToUnit. Preserve the
// reference's asymmetric resolution+1 and resolution-1 denominators.
fn sampleMultiple(tables: *const Tables, a: Atmosphere, radius: f32, mu: f32) V {
    const uv = [2]f32{
        unitToSub(clamp(mu * 0.5 + 0.5), 32),
        unitToSub(clamp((radius - a.bottom_radius) / (a.top_radius - a.bottom_radius)), 32),
    };
    return bilinear(32, 32, &tables.multiple_scattering_lut, uv);
}
fn unitToSub(u: f32, resolution: f32) f32 {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1));
}
fn subToUnit(u: f32, resolution: f32) f32 {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1));
}
// Texture.SampleLevel with samplerLinearClamp: texel centres are at i+0.5.
fn bilinear(comptime width: usize, comptime height: usize, data: *const [height][width][3]f32, uv: [2]f32) V {
    const x = std.math.clamp(uv[0] * float(width) - 0.5, 0, float(width - 1));
    const y = std.math.clamp(uv[1] * float(height) - 0.5, 0, float(height - 1));
    const ix: usize = @intFromFloat(x);
    const iy: usize = @intFromFloat(y);
    const nx = @min(ix + 1, width - 1);
    const ny = @min(iy + 1, height - 1);
    const fx = splat(x - float(ix));
    const fy = splat(y - float(iy));
    const lo = @as(V, data[iy][ix]) * (splat(1) - fx) + @as(V, data[iy][nx]) * fx;
    const hi = @as(V, data[ny][ix]) * (splat(1) - fx) + @as(V, data[ny][nx]) * fx;
    return lo * (splat(1) - fy) + hi * fy;
}

// raySphereIntersectNearest. Double precision in the geometric predicate
// retains the 0.01 km offset when subtracting squared planetary radii.
fn sphere(p: V, d: V, radius: f32) f32 {
    return sphere64(@floatCast(p), @floatCast(d), radius);
}

fn sphere64(pd: W, dd: W, radius: f32) f32 {
    const b = @reduce(.Add, pd * dd);
    const aa = @reduce(.Add, dd * dd);
    const c = @reduce(.Add, pd * pd) - @as(f64, radius) * radius;
    const delta = b * b - aa * c;
    if (delta < 0) return -1;
    const root = @sqrt(delta);
    const near = (-b - root) / aa;
    const far = (-b + root) / aa;
    if (far < 0) return -1;
    return @floatCast(if (near < 0) far else near);
}
fn position(a: Atmosphere, altitude: f32) AtmosphereError!V {
    if (!std.math.isFinite(altitude) or altitude < 0 or altitude > a.top_radius - a.bottom_radius) return error.InvalidAltitude;
    return .{ 0, a.bottom_radius + altitude, 0 };
}
fn unit(v: [3]f32) AtmosphereError!V {
    for (v) |c| if (!std.math.isFinite(c)) return error.InvalidDirection;
    const scale = @max(@abs(v[0]), @max(@abs(v[1]), @abs(v[2])));
    if (scale == 0) return error.InvalidDirection;
    return normalise(@as(V, v) / splat(scale));
}
fn normalise(v: V) V {
    return v / splat(@sqrt(dot(v, v)));
}
fn dot(a: V, b: V) f32 {
    return @reduce(.Add, a * b);
}
fn length(v: V) f32 {
    const wide: @Vector(3, f64) = @floatCast(v);
    return @floatCast(@sqrt(@reduce(.Add, wide * wide)));
}
fn float(value: usize) f32 {
    return @floatFromInt(value);
}
fn splat(value: f32) V {
    return @splat(value);
}
fn clamp(value: f32) f32 {
    return std.math.clamp(value, 0, 1);
}
