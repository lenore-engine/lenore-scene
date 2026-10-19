// The host half of the indirect-light probes: where the probes stand, which
// way their rays go, and the two encodings a probe stores, irradiance as
// second-order spherical harmonics and hit distances on an octahedral map.
//
// The shader carries the same arithmetic; this file is what it is tested
// against. Nothing here touches a device.
//
// Conventions as elsewhere in this module: +Y up, metres, directions unit.

const std = @import("std");
const resources = @import("lenore-resources");
const vec = @import("vec.zig");

const Aabb = resources.Aabb;
const Vec3 = resources.Vec3;

// Past this a grid is a mistake rather than a scene: at 0.5 m it is a volume of
// 20 x 20 x 20 m, and each probe costs 144 bytes of harmonics and 512 of
// moments, about 45 MB at the cap.
pub const max_probes: u32 = 65536;

// A spacing below a centimetre is not a probe grid, it is a voxel volume, and
// the counts it would produce overflow before they are refused.
const min_spacing: f32 = 0.01;

pub const GridError = error{
    // The spacing is not a positive finite distance above `min_spacing`.
    InvalidSpacing,
    // The bounds are empty, inverted or not finite.
    DegenerateBounds,
    // More probes than `max_probes`.
    TooManyProbes,
};

pub const ProbeGrid = struct {
    // Where probe (0, 0, 0) stands; the others step by `spacing` along +X,
    // +Y and +Z.
    origin: Vec3,
    spacing: f32,
    counts: [3]u32,

    // Probes on a regular lattice covering `bounds`, centred in it: as many
    // as fit along each axis at `spacing` with the bounds' corners inside the
    // outermost cells, at least two per axis so a point always has a cell to
    // interpolate in.
    pub fn fit(bounds: Aabb, spacing: f32) GridError!ProbeGrid {
        if (!(spacing >= min_spacing and spacing < std.math.inf(f32))) return error.InvalidSpacing;
        if (!vec.finite(bounds.min) or !vec.finite(bounds.max)) return error.DegenerateBounds;
        const extent = bounds.max - bounds.min;
        if (!@reduce(.And, extent > @as(Vec3, @splat(0)))) return error.DegenerateBounds;

        var counts: [3]u32 = undefined;
        var total: u64 = 1;
        inline for (&counts, 0..) |*axis_count, axis| {
            // Cells, rounded up so the far corner is covered, plus the one
            // probe more than there are cells. Bounded before the cast: an
            // extent over the cap's cube root at the minimum spacing is
            // already refused below, and the float cannot exceed it here.
            const cells = @ceil(extent[axis] / spacing);
            if (cells > @as(f32, @floatFromInt(max_probes))) return error.TooManyProbes;
            axis_count.* = @max(2, @as(u32, @intFromFloat(cells)) + 1);
            total *= axis_count.*;
        }
        if (total > max_probes) return error.TooManyProbes;

        const span: Vec3 = .{
            @as(f32, @floatFromInt(counts[0] - 1)) * spacing,
            @as(f32, @floatFromInt(counts[1] - 1)) * spacing,
            @as(f32, @floatFromInt(counts[2] - 1)) * spacing,
        };
        const centre = (bounds.min + bounds.max) * @as(Vec3, @splat(0.5));
        return .{
            .origin = centre - span * @as(Vec3, @splat(0.5)),
            .spacing = spacing,
            .counts = counts,
        };
    }

    pub fn count(self: ProbeGrid) u32 {
        return self.counts[0] * self.counts[1] * self.counts[2];
    }

    // X fastest, then Y, then Z, which is the order the shader walks a
    // cell's eight corners in.
    pub fn index(self: ProbeGrid, cell: [3]u32) u32 {
        std.debug.assert(cell[0] < self.counts[0] and cell[1] < self.counts[1] and cell[2] < self.counts[2]);
        return cell[0] + self.counts[0] * (cell[1] + self.counts[1] * cell[2]);
    }

    pub fn cellOf(self: ProbeGrid, probe: u32) [3]u32 {
        std.debug.assert(probe < self.count());
        const x = probe % self.counts[0];
        const rest = probe / self.counts[0];
        return .{ x, rest % self.counts[1], rest / self.counts[1] };
    }

    pub fn position(self: ProbeGrid, probe: u32) Vec3 {
        const cell = self.cellOf(probe);
        const steps: Vec3 = .{
            @floatFromInt(cell[0]),
            @floatFromInt(cell[1]),
            @floatFromInt(cell[2]),
        };
        return self.origin + steps * @as(Vec3, @splat(self.spacing));
    }

    // The cell whose eight corners surround `point`, and how far across it
    // the point lies on each axis, in [0, 1]. A point outside the grid is
    // clamped onto its boundary, which is what the shader does too: the
    // outermost probes then light it.
    pub fn locate(self: ProbeGrid, point: Vec3) struct { base: [3]u32, fraction: Vec3 } {
        var base: [3]u32 = undefined;
        var fraction: Vec3 = undefined;
        inline for (0..3) |axis| {
            const steps = (point[axis] - self.origin[axis]) / self.spacing;
            const last: f32 = @floatFromInt(self.counts[axis] - 2);
            const clamped = std.math.clamp(steps, 0, last + 1);
            const cell = @min(@floor(clamped), last);
            base[axis] = @intFromFloat(cell);
            fraction[axis] = clamped - cell;
        }
        return .{ .base = base, .fraction = fraction };
    }
};

// The golden angle, pi (3 - sqrt 5), in radians.
const golden_angle: f32 = std.math.pi * (3.0 - @sqrt(5.0));

// Ray `i` of `n` from a probe: spherical Fibonacci points, equal in area to a
// part in n and with no pole clustering (Keinert et al., "Spherical Fibonacci
// Mapping", SIGGRAPH Asia 2015). Unit length.
pub fn fibonacciDirection(i: u32, n: u32) Vec3 {
    std.debug.assert(i < n);
    const fi: f32 = @floatFromInt(i);
    const fn_: f32 = @floatFromInt(n);
    const y = 1 - (2 * fi + 1) / fn_;
    const radius = @sqrt(@max(0, 1 - y * y));
    const angle = fi * golden_angle;
    return .{ radius * @cos(angle), y, radius * @sin(angle) };
}

// The real spherical harmonics of bands 0 to 2, in the order the shader stores
// them: Y00; Y1-1, Y10, Y11; Y2-2, Y2-1, Y20, Y21, Y22, each a function of a
// unit direction (x, y, z). The constants are the normalizations that make the
// nine functions orthonormal over the sphere, written from their closed forms
// rather than as decimals so they can be read against the definition.
pub const sh_count = 9;

const sh_c0: f32 = 0.5 * @sqrt(1.0 / std.math.pi);
const sh_c1: f32 = @sqrt(3.0 / (4.0 * std.math.pi));
const sh_c2: f32 = 0.5 * @sqrt(15.0 / std.math.pi);
const sh_c3: f32 = 0.25 * @sqrt(5.0 / std.math.pi);
const sh_c4: f32 = 0.25 * @sqrt(15.0 / std.math.pi);

pub fn shBasis(direction: Vec3) [sh_count]f32 {
    const x = direction[0];
    const y = direction[1];
    const z = direction[2];
    return .{
        sh_c0,
        sh_c1 * y,
        sh_c1 * z,
        sh_c1 * x,
        sh_c2 * x * y,
        sh_c2 * y * z,
        sh_c3 * (3 * z * z - 1),
        sh_c2 * x * z,
        sh_c4 * (x * x - y * y),
    };
}

// The clamped-cosine kernel per band: what turns radiance coefficients into
// irradiance coefficients. Ramamoorthi and Hanrahan, "An Efficient
// Representation for Irradiance Environment Maps", SIGGRAPH 2001, equation 8:
// pi, 2 pi / 3, pi / 4 for bands 0, 1, 2.
pub const irradiance_kernel = [3]f32{ std.math.pi, 2.0 * std.math.pi / 3.0, std.math.pi / 4.0 };

fn band(coefficient: usize) usize {
    return if (coefficient == 0) 0 else if (coefficient < 4) 1 else 2;
}

// Irradiance arriving at a surface facing `normal`, from radiance whose SH
// projection is `radiance`. What leaves a diffuse surface of albedo `a` is this
// times a / pi.
pub fn irradiance(radiance: [sh_count]f32, normal: Vec3) f32 {
    const basis = shBasis(normal);
    var total: f32 = 0;
    for (radiance, basis, 0..) |coefficient, value, index| {
        total += irradiance_kernel[band(index)] * coefficient * value;
    }
    return total;
}

// Octahedral mapping of a unit direction onto [-1, 1]^2 (Cigolle et al.,
// "A Survey of Efficient Representations for Independent Unit Vectors",
// JCGT 3(2) 2014): the sphere folded onto an octahedron and the lower half
// unfolded around the upper. Y is the octahedron's pole, the up axis here.
pub fn octEncode(direction: Vec3) [2]f32 {
    const l1 = @abs(direction[0]) + @abs(direction[1]) + @abs(direction[2]);
    var u = direction[0] / l1;
    var v = direction[2] / l1;
    if (direction[1] < 0) {
        const fold_u = (1 - @abs(v)) * signNotZero(u);
        const fold_v = (1 - @abs(u)) * signNotZero(v);
        u = fold_u;
        v = fold_v;
    }
    return .{ u, v };
}

pub fn octDecode(coordinate: [2]f32) Vec3 {
    const u = coordinate[0];
    const v = coordinate[1];
    var direction: Vec3 = .{ u, 1 - @abs(u) - @abs(v), v };
    if (direction[1] < 0) {
        direction[0] = (1 - @abs(v)) * signNotZero(u);
        direction[2] = (1 - @abs(u)) * signNotZero(v);
    }
    const length = @sqrt(@reduce(.Add, direction * direction));
    return direction / @as(Vec3, @splat(length));
}

// The fold needs a side for zero, or the seam's points collapse onto the axis.
fn signNotZero(value: f32) f32 {
    return if (value >= 0) 1 else -1;
}
