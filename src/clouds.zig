// The cloud layers, host side: the low layer's weather and noise maps, the
// high sheet's mipmapped map, and the light that reaches their heights.
//
// The split is Guerrilla's (Schneider, "The Real-time Volumetric Cloudscapes
// of Horizon: Zero Dawn", SIGGRAPH 2015 Advances): a 2D weather map says
// where clouds are and of what kind, and two tiling 3D noises give them
// their shape, so a heap is a volume with lobes on its sides rather than a
// map pushed upward. The noises are built as in Hillaire's generator for
// Frostbite (github.com/sebh/TileableVolumeNoise, main.cpp), which the
// SIGGRAPH 2016 course notes publish for this purpose.
//
//   - The weather map, RGBA8 over one tile of the sky. R is cover:
//     domain-warped gradient noise with a cellular term for separate heaps,
//     equalised to a uniform spread, so "R above 1 - c" is cloud over a share
//     c of the sky and cover never needs a re-bake. G is the kind, 0 a flat
//     layer and 1 a towering heap, a slower field equalised the same way. B
//     is how ragged a cloud is, from a tidy heap to one torn into wisps,
//     equalised too, so "B above 1 - w" is a share w of the clouds. A is a
//     fine noise for the threads a ragged cloud is combed into, equalised;
//     the shader reads it stretched along the wind, so its period across the
//     wind stays kilometres where a volume's would be hundreds of metres.
//   - The shape volume, RGBA8, `shape_side` cubed. R is Perlin-Worley: three
//     octaves of gradient noise lifted to a floor of Worley noise, which gives
//     the cauliflower lobes; G, B and A are Worley fBm at doubling
//     frequencies, which the shader sums to erode R.
//   - The detail volume, RGBA8, `detail_side` cubed: R, G and B are Worley
//     fBm at doubling frequencies, for the edges. A is zero.
//   - The flow volume, RGBA8, `flow_side` cubed: RGB is a divergence-free
//     vector field, the curl of gradient noise (Bridson et al., "Curl-Noise
//     for Procedural Fluid Flow", SIGGRAPH 2007), stored as v * 0.5 + 0.5 and
//     scaled so the longest vector is one. The shader moves its lookups along
//     it, which is what bends a ragged cloud's edges into swirls and
//     filaments instead of round lobes. A is zero.
//
// Every term is periodic on its tile: a whole number of cells across, the
// hash wrapped, and the warp itself periodic.

const std = @import("std");
const atmosphere = @import("atmosphere.zig");

const Allocator = std.mem.Allocator;
const pi = std.math.pi;

pub const CloudError = error{ InvalidMapSide, InvalidMapLength } || Allocator.Error;

// The sides the application bakes at. The weather map is 512 over 20 km, 39 m
// a texel; the volumes are Guerrilla's and Hillaire's sizes. 128 cubed is 8
// MB and the longest of the three bakes.
pub const map_side: u32 = 512;
pub const shape_side: u32 = 128;
pub const detail_side: u32 = 32;
pub const flow_side: u32 = 32;

// Bakes the weather map into `out`, row major, `side` squared texels.
// Deterministic in `seed`. A load-time call: it allocates one float per
// texel for the equalisation.
pub fn bakeMap(gpa: Allocator, out: [][4]u8, side: u32, seed: u32) CloudError!void {
    if (side < 16 or side > 4096) return error.InvalidMapSide;
    if (out.len != @as(usize, side) * side) return error.InvalidMapLength;
    const field = try gpa.alloc(f32, out.len);
    defer gpa.free(field);

    const scale = 1.0 / @as(f32, @floatFromInt(side));
    // The warp moves the sample by up to about a twelfth of the tile, which
    // bends the cells' edges into the shapes of cumulus fields rather than a
    // lattice of round blobs.
    for (field, 0..) |*value, index| {
        const u = (@as(f32, @floatFromInt(index % side)) + 0.5) * scale;
        const v = (@as(f32, @floatFromInt(index / side)) + 0.5) * scale;
        const wu = u + 0.08 * fbm(u, v, 4, 3, seed +% 101);
        const wv = v + 0.08 * fbm(u, v, 4, 3, seed +% 202);
        value.* = fbm(wu, wv, 8, 5, seed) + 0.35 * (cells(wu, wv, 24, seed +% 303) - 0.5);
    }
    equalise(field, out, 0);

    for (field, 0..) |*value, index| {
        const u = (@as(f32, @floatFromInt(index % side)) + 0.5) * scale;
        const v = (@as(f32, @floatFromInt(index / side)) + 0.5) * scale;
        value.* = fbm(u, v, 4, 3, seed +% 404);
    }
    equalise(field, out, 1);

    // Finer than the kind, so neighbouring heaps of one field can differ.
    for (field, 0..) |*value, index| {
        const u = (@as(f32, @floatFromInt(index % side)) + 0.5) * scale;
        const v = (@as(f32, @floatFromInt(index / side)) + 0.5) * scale;
        value.* = fbm(u, v, 12, 3, seed +% 505);
    }
    equalise(field, out, 2);

    for (field, 0..) |*value, index| {
        const u = (@as(f32, @floatFromInt(index % side)) + 0.5) * scale;
        const v = (@as(f32, @floatFromInt(index / side)) + 0.5) * scale;
        value.* = fbm(u, v, 64, 2, seed +% 606);
    }
    equalise(field, out, 3);
}

// The high sheet's RGBA8 map includes every level down to one texel.
pub const high_side: u32 = 2048;
pub const fibre_side: u32 = 512;
pub const fibreLength = highLength;

pub fn highLength(side: u32) usize {
    var n = side;
    var total: usize = 0;
    while (n > 0) : (n >>= 1) total += @as(usize, n) * n;
    return total;
}

// R and G hold cirrus and veil density; B and A are equalised placement and
// kind on a larger tile. Rows can run independently; equalisation and box
// filtering follow after all of them have finished.
pub fn bakeHigh(gpa: Allocator, io: std.Io, out: [][4]u8, side: u32, seed: u32, workers: usize) CloudError!void {
    if (side < 64 or side > 4096 or !std.math.isPowerOfTwo(side)) return error.InvalidMapSide;
    if (out.len != highLength(side)) return error.InvalidMapLength;
    const count = @as(usize, side) * side;
    const fields = try gpa.alloc(f32, 2 * count);
    defer gpa.free(fields);
    const ranges: usize = @max(1, @min(workers, 16));
    var futures: [15]std.Io.Future(void) = undefined;
    var started: usize = 0;
    {
        defer for (futures[0..started]) |*future| future.await(io);
        for (0..ranges) |i| {
            const begin: u32 = @intCast(@as(usize, side) * i / ranges);
            const end: u32 = @intCast(@as(usize, side) * (i + 1) / ranges);
            if (i + 1 == ranges) {
                highRows(out[0..count], fields, side, seed, begin, end);
            } else {
                futures[i] = io.async(highRows, .{ out[0..count], fields, side, seed, begin, end });
                started += 1;
            }
        }
    }
    equalise(fields[0..count], out[0..count], 2);
    equalise(fields[count..], out[0..count], 3);
    highMips(out, side);
}

fn highMips(out: [][4]u8, side: u32) void {
    const count = @as(usize, side) * side;
    var n = side;
    var parent: usize = 0;
    var next = count;
    while (n > 1) {
        const half = n / 2;
        for (0..half) |y| for (0..half) |x| {
            for (0..4) |c| {
                const a = parent + 2 * y * n + 2 * x;
                const sum = @as(u16, out[a][c]) + out[a + 1][c] + out[a + n][c] + out[a + n + 1][c];
                out[next + y * half + x][c] = @intCast((sum + 2) / 4);
            }
        };
        parent = next;
        next += @as(usize, half) * half;
        n = half;
    }
}

// B and A go to `fields` (placement, then kind) for the equalisation that
// follows; they are smooth fields read on the macro tile.
fn highRows(out: [][4]u8, fields: []f32, side: u32, seed: u32, begin: u32, end: u32) void {
    const scale = 1.0 / @as(f32, @floatFromInt(side));
    const count = @as(usize, side) * side;
    for (out[@as(usize, begin) * side .. @as(usize, end) * side], @as(usize, begin) * side..) |*texel, i| {
        const u = (@as(f32, @floatFromInt(i % side)) + 0.5) * scale;
        const v = (@as(f32, @floatFromInt(i / side)) + 0.5) * scale;
        const veil = smooth(0.3, 0.7, fbm(u, v, 2, 4, seed +% 303) * 0.5 + 0.5);
        const streak = fbmAnisotropic(u, v, 3, 48, 2, seed +% 404) * 0.5 + 0.5;
        texel.* = .{ unorm(highCirrus(u, v, seed)), unorm(veil * (0.8 + 0.2 * streak)), 0, 0 };
        fields[i] = fbm(u, v, 3, 4, seed +% 701);
        fields[count + i] = fbm(u, v, 2, 3, seed +% 802);
    }
}

fn highCirrus(u: f32, v: f32, seed: u32) f32 {
    return 1 - spineRemainder(u, v, seed) * webRemainder(u, v, seed +% 505);
}

// One strand in its own frame: t from the head (0) to the tail's end (1),
// d across its centreline. The tail spreads as it trails; a bright core
// and a faint fringe three times wider; every factor reaches zero at both
// ends, so no strand has a cut edge. Past 6.5 widths the fringe is under
// half a byte and the strand is skipped.
fn strandShape(t: f32, d: f32, width: f32, bumps: f32, h: u32) f32 {
    if (t <= 0 or t >= 1) return 0;
    const x = d / (width * (0.6 + 1.4 * t));
    if (@abs(x) > 6.5) return 0;
    const profile = @max(@exp(-x * x), 0.2 * @exp(-x * x / 9));
    const n = strandNoise(bumps * t, h);
    return smooth(0, 0.1, t) * ((1 - t) * @sqrt(1 - t)) * (0.6 + 0.4 * n) * profile;
}

const strand_reach = 6.5 * 2.0;

// The thin web between the spines: straight strands without branches.
fn webRemainder(u: f32, v: f32, seed: u32) f32 {
    const nx = 4;
    const ny = 40;
    const cx: i32 = @intFromFloat(@floor(u * nx));
    const cy: i32 = @intFromFloat(@floor(v * ny));
    var remaining: f32 = 1;
    // The widest tail's fringe reaches past the next cell across.
    var dy: i32 = -2;
    while (dy <= 2) : (dy += 1) {
        // Heads lie downwind; a one-cell tail reaches from the next cell.
        var dx: i32 = 0;
        while (dx <= 1) : (dx += 1) {
            const x = cx + dx;
            const y = cy + dy;
            const h = hash(@intCast(@mod(x, nx)), @intCast(@mod(y, ny)), seed);
            // A quarter of the cells: a regular web of equal lines reads
            // as hatching.
            if (h % 4 != 0) continue;
            const head_u = (@as(f32, @floatFromInt(x)) + hashShare(h, 1)) / nx;
            const head_v = (@as(f32, @floatFromInt(y)) + 0.25 + 0.5 * hashShare(h, 2)) / ny;
            const length = (0.5 + 0.5 * hashShare(h, 3)) / nx;
            const width = (0.05 + 0.07 * hashShare(h, 4)) / ny;
            const strength = 0.15 + 0.15 * hashShare(h, 5);
            const t = (head_u - u) / length;
            remaining *= 1 - strength * strandShape(t, v - head_v, width, 6, h);
        }
    }
    return remaining;
}

// Cirrus uncinus: a spine with barbs, the barbs with barbules. Barbs leave
// the spine at an angle and trail upwind of where they leave it, as ice
// falling out of the spine lags in slower air below, and they hook back
// toward the wind for the same reason. Three kinds by hash: a feather
// (barbs along one side), a brush (barbs fanned both ways near the head)
// and a plain streak with few.
const Kind = enum { feather, brush, streak };

fn spineRemainder(u: f32, v: f32, seed: u32) f32 {
    const nx = 2;
    const ny = 14;
    const cx: i32 = @intFromFloat(@floor(u * nx));
    const cy: i32 = @intFromFloat(@floor(v * ny));
    var remaining: f32 = 1;
    // Barbs reach up to about 3.9 cells across from their spine's head
    // (see `reach` below); heads five cells away are at least 4.25 away.
    var dy: i32 = -4;
    while (dy <= 4) : (dy += 1) {
        var dx: i32 = 0;
        while (dx <= 2) : (dx += 1) {
            const x = cx + dx;
            const y = cy + dy;
            const h = hash(@intCast(@mod(x, nx)), @intCast(@mod(y, ny)), seed);
            if (h % 3 == 0) continue;
            const head_u = (@as(f32, @floatFromInt(x)) + hashShare(h, 1)) / nx;
            const head_v = (@as(f32, @floatFromInt(y)) + 0.25 + 0.5 * hashShare(h, 2)) / ny;
            const length = (0.6 + 0.6 * hashShare(h, 3)) / nx;
            const width = (0.06 + 0.09 * hashShare(h, 4)) / ny;
            const strength = 0.6 + 0.4 * hashShare(h, 5);
            const t = (head_u - u) / length;
            // Barbs leave at t <= 0.9 and run at most 0.18 of the spine
            // upwind; a brush's barbules can turn a little downwind of its
            // head.
            if (t <= -0.05 or t >= 1.15) continue;
            const d = v - head_v;
            // Across, a barb of length l reaches l * sin (<= 0.94) plus its
            // hook (<= 0.5 l) plus its barbules (<= 0.5 l): under 2 l, and
            // its fringe.
            const longest = 0.18 * length;
            const reach = @max(strand_reach * width, 2 * longest + strand_reach * 0.5 * width);
            if (@abs(d) > reach) continue;

            const pick = hashShare(h, 6);
            const kind: Kind = if (pick < 0.45) .feather else if (pick < 0.75) .brush else .streak;
            // A branched spine is mostly its barbs, so its own line is
            // narrower and fainter than a plain streak's.
            const spine: f32 = if (kind == .streak) 1 else 0.6;
            remaining *= 1 - spine * strength * strandShape(t, d, spine * width, 6, h);
            const count: u32 = switch (kind) {
                .feather => 14 + h % 8,
                .brush => 10 + h % 6,
                .streak => h % 4,
            };
            const side: f32 = if (h & 2 == 0) 1 else -1;
            for (0..count) |i| {
                const c = hash(h, @intCast(i), 0x3c1);
                const at = switch (kind) {
                    .feather, .streak => 0.05 + 0.85 * hashShare(c, 1),
                    .brush => 0.02 + 0.33 * hashShare(c, 1),
                };
                const origin = [2]f32{ head_u - at * length, head_v };
                // The longest barb's span along the wind, from `at` alone:
                // most barbs of a spine near the texel are still far from it.
                const span = 2 * longest + strand_reach * 0.5 * width;
                if (u > origin[0] + 0.5 * longest + strand_reach * 0.5 * width or u < origin[0] - span) continue;
                const barb_length = length * (0.1 + 0.08 * hashShare(c, 3));
                const barb_width = width * (0.3 + 0.2 * hashShare(c, 4));
                const margin = 2 * barb_length + strand_reach * barb_width;
                if (u > origin[0] + 0.5 * barb_length + strand_reach * barb_width or u < origin[0] - margin) continue;
                if (@abs(v - origin[1]) > margin) continue;
                const barb_side: f32 = if (kind == .brush and c & 1 == 0) -side else side;
                const degrees = switch (kind) {
                    .feather => 25 + 30 * hashShare(c, 2),
                    .brush => 30 + 40 * hashShare(c, 2),
                    .streak => 15 + 20 * hashShare(c, 2),
                };
                const angle = degrees * pi / 180;
                const hook = 0.2 + 0.3 * hashShare(c, 5);
                const barb_strength = strength * (0.6 + 0.4 * hashShare(c, 6)) * (1 - 0.5 * at);
                const toward = [2]f32{ -@cos(angle), barb_side * @sin(angle) };
                remaining *= barbRemainder(u, v, origin, toward, barb_length, barb_width, hook, barb_strength, c, kind != .streak);
            }
        }
    }
    return remaining;
}

// A barb from `origin` along the unit `toward`, hooking upwind by `hook`
// of its length at its end; with `branched`, barbules leave it on its outer
// side the same way barbs leave the spine.
fn barbRemainder(u: f32, v: f32, origin: [2]f32, toward: [2]f32, length: f32, width: f32, hook: f32, strength: f32, h: u32, branched: bool) f32 {
    const ru = u - origin[0];
    const rv = v - origin[1];
    // `normal` points to the barb's upwind side, where the hook bends it.
    const normal = [2]f32{ -@abs(toward[1]), toward[0] * std.math.sign(toward[1]) };
    const a = ru * toward[0] + rv * toward[1];
    const b = ru * normal[0] + rv * normal[1];
    const t = a / length;
    if (t <= 0 or t >= 1.2) return 1;
    const offset = b - hook * length * t * t;
    // Barbules run straight while the barb hooks, so the hook's growth
    // between where they leave and t = 1.2 adds to their reach.
    const reach = strand_reach * width + if (branched) 0.5 * length + 1.44 * hook * length + strand_reach * 0.6 * width else 0;
    if (@abs(offset) > reach) return 1;
    var remaining = 1 - strength * strandShape(t, offset, width, 4, h);
    if (!branched) return remaining;
    const count = 3 + h % 4;
    for (0..count) |j| {
        const g = hash(h, @intCast(j), 0x77);
        const at = 0.1 + 0.7 * hashShare(g, 1);
        const start = [2]f32{
            origin[0] + toward[0] * at * length + normal[0] * hook * length * at * at,
            origin[1] + toward[1] * at * length + normal[1] * hook * length * at * at,
        };
        // The longest barbule's box before its trigonometry.
        const box = 0.5 * length + strand_reach * 0.6 * width;
        if (@abs(u - start[0]) > box or @abs(v - start[1]) > box) continue;
        const turn = (20 + 20 * hashShare(g, 2)) * pi / 180;
        // Barbules turn further from the wind than their barb, on the side
        // away from the hook.
        const away = if (toward[1] >= 0) -turn else turn;
        const turned = [2]f32{
            toward[0] * @cos(away) - toward[1] * @sin(away),
            toward[0] * @sin(away) + toward[1] * @cos(away),
        };
        const sub_length = length * (0.3 + 0.2 * hashShare(g, 3));
        const sub_width = width * (0.45 + 0.15 * hashShare(g, 4));
        const sub_strength = strength * (0.5 + 0.3 * hashShare(g, 5)) * (1 - 0.5 * at);
        remaining *= barbRemainder(u, v, start, turned, sub_length, sub_width, 0.1, sub_strength, g, false);
    }
    return remaining;
}

fn strandNoise(at: f32, seed: u32) f32 {
    const cell: u32 = @intFromFloat(@floor(at));
    return std.math.lerp(hashShare(seed, cell % 6 + 10), hashShare(seed, (cell + 1) % 6 + 10), smooth(0, 1, at - @floor(at)));
}

// R is equalised so contrast controls the same range for every seed.
// Other channels are zero; the chain uses the high map's rounded box means.
pub fn bakeHighFibres(gpa: Allocator, io: std.Io, out: [][4]u8, side: u32, seed: u32, workers: usize) CloudError!void {
    if (side < 64 or side > 4096 or !std.math.isPowerOfTwo(side)) return error.InvalidMapSide;
    if (out.len != fibreLength(side)) return error.InvalidMapLength;
    const count = @as(usize, side) * side;
    const field = try gpa.alloc(f32, count);
    defer gpa.free(field);
    const ranges: usize = @max(1, @min(workers, 16));
    var futures: [15]std.Io.Future(void) = undefined;
    var started: usize = 0;
    {
        defer for (futures[0..started]) |*future| future.await(io);
        for (0..ranges) |i| {
            const begin = @as(usize, side) * i / ranges;
            const end = @as(usize, side) * (i + 1) / ranges;
            if (i + 1 == ranges) {
                fibreRows(field, side, seed, begin, end);
            } else {
                futures[i] = io.async(fibreRows, .{ field, side, seed, begin, end });
                started += 1;
            }
        }
    }
    @memset(out, .{ 0, 0, 0, 0 });
    equalise(field, out[0..count], 0);
    highMips(out, side);
}

fn fibreRows(field: []f32, side: u32, seed: u32, begin: usize, end: usize) void {
    const scale = 1 / @as(f32, @floatFromInt(side));
    for (field[begin * side .. end * side], begin * side..) |*value, i| {
        const u = (@as(f32, @floatFromInt(i % side)) + 0.5) * scale;
        const v = (@as(f32, @floatFromInt(i / side)) + 0.5) * scale;
        const ridge = 1 - @abs(fbmAnisotropic(u, v, 8, 64, 2, seed));
        value.* = ridge * ridge * ridge * ridge;
    }
}

pub const BendWave = extern struct { kx: f32, ky: f32, amplitude: f32, phase: f32 };

// Three octaves of six waves: 8-20 km, 3-7 km and 1-2.5 km on the 40 km
// tile. Each octave carries a share of the bend's slope, so the strands
// curve at every scale by angles of one order, which is what reads as
// fractal; amplitudes fall with wavelength.
pub const bend_waves = 18;
const bend_octaves = [3]struct { low: f32, high: f32, share: f32 }{
    .{ .low = 2, .high = 5, .share = 0.55 },
    .{ .low = 6, .high = 14, .share = 0.3 },
    .{ .low = 16, .high = 40, .share = 0.15 },
};

// The sum of gradient magnitudes is bend: d psi/d across >= 1 - bend
// prevents folds. The gradient perturbation lies in a radius-bend disc
// about (0, 1); the strand angle is bounded by asin(bend).
pub fn highBend(seed: u32, period: f32, bend: f32) [bend_waves]BendWave {
    std.debug.assert(period > 0 and bend >= 0 and bend <= 0.9);
    var waves: [bend_waves]BendWave = undefined;
    for (bend_octaves, 0..) |octave, o| {
        const group = waves[o * 6 ..][0..6];
        var bound: f32 = 0;
        for (group, 0..) |*wave, i| {
            const h = hash(seed, @intCast(o * 6 + i), 0x514b);
            // Integer cycles per tile keep psi - across periodic. Directions
            // stay within 70 degrees of the wind's so every wave bends the
            // strands along their length rather than only spacing them.
            var salt: u32 = 0;
            var k: [2]f32 = undefined;
            while (true) : (salt += 1) {
                const radius = octave.low + (octave.high - octave.low) * hashShare(h, 10 + salt);
                const angle = (hashShare(h, 40 + salt) * 2 - 1) * 1.22;
                k = .{ @round(radius * @cos(angle)), @round(radius * @sin(angle)) };
                const norm = @sqrt(k[0] * k[0] + k[1] * k[1]);
                if (k[0] != 0 and norm >= octave.low and norm <= octave.high) break;
            }
            const norm = @sqrt(k[0] * k[0] + k[1] * k[1]);
            const amplitude = (0.5 + hashShare(h, 1)) / norm;
            wave.* = .{ .kx = k[0], .ky = k[1], .amplitude = amplitude, .phase = 2 * pi * hashShare(h, 2) };
            bound += amplitude * 2 * pi * norm / period;
        }
        for (group) |*wave| wave.amplitude *= octave.share * bend / bound;
    }
    return waves;
}

pub fn highPsi(waves: [bend_waves]BendWave, period: f32, along: f32, across: f32) f32 {
    var psi = across;
    for (waves) |wave| {
        // Reducing cycles before sin keeps horizon distances and drift from
        // growing the trigonometric argument on either host or device.
        const cycles = (wave.kx * along + wave.ky * across) / period;
        psi += wave.amplitude * @sin(2 * pi * wrapUnit(cycles) + wave.phase);
    }
    return psi;
}

fn hashShare(h: u32, salt: u32) f32 {
    return @as(f32, @floatFromInt(hash(h, salt, 0x917ab))) / 4294967296.0;
}

fn smooth(low: f32, high: f32, value: f32) f32 {
    const t = std.math.clamp((value - low) / (high - low), 0, 1);
    return t * t * (3 - 2 * t);
}

// Placement in metres in a unit wind's frame: offset.x along it, offset.y
// across it. A zero period turns the layer off, as on the device.
pub const HighPlacement = struct {
    offset: [2]f32 = .{ 0, 0 },
    period: f32 = 0,
    macro_period: f32 = 173000,
    cover: f32 = 0.5,
    sheet: f32 = 0.3,
    cirrus_depth: f32 = 0.3,
    sheet_depth: f32 = 0.15,
    wind: [2]f32 = .{ 1, 0 },
    fibre_period: [2]f32 = .{ 8000, 2000 },
    fibre_contrast: f32 = 0.35,
    bend: [bend_waves]BendWave = @splat(.{ .kx = 0, .ky = 0, .amplitude = 0, .phase = 0 }),
};

pub fn highDepth(texels: []const [4]u8, side: u32, fibres: []const [4]u8, fibre_size: u32, placement: HighPlacement, x: f32, z: f32) f32 {
    const p = placement;
    if (p.period <= 0) return 0;
    const along = x * p.wind[0] + z * p.wind[1] - p.offset[0];
    const across = -x * p.wind[1] + z * p.wind[0] - p.offset[1];
    const m = highSample(texels, side, along / p.macro_period + 0.37, across / p.macro_period + 0.71);
    const psi = highPsi(p.bend, p.period, along, across);
    // The veil shares the bent coordinates so its streaks follow the strands.
    const d = highSample(texels, side, along / p.period, psi / p.period);
    const f = highSample(fibres, fibre_size, along / p.fibre_period[0], psi / p.fibre_period[1])[0];
    const cirrus = d[0] * (1 - p.fibre_contrast + p.fibre_contrast * f);
    const mask = highShare(p.cover, m[2], 0.08);
    const share = highShare(p.sheet, m[3], 0.1);
    return mask * (p.cirrus_depth * cirrus * (1 - share) + p.sheet_depth * d[1] * share);
}

fn highShare(share: f32, value: f32, edge: f32) f32 {
    if (share <= 0) return 0;
    if (share >= 1) return 1;
    return smooth(1 - share - edge, 1 - share + edge, value);
}

fn highSample(texels: []const [4]u8, side: u32, u: f32, v: f32) [4]f32 {
    const x = wrapUnit(u) * @as(f32, @floatFromInt(side)) - 0.5;
    const y = wrapUnit(v) * @as(f32, @floatFromInt(side)) - 0.5;
    const ix: i64 = @intFromFloat(@floor(x));
    const iy: i64 = @intFromFloat(@floor(y));
    const tx = x - @floor(x);
    const ty = y - @floor(y);
    var result: [4]f32 = undefined;
    for (&result, 0..) |*value, c| {
        var rows: [2]f32 = undefined;
        for (&rows, 0..) |*row, j| {
            const a: usize = @intCast(@mod(iy + @as(i64, @intCast(j)), side) * side + @mod(ix, side));
            const b: usize = @intCast(@mod(iy + @as(i64, @intCast(j)), side) * side + @mod(ix + 1, side));
            row.* = std.math.lerp(@as(f32, @floatFromInt(texels[a][c])), @as(f32, @floatFromInt(texels[b][c])), tx) / 255;
        }
        value.* = std.math.lerp(rows[0], rows[1], ty);
    }
    return result;
}

// Ice sends much of the extinguished direct beam forward, where it can
// still reach the ground near the sun's direction. Cap the lost light to
// account for that share without tracing the forward scattering on the host.
pub fn highSunShare(depth: f32, toward_sun_y: f32, cap: f32) f32 {
    return 1 - @min(cap, 1 - @exp(-depth / @max(toward_sun_y, 0.05)));
}

// Bakes the shape volume into `out`, `side` cubed, x fastest then y then z.
// Allocates nothing. The slices along z are split over up to `workers`
// ranges, the last on this thread; every texel depends only on its own
// position, so the bytes are the same however it is split.
pub fn bakeShape(io: std.Io, out: [][4]u8, side: u32, seed: u32, workers: usize) CloudError!void {
    if (side < 8 or side > 512) return error.InvalidMapSide;
    if (out.len != @as(usize, side) * side * side) return error.InvalidMapLength;

    const max_ranges = 16;
    const ranges: u32 = @intCast(@max(@min(@min(workers, side), max_ranges), 1));
    var bounds: [max_ranges + 1]u32 = undefined;
    for (bounds[0 .. ranges + 1], 0..) |*bound, index| {
        bound.* = @intCast(@as(u64, side) * index / ranges);
    }

    var futures: [max_ranges - 1]std.Io.Future(void) = undefined;
    var started: usize = 0;
    defer for (futures[0..started]) |*future| future.await(io);
    for (0..ranges) |index| {
        const begin = bounds[index];
        const end = bounds[index + 1];
        if (index + 1 == ranges) {
            shapeSlices(out, side, seed, begin, end);
            break;
        }
        futures[index] = io.async(shapeSlices, .{ out, side, seed, begin, end });
        started += 1;
    }
}

// Slices `begin` up to `end` of the shape volume.
fn shapeSlices(out: [][4]u8, side: u32, seed: u32, begin: u32, end: u32) void {
    // Hillaire's constants: gradient noise at 8 cells, 3 octaves; its Worley
    // floor at 4 cells times 2, 8 and 14; the fBm channels at 4 cells times
    // 1, 2, 4, 8 and 16.
    const scale = 1.0 / @as(f32, @floatFromInt(side));
    const slice = @as(usize, side) * side;
    for (out[begin * slice .. end * slice], begin * slice..) |*texel, index| {
        const p = volumePoint(index, side, scale);
        const perlin = std.math.clamp(fbm3(p, 8, 3, seed) * 0.5 + 0.5, 0, 1);
        const floor = worleyFbm(p, .{ 8, 32, 56 }, .{ 0.625, 0.25, 0.125 }, seed +% 11);
        const perlin_worley = remap(perlin, 0, 1, floor, 1);
        const w = [5]f32{
            1 - worley(p, 4, seed +% 21),
            1 - worley(p, 8, seed +% 22),
            1 - worley(p, 16, seed +% 23),
            1 - worley(p, 32, seed +% 24),
            1 - worley(p, 64, seed +% 25),
        };
        texel.* = .{
            unorm(perlin_worley),
            unorm(w[0] * 0.625 + w[1] * 0.25 + w[2] * 0.125),
            unorm(w[1] * 0.625 + w[2] * 0.25 + w[3] * 0.125),
            unorm(w[2] * 0.75 + w[3] * 0.25),
        };
    }
}

// Bakes the detail volume into `out`, `side` cubed, in the shape volume's
// order. Hillaire's erosion texture: Worley at 2 cells times 1, 2, 4 and 8.
pub fn bakeDetail(out: [][4]u8, side: u32, seed: u32) CloudError!void {
    if (side < 8 or side > 512) return error.InvalidMapSide;
    if (out.len != @as(usize, side) * side * side) return error.InvalidMapLength;
    const scale = 1.0 / @as(f32, @floatFromInt(side));
    for (out, 0..) |*texel, index| {
        const p = volumePoint(index, side, scale);
        const w = [4]f32{
            1 - worley(p, 2, seed +% 31),
            1 - worley(p, 4, seed +% 32),
            1 - worley(p, 8, seed +% 33),
            1 - worley(p, 16, seed +% 34),
        };
        texel.* = .{
            unorm(w[0] * 0.625 + w[1] * 0.25 + w[2] * 0.125),
            unorm(w[1] * 0.625 + w[2] * 0.25 + w[3] * 0.125),
            unorm(w[2] * 0.75 + w[3] * 0.25),
            0,
        };
    }
}

// Bakes the flow volume into `out`, in the shape volume's order. The
// potential is three gradient fBm fields at 4 cells, 2 octaves, and the curl
// is taken by central differences a quarter of a texel wide, which is
// periodic because the fields are. Two passes, the first for the longest
// vector, so it allocates nothing.
pub fn bakeFlow(out: [][4]u8, side: u32, seed: u32) CloudError!void {
    if (side < 8 or side > 512) return error.InvalidMapSide;
    if (out.len != @as(usize, side) * side * side) return error.InvalidMapLength;
    const scale = 1.0 / @as(f32, @floatFromInt(side));
    var longest: f32 = 0;
    for (0..out.len) |index| {
        const v = curl(volumePoint(index, side, scale), 0.25 * scale, seed);
        longest = @max(longest, @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
    }
    const inverse = if (longest > 0) 1 / longest else 0;
    for (out, 0..) |*texel, index| {
        const v = curl(volumePoint(index, side, scale), 0.25 * scale, seed);
        texel.* = .{
            unorm(v[0] * inverse * 0.5 + 0.5),
            unorm(v[1] * inverse * 0.5 + 0.5),
            unorm(v[2] * inverse * 0.5 + 0.5),
            0,
        };
    }
}

fn curl(p: [3]f32, h: f32, seed: u32) [3]f32 {
    // d[i][j]: the derivative of the potential's component i along axis j.
    var d: [3][3]f32 = undefined;
    for (0..3) |i| {
        const component_seed = seed +% 41 +% @as(u32, @intCast(i)) *% 131;
        for (0..3) |j| {
            var ahead = p;
            var behind = p;
            ahead[j] += h;
            behind[j] -= h;
            d[i][j] = (fbm3(ahead, 4, 2, component_seed) - fbm3(behind, 4, 2, component_seed)) / (2 * h);
        }
    }
    return .{ d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1] };
}

fn volumePoint(index: usize, side: u32, scale: f32) [3]f32 {
    const s: usize = side;
    return .{
        (@as(f32, @floatFromInt(index % s)) + 0.5) * scale,
        (@as(f32, @floatFromInt(index / s % s)) + 0.5) * scale,
        (@as(f32, @floatFromInt(index / (s * s))) + 0.5) * scale,
    };
}

fn remap(value: f32, low: f32, high: f32, new_low: f32, new_high: f32) f32 {
    return new_low + (value - low) / (high - low) * (new_high - new_low);
}

fn unorm(value: f32) u8 {
    return @intFromFloat(@round(255 * std.math.clamp(value, 0, 1)));
}

// What reaches a layer at `altitude_km` from a sun of unit irradiance above
// the atmosphere, and what the air between it and the ground takes away from
// a cloud seen through it.
pub const LayerLight = struct {
    // The sun's share at the layer, per channel. Zero once the planet hides
    // the sun from that height, which is a few minutes after it has set at
    // the ground: a layer at 2 km sees about 1.4 degrees further round.
    sun: [3]f32,
    // The air's mean extinction per kilometre on any path from the ground up
    // to the layer. For a density falling as exp(-y / H) the mean over
    // heights 0 to h is H (1 - exp(-h / H)) / h, the same whatever the
    // path's slope, so a cloud's distance times this is its optical depth.
    // The planet's curvature is left out; it matters only near the horizon,
    // where the layer is far enough to be lost in haze either way.
    haze: [3]f32,
};

pub fn layerLight(
    tables: *const atmosphere.Tables,
    a: atmosphere.Atmosphere,
    altitude_km: f32,
    toward_sun: [3]f32,
) atmosphere.AtmosphereError!LayerLight {
    const sun = try atmosphere.transmittance(tables, a, altitude_km, toward_sun);
    const rayleigh = meanDensity(altitude_km, a.rayleigh_scale_height);
    const mie = meanDensity(altitude_km, a.mie_scale_height);
    var haze: [3]f32 = undefined;
    for (&haze, a.rayleigh_scattering, a.mie_extinction) |*h, r, m| h.* = r * rayleigh + m * a.aerosol * mie;
    return .{ .sun = sun, .haze = haze };
}

fn meanDensity(altitude_km: f32, scale_height: f32) f32 {
    // The limit at the ground is one; the quotient loses precision well
    // before a metre.
    if (altitude_km < 1.0e-3) return 1;
    return scale_height * (1 - @exp(-altitude_km / scale_height)) / altitude_km;
}

// Spreads a channel uniformly over 0-255 by rank: a histogram of 4096 bins
// between the field's extremes and its running sum. Equal texels land on one
// value, so the spread is uniform to within a bin.
fn equalise(field: []const f32, out: [][4]u8, channel: usize) void {
    const bins = 4096;
    var counts: [bins]u32 = @splat(0);
    const low, const high = extremes(field);
    const width = @max(high - low, 1.0e-12);
    for (field) |value| counts[bin(value, low, width, bins)] += 1;

    // The value of a bin is the share of texels below its middle.
    var below: [bins]u8 = undefined;
    var running: u64 = 0;
    const total: f64 = @floatFromInt(field.len);
    for (&below, counts) |*b, count| {
        const middle = @as(f64, @floatFromInt(running)) + @as(f64, @floatFromInt(count)) * 0.5;
        b.* = @intFromFloat(@round(255 * middle / total));
        running += count;
    }
    for (field, out) |value, *texel| texel[channel] = below[bin(value, low, width, bins)];
}

fn extremes(field: []const f32) struct { f32, f32 } {
    var low = field[0];
    var high = field[0];
    for (field) |value| {
        low = @min(low, value);
        high = @max(high, value);
    }
    return .{ low, high };
}

fn bin(value: f32, low: f32, width: f32, comptime bins: u32) usize {
    const at: u32 = @intFromFloat(@floor((value - low) / width * bins));
    return @min(at, bins - 1);
}

// Fractal sum of gradient noise, periodic on the unit square: octave k has
// `period` times 2^k cells across and half the amplitude of the one before.
// Divided by the sum of the amplitudes, so within about +-0.7.
fn fbm(u: f32, v: f32, period: u32, octaves: u32, seed: u32) f32 {
    var sum: f32 = 0;
    var amplitude: f32 = 1;
    var total: f32 = 0;
    var cells_across = period;
    for (0..octaves) |octave| {
        const x = wrapUnit(u) * @as(f32, @floatFromInt(cells_across));
        const y = wrapUnit(v) * @as(f32, @floatFromInt(cells_across));
        sum += amplitude * gradientNoise(x, y, cells_across, seed +% @as(u32, @intCast(octave)) *% 7919);
        total += amplitude;
        amplitude *= 0.5;
        cells_across *= 2;
    }
    return sum / total;
}

// Perlin's improved noise in two dimensions (quintic fade), with gradients
// at a hashed angle and the lattice wrapped every `period` cells.
fn gradientNoise(x: f32, y: f32, period: u32, seed: u32) f32 {
    const fx = @floor(x);
    const fy = @floor(y);
    const tx = x - fx;
    const ty = y - fy;
    const ix: u32 = @as(u32, @intFromFloat(fx)) % period;
    const iy: u32 = @as(u32, @intFromFloat(fy)) % period;
    const jx = (ix + 1) % period;
    const jy = (iy + 1) % period;

    const n00 = corner(ix, iy, tx, ty, seed);
    const n10 = corner(jx, iy, tx - 1, ty, seed);
    const n01 = corner(ix, jy, tx, ty - 1, seed);
    const n11 = corner(jx, jy, tx - 1, ty - 1, seed);
    const sx = fade(tx);
    const sy = fade(ty);
    return std.math.lerp(std.math.lerp(n00, n10, sx), std.math.lerp(n01, n11, sx), sy);
}

fn corner(ix: u32, iy: u32, dx: f32, dy: f32, seed: u32) f32 {
    const angle = @as(f32, @floatFromInt(hash(ix, iy, seed))) * (2 * pi / 4294967296.0);
    return @cos(angle) * dx + @sin(angle) * dy;
}

fn fade(t: f32) f32 {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// One minus the distance to the nearest of one hashed point per cell, in
// cell widths, held to [0, 1]: round heaps, highest at their points.
fn cells(u: f32, v: f32, across: u32, seed: u32) f32 {
    const x = wrapUnit(u) * @as(f32, @floatFromInt(across));
    const y = wrapUnit(v) * @as(f32, @floatFromInt(across));
    const fx = @floor(x);
    const fy = @floor(y);
    const cx: i64 = @intFromFloat(fx);
    const cy: i64 = @intFromFloat(fy);
    var nearest: f32 = 4;
    var oy: i64 = -1;
    while (oy <= 1) : (oy += 1) {
        var ox: i64 = -1;
        while (ox <= 1) : (ox += 1) {
            const wx: u32 = @intCast(@mod(cx + ox, across));
            const wy: u32 = @intCast(@mod(cy + oy, across));
            const h = hash(wx, wy, seed);
            const px = fx + @as(f32, @floatFromInt(ox)) + @as(f32, @floatFromInt(h & 0xffff)) / 65536.0;
            const py = fy + @as(f32, @floatFromInt(oy)) + @as(f32, @floatFromInt(h >> 16)) / 65536.0;
            nearest = @min(nearest, (px - x) * (px - x) + (py - y) * (py - y));
        }
    }
    return 1 - @min(@sqrt(nearest), 1);
}

fn wrapUnit(t: f32) f32 {
    return t - @floor(t);
}

// Chris Wellons' lowbias32 finaliser (hash-prospector) over the lattice
// coordinates and the seed mixed by odd constants.
fn hash(x: u32, y: u32, seed: u32) u32 {
    var h = x *% 0x8da6b343 ^ y *% 0xd8163841 ^ seed *% 0xcb1ab31f;
    h ^= h >> 16;
    h *%= 0x7feb352d;
    h ^= h >> 15;
    h *%= 0x846ca68b;
    h ^= h >> 16;
    return h;
}

// Gradient noise in three dimensions with the quintic fade, periodic every
// `period` cells, the gradient one of the twelve cube edge directions (Perlin,
// "Improving Noise", SIGGRAPH 2002), within about +-1.
fn gradientNoise3(p: [3]f32, period: u32) f32 {
    var cell: [3]u32 = undefined;
    var next: [3]u32 = undefined;
    var t: [3]f32 = undefined;
    for (0..3) |axis| {
        const f = @floor(p[axis]);
        t[axis] = p[axis] - f;
        cell[axis] = @as(u32, @intFromFloat(f)) % period;
        next[axis] = (cell[axis] + 1) % period;
    }
    var corners: [8]f32 = undefined;
    for (&corners, 0..) |*c, index| {
        const ox: u1 = @intCast(index & 1);
        const oy: u1 = @intCast(index >> 1 & 1);
        const oz: u1 = @intCast(index >> 2 & 1);
        const x = if (ox == 1) next[0] else cell[0];
        const y = if (oy == 1) next[1] else cell[1];
        const z = if (oz == 1) next[2] else cell[2];
        const d = [3]f32{ t[0] - @as(f32, @floatFromInt(ox)), t[1] - @as(f32, @floatFromInt(oy)), t[2] - @as(f32, @floatFromInt(oz)) };
        c.* = edgeGradient(hash3(x, y, z, 0x9e3779b9), d);
    }
    const sx = fade(t[0]);
    const sy = fade(t[1]);
    const sz = fade(t[2]);
    const x00 = std.math.lerp(corners[0], corners[1], sx);
    const x10 = std.math.lerp(corners[2], corners[3], sx);
    const x01 = std.math.lerp(corners[4], corners[5], sx);
    const x11 = std.math.lerp(corners[6], corners[7], sx);
    return std.math.lerp(std.math.lerp(x00, x10, sy), std.math.lerp(x01, x11, sy), sz);
}

fn edgeGradient(h: u32, d: [3]f32) f32 {
    return switch (h % 12) {
        0 => d[0] + d[1],
        1 => -d[0] + d[1],
        2 => d[0] - d[1],
        3 => -d[0] - d[1],
        4 => d[0] + d[2],
        5 => -d[0] + d[2],
        6 => d[0] - d[2],
        7 => -d[0] - d[2],
        8 => d[1] + d[2],
        9 => -d[1] + d[2],
        10 => d[1] - d[2],
        else => -d[1] - d[2],
    };
}

// Octaves of `gradientNoise3` on the unit cube, doubling from `period` cells
// and halving in amplitude, divided by the amplitudes' sum.
fn fbm3(p: [3]f32, period: u32, octaves: u32, seed: u32) f32 {
    var sum: f32 = 0;
    var amplitude: f32 = 1;
    var total: f32 = 0;
    var across = period;
    for (0..octaves) |octave| {
        // The seed shifts the lattice by whole tiles' worth of cells, which
        // keeps the period and changes which hashes a point meets.
        const shift: f32 = @floatFromInt((seed +% @as(u32, @intCast(octave)) *% 7919) % 997);
        const n = @as(f32, @floatFromInt(across));
        const q = [3]f32{ wrapUnit(p[0]) * n + shift * n, wrapUnit(p[1]) * n, wrapUnit(p[2]) * n };
        sum += amplitude * gradientNoise3(q, across);
        total += amplitude;
        amplitude *= 0.5;
        across *= 2;
    }
    return sum / total;
}

// The distance to the nearest of one hashed point per cell, in cell widths,
// over a grid `across` cells to the unit cube and wrapping, held to [0, 1].
fn worley(p: [3]f32, across: u32, seed: u32) f32 {
    const n: f32 = @floatFromInt(across);
    const x = [3]f32{ wrapUnit(p[0]) * n, wrapUnit(p[1]) * n, wrapUnit(p[2]) * n };
    const c = [3]i64{ @intFromFloat(@floor(x[0])), @intFromFloat(@floor(x[1])), @intFromFloat(@floor(x[2])) };
    var nearest: f32 = 4;
    var dz: i64 = -1;
    while (dz <= 1) : (dz += 1) {
        var dy: i64 = -1;
        while (dy <= 1) : (dy += 1) {
            var dx: i64 = -1;
            while (dx <= 1) : (dx += 1) {
                const k = [3]i64{ c[0] + dx, c[1] + dy, c[2] + dz };
                const h = hash3(@intCast(@mod(k[0], across)), @intCast(@mod(k[1], across)), @intCast(@mod(k[2], across)), seed);
                const offset = [3]f32{
                    @as(f32, @floatFromInt(h & 0x3ff)) / 1024.0,
                    @as(f32, @floatFromInt(h >> 10 & 0x3ff)) / 1024.0,
                    @as(f32, @floatFromInt(h >> 20 & 0x3ff)) / 1024.0,
                };
                var squared: f32 = 0;
                for (0..3) |axis| {
                    const d = @as(f32, @floatFromInt(k[axis])) + offset[axis] - x[axis];
                    squared += d * d;
                }
                nearest = @min(nearest, squared);
            }
        }
    }
    return @min(@sqrt(nearest), 1);
}

fn worleyFbm(p: [3]f32, comptime across: [3]u32, weights: [3]f32, seed: u32) f32 {
    var sum: f32 = 0;
    inline for (across, weights, 0..) |a, w, i| sum += w * (1 - worley(p, a, seed +% @as(u32, i)));
    return sum;
}

fn hash3(x: u32, y: u32, z: u32, seed: u32) u32 {
    return hash(x ^ z *% 0x27d4eb2d, y, seed);
}

// Separate lattice periods keep the fibres long along the wind and fine
// across it without losing the unit tile's wrap.
fn fbmAnisotropic(u: f32, v: f32, px: u32, py: u32, octaves: u32, seed: u32) f32 {
    var sum: f32 = 0;
    var total: f32 = 0;
    var amplitude: f32 = 1;
    var nx = px;
    var ny = py;
    for (0..octaves) |k| {
        sum += amplitude * gradientNoiseAnisotropic(wrapUnit(u) * @as(f32, @floatFromInt(nx)), wrapUnit(v) * @as(f32, @floatFromInt(ny)), nx, ny, seed +% @as(u32, @intCast(k)) *% 7919);
        total += amplitude;
        amplitude *= 0.5;
        nx *= 2;
        ny *= 2;
    }
    return sum / total;
}

fn gradientNoiseAnisotropic(x: f32, y: f32, period_x: u32, period_y: u32, seed: u32) f32 {
    const fx = @floor(x);
    const fy = @floor(y);
    const tx = x - fx;
    const ty = y - fy;
    const ix: u32 = @as(u32, @intFromFloat(fx)) % period_x;
    const iy: u32 = @as(u32, @intFromFloat(fy)) % period_y;
    const jx = (ix + 1) % period_x;
    const jy = (iy + 1) % period_y;

    const n00 = corner(ix, iy, tx, ty, seed);
    const n10 = corner(jx, iy, tx - 1, ty, seed);
    const n01 = corner(ix, jy, tx, ty - 1, seed);
    const n11 = corner(jx, jy, tx - 1, ty - 1, seed);
    const sx = fade(tx);
    const sy = fade(ty);
    return std.math.lerp(std.math.lerp(n00, n10, sx), std.math.lerp(n01, n11, sx), sy);
}
