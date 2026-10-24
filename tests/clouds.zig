const std = @import("std");
const scene = @import("lenore-scene");

const testing = std.testing;
const clouds = scene.clouds;

const side = 64;

fn bake(seed: u32) ![side * side][4]u8 {
    var map: [side * side][4]u8 = undefined;
    try clouds.bakeMap(testing.allocator, &map, side, seed);
    return map;
}

test "the map is the same for the same seed and differs for another" {
    const a = try bake(7);
    const b = try bake(7);
    const c = try bake(8);
    try testing.expectEqualSlices([4]u8, &a, &b);
    try testing.expect(!std.mem.eql([4]u8, &a, &c));
}

test "cover is a share of the sky: R above 1 - c covers about c" {
    const map = try bake(1);
    for ([_]f32{ 0.2, 0.5, 0.8 }) |cover| {
        const threshold: u8 = @intFromFloat(@round(255 * (1 - cover)));
        var above: usize = 0;
        for (map) |texel| {
            if (texel[0] >= threshold) above += 1;
        }
        const share = @as(f32, @floatFromInt(above)) / (side * side);
        try testing.expectApproxEqAbs(cover, share, 0.02);
    }
}

// The step across the tile's edge, from the last column to the first and the
// last row to the first, against the step between neighbours inside it. A
// seam shows as a mean jump several times the interior's.
test "the map repeats without a seam in either direction" {
    const map = try bake(3);
    for (0..4) |channel| {
        var inside: f64 = 0;
        var across: f64 = 0;
        for (0..side) |row| {
            for (0..side - 1) |column| inside += step(map, row, column, row, column + 1, channel);
            across += step(map, row, side - 1, row, 0, channel);
        }
        var inside_rows: f64 = 0;
        var across_rows: f64 = 0;
        for (0..side) |column| {
            for (0..side - 1) |row| inside_rows += step(map, row, column, row + 1, column, channel);
            across_rows += step(map, side - 1, column, 0, column, channel);
        }
        try testing.expect(across / side < 2 * inside / (side * (side - 1)));
        try testing.expect(across_rows / side < 2 * inside_rows / (side * (side - 1)));
    }
}

// Raggedness is equalised as cover is, so a share of the clouds is ragged
// whatever the knob is set to.
test "raggedness is spread uniformly: B above 1 - w is about w" {
    const map = try bake(2);
    for ([_]f32{ 0.25, 0.5, 0.75 }) |share| {
        const threshold: u8 = @intFromFloat(@round(255 * (1 - share)));
        var above: usize = 0;
        for (map) |texel| {
            if (texel[2] >= threshold) above += 1;
        }
        try testing.expectApproxEqAbs(share, @as(f32, @floatFromInt(above)) / (side * side), 0.02);
    }
}

fn step(map: [side * side][4]u8, r0: usize, c0: usize, r1: usize, c1: usize, channel: usize) f64 {
    const a: f64 = @floatFromInt(map[r0 * side + c0][channel]);
    const b: f64 = @floatFromInt(map[r1 * side + c1][channel]);
    return @abs(a - b);
}

// The volumes at 16 a side, small enough for a debug build. Same seam test,
// along each of the three axes.
const volume_side = 16;
const Volume = [volume_side * volume_side * volume_side][4]u8;

fn at(volume: *const Volume, x: usize, y: usize, z: usize) [4]u8 {
    return volume[(z * volume_side + y) * volume_side + x];
}

fn expectTiles(volume: *const Volume, channels: usize) !void {
    const n = volume_side;
    for (0..channels) |channel| {
        for (0..3) |axis| {
            var inside: f64 = 0;
            var across: f64 = 0;
            for (0..n) |a| for (0..n) |b| {
                for (0..n - 1) |i| inside += difference(volume, axis, a, b, i, i + 1, channel);
                across += difference(volume, axis, a, b, n - 1, 0, channel);
            };
            try testing.expect(across / (n * n) < 2 * inside / (n * n * (n - 1)));
        }
    }
}

fn difference(volume: *const Volume, axis: usize, a: usize, b: usize, i: usize, j: usize, channel: usize) f64 {
    const p = switch (axis) {
        0 => .{ at(volume, i, a, b), at(volume, j, a, b) },
        1 => .{ at(volume, a, i, b), at(volume, a, j, b) },
        else => .{ at(volume, a, b, i), at(volume, a, b, j) },
    };
    const first: f64 = @floatFromInt(p[0][channel]);
    const second: f64 = @floatFromInt(p[1][channel]);
    return @abs(first - second);
}

test "the shape and detail volumes repeat without a seam on every axis" {
    var shape: Volume = undefined;
    try clouds.bakeShape(testing.io, &shape, volume_side, 5, 1);
    try expectTiles(&shape, 4);
    var detail: Volume = undefined;
    try clouds.bakeDetail(&detail, volume_side, 5);
    try expectTiles(&detail, 3);
    for (detail) |texel| try testing.expectEqual(@as(u8, 0), texel[3]);
}

// The slices are split over workers by position only, so any split, more
// workers than slices included, gives the bytes of one range.
test "the shape volume is the same however its slices are split" {
    var whole: Volume = undefined;
    try clouds.bakeShape(testing.io, &whole, volume_side, 3, 1);
    for ([_]usize{ 3, 16, 64 }) |workers| {
        var split: Volume = undefined;
        try clouds.bakeShape(testing.io, &split, volume_side, 3, workers);
        try testing.expectEqualSlices([4]u8, &whole, &split);
    }
}

// The flow tiles like the other volumes, its longest vector is one, and it
// has no divergence: per texel, by central differences over the stored
// field, against the size of one axis's difference.
test "the flow volume tiles, fills its range and is divergence-free" {
    var flow: Volume = undefined;
    try clouds.bakeFlow(&flow, volume_side, 4);
    try expectTiles(&flow, 3);
    var longest: f32 = 0;
    for (flow) |texel| {
        try testing.expectEqual(@as(u8, 0), texel[3]);
        var squared: f32 = 0;
        for (texel[0..3]) |c| {
            const v = @as(f32, @floatFromInt(c)) / 255 * 2 - 1;
            squared += v * v;
        }
        longest = @max(longest, @sqrt(squared));
    }
    try testing.expectApproxEqAbs(@as(f32, 1), longest, 0.02);

    // At 32 a side, so the finer octave's cells are four texels across and
    // the differences resolve them.
    const n = 32;
    const fine = try testing.allocator.alloc([4]u8, n * n * n);
    defer testing.allocator.free(fine);
    try clouds.bakeFlow(fine, n, 4);
    var divergence: f64 = 0;
    var spread: f64 = 0;
    for (0..n) |z| for (0..n) |y| for (0..n) |x| {
        var sum: f64 = 0;
        for (0..3) |axis| {
            var ahead = [3]usize{ x, y, z };
            var behind = ahead;
            ahead[axis] = (ahead[axis] + 1) % n;
            behind[axis] = (behind[axis] + n - 1) % n;
            const a: f64 = @floatFromInt(fine[(ahead[2] * n + ahead[1]) * n + ahead[0]][axis]);
            const b: f64 = @floatFromInt(fine[(behind[2] * n + behind[1]) * n + behind[0]][axis]);
            sum += a - b;
            spread += @abs(a - b);
        }
        divergence += @abs(sum);
    };
    // Measured 0.23 of a term's mean size; three independent terms would
    // sum to more than one.
    try testing.expect(divergence < 0.35 * spread / 3);
}

// Perlin-Worley is the gradient noise lifted to a floor of Worley noise, so it
// never falls below its own Worley fBm, and both spread over a good part of
// the range rather than sitting at one value.
test "the shape's Perlin-Worley spreads and stays above its floor's least" {
    var shape: Volume = undefined;
    try clouds.bakeShape(testing.io, &shape, volume_side, 9, 1);
    var low: u8 = 255;
    var high: u8 = 0;
    for (shape) |texel| {
        low = @min(low, texel[0]);
        high = @max(high, texel[0]);
    }
    try testing.expect(high - low > 100);
    try testing.expect(low > 0);
}

test "a volume of the wrong length or side is refused" {
    var small: [4 * 4 * 4][4]u8 = undefined;
    try testing.expectError(error.InvalidMapSide, clouds.bakeShape(testing.io, &small, 4, 0, 1));
    var short: [volume_side * volume_side * volume_side - 1][4]u8 = undefined;
    try testing.expectError(error.InvalidMapLength, clouds.bakeDetail(&short, volume_side, 0));
}

test "a map of the wrong length or side is refused" {
    var small: [8 * 8][4]u8 = undefined;
    try testing.expectError(error.InvalidMapSide, clouds.bakeMap(testing.allocator, &small, 8, 0));
    var short: [side * side - 1][4]u8 = undefined;
    try testing.expectError(error.InvalidMapLength, clouds.bakeMap(testing.allocator, &short, side, 0));
}

test "the layer sees the sun after the ground has lost it, and less air than the ground" {
    const a: scene.Atmosphere = .{};
    const tables = try testing.allocator.create(scene.Tables);
    defer testing.allocator.destroy(tables);
    try tables.build(a);

    // One degree under the horizon: gone at the ground, still lit at 2 km.
    const below = [3]f32{ @cos(std.math.degreesToRadians(-1.0)), @sin(std.math.degreesToRadians(-1.0)), 0 };
    const ground = try clouds.layerLight(tables, a, scene.planet_radius_offset, below);
    const layer = try clouds.layerLight(tables, a, 2, below);
    try testing.expectEqual(@as(f32, 0), ground.sun[0]);
    try testing.expect(layer.sun[0] > 0);
    // Reddened: the long wavelengths come through better.
    try testing.expect(layer.sun[0] > layer.sun[2]);

    // At the ground the mean is the sea-level extinction itself.
    for (0..3) |c| {
        const sea = a.rayleigh_scattering[c] + a.mie_extinction[c] * a.aerosol;
        try testing.expectApproxEqRel(sea, ground.haze[c], 0.01);
        try testing.expect(layer.haze[c] < ground.haze[c]);
    }
}

const high_length = clouds.highLength(side);

fn bakeHigh(seed: u32, workers: usize) ![high_length][4]u8 {
    var out: [high_length][4]u8 = undefined;
    try clouds.bakeHigh(testing.allocator, testing.io, &out, side, seed, workers);
    return out;
}

test "the high map is identical with one, three and sixteen workers" {
    const one = try bakeHigh(3, 1);
    for ([_]usize{ 3, 16 }) |workers| {
        const split = try bakeHigh(3, workers);
        try testing.expectEqualSlices([4]u8, &one, &split);
    }
}

test "highLength includes every mip and every child is the rounded mean" {
    try testing.expectEqual(@as(usize, 1), clouds.highLength(1));
    try testing.expectEqual(@as(usize, 5461), clouds.highLength(64));
    try testing.expectEqual(@as(usize, 1398101), clouds.highLength(1024));
    const map = try bakeHigh(3, 1);
    var n: usize = side;
    var parent: usize = 0;
    var next: usize = n * n;
    while (n > 1) {
        const half = n / 2;
        for (0..half) |y| for (0..half) |x| for (0..4) |c| {
            const at_parent = parent + y * 2 * n + x * 2;
            const sum = @as(u16, map[at_parent][c]) + map[at_parent + 1][c] + map[at_parent + n][c] + map[at_parent + n + 1][c];
            try testing.expectEqual(@as(u8, @intCast((sum + 2) / 4)), map[next + y * half + x][c]);
        };
        parent = next;
        next += half * half;
        n = half;
    }
    try testing.expectEqual(map.len, next);
}

test "high density tiles, has gaps and strands, and macro cover is a share" {
    const high = try bakeHigh(3, 1);
    const map: [side * side][4]u8 = high[0 .. side * side].*;
    for (0..2) |c| {
        var inside: [2]f64 = .{ 0, 0 };
        var seam: [2]f64 = .{ 0, 0 };
        for (0..side) |i| {
            seam[0] += step(map, i, side - 1, i, 0, c);
            seam[1] += step(map, side - 1, i, 0, i, c);
            for (0..side - 1) |j| {
                inside[0] += step(map, i, j, i, j + 1, c);
                inside[1] += step(map, j, i, j + 1, i, c);
            }
        }
        for (inside, seam) |a, b| try testing.expect(b / side <= 2 * a / (side * (side - 1)));
    }
    var gaps: usize = 0;
    var strands: usize = 0;
    for (map) |texel| {
        if (texel[0] <= 5) gaps += 1;
        if (texel[0] > 153) strands += 1;
    }
    try testing.expect(gaps > side * side * 3 / 10);
    try testing.expect(strands > 0);
    for ([_]f32{ 0.2, 0.5, 0.8 }) |cover| {
        var above: usize = 0;
        for (map) |texel| {
            if (@as(f32, @floatFromInt(texel[2])) / 255 > 1 - cover) above += 1;
        }
        try testing.expectApproxEqAbs(cover, @as(f32, @floatFromInt(above)) / (side * side), 0.02);
    }
}

test "highDepth wraps bilinearly in the wind frame and clamps endpoint shares" {
    const fibres = [_][4]u8{.{ 255, 0, 0, 0 }};
    const texels = [_][4]u8{ .{ 0, 255, 255, 0 }, .{ 255, 0, 255, 255 }, .{ 0, 255, 255, 0 }, .{ 255, 0, 255, 255 } };
    var p: clouds.HighPlacement = .{ .period = 8, .macro_period = 16, .cover = 1, .sheet = 0, .cirrus_depth = 0.4, .sheet_depth = 0.2, .wind = .{ 0, 1 }, .offset = .{ 1, 2 } };
    // Along = z - 1, across = -x - 2. These land at fine texel centres.
    try testing.expectApproxEqAbs(@as(f32, 0), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.4), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 7), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.4), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, -1), 1e-6);
    p.sheet = 1;
    try testing.expectApproxEqAbs(@as(f32, 0.2), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 3), 1e-6);
    p.cover = 0;
    try testing.expectEqual(@as(f32, 0), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 3));
    p.cover = 1;
    p.period = 0;
    try testing.expectEqual(@as(f32, 0), clouds.highDepth(&texels, 2, &fibres, 1, p, -4, 3));

    // At world (0,0), macro UV is (0.37,0.71): interpolation weights
    // are x=0.24, y=0.92 on this 2x2 map, independently of the fine tile.
    const macro_texels = [_][4]u8{ .{ 255, 255, 0, 0 }, .{ 255, 255, 255, 0 }, .{ 255, 255, 0, 255 }, .{ 255, 255, 255, 255 } };
    p = .{ .period = 8, .macro_period = 16, .cover = 0.76, .sheet = 0.08, .cirrus_depth = 0.4, .sheet_depth = 0.2 };
    // Both masks are at their midpoints, so depth is half of the mean.
    try testing.expectApproxEqAbs(@as(f32, 0.15), clouds.highDepth(&macro_texels, 2, &fibres, 1, p, 0, 0), 1e-6);

    // Constant channels isolate both smooth masks from the fine lookup.
    const masks = [_][4]u8{.{ 255, 255, 128, 128 }};
    p = .{ .period = 8, .cover = 0.5, .sheet = 0.5, .cirrus_depth = 0.4, .sheet_depth = 0.2 };
    const b: f32 = 128.0 / 255.0;
    const mask_t = (b - 0.42) / 0.16;
    const share_t = (b - 0.4) / 0.2;
    const mask = mask_t * mask_t * (3 - 2 * mask_t);
    const share = share_t * share_t * (3 - 2 * share_t);
    try testing.expectApproxEqAbs(mask * (0.4 * (1 - share) + 0.2 * share), clouds.highDepth(&masks, 1, &fibres, 1, p, 3, 7), 1e-6);
}

test "highSunShare leaves clear sun whole and caps low-sun loss" {
    try testing.expectEqual(@as(f32, 1), clouds.highSunShare(0, 1, 0.15));
    try testing.expectApproxEqAbs(@as(f32, 0.85), clouds.highSunShare(1, 1, 0.15), 1e-6);
    try testing.expectApproxEqAbs(@exp(@as(f32, -0.1)), clouds.highSunShare(0.1, 1, 0.15), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.85), clouds.highSunShare(0.1, 0.01, 0.15), 1e-6);
    try testing.expectEqual(clouds.highSunShare(0.001, 0.05, 0.15), clouds.highSunShare(0.001, 0.001, 0.15));
}

test "high maps refuse invalid sides and truncated mip chains" {
    var out: [high_length][4]u8 = undefined;
    for ([_]u32{ 32, 65, 8192 }) |n| try testing.expectError(error.InvalidMapSide, clouds.bakeHigh(testing.allocator, testing.io, &out, n, 3, 1));
    try testing.expectError(error.InvalidMapLength, clouds.bakeHigh(testing.allocator, testing.io, out[0 .. out.len - 1], side, 3, 1));
}

test "high bend is deterministic, bounded and periodic" {
    const period: f32 = 40000;
    const bend: f32 = 0.7;
    const waves = clouds.highBend(19, period, bend);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&waves), std.mem.asBytes(&clouds.highBend(19, period, bend)));
    var sum: f32 = 0;
    const ranges = [3][2]f32{ .{ 2, 5 }, .{ 6, 14 }, .{ 16, 40 } };
    for (waves, 0..) |wave, i| {
        try testing.expectEqual(@floor(wave.kx), wave.kx);
        try testing.expectEqual(@floor(wave.ky), wave.ky);
        try testing.expect(wave.kx != 0);
        const norm = @sqrt(wave.kx * wave.kx + wave.ky * wave.ky);
        try testing.expect(norm >= ranges[i / 6][0] and norm <= ranges[i / 6][1]);
        sum += @abs(wave.amplitude) * 2 * std.math.pi * norm / period;
    }
    try testing.expectApproxEqRel(bend, sum, 1e-4);
    const h: f32 = 16;
    for (0..64) |y| for (0..64) |x| {
        const u = @as(f32, @floatFromInt(x)) * period / 64;
        const v = @as(f32, @floatFromInt(y)) * period / 64;
        const dv = (clouds.highPsi(waves, period, u, v + h) - clouds.highPsi(waves, period, u, v - h)) / (2 * h);
        const du = (clouds.highPsi(waves, period, u + h, v) - clouds.highPsi(waves, period, u - h, v)) / (2 * h);
        try testing.expect(dv >= 1 - bend - 1e-3);
        try testing.expect(@abs(du) <= bend + 1e-3);
        // A tile away the f32 phase of eighteen waves rounds by up to
        // 0.043 m at seed 19, far under the fibres' 30 m.
        const displacement = clouds.highPsi(waves, period, u, v) - v;
        try testing.expectApproxEqAbs(displacement, clouds.highPsi(waves, period, u + period, v) - v, 0.1);
        try testing.expectApproxEqAbs(displacement, clouds.highPsi(waves, period, u, v + period) - (v + period), 0.1);
    };
    const straight = clouds.highBend(19, period, 0);
    try testing.expectEqual(@as(f32, 37), clouds.highPsi(straight, period, 9, 37));
}

fn checkAnisotropy(map: []const [4]u8, n: usize, ratio: f64) !void {
    var along: f64 = 0;
    var across: f64 = 0;
    for (0..n) |y| for (0..n) |x| {
        along += @abs(@as(f64, @floatFromInt(map[y * n + x][0])) - @as(f64, @floatFromInt(map[y * n + (x + 1) % n][0])));
        across += @abs(@as(f64, @floatFromInt(map[y * n + x][0])) - @as(f64, @floatFromInt(map[((y + 1) % n) * n + x][0])));
    };
    try testing.expect(across >= ratio * along);
}

test "high strands have fine cores, gaps and greater variation across wind" {
    const n = 256;
    const map = try testing.allocator.alloc([4]u8, clouds.highLength(n));
    defer testing.allocator.free(map);
    try clouds.bakeHigh(testing.allocator, testing.io, map, n, 19, 3);
    // Barbs leave their spines at 15 to 70 degrees, so the strands vary
    // across the wind less than the fibres do: 1.93 at seed 19.
    try checkAnisotropy(map, n, 1.5);
    var gaps: usize = 0;
    var cores: usize = 0;
    for (map[0 .. n * n]) |texel| {
        if (@as(f32, @floatFromInt(texel[0])) < 0.02 * 255) gaps += 1;
        if (texel[0] > 0.6 * 255) cores += 1;
    }
    try testing.expect(gaps >= n * n * 3 / 10);
    try testing.expect(cores > 0);
}

test "high strand ends are continuous along wind at full resolution" {
    const n = 1024;
    const map = try testing.allocator.alloc([4]u8, clouds.highLength(n));
    defer testing.allocator.free(map);
    try clouds.bakeHigh(testing.allocator, testing.io, map, n, 19, 16);
    // In t, the ramp slope is <= 15, tail <= 1.5, and brightness
    // <= 0.4 * 6 * 1.5 = 3.6. Widening adds <= (1.4/0.6)*2/e.
    // Divide their sum D by L: L >= 0.3 main, 0.125 web.
    // At most three main and two web columns reach a point. Across rows,
    // the central profile <= 1 and neighbours are >= 0.25 cells away.
    // Their total profiles <= 1+2*max(exp(-(0.25/w)^2),
    // 0.2*exp(-(0.25/(3*w))^2)), with w <= 0.3 main, 0.24 web.
    // Union's derivative <= the sum of contribution derivatives. Thus the
    // step <= D*(3*Pmain/0.3 + 2*0.45*Pweb/0.125)/1024 + 1/255,
    // or 0.68681384 including byte rounding.
    const d = 15 + 1.5 + 3.6 + (1.4 / 0.6) * 2 / @exp(@as(f64, 1));
    const pm = 1 + 2 * @max(@exp(-std.math.pow(f64, 0.25 / 0.3, 2)), 0.2 * @exp(-std.math.pow(f64, 0.25 / 0.9, 2)));
    const pw = 1 + 2 * @max(@exp(-std.math.pow(f64, 0.25 / 0.24, 2)), 0.2 * @exp(-std.math.pow(f64, 0.25 / 0.72, 2)));
    const bound = d * (3 * pm / 0.3 + 2 * 0.45 * pw / 0.125) / n + 1.0 / 255.0;
    var largest: u8 = 0;
    for (0..n) |y| for (0..n) |x| {
        const a = map[y * n + x][0];
        const b = map[y * n + (x + 1) % n][0];
        largest = @max(largest, @max(a, b) - @min(a, b));
    };
    try testing.expect(@as(f64, @floatFromInt(largest)) / 255 < bound);
}

test "high fibres are split independent, equalised, anisotropic and mip filtered" {
    const n = 256;
    const map = try testing.allocator.alloc([4]u8, clouds.fibreLength(n));
    defer testing.allocator.free(map);
    const split = try testing.allocator.alloc([4]u8, map.len);
    defer testing.allocator.free(split);
    try clouds.bakeHighFibres(testing.allocator, testing.io, map, n, 19, 1);
    for ([_]usize{ 3, 16 }) |workers| {
        try clouds.bakeHighFibres(testing.allocator, testing.io, split, n, 19, workers);
        try testing.expectEqualSlices([4]u8, map, split);
    }
    try checkAnisotropy(map, n, 3);
    for (map) |texel| try testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, texel[1..]);
    for ([_]f32{ 0.2, 0.5, 0.8 }) |c| {
        var above: usize = 0;
        for (map[0 .. n * n]) |texel| {
            if (@as(f32, @floatFromInt(texel[0])) / 255 > 1 - c) above += 1;
        }
        try testing.expectApproxEqAbs(c, @as(f32, @floatFromInt(above)) / (n * n), 0.02);
    }
    var width: usize = n;
    var parent: usize = 0;
    var next: usize = n * n;
    while (width > 1) {
        const half = width / 2;
        for (0..half) |y| for (0..half) |x| {
            const a = parent + 2 * y * width + 2 * x;
            const sum = @as(u16, map[a][0]) + map[a + 1][0] + map[a + width][0] + map[a + width + 1][0];
            try testing.expectEqual(@as(u8, @intCast((sum + 2) / 4)), map[next + y * half + x][0]);
        };
        parent = next;
        next += half * half;
        width = half;
    }
    var seam: [2]f64 = .{ 0, 0 };
    var inside: [2]f64 = .{ 0, 0 };
    for (0..n) |y| for (0..n) |x| {
        const a: f64 = @floatFromInt(map[y * n + x][0]);
        const dx = @abs(a - @as(f64, @floatFromInt(map[y * n + (x + 1) % n][0])));
        const dy = @abs(a - @as(f64, @floatFromInt(map[((y + 1) % n) * n + x][0])));
        if (x + 1 == n) seam[0] += dx else inside[0] += dx;
        if (y + 1 == n) seam[1] += dy else inside[1] += dy;
    };
    for (seam, inside) |a, b| try testing.expect(a / n <= 2 * b / (n * (n - 1)));
    try testing.expectError(error.InvalidMapSide, clouds.bakeHighFibres(testing.allocator, testing.io, map, 65, 19, 1));
    try testing.expectError(error.InvalidMapLength, clouds.bakeHighFibres(testing.allocator, testing.io, map[0 .. map.len - 1], n, 19, 1));
}

test "highDepth bends a stripe and modulates only cirrus with fibres" {
    const map = [_][4]u8{ .{ 0, 255, 255, 0 }, .{ 0, 255, 255, 0 }, .{ 255, 255, 255, 0 }, .{ 255, 255, 255, 0 } };
    const fibres = [_][4]u8{.{ 128, 0, 0, 0 }};
    var p: clouds.HighPlacement = .{ .period = 100, .cover = 1, .sheet = 0, .cirrus_depth = 0.3 };
    p.bend[0] = .{ .kx = 2, .ky = 0, .amplitude = 10, .phase = 0 };
    // At s=12.5 the wave displaces psi by +10. The peak at psi=75
    // therefore moves from world across=75 to across=65.
    const expected: f32 = 0.3 * (0.65 + 0.35 * 128.0 / 255.0);
    const peak = clouds.highDepth(&map, 2, &fibres, 1, p, 12.5, 65);
    try testing.expectApproxEqAbs(expected, peak, 1e-6);
    try testing.expect(peak > clouds.highDepth(&map, 2, &fibres, 1, p, 12.5, 75));
    p.bend = @splat(.{ .kx = 0, .ky = 0, .amplitude = 0, .phase = 0 });
    try testing.expectApproxEqAbs(expected, clouds.highDepth(&map, 2, &fibres, 1, p, 12.5, 75), 1e-6);
    p.sheet = 1;
    try testing.expectApproxEqAbs(p.sheet_depth, clouds.highDepth(&map, 2, &fibres, 1, p, 12.5, 75), 1e-6);
}
