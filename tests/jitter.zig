const std = @import("std");
const scene = @import("lenore-scene");

const testing = std.testing;
const jitter = scene.jitter;

test "the radical inverse mirrors the digits about the point" {
    try testing.expectEqual(@as(f32, 0.5), jitter.radicalInverse(1, 2));
    try testing.expectEqual(@as(f32, 0.25), jitter.radicalInverse(2, 2));
    try testing.expectEqual(@as(f32, 0.75), jitter.radicalInverse(3, 2));
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), jitter.radicalInverse(1, 3), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), jitter.radicalInverse(2, 3), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 9.0), jitter.radicalInverse(3, 3), 1e-7);
}

test "offsets stay inside the pixel and repeat with the cycle" {
    for (0..64) |frame| {
        const offset = jitter.subpixelOffset(frame, 8);
        for (offset) |axis| try testing.expect(axis >= -0.5 and axis < 0.5);
        try testing.expectEqual(offset, jitter.subpixelOffset(frame + 8, 8));
    }
}

// Over a cycle the offsets average near the pixel's centre: a sequence whose
// mean sat off it would shift the accumulated picture by that much.
test "a cycle of eight is centred on the pixel" {
    var sum: [2]f32 = .{ 0, 0 };
    for (0..8) |frame| {
        const offset = jitter.subpixelOffset(frame, 8);
        sum[0] += offset[0];
        sum[1] += offset[1];
    }
    try testing.expect(@abs(sum[0] / 8) < 0.07);
    try testing.expect(@abs(sum[1] / 8) < 0.07);
}

test "the device offset is two over the extent per pixel" {
    const pixels = jitter.subpixelOffset(3, 8);
    const device = jitter.deviceOffset(3, 8, 1000, 500);
    try testing.expectApproxEqAbs(pixels[0] * 2 / 1000, device[0], 1e-9);
    try testing.expectApproxEqAbs(pixels[1] * 2 / 500, device[1], 1e-9);
}
