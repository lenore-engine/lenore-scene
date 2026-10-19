const std = @import("std");

// Where inside its pixel a frame samples, for a temporal pass that gathers
// several frames into one: each frame moves the projection by a sub-pixel
// offset, and over a cycle of frames the offsets cover the pixel evenly.
//
// The Halton sequence in bases 2 and 3, the pair a temporal upscaler is usually
// driven by: consecutive points stay apart, and any prefix covers the square
// about as evenly as a prefix can. Index zero is skipped, since every radical
// inverse of zero is zero and that point would sit on the pixel's corner.

// The radical inverse of `index` in `base`: its digits mirrored about the point.
// For base 2, 1 is 0.5, 2 is 0.25, 3 is 0.75.
pub fn radicalInverse(index: u64, base: u64) f32 {
    std.debug.assert(base >= 2);
    var remaining = index;
    var scale: f64 = 1;
    var result: f64 = 0;
    while (remaining > 0) {
        scale /= @floatFromInt(base);
        result += @as(f64, @floatFromInt(remaining % base)) * scale;
        remaining /= base;
    }
    return @floatCast(result);
}

// The offset for `frame` in a cycle of `phases` frames, in pixels, each axis
// in [-0.5, 0.5): the pixel's centre is the origin.
pub fn subpixelOffset(frame: u64, phases: u32) [2]f32 {
    std.debug.assert(phases > 0);
    const index = frame % phases + 1;
    return .{ radicalInverse(index, 2) - 0.5, radicalInverse(index, 3) - 0.5 };
}

// The same offset in device coordinates for a target of `width` by `height`
// pixels: a pixel spans two over its count, since device coordinates run from
// -1 to 1.
pub fn deviceOffset(frame: u64, phases: u32, width: u32, height: u32) [2]f32 {
    const pixels = subpixelOffset(frame, phases);
    return .{
        pixels[0] * 2 / @as(f32, @floatFromInt(width)),
        pixels[1] * 2 / @as(f32, @floatFromInt(height)),
    };
}
