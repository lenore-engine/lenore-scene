const std = @import("std");
const scene = @import("lenore-scene");

const testing = std.testing;
const Photocell = scene.Photocell;
const Iris = scene.Iris;

// Eight across, which is the grid the engine's meter pass produces. Nothing here
// depends on that number beyond the cells the cases name.
//
// Typed `u32` because that is what the pass declares its cell side as, and the
// coercion to the comptime `usize` the reading takes is then exercised by every
// case below rather than left to the one caller that does it.
const side: u32 = 8;

const cell: Photocell = .{};

fn flat(value: f32) [side * side]f32 {
    return @splat(value);
}

test "a uniform frame meters at its own luminance whatever the weighting" {
    const cells = flat(0.4);
    try testing.expectApproxEqRel(@as(f32, 0.4), cell.read(side, &cells), 1.0e-5);
}

test "a fixture in the picture cannot decide the exposure" {
    // An interior near 0.5 with one ceiling panel in the middle of the frame
    // radiating 28 times that. Four of 64 cells, which at the centre of the
    // pattern is about an eighth of the metering weight, so the default fifth
    // takes all of it and the reading is the room's own.
    var cells = flat(0.5);
    for ([_]usize{ 3 * side + 3, 3 * side + 4, 4 * side + 3, 4 * side + 4 }) |index|
        cells[index] = 14.0;

    const measured = cell.read(side, &cells);
    try testing.expectApproxEqRel(@as(f32, 0.5), measured, 1.0e-3);

    // Which is the property the trim exists for, stated where it bites: the
    // exposure that answers this reading leaves the panel above a bloom
    // threshold of one in the exposed scale. A plain weighted mean of the same
    // frame reads about four times as high, and the exposure that answers that
    // puts the only source in the picture below its own glow.
    try testing.expect(14.0 * (0.10 / measured) > 1.0);

    // And the mean it is being held against really is that much higher, so the
    // case is the one described and not a frame the trim never mattered to.
    const untrimmed: Photocell = .{ .trim = 0.0 };
    try testing.expect(untrimmed.read(side, &cells) > 3.0 * measured);
}

test "a fixture that is most of the picture does decide it" {
    // Pointing the camera up into a panel. Past the trim there is nothing left
    // but panel, so the iris stops down, which is what a camera does.
    var cells = flat(14.0);
    for (0..2) |row| {
        for (0..side) |column| cells[row * side + column] = 0.5;
    }
    try testing.expect(cell.read(side, &cells) > 5.0);
}

test "the trim is a ratio, so a frame twice as bright meters twice as high" {
    // The cut is taken by rank rather than against a level, so it holds under
    // scaling, which is what keeps the meter able to follow a room getting
    // brighter across all of it.
    var dim = flat(0.3);
    dim[3 * side + 3] = 9.0;
    var bright: [side * side]f32 = undefined;
    for (&bright, dim) |*out, value| out.* = value * 2.0;

    try testing.expectApproxEqRel(2.0 * cell.read(side, &dim), cell.read(side, &bright), 1.0e-4);
}

test "the middle of the frame weighs more than a corner" {
    // Read on the dim side, because the bright side is what the trim removes: a
    // patch of shadow in the middle of the picture pulls the reading down
    // further than the same patch in a corner does.
    var middle = flat(0.5);
    for (3..5) |row| {
        for (3..5) |column| middle[row * side + column] = 0.05;
    }
    var corner = flat(0.5);
    for (0..2) |row| {
        for (0..2) |column| corner[row * side + column] = 0.05;
    }

    const uniform = cell.read(side, &flat(0.5));
    try testing.expect(cell.read(side, &middle) < cell.read(side, &corner));
    try testing.expect(cell.read(side, &corner) < uniform);
}

test "a wide spread is a plain average and a narrow one is a spot meter" {
    // The two ends of the one number, which is the claim that the pattern covers
    // the three meters a camera is described by rather than only the middle one.
    var cells = flat(1.0);
    for (0..2) |row| {
        for (0..2) |column| cells[row * side + column] = 0.0;
    }
    // No trim, so the reading is the weighting alone.
    const average: Photocell = .{ .spread = 1.0e4, .trim = 0.0 };
    const spot: Photocell = .{ .spread = 0.05, .trim = 0.0 };

    // Four dark cells of 64, weighed equally.
    try testing.expectApproxEqRel(@as(f32, 60.0 / 64.0), average.read(side, &cells), 1.0e-3);
    // A spot in the middle cannot see a corner at all.
    try testing.expectApproxEqRel(@as(f32, 1.0), spot.read(side, &cells), 1.0e-4);
}

fn settled(iris: *Iris, measured: f32, seconds: f32) void {
    const step: f32 = 1.0 / 120.0;
    var elapsed: f32 = 0.0;
    while (elapsed < seconds) : (elapsed += step) iris.advance(measured, step);
}

// The rates and the damping are the defaults; the four that are not are a look,
// and these are one interior's.
const settings: Iris.Settings = .{
    .key = 0.08,
    .least = 0.048,
    .most = 0.772,
    .start = 0.193,
};

test "the first measurement is taken rather than approached" {
    var iris: Iris = .init(settings);
    try testing.expectApproxEqRel(@as(f32, 0.193), iris.exposure(), 1.0e-5);

    iris.advance(0.4, 1.0 / 120.0);
    // Key over measured, in one step, with no sweep from where it started.
    try testing.expectApproxEqRel(@as(f32, 0.08 / 0.4), iris.exposure(), 1.0e-5);
}

test "a still scene settles on the key and stays there" {
    var iris: Iris = .init(settings);
    iris.advance(0.4, 1.0 / 120.0);
    settled(&iris, 0.4, 2.0);
    try testing.expectApproxEqRel(@as(f32, 0.08 / 0.4), iris.exposure(), 1.0e-3);

    // A second of the same frame does not move it.
    const before = iris.exposure();
    settled(&iris, 0.4, 1.0);
    try testing.expectApproxEqRel(before, iris.exposure(), 1.0e-4);
}

test "it stops down faster than it opens up" {
    // The same step in the logarithm, taken each way, measured over the same
    // interval. This is the asymmetry the camera is recognised by, so it is
    // pinned rather than left to the two rate constants being different.
    const dim: f32 = 0.2;
    const bright: f32 = 0.8;

    var closing: Iris = .init(settings);
    closing.advance(dim, 1.0 / 120.0);
    settled(&closing, bright, 0.15);
    const closed = @abs(@log(closing.exposure()) - @log(0.08 / dim));

    var opening: Iris = .init(settings);
    opening.advance(bright, 1.0 / 120.0);
    settled(&opening, dim, 0.15);
    const opened = @abs(@log(opening.exposure()) - @log(0.08 / bright));

    // After the same time, the iris that had to stop down has covered more of
    // its step than the one that had to open.
    try testing.expect(closed > opened);
}

test "the response overshoots its mark and comes back" {
    var iris: Iris = .init(settings);
    iris.advance(0.8, 1.0 / 120.0);

    const target = @log(@as(f32, 0.08 / 0.2));
    const step: f32 = 1.0 / 120.0;
    var past = false;
    var elapsed: f32 = 0.0;
    while (elapsed < 3.0) : (elapsed += step) {
        iris.advance(0.2, step);
        if (@log(iris.exposure()) > target + 1.0e-4) past = true;
    }
    try testing.expect(past);
    // And it is back on its mark by the end rather than ringing.
    try testing.expectApproxEqRel(@as(f32, 0.08 / 0.2), iris.exposure(), 1.0e-3);
}

test "the stops hold, and a frame with no light in it is one of them" {
    var iris: Iris = .init(settings);
    // Darker than the iris can open for.
    iris.advance(1.0e-6, 1.0 / 120.0);
    settled(&iris, 1.0e-6, 2.0);
    try testing.expectApproxEqRel(settings.most, iris.exposure(), 1.0e-5);

    // A frame that measured exactly nothing is the same case and not an infinity.
    var dark: Iris = .init(settings);
    dark.advance(0.0, 1.0 / 120.0);
    try testing.expectApproxEqRel(settings.most, dark.exposure(), 1.0e-5);
    try testing.expect(std.math.isFinite(dark.exposure()));

    // And the other end, against a frame far brighter than the key.
    var bright: Iris = .init(settings);
    bright.advance(100.0, 1.0 / 120.0);
    settled(&bright, 100.0, 2.0);
    try testing.expectApproxEqRel(settings.least, bright.exposure(), 1.0e-5);
}

test "an iris driven into a stop leaves it as soon as the frame changes" {
    // The velocity is dropped at the stop, so what follows is the response to
    // the new frame and not a second of unwinding.
    var iris: Iris = .init(settings);
    iris.advance(1.0e-6, 1.0 / 120.0);
    settled(&iris, 1.0e-6, 2.0);

    settled(&iris, 0.4, 0.25);
    try testing.expect(iris.exposure() < settings.most);
}

test "the two halves compose into what a camcorder does" {
    // The one case that needs both: a pan from a lit hall into a dark corridor
    // stops the iris nowhere near its bound, and back again returns it. Neither
    // half states this, because the reading and the servo are separately correct
    // and what is checked here is that they are on the same scale.
    var iris: Iris = .init(settings);
    const hall = cell.read(side, &flat(0.5));
    const corridor = cell.read(side, &flat(0.05));

    iris.advance(hall, 1.0 / 120.0);
    const lit = iris.exposure();
    try testing.expect(lit > settings.least and lit < settings.most);

    settled(&iris, corridor, 3.0);
    const dark = iris.exposure();
    try testing.expect(dark > lit);
    try testing.expect(dark <= settings.most);

    settled(&iris, hall, 3.0);
    try testing.expectApproxEqRel(lit, iris.exposure(), 1.0e-3);
}

test "a clicked iris reports the detents and nothing between them" {
    // The property the quantized display path needs: the exposure takes a few
    // values rather than a continuum, so a band boundary in the picture is still
    // while the servo is between clicks.
    var clicked: Iris = .init(.{
        .key = settings.key,
        .least = settings.least,
        .most = settings.most,
        .start = settings.start,
        // Thirds of a stop, which is the finest detent an aperture ring has.
        .click = 1.0 / 3.0,
    });

    // Every value it can report is a third of a stop above the one below it, so
    // the logarithm base two of each is a multiple of a third.
    var seen: usize = 0;
    var previous: f32 = 0.0;
    clicked.advance(0.8, 1.0 / 120.0);
    var elapsed: f32 = 0.0;
    while (elapsed < 3.0) : (elapsed += 1.0 / 120.0) {
        clicked.advance(0.2, 1.0 / 120.0);
        const reported = clicked.exposure();
        const stops = @log2(reported) * 3.0;
        try testing.expectApproxEqAbs(@round(stops), stops, 1.0e-4);
        if (reported != previous) {
            seen += 1;
            previous = reported;
        }
    }

    // And it really moved: a click that reported one value throughout would pass
    // the test above for the wrong reason.
    try testing.expect(seen > 1);

    // Against the same run without detents, which has to pass through many more
    // distinct values over the same sweep.
    var smooth: Iris = .init(settings);
    smooth.advance(0.8, 1.0 / 120.0);
    var distinct: usize = 0;
    var last: f32 = 0.0;
    elapsed = 0.0;
    while (elapsed < 3.0) : (elapsed += 1.0 / 120.0) {
        smooth.advance(0.2, 1.0 / 120.0);
        if (smooth.exposure() != last) {
            distinct += 1;
            last = smooth.exposure();
        }
    }
    try testing.expect(distinct > seen * 10);
}

test "a click does not move where the iris settles, only what it reports" {
    // The servo is untouched: the detent is on the reading and not on the state,
    // so a clicked iris and a smooth one settle on the same place and only the
    // reported value is rounded to the nearest one.
    var clicked: Iris = .init(.{
        .key = settings.key,
        .least = settings.least,
        .most = settings.most,
        .start = settings.start,
        .click = 1.0 / 3.0,
    });
    var smooth: Iris = .init(settings);

    clicked.advance(0.4, 1.0 / 120.0);
    smooth.advance(0.4, 1.0 / 120.0);
    settled(&clicked, 0.4, 2.0);
    settled(&smooth, 0.4, 2.0);

    try testing.expectEqual(smooth.log_exposure, clicked.log_exposure);
    // And the reading is within half a click of the state it stands for.
    const stops = @abs(@log2(clicked.exposure()) - @log2(smooth.exposure()));
    try testing.expect(stops <= 1.0 / 6.0 + 1.0e-5);
}
