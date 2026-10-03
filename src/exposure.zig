// The camera's meter and its iris: what a frame's light is read as, and what an
// exposure does about it.
//
// A pass elsewhere measures the frame and hands back a coarse grid of mean
// luminances. Everything here is what a camera does with one. The two halves are
// separate because they answer different questions and an application may want
// only one of them: `Photocell` decides which part of the picture votes, and
// `Iris` decides how fast the exposure follows what it voted for.
//
// No graphics API and no device. This is arithmetic over a grid and an interval,
// which is the whole of what can be wrong with it, so it is tested on the host
// against hand-worked cases.
//
// Radiance, exposure and luminance here are all linear and unbounded above. What
// a display operator does afterwards is not this file's business.

const std = @import("std");

// What the camera meters, out of the grid the pass handed back.
//
// Centre weighted with a trimmed mean, which is what a photocell in a consumer
// camera behaves like. The two numbers are the whole of the pattern, and the
// range between them covers the three meters a camera is usually described by: a
// wide `spread` is a plain average, a narrow one is a spot meter, and the
// default is the centre weighting in between.
//
// The weight falls off as a Gaussian in the distance from the centre of the
// frame, measured in the frame's own coordinates rather than in pixels. So on a
// wide window the pattern is a wide ellipse, which is what a camera's is: the
// metering follows the picture and not the aspect ratio.
pub const Photocell = struct {
    // The falloff of the weight, in frame widths, applied to the distance from
    // the centre.
    //
    // At 0.45 the corners of a square frame carry about a seventh of what the
    // middle does, which leaves them able to move the exposure without being
    // able to decide it.
    spread: f32 = 0.45,

    // The share of the weight, taken from the bright end, that does not vote.
    //
    // **A light fixture in the picture may not decide the exposure**, and a
    // plain weighted mean lets it. A source radiating tens of times what the
    // surfaces it lights hold will lift a weighted mean severalfold while
    // covering a sixteenth of the frame, and the exposure that answers that is
    // one where the room is too dark to read and the source is no longer bright
    // enough to bloom.
    //
    // So the brightest `trim` of the weight is dropped and the rest averaged.
    // The reason it is a trim rather than a clamp against the frame's own level
    // is that a clamp has a fixed point: the clamped cells still carry their
    // share of the mean they are being measured against, and where the bright
    // cells are more than a `1 / headroom` of the weight the iteration does not
    // converge at all. A trim has no such loop, because the cells that are
    // dropped contribute nothing to what remains.
    //
    // It has to exceed the share of the metering weight the brightest fixture in
    // the frame covers. A fifth is what one interior of ceiling panels needed,
    // where a panel in the middle of the picture took about an eighth.
    //
    // What a trim costs is the case where the fixture really is most of the
    // picture. A camera pointed up into a light should stop down, and it still
    // does: past the trim there is nothing left but fixture.
    trim: f32 = 0.2,

    // One reading, from a square grid of cell means held row major.
    //
    // `side` is comptime because the pass that fills the grid fixes it at
    // compile time, and taking it that way is what lets the scratch array be
    // exactly the size of the grid. A runtime side would need a bound and a
    // check against it, and that check is removed in the shipping build.
    pub fn read(self: Photocell, comptime side: usize, cells: *const [side * side]f32) f32 {
        if (side == 0) return 0.0;

        var samples: [side * side]Sample = undefined;
        var weight_total: f64 = 0.0;
        const across: f32 = @floatFromInt(side);
        for (0..side) |row| {
            for (0..side) |column| {
                const index = row * side + column;
                const x = (@as(f32, @floatFromInt(column)) + 0.5) / across - 0.5;
                const y = (@as(f32, @floatFromInt(row)) + 0.5) / across - 0.5;
                const weight = self.centreWeight(x * x + y * y);
                samples[index] = .{ .luminance = cells[index], .weight = weight };
                weight_total += weight;
            }
        }
        if (!(weight_total > 0.0)) return 0.0;

        std.mem.sortUnstable(Sample, &samples, {}, Sample.dimmerFirst);

        // The weight to average over, from the dim end. The boundary sample is
        // counted in part rather than in or out, so the reading moves
        // continuously as a cell brightens through the cut instead of stepping
        // when it crosses.
        const wanted = weight_total * (1.0 - std.math.clamp(@as(f64, self.trim), 0.0, 1.0));
        var taken: f64 = 0.0;
        var total: f64 = 0.0;
        for (samples) |sample| {
            // The walk stops when the budget is spent and not when a sample
            // contributes nothing. The order is by luminance and says nothing
            // about the weights, so a cell the pattern gives no weight at all
            // sits wherever its brightness puts it, and stopping there would
            // throw away every cell behind it. A narrow pattern underflows the
            // weight of a corner to zero, which is how this is reached.
            const remaining = wanted - taken;
            if (!(remaining > 0.0)) break;
            const share = @min(@as(f64, sample.weight), remaining);
            if (!(share > 0.0)) continue;
            total += share * @as(f64, sample.luminance);
            taken += share;
        }
        if (!(taken > 0.0)) return 0.0;
        return @floatCast(total / taken);
    }

    // The falloff, given the squared distance from the centre of the frame.
    fn centreWeight(self: Photocell, squared: f32) f32 {
        return @exp(-squared / (self.spread * self.spread));
    }
};

const Sample = struct {
    luminance: f32,
    weight: f32,

    fn dimmerFirst(_: void, a: Sample, b: Sample) bool {
        return a.luminance < b.luminance;
    }
};

// What a camera's iris does, and the three things about it worth having.
//
// **It stops down faster than it opens up.** Point one at a window and the
// picture snaps dark; turn away and it takes a second or more to come back. That
// asymmetry is what makes the adaptation read as a camera rather than as a slow
// fade, and it is why the two rates below are separate numbers.
//
// **It overshoots and settles.** An iris is a servo and not a filter: it runs
// past its mark and comes back. A first-order lag would reach the same place and
// never do that.
//
// **It has ends.** A room a level can produce is far darker than any iris could
// open for, and a camera pointed into one records a dark room rather than a
// lifted grey one. The bounds are what keep black corridors black.
pub const Iris = struct {
    // The mean luminance the metered part of the frame is driven towards.
    //
    // Exposure is this over what the frame measures, so a frame already at the
    // key is left alone. It is a look and not a physical quantity: it decides
    // where an ordinary lit room sits on the tone curve, and every other level of
    // brightness is relative to it.
    key: f32,

    // What the iris may reach, as multipliers on radiance.
    //
    // The upper bound is the one that does the work: an unlit room meters near
    // zero and the exposure it asks for is unbounded, so without a stop the
    // darkest rooms would be the brightest pictures.
    least: f32,
    most: f32,

    // How fast the iris moves, as the natural frequency of a second-order servo,
    // in radians a second. `closing` is used when the frame is brighter than the
    // key and the iris has to stop down.
    //
    // For a servo damped at `damping` the response settles in about four over the
    // product of the two, so the defaults are roughly a third of a second down
    // and a second and a half back up, which is the asymmetry a consumer camera
    // is recognised by.
    closing: f32 = 19.0,
    opening: f32 = 3.8,

    // Under one, so the response overshoots and comes back. At 0.7 the overshoot
    // is a few per cent of the step, which is visible in a pan between a dark
    // corridor and a lit hall and is not a wobble.
    damping: f32 = 0.7,

    // The size of a click, in stops, or zero for an exposure that moves
    // continuously.
    //
    // A real aperture ring has detents, and this is that. What it is for is
    // downstream: a display path that quantizes the picture, whether to a palette
    // or to a fixed number of bands, moves every band boundary in the frame at
    // once when the exposure drifts. Every surface then steps between bands at
    // its own moment and the whole image reads as breathing. Clicking the
    // exposure instead makes the adaptation happen in a few visible jumps and
    // leaves the boundaries still between them.
    //
    // It is a decision the display path forces and the servo has to carry,
    // because the alternative is quantizing in the shader, where a constant
    // exposure the application chose would be silently rounded to something
    // else. Zero is exact and is the default: an application whose picture is
    // continuous wants no detents at all.
    //
    // The servo itself stays continuous. Only what `exposure` reports is
    // snapped, so a click does not become a state the response has to settle
    // out of.
    click: f32 = 0.0,

    // Where the iris is now, and how fast it is moving, both in the logarithm of
    // the exposure. In the log because a stop is a ratio: a servo working on the
    // exposure itself would move twice as fast through the bright end of its
    // range as through the dark one, and the picture would read as two different
    // cameras.
    log_exposure: f32,
    velocity: f32,

    // Whether any frame has been measured yet. The first measurement is taken
    // rather than approached: a camera is already exposed when it is switched on,
    // and sweeping to the first reading would put a fade over the start of a run.
    started: bool = false,

    pub const Settings = struct {
        key: f32,
        least: f32,
        most: f32,
        // What the iris shows before anything has been measured, which is the
        // first frame or two of a run and every frame of a build that does not
        // meter.
        start: f32,
        closing: f32 = 19.0,
        opening: f32 = 3.8,
        damping: f32 = 0.7,
        click: f32 = 0.0,
    };

    pub fn init(settings: Settings) Iris {
        return .{
            .key = settings.key,
            .least = settings.least,
            .most = settings.most,
            .closing = settings.closing,
            .opening = settings.opening,
            .damping = settings.damping,
            .click = settings.click,
            .log_exposure = @log(std.math.clamp(settings.start, settings.least, settings.most)),
            .velocity = 0.0,
        };
    }

    // What the picture is multiplied by.
    //
    // Snapped to the click, when there is one. In stops, so the detents are
    // evenly spaced in the ratio a stop is and not in the exposure itself: a
    // fixed step in the multiplier would be a whole stop at the dark end of the
    // range and a hundredth of one at the bright end.
    pub fn exposure(self: Iris) f32 {
        const continuous = @exp(self.log_exposure);
        if (!(self.click > 0.0)) return continuous;

        // The logarithm exists because the servo's own state is one, so the
        // exposure is positive by construction and this needs no guard.
        const stops = self.log_exposure / @log(@as(f32, 2.0));
        return std.math.pow(f32, 2.0, @round(stops / self.click) * self.click);
    }

    // One frame of the servo, given what the frame metered and how long it took.
    //
    // `measured` is a reading from `Photocell.read`, or any other weighting of
    // the same grid. A frame that measured nothing at all is not a reason to open
    // the iris to its stop: it is what a lens cap looks like, and what an unlit
    // room looks like, and `most` answers both.
    pub fn advance(self: *Iris, measured: f32, seconds: f32) void {
        const wanted = @log(self.wantedExposure(measured));
        if (!self.started) {
            self.started = true;
            self.log_exposure = wanted;
            self.velocity = 0.0;
            return;
        }
        if (!(seconds > 0.0)) return;

        // Stopping down is the fast direction: the frame is brighter than the
        // iris is set for, so the exposure it wants is the smaller one.
        const rate = if (wanted < self.log_exposure) self.closing else self.opening;

        // Semi-implicit Euler, which is what keeps a spring stable at a frame's
        // step where the explicit form gains energy. The step is not clamped
        // here: the caller's own delta already is, and a servo that silently
        // ignored a long frame would drift from the wall clock.
        const offset = wanted - self.log_exposure;
        self.velocity += (rate * rate * offset - 2.0 * self.damping * rate * self.velocity) * seconds;
        self.log_exposure += self.velocity * seconds;

        // The stops are on the position and not on what it is approaching, so an
        // iris driven into one stays there rather than winding up: the velocity
        // is dropped as well, or it would spend the next second unwinding a speed
        // it never travelled at.
        const lowest = @log(self.least);
        const highest = @log(self.most);
        if (self.log_exposure < lowest) {
            self.log_exposure = lowest;
            self.velocity = @max(self.velocity, 0.0);
        } else if (self.log_exposure > highest) {
            self.log_exposure = highest;
            self.velocity = @min(self.velocity, 0.0);
        }
    }

    fn wantedExposure(self: Iris, measured: f32) f32 {
        // A frame that measured nothing would divide by zero. The bound above is
        // the answer either way, so it is taken directly rather than through an
        // infinity.
        if (!(measured > 0.0)) return self.most;
        return std.math.clamp(self.key / measured, self.least, self.most);
    }
};
