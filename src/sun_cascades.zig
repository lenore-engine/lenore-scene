// The sun's shadow as up to three maps around the camera, and which of them is
// baked in a frame.
//
// One map fitted to the whole scene gives every caster the same texel, which
// over a kilometre is far too coarse near the eye. A near square follows the
// camera at the finest texel, a middle one around it at a coarser one, and the
// last covers the whole scene sphere as the single map always did. A cascade
// as wide as the scene is no cascade at all, so a small scene keeps one map,
// fitted as before.
//
// Each cascade keeps two layers so the lookup can fade from the map before a
// bake to the one it made, exactly as the single map did. A cascade is baked
// when the sun has turned past `resample_degrees` since its bake, when the
// camera has walked a quarter of its square from the square's centre, or when
// the casters changed. At most one cascade is baked per frame, so three bakes
// never land in the same frame.
//
// No graphics API. The engine carries the result to the device.

const std = @import("std");
const resources = @import("lenore-resources");
const sun_shadow = @import("sun_shadow.zig");

const Sphere = resources.Sphere;
const Vec3 = resources.Vec3;
const SunShadowFit = sun_shadow.SunShadowFit;
const FitError = sun_shadow.FitError;

pub const max_cascades = 3;

// How far the camera may walk from a square's centre, as a share of the
// square's width, before the square is moved. A quarter leaves the camera at
// least a quarter of the width from every edge, and the lookup blends into the
// next cascade over the last tenth.
pub const recentre_share: f32 = 0.25;

// Which image a cascade is drawn into. The near cascade has the large map to
// itself; the coarser ones share the small one, two layers each.
pub const Image = enum { fine, coarse };

pub const Settings = struct {
    // World widths of the near and middle squares. A width at or above the
    // scene sphere's diameter turns that cascade and every coarser one into
    // the single map around the scene.
    //
    // Provisional: 24 m gives Reclaim's base the texel its single map had at
    // 4096 (about 6 mm) wherever the player stands, and 128 m keeps a shed's
    // shadow at about 6 cm across the yard at 2048.
    widths: [2]f32 = .{ 24, 128 },
    fine_size: u32,
    coarse_size: u32,
};

pub const Cascade = struct {
    // The fit the layer at `layer` holds, null before the first bake.
    shown: ?SunShadowFit = null,
    // The fit the other layer holds, faded from after a bake.
    previous: ?SunShadowFit = null,
    // The fit waiting to be baked, if one is.
    target: ?SunShadowFit = null,
    // Set when the casters changed since `shown` was baked, so the next bake
    // fades from nothing. A bake only for the sun's turn or the square's move
    // keeps `shown` to fade from: both maps are corrected to the current sun
    // and differ only in their rasterization.
    invalid: bool = false,
    // Which of the cascade's two layers `shown` is in.
    layer: u32 = 0,
    // Seconds since `shown` was baked, while `previous` is faded from.
    fading: f32 = 0,
    // Frames `target` has waited, so a cascade passed over is served next.
    waiting: u32 = 0,
    image: Image = .fine,
    // The first of the cascade's two layers in its image.
    base_layer: u32 = 0,
    // Zero for the cascade around the scene sphere, which never moves.
    width: f32 = 0,

    // The newer map's share of the lookup.
    pub fn share(self: *const Cascade) f32 {
        if (self.previous == null) return 1;
        return @min(1, self.fading / sun_shadow.fade_seconds);
    }

    // The layer of `shown`, and of `previous`, in the cascade's image.
    pub fn shownLayer(self: *const Cascade) u32 {
        return self.base_layer + self.layer;
    }

    pub fn previousLayer(self: *const Cascade) u32 {
        return self.base_layer + if (self.previous != null) 1 - self.layer else self.layer;
    }

    // The layer the next bake writes: the one not shown.
    pub fn bakeLayer(self: *const Cascade) u32 {
        return self.base_layer + if (self.shown != null) 1 - self.layer else self.layer;
    }

    // What a staleness test compares against: the fit about to be baked, or
    // the one shown.
    fn latest(self: *const Cascade) ?SunShadowFit {
        return self.target orelse self.shown;
    }
};

pub const SunCascades = struct {
    cascades: [max_cascades]Cascade = @splat(.{}),
    count: u32 = 0,
    // The sphere the fits were made for. Another one changes every fit and
    // what each casts, so nothing is faded across it.
    scene: Sphere = .{ .centre = .{ 0, 0, 0 }, .radius = 0 },

    // Brings every cascade up to the sun at `toward` and the camera at `at`.
    // A cascade that needs a new fit is given one as its target; nothing is
    // baked here. `toward` needs no particular length.
    pub fn refit(self: *SunCascades, scene: Sphere, toward: Vec3, at: Vec3, settings: Settings) FitError!void {
        const diameter = 2 * scene.radius;
        var count: u32 = 1;
        for (settings.widths) |width| {
            if (width >= diameter) break;
            count += 1;
        }

        const same_scene = self.count == count and
            @reduce(.And, self.scene.centre == scene.centre) and self.scene.radius == scene.radius;
        if (!same_scene) {
            self.* = .{ .count = count, .scene = scene };
            for (self.cascades[0..count], 0..) |*cascade, index| {
                const last = index + 1 == count;
                cascade.width = if (last) 0 else settings.widths[index];
                cascade.image = if (index == 0) .fine else .coarse;
                cascade.base_layer = if (index == 0) 0 else @intCast((index - 1) * 2);
            }
        }

        for (self.cascades[0..count]) |*cascade| {
            const size = if (cascade.image == .fine) settings.fine_size else settings.coarse_size;
            if (cascade.latest()) |latest| {
                const turned = latest.stale(toward);
                const walked = cascade.width > 0 and latest.lateralDistance(at) > cascade.width * recentre_share;
                if (!turned and !walked) continue;
            }
            cascade.target = if (cascade.width > 0)
                try SunShadowFit.computeCascade(scene, toward, size, cascade.width, at)
            else
                try SunShadowFit.compute(scene, toward, size);
        }
    }

    // Every cascade is baked again with the fit it shows: the casters moved,
    // the bias changed, or shadows came back on after frames in which nothing
    // kept the maps current. Nothing is faded across it.
    pub fn invalidate(self: *SunCascades) void {
        for (self.cascades[0..self.count]) |*cascade| {
            if (cascade.latest()) |latest| cascade.target = latest;
            cascade.invalid = true;
            cascade.previous = null;
        }
    }

    // Moves every fade on by `delta` seconds and lets an older map go once
    // its fade is done.
    pub fn advance(self: *SunCascades, delta: f32) void {
        for (self.cascades[0..self.count]) |*cascade| {
            if (cascade.previous == null) continue;
            cascade.fading += delta;
            if (cascade.fading >= sun_shadow.fade_seconds) cascade.previous = null;
        }
    }

    // The cascade to bake next: the waiting one that has waited longest, the
    // finer on a tie, or the near one when none waits. A frame orders its
    // casters for this one before it knows whether casters moved, which would
    // make every cascade wait.
    pub fn candidate(self: *const SunCascades) u32 {
        var chosen: ?u32 = null;
        for (self.cascades[0..self.count], 0..) |*cascade, index| {
            if (cascade.target == null) continue;
            if (chosen) |best| {
                if (cascade.waiting <= self.cascades[best].waiting) continue;
            }
            chosen = @intCast(index);
        }
        return chosen orelse 0;
    }

    // Bakes cascade `index` if it waits, and makes every other waiting one
    // wait a frame more, so casters that move in every frame still bake all
    // three in turn rather than the near one alone. False when it did not
    // wait.
    pub fn serve(self: *SunCascades, index: u32) bool {
        if (self.cascades[index].target == null) return false;
        for (self.cascades[0..self.count], 0..) |*cascade, other| {
            if (cascade.target != null and other != index) cascade.waiting += 1;
        }
        self.baked(index);
        return true;
    }

    // `candidate` and `serve` together, for a caller that orders nothing
    // between them: the cascade baked, or null when none waited.
    pub fn next(self: *SunCascades) ?u32 {
        const index = self.candidate();
        return if (self.serve(index)) index else null;
    }

    // Records that cascade `index` was baked with its target into
    // `bakeLayer`, which becomes the layer shown.
    pub fn baked(self: *SunCascades, index: u32) void {
        const cascade = &self.cascades[index];
        const target = cascade.target orelse return;
        if (cascade.shown) |shown| {
            cascade.previous = if (cascade.invalid) null else shown;
            cascade.layer = 1 - cascade.layer;
        } else cascade.previous = null;
        cascade.shown = target;
        cascade.target = null;
        cascade.invalid = false;
        cascade.fading = 0;
        cascade.waiting = 0;
    }

    pub fn active(self: *const SunCascades) []const Cascade {
        return self.cascades[0..self.count];
    }
};
