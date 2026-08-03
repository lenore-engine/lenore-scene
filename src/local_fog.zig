// Bounded volumes of participating media: an oriented ellipsoid with a dense
// core and a shell that fades to nothing, and the ray arithmetic over it.
//
// Authoring and use are two types. `LocalFogVolume` is what a level states, and
// it is checked once; `LocalFogVolume.Placed` is what every ray sees, and it
// carries the world-to-unit transform already inverted and the world bounds
// already taken. Nothing on the second one validates or inverts anything, which
// is the point: the density of one volume along one segment is evaluated eight
// times per segment per ray.
//
// Unit space is where the ellipsoid is the unit sphere. Radius there is what the
// density profile is a function of, and the ray parameter stays in world units
// through it because the transform is linear.

const std = @import("std");
const zm = @import("zmath");
const resources = @import("lenore-resources");
const vec = @import("vec.zig");

const Aabb = resources.Aabb;
const Frustum = @import("frustum.zig").Frustum;
const Vec3 = resources.Vec3;

// A quaternion this short cannot be normalized into an orientation. Its square
// is what is compared, so the bound is on the square.
const min_rotation_length_squared: f32 = 1e-8;

// Below this, the ray's direction in unit space has no length worth solving a
// quadratic against: the volume would have to be wider than 10^6 world units
// along that axis for a unit world direction to shrink this far.
const min_unit_direction_squared: f32 = 1e-12;

// Positive abscissae and weights of the four-point Gauss-Legendre rule on
// [-1, 1], which comes in symmetric pairs. Four points rather than two because
// a volume small enough to fall inside one march segment puts both abscissae of
// a two-point rule in the flat core and misses the shell entirely.
//
// These are not tuned numbers and are pinned by their defining property: a rule
// of n points is exact for polynomials of degree below 2n, so this one
// integrates a degree-seven polynomial exactly, and the test asserts that.
const gauss_abscissae = [2]f64{ 0.3399810435848563, 0.8611363115940526 };
const gauss_weights = [2]f64{ 0.6521451548625461, 0.3478548451374538 };

pub const LocalFogError = error{
    // The centre, the radii or the rotation do not place an ellipsoid.
    InvalidVolumePlacement,

    // The density, the shell width or the albedo do not describe a medium.
    InvalidVolumeMedium,

    // The froxel depth distribution has no range to distribute over.
    InvalidFroxelDepth,

    // The collection is full.
    TooManyLocalFogVolumes,

    // The index names no volume in the collection.
    NoSuchLocalFogVolume,
};

pub const LocalFogVolume = struct {
    centre: Vec3 = @splat(0),

    // Semi-axes, before rotation. All three positive.
    radii: Vec3 = @splat(1),

    // Normalized on placement, so any non-zero quaternion is accepted.
    rotation: zm.Quat = .{ 0, 0, 0, 1 },

    // Extinction at the core, per world unit.
    density: f32 = 0.1,

    // Fraction of the radius over which the density falls to zero, in (0, 1].
    // One fades from the very centre; small values keep a dense core and fade
    // only at the boundary. Zero is excluded because the fade divides by it, and
    // a volume with a hard edge is a shape the raymarch cannot sample cleanly
    // anyway.
    edge_falloff: f32 = 0.25,

    // Fraction of extinction that scatters rather than being absorbed, per
    // channel, in [0, 1].
    scattering_albedo: Vec3 = @splat(1),

    // Whether the medium is carried by the wind that animates the global fog.
    wind_noise: bool = false,

    pub fn validate(self: LocalFogVolume) LocalFogError!void {
        if (!vec.finite(self.centre)) return error.InvalidVolumePlacement;
        if (!vec.finite(self.radii) or !@reduce(.And, self.radii > @as(Vec3, @splat(0))))
            return error.InvalidVolumePlacement;
        if (!@reduce(.And, @abs(self.rotation) < @as(zm.Quat, @splat(std.math.inf(f32)))))
            return error.InvalidVolumePlacement;
        if (zm.lengthSq4(self.rotation)[0] < min_rotation_length_squared)
            return error.InvalidVolumePlacement;

        if (!(self.density > 0 and self.density < std.math.inf(f32)))
            return error.InvalidVolumeMedium;
        if (!(self.edge_falloff > 0 and self.edge_falloff <= 1))
            return error.InvalidVolumeMedium;
        if (!vec.finite(self.scattering_albedo) or
            !@reduce(.And, self.scattering_albedo >= @as(Vec3, @splat(0))) or
            !@reduce(.And, self.scattering_albedo <= @as(Vec3, @splat(1))))
            return error.InvalidVolumeMedium;
    }

    // The form every ray uses. Checked here and nowhere after.
    pub fn place(self: LocalFogVolume) LocalFogError!Placed {
        try self.validate();

        const rotation = zm.normalize4(self.rotation);

        // `composeTransform` builds scale, then rotation, then translation, so
        // the inverse undoes them in the opposite order. Built directly rather
        // than by inverting the product: the factors are each trivially
        // invertible, and a general inverse of a matrix that is known to be one
        // of these is both slower and less exact.
        const world_to_unit = zm.mul(zm.mul(
            zm.translation(-self.centre[0], -self.centre[1], -self.centre[2]),
            zm.quatToMat(zm.conjugate(rotation)),
        ), zm.scaling(1 / self.radii[0], 1 / self.radii[1], 1 / self.radii[2]));

        // The forward linear part sends the unit sphere to the ellipsoid, so the
        // support along world axis i is the length of its i-th column: the most
        // that axis can be reached by any unit vector run through it. Exact,
        // where a sphere of the largest radius would be loose by the ratio
        // between the axes.
        const linear = zm.mul(
            zm.scaling(self.radii[0], self.radii[1], self.radii[2]),
            zm.quatToMat(rotation),
        );
        var half: Vec3 = undefined;
        inline for (0..3) |axis| {
            half[axis] = @sqrt(linear[0][axis] * linear[0][axis] +
                linear[1][axis] * linear[1][axis] +
                linear[2][axis] * linear[2][axis]);
        }

        return .{
            .world_to_unit = world_to_unit,
            .bounds = .{ .min = self.centre - half, .max = self.centre + half },
            .density = self.density,
            .edge_falloff = self.edge_falloff,
            .scattering_albedo = self.scattering_albedo,
            .wind_noise = self.wind_noise,
        };
    }

    pub const Placed = struct {
        world_to_unit: zm.Mat,

        // The world box the ellipsoid occupies, tight on every axis.
        bounds: Aabb,

        density: f32,
        edge_falloff: f32,
        scattering_albedo: Vec3,
        wind_noise: bool,

        // Density at a radius in unit space, which is one at the boundary. The
        // shell is the last `edge_falloff` of that radius and falls through a
        // smoothstep, so the profile meets both the core and the boundary with a
        // zero derivative and a march cannot see where the shell begins.
        pub fn shellDensity(self: Placed, radius: f32) f32 {
            if (radius >= 1) return 0;
            const shell_start = 1 - self.edge_falloff;
            const x = std.math.clamp((radius - shell_start) / self.edge_falloff, 0, 1);
            return self.density * (1 - x * x * (3 - 2 * x));
        }

        pub fn densityAt(self: Placed, world: Vec3) f32 {
            return self.shellDensity(self.unitRadius(world));
        }

        // The world-distance span over which the ray is inside the volume,
        // clipped to [0, limit], or null when it never is.
        //
        // Solved in unit space, where the volume is the unit sphere. The
        // direction is deliberately not renormalized there: leaving its length
        // as the inverse scale is what keeps the parameter a world distance.
        pub fn chordSpan(self: Placed, origin: Vec3, direction: Vec3, limit: f32) ?[2]f32 {
            const o = self.toUnit(origin, 1);
            const d = self.toUnit(direction, 0);

            const a = @reduce(.Add, d * d);
            if (a < min_unit_direction_squared) return null;
            const b = @reduce(.Add, o * d);
            const c = @reduce(.Add, o * o) - 1;

            const discriminant = b * b - a * c;
            if (!(discriminant > 0)) return null;

            const root = @sqrt(discriminant);
            const near = @max((-b - root) / a, 0);
            const far = @min((-b + root) / a, limit);
            if (!(far > near)) return null;
            return .{ near, far };
        }

        // Optical depth this volume adds over the ray segment [from, to]: the
        // integral of the density profile along the part of the segment that is
        // inside the volume.
        pub fn segmentTau(self: Placed, origin: Vec3, direction: Vec3, from: f32, to: f32) f32 {
            const span = self.chordSpan(origin, direction, std.math.floatMax(f32)) orelse return 0;
            const near = @max(span[0], from);
            const far = @min(span[1], to);
            if (!(far > near)) return 0;

            const middle = 0.5 * (near + far);
            const half = 0.5 * (far - near);

            var sum: f32 = 0;
            inline for (gauss_abscissae, gauss_weights) |abscissa, weight| {
                const offset = half * @as(f32, @floatCast(abscissa));
                const w: f32 = @floatCast(weight);
                sum += w * self.shellDensity(self.unitRadius(pointAt(origin, direction, middle - offset)));
                sum += w * self.shellDensity(self.unitRadius(pointAt(origin, direction, middle + offset)));
            }
            return sum * half;
        }

        pub fn visible(self: Placed, frustum: *const Frustum) bool {
            return frustum.intersectsAabb(self.bounds);
        }

        fn unitRadius(self: Placed, world: Vec3) f32 {
            const local = self.toUnit(world, 1);
            return @sqrt(@reduce(.Add, local * local));
        }

        // A point carries one in the fourth lane under a row-vector transform, a
        // direction carries zero. The lane is dropped on the way out, so what
        // comes back is the three that matter.
        fn toUnit(self: Placed, v: Vec3, w: f32) Vec3 {
            const wide = zm.mul(zm.f32x4(v[0], v[1], v[2], w), self.world_to_unit);
            return .{ wide[0], wide[1], wide[2] };
        }
    };
};

fn pointAt(origin: Vec3, direction: Vec3, t: f32) Vec3 {
    return origin + direction * @as(Vec3, @splat(t));
}

// A fixed-capacity set of placed volumes. The capacity is the caller's, because
// what bounds it is whatever consumes the set rather than anything here.
//
// Only the placed form is kept. What a level authored is the level's to hold: a
// second copy here would be a second answer to where a volume is, and the two
// would drift the first time one of them was edited.
pub fn LocalFogVolumes(comptime capacity: usize) type {
    return struct {
        const Self = @This();

        items: [capacity]LocalFogVolume.Placed = undefined,
        len: usize = 0,

        // The volume is placed before anything is written, so a rejected volume
        // leaves the set exactly as it was.
        pub fn add(self: *Self, volume: LocalFogVolume) LocalFogError!u32 {
            if (self.len == capacity) return error.TooManyLocalFogVolumes;
            const placed = try volume.place();
            self.items[self.len] = placed;
            self.len += 1;
            return @intCast(self.len - 1);
        }

        pub fn set(self: *Self, index: usize, volume: LocalFogVolume) LocalFogError!void {
            if (index >= self.len) return error.NoSuchLocalFogVolume;
            self.items[index] = try volume.place();
        }

        pub fn get(self: *const Self, index: usize) ?*const LocalFogVolume.Placed {
            if (index >= self.len) return null;
            return &self.items[index];
        }

        pub fn slice(self: *const Self) []const LocalFogVolume.Placed {
            return self.items[0..self.len];
        }

        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        // Extinction at a point from every volume that covers it. They add:
        // overlapping media are more opaque than either alone.
        pub fn densityAt(self: *const Self, world: Vec3) f32 {
            var total: f32 = 0;
            for (self.slice()) |*volume| total += volume.densityAt(world);
            return total;
        }
    };
}

// How a froxel grid distributes its depth slices over distance.
//
// Exponential rather than uniform, so slice thickness grows with distance and
// the near field, where a fixed angular footprint covers the least world space,
// gets the most slices. The near plane of the distribution is separate from the
// camera's: a froxel grid starting at a camera near plane of a centimetre would
// spend most of its depth on the first metre.
pub const FroxelDepth = struct {
    near: f32,
    range: f32,
    slices: u32,

    pub fn init(near: f32, range: f32, slices: u32) LocalFogError!FroxelDepth {
        if (!(near > 0 and near < range and range < std.math.inf(f32)))
            return error.InvalidFroxelDepth;
        if (slices < 2) return error.InvalidFroxelDepth;
        return .{ .near = near, .range = range, .slices = slices };
    }

    // Distance at a slice index, from `near` at zero to `range` at the last one.
    pub fn distanceAt(self: FroxelDepth, index: f32) f32 {
        const normalized = std.math.clamp(index / @as(f32, @floatFromInt(self.slices - 1)), 0, 1);
        return self.near * std.math.pow(f32, self.range / self.near, normalized);
    }

    // The texture coordinate that samples the slice a distance falls in, with
    // the half-texel offset that puts a slice index at the centre of its texel
    // rather than on the seam between two.
    pub fn textureZ(self: FroxelDepth, distance: f32) f32 {
        const clamped = std.math.clamp(distance, self.near, self.range);
        const normalized = @log(clamped / self.near) / @log(self.range / self.near);
        const slices: f32 = @floatFromInt(self.slices);
        return (normalized * (slices - 1) + 0.5) / slices;
    }

    // The inverse of `textureZ`. That one puts an index at (index + 0.5) /
    // slices, so this takes the index back out and asks for its distance.
    pub fn distanceAtTextureZ(self: FroxelDepth, z: f32) f32 {
        return self.distanceAt(z * @as(f32, @floatFromInt(self.slices)) - 0.5);
    }
};
