// A punctual light as the scene holds it: what it emits, and where it is once
// the node that references it has been placed.
//
// The three kinds carry different data and only their own, so there is no tag to
// consult before touching a field and no field that means one thing for a point
// light and another for a directional one. That is what the kinds are for: a
// directional light has no position to place and no range to cut it off, and
// here it has neither.
//
// Directions point the way the light travels, which is the same convention the
// document a light is loaded from uses. A sun shadow fit wants the reverse, the
// direction toward the source.

const std = @import("std");
const zm = @import("zmath");
const resources = @import("lenore-resources");
const vec = @import("vec.zig");

const Vec3 = resources.Vec3;

// Below this, a direction has no length worth normalizing. Its square, 1e-12,
// is far from where an f32 underflows.
const min_direction_length_squared: f32 = 1e-12;

// Cones narrower than this gap in cosine collapse to a hard edge. The blend
// between the two angles divides by their difference, and an edge that thin is
// indistinguishable from a step anyway.
const min_cone_cosine_gap: f32 = 1e-4;

// Where an unwindowed inverse-square falloff has dropped far enough to stop.
// Radiance is scene-referred, so this is a fraction of a unit emitter at one
// unit: past it the windowed falloff is under a tonemapped display's step.
const min_visible_radiance: f32 = 0.005;

pub const LightError = error{
    // Colour or intensity is negative, or not finite. Both come from an asset.
    InvalidLightEmission,

    // The range is not a positive, finite distance.
    InvalidLightRange,

    // The position is not finite.
    InvalidLightPosition,

    // The direction has no length, or is not finite. A light with no direction
    // is not a light pointing somewhere by default.
    DegenerateLightDirection,

    // The cone angles are not 0 <= inner < outer <= pi/2.
    InvalidSpotCone,
};

pub const Light = struct {
    // Linear, and not premultiplied by the intensity: the two are authored
    // separately and a tonemapper wants them that way round.
    colour: Vec3,
    intensity: f32,
    kind: Kind,

    pub const Kind = union(enum) {
        // Unit, and pointing the way the light travels.
        directional: Vec3,
        point: Point,
        spot: Spot,
    };

    pub const Point = struct {
        position: Vec3,

        // Distance past which the light contributes nothing. Authored, or
        // derived with `rangeFor` when the asset leaves it out.
        range: f32,
    };

    pub const Spot = struct {
        position: Vec3,

        // Unit, the axis of the cone, pointing the way the light travels.
        direction: Vec3,
        range: f32,

        // Cosines rather than angles, because what reads them tests a dot
        // product against the axis and would otherwise take two cosines per
        // sample. `cos_inner` is the larger of the two: cosine falls as the
        // angle opens.
        cos_inner: f32,
        cos_outer: f32,
    };

    pub const SpotParams = struct {
        position: Vec3,
        direction: Vec3,
        range: f32,

        // From the axis, in radians, so the full cone is twice the outer angle.
        inner_angle: f32,
        outer_angle: f32,
    };

    pub fn directional(colour: Vec3, intensity: f32, direction: Vec3) LightError!Light {
        try checkEmission(colour, intensity);
        return .{
            .colour = colour,
            .intensity = intensity,
            .kind = .{ .directional = try normalize(direction) },
        };
    }

    pub fn point(colour: Vec3, intensity: f32, position: Vec3, range: f32) LightError!Light {
        try checkEmission(colour, intensity);
        try checkRange(range);
        if (!vec.finite(position)) return error.InvalidLightPosition;
        return .{
            .colour = colour,
            .intensity = intensity,
            .kind = .{ .point = .{ .position = position, .range = range } },
        };
    }

    pub fn spot(colour: Vec3, intensity: f32, params: SpotParams) LightError!Light {
        try checkEmission(colour, intensity);
        try checkRange(params.range);
        if (!vec.finite(params.position)) return error.InvalidLightPosition;

        if (!(params.inner_angle >= 0 and
            params.inner_angle < params.outer_angle and
            params.outer_angle <= std.math.pi / 2.0))
            return error.InvalidSpotCone;

        const cos_outer = @cos(params.outer_angle);
        return .{
            .colour = colour,
            .intensity = intensity,
            .kind = .{
                .spot = .{
                    .position = params.position,
                    .direction = try normalize(params.direction),
                    .range = params.range,
                    // The angles are already ordered, so this only widens a cone
                    // whose two angles are close enough that their cosines are not.
                    .cos_inner = @max(@cos(params.inner_angle), cos_outer + min_cone_cosine_gap),
                    .cos_outer = cos_outer,
                },
            },
        };
    }

    // The same light placed by a node transform. This is the step between the
    // document, where a light is stated in the space of the node that references
    // it, and the scene, where it has to be in world space.
    //
    // The transform moves and turns a light and does not touch what it emits.
    // KHR_lights_punctual, "Light Shared Properties": light properties are
    // unaffected by node transforms, and `range` and `intensity` in particular
    // do not change with scale. So a range is a world distance an artist sets,
    // and a scaled node keeps it.
    //
    // A direction is carried by the matrix itself rather than by its inverse
    // transpose. That rule belongs to normals, which have to stay perpendicular
    // to a surface that is being skewed; the axis of a cone is an ordinary
    // direction and follows the geometry.
    pub fn placed(self: Light, model: zm.Mat) LightError!Light {
        return .{
            .colour = self.colour,
            .intensity = self.intensity,
            .kind = switch (self.kind) {
                .directional => |direction| .{ .directional = try normalize(rotate(direction, model)) },
                .point => |source| .{ .point = .{
                    .position = try transform(source.position, model),
                    .range = source.range,
                } },
                .spot => |source| .{ .spot = .{
                    .position = try transform(source.position, model),
                    .direction = try normalize(rotate(source.direction, model)),
                    .range = source.range,
                    .cos_inner = source.cos_inner,
                    .cos_outer = source.cos_outer,
                } },
            },
        };
    }

    // The range to give a light whose asset states none, which the format allows.
    // It is the distance at which an unwindowed inverse-square falloff reaches
    // `min_visible_radiance`: intensity / r^2 = threshold, so r = sqrt(intensity
    // / threshold).
    pub fn rangeFor(intensity: f32) f32 {
        return @sqrt(@max(intensity, 0) / min_visible_radiance);
    }
};

// How the sun is drawn as a disk, as distinct from how it lights anything. The
// direction, colour and intensity stay on the light that was designated the sun.
pub const SunAppearance = struct {
    // 0.266 degrees, half of the Sun's apparent angular diameter of about
    // 0.53 degrees from Earth.
    angular_radius: f32 = std.math.degreesToRadians(0.266),

    // Multiplies the disk's radiance only, leaving the light it casts alone. An
    // art control: the true ratio between a solar disk and its illumination is
    // far outside what a display reproduces.
    disk_radiance_scale: f32 = 1,

    pub fn validate(self: SunAppearance) error{InvalidSunAppearance}!void {
        if (!(self.angular_radius > 0 and self.angular_radius < std.math.pi / 2.0))
            return error.InvalidSunAppearance;
        if (!(self.disk_radiance_scale >= 0 and self.disk_radiance_scale < std.math.inf(f32)))
            return error.InvalidSunAppearance;
    }
};

fn checkEmission(colour: Vec3, intensity: f32) LightError!void {
    if (!vec.finite(colour) or !@reduce(.And, colour >= @as(Vec3, @splat(0))))
        return error.InvalidLightEmission;
    if (!(intensity >= 0 and intensity < std.math.inf(f32)))
        return error.InvalidLightEmission;
}

fn checkRange(range: f32) LightError!void {
    if (!(range > 0 and range < std.math.inf(f32))) return error.InvalidLightRange;
}

fn normalize(direction: Vec3) LightError!Vec3 {
    if (!vec.finite(direction)) return error.DegenerateLightDirection;
    const length_squared = @reduce(.Add, direction * direction);
    if (length_squared < min_direction_length_squared) return error.DegenerateLightDirection;
    return direction / @as(Vec3, @splat(@sqrt(length_squared)));
}

// The matrix comes from a document's node walk, so it is asset data and the
// point it produces is checked once here. A constructor validated the position
// it was given; nothing downstream re-checks the placed one.
fn transform(position: Vec3, model: zm.Mat) LightError!Vec3 {
    const placed = zm.mul(zm.f32x4(position[0], position[1], position[2], 1), model);
    const world: Vec3 = .{ placed[0], placed[1], placed[2] };
    if (!vec.finite(world)) return error.InvalidLightPosition;
    return world;
}

fn rotate(direction: Vec3, model: zm.Mat) Vec3 {
    const turned = zm.mul(zm.f32x4(direction[0], direction[1], direction[2], 0), model);
    return .{ turned[0], turned[1], turned[2] };
}
