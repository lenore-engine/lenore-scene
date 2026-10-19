const ambient_occlusion = @import("ambient_occlusion.zig");
const atmosphere = @import("atmosphere.zig");
const bounds = @import("bounds.zig");
const camera = @import("camera.zig");
const draw_batches = @import("draw_batches.zig");
const draw_order = @import("draw_order.zig");
const exposure = @import("exposure.zig");
const frustum = @import("frustum.zig");
const joint_offsets = @import("joint_offsets.zig");
pub const jitter = @import("jitter.zig");
const light = @import("light.zig");
const local_shadow = @import("local_shadow.zig");
const picking = @import("picking.zig");
pub const probe_grid = @import("probe_grid.zig");
const sun_shadow = @import("sun_shadow.zig");
const transform = @import("transform.zig");

pub const AmbientOcclusionSettings = ambient_occlusion.AmbientOcclusionSettings;
pub const AmbientOcclusionReconstruction = ambient_occlusion.Reconstruction;
pub const AmbientOcclusionReconstructionError = ambient_occlusion.ReconstructionError;

pub const Atmosphere = atmosphere.Atmosphere;
pub const AtmosphereError = atmosphere.AtmosphereError;
pub const Tables = atmosphere.Tables;
pub const planet_radius_offset = atmosphere.planet_radius_offset;
pub const transmittance = atmosphere.transmittance;
pub const skyRadiance = atmosphere.skyRadiance;
pub const skyIrradiance = atmosphere.skyIrradiance;
pub const skyIrradianceResolution = atmosphere.skyIrradianceResolution;
pub const sunAtGround = atmosphere.sunAtGround;

pub const worldAabb = bounds.worldAabb;
pub const unionAabb = bounds.unionAabb;
pub const sphereAroundAabb = bounds.sphereAroundAabb;

pub const Camera = camera.Camera;
pub const Anchor = camera.Anchor;
pub const Placement = camera.Placement;
pub const Projection = camera.Projection;
pub const RayBasis = camera.RayBasis;
pub const ProjectionError = camera.ProjectionError;

pub const DrawBatches = draw_batches.DrawBatches;
pub const DrawBatchError = draw_batches.BuildError;
pub const FaceCulling = draw_batches.FaceCulling;
pub const FrontFace = draw_batches.FrontFace;

pub const Layer = draw_order.Layer;
pub const DrawKey = draw_order.Key;
pub const DrawOrderError = draw_order.OrderError;
pub const depthOf = draw_order.depthOf;
pub const orderDraws = draw_order.order;

pub const Photocell = exposure.Photocell;
pub const Iris = exposure.Iris;

pub const Frustum = frustum.Frustum;

pub const assignJointOffsets = joint_offsets.assignJointOffsets;
pub const JointOffsetError = joint_offsets.JointOffsetError;
pub const no_joint_base = joint_offsets.no_joint_base;

pub const Light = light.Light;
pub const LightError = light.LightError;
pub const SunAppearance = light.SunAppearance;

pub const LocalShadowFit = local_shadow.LocalShadowFit;
pub const LocalShadowError = local_shadow.LocalShadowError;
pub const assignShadowLayers = local_shadow.assignLayers;
pub const shadow_face_axes = local_shadow.face_axes;
// The near plane of every local layer, which the lookup needs to turn a stored
// depth back into a distance.
pub const local_shadow_near = local_shadow.near;

pub const Ray = picking.Ray;
pub const Viewport = picking.Viewport;
pub const Intersection = picking.Intersection;
pub const PickError = picking.Error;
pub const cameraRay = picking.cameraRay;
pub const intersectAabb = picking.intersectAabb;
pub const intersectInstance = picking.intersectInstance;

pub const SunShadowFit = sun_shadow.SunShadowFit;
pub const SunShadowSettings = sun_shadow.SunShadowSettings;
pub const FitError = sun_shadow.FitError;

pub const Transform = transform.Transform;
pub const rotationFromEulerZyx = transform.rotationFromEulerZyx;

pub const split_sum = @import("split_sum.zig");
