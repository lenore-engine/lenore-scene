const bounds = @import("bounds.zig");
const camera = @import("camera.zig");
const draw_batches = @import("draw_batches.zig");
const draw_order = @import("draw_order.zig");
const fog = @import("fog.zig");
const frustum = @import("frustum.zig");
const joint_offsets = @import("joint_offsets.zig");
const light = @import("light.zig");
const local_fog = @import("local_fog.zig");
const picking = @import("picking.zig");
const sun_shadow = @import("sun_shadow.zig");
const transform = @import("transform.zig");

pub const worldAabb = bounds.worldAabb;
pub const unionAabb = bounds.unionAabb;
pub const sphereAroundAabb = bounds.sphereAroundAabb;

pub const Camera = camera.Camera;
pub const Anchor = camera.Anchor;
pub const Placement = camera.Placement;
pub const Projection = camera.Projection;
pub const ProjectionError = camera.ProjectionError;

pub const DrawBatches = draw_batches.DrawBatches;
pub const DrawBatchError = draw_batches.BuildError;
pub const FaceCulling = draw_batches.FaceCulling;

pub const Layer = draw_order.Layer;
pub const DrawKey = draw_order.Key;
pub const DrawOrderError = draw_order.OrderError;
pub const depthOf = draw_order.depthOf;
pub const orderDraws = draw_order.order;

pub const FogSettings = fog.FogSettings;
pub const VolumetricSettings = fog.VolumetricSettings;
pub const FogError = fog.FogError;

pub const Frustum = frustum.Frustum;

pub const assignJointOffsets = joint_offsets.assignJointOffsets;
pub const JointOffsetError = joint_offsets.JointOffsetError;
pub const no_joint_base = joint_offsets.no_joint_base;

pub const Light = light.Light;
pub const LightError = light.LightError;
pub const SunAppearance = light.SunAppearance;

pub const LocalFogVolume = local_fog.LocalFogVolume;
pub const LocalFogVolumes = local_fog.LocalFogVolumes;
pub const FroxelDepth = local_fog.FroxelDepth;
pub const LocalFogError = local_fog.LocalFogError;

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
