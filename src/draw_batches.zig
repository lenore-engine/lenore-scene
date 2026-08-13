// Turns a final draw order into contiguous instance runs. Culling and sorting
// happen before this step; reordering here would invalidate transparent depth
// order. Only adjacent draws with identical GPU-facing state coalesce.
//
// The order arrives partitioned, visible draws first, and a run is not allowed
// to span that boundary. A batch is therefore wholly visible or wholly culled,
// and the visible ones are a prefix: a camera pass records that prefix and a
// shadow bake records all of them. Letting one run straddle the boundary would
// leave a batch that neither pass can record correctly, since an instance range
// is one number and cannot say that half of it is on screen.
//
// The resource identifiers are parameters because scene planning only compares
// their identity. It neither resolves them nor depends on the backend that owns
// them.

const std = @import("std");

// Which polygon side the recorder drops. A double-sided or conservatively
// skinned draw drops neither. This belongs in the batch key: one draw command
// has one culling state.
pub const FaceCulling = enum {
    none,
    back,
    front,
};

// Which winding the rasterizer calls the front of a triangle.
//
// glTF 2.0, section 3.7.4: the determinant of the node's global transform
// defines the winding order of that primitive, counter-clockwise where it is
// positive and clockwise where it is not. Part of that product is baked into
// the geometry when it is loaded; what is left is the instance transform, whose
// sign is per instance and per frame, which is what puts this in the batch key
// beside the culling state.
//
// A state of its own rather than exchanging `back` for `front` above. The two
// would drop the same triangles, but the winding also decides what the fragment
// stage is told about facing, and a double-sided material draws with no culling
// at all and still asks: there is no side to exchange there, and folding the
// two would light a mirrored double-sided surface as though its back were its
// front.
pub const FrontFace = enum {
    counter_clockwise,
    clockwise,
};

pub const BuildError = error{
    // An order entry does not name a draw. Every index in it must belong to the
    // source list, whichever side of the visibility boundary it lies on.
    DrawIndexOutOfRange,

    // The boundary is a position in `order` and is used to slice it, so a value
    // past its end would produce a prefix longer than the list. It cannot be
    // justified by construction here: the count and the order arrive as two
    // independent arguments and nothing in the signature says they were produced
    // together.
    VisibleCountOutOfRange,

    // first_instance and instance_count are u32 in the draw contract. Refuse a
    // list that cannot be represented before narrowing any offset into it.
    TooManyInstances,

    // No output is written on this error. The caller can size for the worst
    // case with one batch per ordered draw or retain a measured scene capacity.
    BatchCapacityExceeded,
};

// The types shared by one scene planner and its renderer consumer. MeshId and
// MaterialId may be indices, generational handles or another equality-comparable
// identity; their representation is not interpreted here.
pub fn DrawBatches(comptime MeshId: type, comptime MaterialId: type) type {
    return struct {
        // One candidate before culling and ordering. `order` addresses these
        // values, and that order is also the order in which the instance ring
        // is packed.
        pub const Draw = struct {
            mesh: MeshId,
            material: MaterialId,
            face_culling: FaceCulling,
            front_face: FrontFace,
        };

        // One command's scene-owned policy. The backend resolves the resource
        // identities, chooses the pipeline and descriptor set, and records the
        // command without sorting or allocating.
        pub const Batch = struct {
            mesh: MeshId,
            material: MaterialId,
            face_culling: FaceCulling,
            front_face: FrontFace,
            first_instance: u32,
            instance_count: u32,
        };

        // What `build` wrote, with the batches a camera pass records first.
        pub const Batches = struct {
            batches: []Batch,
            // How many of `batches` are visible. The rest hold the draws behind
            // the camera's boundary and are recorded by the passes that do not
            // use the camera.
            visible: usize,

            pub fn visibleBatches(self: Batches) []Batch {
                return self.batches[0..self.visible];
            }
        };

        // Coalesces adjacent entries of `order` without changing their order.
        // `visible` is how many of its leading entries the camera can see, which
        // is where a run is cut whether or not the state continues across it.
        //
        // The function validates the complete input and output capacity before
        // writing the first batch, so every error leaves `destination`
        // untouched.
        pub fn build(
            draws: []const Draw,
            order: []const u32,
            visible: usize,
            destination: []Batch,
        ) BuildError!Batches {
            if (order.len > std.math.maxInt(u32)) return error.TooManyInstances;
            if (visible > order.len) return error.VisibleCountOutOfRange;

            var batch_count: usize = 0;
            var visible_batches: usize = 0;
            var previous: ?Draw = null;
            for (order, 0..) |draw_index, index| {
                if (draw_index >= draws.len) return error.DrawIndexOutOfRange;
                const current = draws[draw_index];
                if (previous == null or index == visible or !sameState(previous.?, current)) {
                    batch_count += 1;
                    if (index < visible) visible_batches += 1;
                }
                previous = current;
            }
            if (batch_count > destination.len) return error.BatchCapacityExceeded;

            var batch_index: usize = 0;
            var first: usize = 0;
            while (first < order.len) {
                const representative = draws[order[first]];
                var end = first + 1;
                while (end < order.len and
                    end != visible and
                    sameState(representative, draws[order[end]]))
                    end += 1;

                destination[batch_index] = .{
                    .mesh = representative.mesh,
                    .material = representative.material,
                    .face_culling = representative.face_culling,
                    .front_face = representative.front_face,
                    .first_instance = @intCast(first),
                    .instance_count = @intCast(end - first),
                };
                batch_index += 1;
                first = end;
            }
            std.debug.assert(batch_index == batch_count);
            return .{ .batches = destination[0..batch_count], .visible = visible_batches };
        }

        fn sameState(a: Draw, b: Draw) bool {
            return a.mesh == b.mesh and
                a.material == b.material and
                a.face_culling == b.face_culling and
                a.front_face == b.front_face;
        }
    };
}
