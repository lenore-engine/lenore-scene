// Turns a final visible draw order into contiguous instance runs. Culling and
// sorting happen before this step; reordering here would invalidate transparent
// depth order. Only adjacent draws with identical GPU-facing state coalesce.
//
// The resource identifiers are parameters because scene planning only compares
// their identity. It neither resolves them nor depends on the backend that owns
// them.

const std = @import("std");

// Which polygon side the recorder drops. A reflected transform exchanges front
// and back, while a double-sided or conservatively skinned draw drops neither.
// This belongs in the batch key: one draw command has one culling state.
pub const FaceCulling = enum {
    none,
    back,
    front,
};

pub const BuildError = error{
    // An order entry does not name a draw. The order can be a culled subset, but
    // every surviving index must still belong to the source list.
    DrawIndexOutOfRange,

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
        };

        // One command's scene-owned policy. The backend resolves the resource
        // identities, chooses the pipeline and descriptor set, and records the
        // command without sorting or allocating.
        pub const Batch = struct {
            mesh: MeshId,
            material: MaterialId,
            face_culling: FaceCulling,
            first_instance: u32,
            instance_count: u32,
        };

        // Coalesces adjacent entries of `order` without changing their order.
        // The function validates the complete input and output capacity before
        // writing the first batch, so every error leaves `destination`
        // untouched.
        pub fn build(
            draws: []const Draw,
            order: []const u32,
            destination: []Batch,
        ) BuildError![]Batch {
            if (order.len > std.math.maxInt(u32)) return error.TooManyInstances;

            var batch_count: usize = 0;
            var previous: ?Draw = null;
            for (order) |draw_index| {
                if (draw_index >= draws.len) return error.DrawIndexOutOfRange;
                const current = draws[draw_index];
                if (previous == null or !sameState(previous.?, current))
                    batch_count += 1;
                previous = current;
            }
            if (batch_count > destination.len) return error.BatchCapacityExceeded;

            var batch_index: usize = 0;
            var first: usize = 0;
            while (first < order.len) {
                const representative = draws[order[first]];
                var end = first + 1;
                while (end < order.len and sameState(representative, draws[order[end]]))
                    end += 1;

                destination[batch_index] = .{
                    .mesh = representative.mesh,
                    .material = representative.material,
                    .face_culling = representative.face_culling,
                    .first_instance = @intCast(first),
                    .instance_count = @intCast(end - first),
                };
                batch_index += 1;
                first = end;
            }
            std.debug.assert(batch_index == batch_count);
            return destination[0..batch_count];
        }

        fn sameState(a: Draw, b: Draw) bool {
            return a.mesh == b.mesh and
                a.material == b.material and
                a.face_culling == b.face_culling;
        }
    };
}
