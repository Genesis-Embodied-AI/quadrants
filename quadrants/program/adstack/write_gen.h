#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/transforms/static_adstack_analysis.h"

namespace quadrants::lang {

class LaunchContextBuilder;
class Program;

// Apply the captured per-task loop trip-count clip to `effective_rows`. Each loop iteration of an adstack
// task claims at most one row at the LCA-block, so the heap needs at most `trip_count` rows regardless of
// how many cells of an oversized gating SNode/ndarray the reducer counted. Two trip-count sources, picked
// in order: `bound_expr.loop_iter_static` (compile-time-known constant, integer compare) and
// `bound_expr.loop_iter_size_expr` (per-launch tree walk via `evaluate_adstack_size_expr`). Both are
// gated by `dispatched_threads_ceiling` so a `dynamic_gpu_range_for` that exceeds the dispatch cap and
// serialises iterations across threads (each thread reaches the LCA-block multiple times) does not
// accidentally undersize the heap; pass `std::numeric_limits<std::size_t>::max()` to disable the
// ceiling. No-op when the static field is zero AND the SizeExpr is empty (the analyzer leaves both
// unset for shapes the compile-time path cannot cover) - the caller's pre-clip `effective_rows` is left
// unchanged so the runtime falls through to the unclipped reducer count.
void clip_effective_rows_by_loop_trip_count(std::size_t &effective_rows,
                                            const StaticAdStackBoundExpr &bound_expr,
                                            std::size_t dispatched_threads_ceiling,
                                            Program *prog,
                                            LaunchContextBuilder *ctx);

// Adstack-cache invalidation bump. Called from each backend's kernel launcher BEFORE the per-task
// `publish_adstack_metadata` loop runs, so the per-task metadata cache (`Program::*PerTaskAdStackCacheEntry`) snapshots
// the latest counters at record time and the next lookup detects any drift. Two sources contribute:
//
//   - SNode writes: every task in the kernel lists its compile-time `snode_writes` set (computed at codegen via
//     `irpass::analysis::gather_snode_read_writes`), bumped per id; covers `SizeExpr::FieldLoad` cache invalidation.
//   - ndarray data writes: every arg slot the kernel writes to (`OffloadedTask::arr_writes` on LLVM-GPU, the kernel-
//     level `ctx_attribs.arr_access` WRITE bits on SPIR-V) bumps the bound `DeviceAllocation`'s data generation.
//     SPIR-V also bumps on the `kNone` READ branch to catch host-driven mutations of raw numpy / torch buffers blitted
//     between launches; covers `SizeExpr::ExternalTensorRead` invalidation.
//
// The two helpers share the same Program-level effect; their signatures differ only because the codegen-time write
// sets are stored in different per-backend structs. Forward-only kernels (no adstack tasks) still call these to keep
// counters monotone, which is cheap (one map insert per snode_id at most).
void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<OffloadedTask> &offloaded_tasks);
// CPU launcher overload: per-task snode_writes / arr_writes / arr_reads are stored as separate parallel vectors on
// the launcher `Context` rather than as `OffloadedTask` clones, for legacy reasons documented in the CPU `Context`
// struct.
void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<std::vector<int>> &snode_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_reads_per_task);
void bump_writes_for_kernel_spirv(
    Program *prog,
    LaunchContextBuilder *ctx,
    const std::vector<spirv::TaskAttributes> &task_attribs,
    const std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>> &arr_access);

}  // namespace quadrants::lang
