#include "quadrants/program/adstack/write_gen.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/adstack/cache.h"
#include "quadrants/program/adstack/eval.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/transforms/static_adstack_analysis.h"

namespace quadrants::lang {

void clip_effective_rows_by_loop_trip_count(std::size_t &effective_rows,
                                            const StaticAdStackBoundExpr &bound_expr,
                                            std::size_t dispatched_threads_ceiling,
                                            Program *prog,
                                            LaunchContextBuilder *ctx) {
  if (bound_expr.loop_iter_static > 0) {
    // Compile-time trip count: integer compare, no per-launch eval cost. Constant `SizeExpr` shapes are
    // already collapsed into this field by the analyzer so they short-circuit the runtime eval below.
    const std::size_t loop_iter_static = static_cast<std::size_t>(bound_expr.loop_iter_static);
    if (loop_iter_static <= dispatched_threads_ceiling) {
      effective_rows = std::min<std::size_t>(effective_rows, loop_iter_static);
    }
    return;
  }
  if (bound_expr.loop_iter_size_expr.nodes.empty() || prog == nullptr || ctx == nullptr) {
    // Runtime tree empty or no resolution context: the analyzer left this field unset for shapes the
    // compile-time path could not cover (or the caller did not supply a `Program` / `LaunchContextBuilder`),
    // so leave `effective_rows` alone and let the caller fall back to the unclipped reducer count.
    return;
  }
  // Runtime-bounded clip: evaluate the captured trip-count `SizeExpr` only when the static field is unset
  // (the analyzer leaves `loop_iter_static == 0` for shapes the compile-time path cannot cover, e.g.
  // `for j in range(field[i])` / `for k in range(arr.shape[axis])`). Cost = one tree walk per launch,
  // dominated by host scalar reads through `SNodeRwAccessorsBank` on `FieldLoad` / `ExternalTensorRead`
  // nodes (CPU: a memory load; CUDA / AMDGPU: a 4-8 byte DtoH). The evaluator returns -1 when the tree
  // references state that is not host-resolvable from `ctx`; in that case we leave `effective_rows`
  // unclipped from this source.
  const int64_t evaluated = evaluate_adstack_size_expr(bound_expr.loop_iter_size_expr, prog, ctx);
  if (evaluated > 0 && static_cast<std::size_t>(evaluated) <= dispatched_threads_ceiling) {
    effective_rows = std::min<std::size_t>(effective_rows, static_cast<std::size_t>(evaluated));
  }
}

void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<OffloadedTask> &offloaded_tasks) {
  if (prog == nullptr) {
    return;
  }
  auto bump_data_ptr = [&](int arg_id) {
    ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(data_key);
    if (it != ctx->array_ptrs.end() && it->second != nullptr) {
      prog->adstack_cache().bump_ndarray_data_gen(it->second);
    }
  };
  for (const auto &task : offloaded_tasks) {
    for (int snode_id : task.snode_writes) {
      prog->adstack_cache().bump_snode_write_gen(snode_id);
    }
    for (int arg_id : task.arr_writes) {
      bump_data_ptr(arg_id);
    }
    // Read-only `DevAllocType::kNone` args also need a bump: the user's host array is either H2D-blitted to a
    // temporary device buffer (CUDA / AMDGPU) or read directly (CPU), and in both cases the data pointer used as
    // the cache key is stable across launches, so a content mutation the user performed outside Quadrants's
    // tracking is invisible to the metadata cache without an explicit bump. Mirrors the SPIR-V `kone_h2d_blit`
    // rule in `bump_writes_for_kernel_spirv`.
    for (int arg_id : task.arr_reads) {
      auto type_it = ctx->device_allocation_type.find(arg_id);
      if (type_it == ctx->device_allocation_type.end() ||
          type_it->second != LaunchContextBuilder::DevAllocType::kNone) {
        continue;
      }
      bump_data_ptr(arg_id);
    }
  }
}

void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<std::vector<int>> &snode_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_reads_per_task) {
  if (prog == nullptr) {
    return;
  }
  // Skip the per-task / per-arg gen-counter bumps when the adstack cache has never been recorded into: the bumps only
  // exist so a later cache lookup can detect drift, and with no entries to drift against the work is wasted.
  // Forward-only kernels (no `record_*` ever called against this `AdStackCache`) hit this gate on every launch and
  // pay zero per-arg hashmap lookup, which matters on the CPU LLVM ndarray path where every kernel-bound arg slot
  // would otherwise show up in `arr_writes_per_task` / `arr_reads_per_task`. The flag is one-way: the first sizer,
  // per-task, or max-reducer recording flips it true and every subsequent launch resumes the unconditional bump path
  // so a later cache lookup never sees a missed bump.
  AdStackCache &cache = prog->adstack_cache();
  if (!cache.has_any_recordings()) {
    return;
  }
  // Per-id refinement of the program-wide `has_any_recordings()` gate. In a mixed program where some autodiff kernel
  // has recorded but the kernel about to launch is forward-only, we still walk this kernel's static write set but skip
  // the bump for any id no cached entry observes. The observation footprint grows monotonically as `record_*` fires;
  // an id newly added to it on launch N is bumped from launch N+1 onward, and the launch-N record itself snapshots the
  // current (un-bumped) gen, so a future kernel write to that id will bump-then-mismatch and the cache will invalidate
  // exactly when it should. The ndarray loops are gated collectively on `any_external_read_observed()` rather than
  // per-arg because the per-launch `array_ptrs.find` cost is what dominates on ndarray-heavy CPU workloads; if no
  // cached entry depends on an ExternalRead, every find in the arg loops is wasted.
  for (const auto &task_snodes : snode_writes_per_task) {
    for (int snode_id : task_snodes) {
      if (cache.is_snode_observed(snode_id)) {
        cache.bump_snode_write_gen(snode_id);
      }
    }
  }
  if (!cache.any_external_read_observed()) {
    return;
  }
  // Per-devalloc refinement on top of the `any_external_read_observed()` gate above: after the unavoidable `find`
  // resolves the arg slot to a `DeviceAllocation *`, skip the bump update for devallocs no cached entry observes.
  // Cache entries hold `observed_devalloc` per ExternalReadObs / `arg_gens` snapshot, aggregated into
  // `observed_devalloc_ptrs_` at record time. A first-time observation of `devalloc` adds it to the set, so the
  // immediately-following kernel write picks up the gate and the cache lookup detects the mismatch on the next
  // launch.
  auto bump_data_ptr = [&](int arg_id) {
    ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(data_key);
    if (it != ctx->array_ptrs.end() && it->second != nullptr && cache.is_devalloc_observed(it->second)) {
      cache.bump_ndarray_data_gen(it->second);
    }
  };
  for (const auto &task_args : arr_writes_per_task) {
    for (int arg_id : task_args) {
      bump_data_ptr(arg_id);
    }
  }
  // Read-only `DevAllocType::kNone` args: see the comment in the CUDA / AMDGPU overload for why CPU LLVM also
  // needs the bump. Empty `arr_reads_per_task` is the legal cache-miss path (offline-cache load that did not
  // capture per-task arr_reads); skip the loop without raising.
  for (const auto &task_args : arr_reads_per_task) {
    for (int arg_id : task_args) {
      auto type_it = ctx->device_allocation_type.find(arg_id);
      if (type_it == ctx->device_allocation_type.end() ||
          type_it->second != LaunchContextBuilder::DevAllocType::kNone) {
        continue;
      }
      bump_data_ptr(arg_id);
    }
  }
}

void bump_writes_for_kernel_spirv(
    Program *prog,
    LaunchContextBuilder *ctx,
    const std::vector<spirv::TaskAttributes> &task_attribs,
    const std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>> &arr_access) {
  if (prog == nullptr) {
    return;
  }
  for (const auto &task : task_attribs) {
    for (int snode_id : task.snode_writes) {
      prog->adstack_cache().bump_snode_write_gen(snode_id);
    }
  }
  for (const auto &kv : arr_access) {
    const std::vector<int> &indices = kv.first;
    uint32_t access = uint32_t(kv.second);
    QD_ASSERT(indices.size() == 1);
    int arg_id = indices[0];
    bool kernel_writes = (access & uint32_t(irpass::ExternalPtrAccess::WRITE)) != 0;
    bool kone_h2d_blit = (access & uint32_t(irpass::ExternalPtrAccess::READ)) != 0 &&
                         ctx->device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone;
    if (!kernel_writes && !kone_h2d_blit) {
      continue;
    }
    ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
    auto it = ctx->array_ptrs.find(data_key);
    if (it != ctx->array_ptrs.end()) {
      prog->adstack_cache().bump_ndarray_data_gen(it->second);
    }
  }
}

}  // namespace quadrants::lang
