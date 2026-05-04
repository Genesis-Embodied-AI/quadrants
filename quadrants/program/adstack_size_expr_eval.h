#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/program/program.h"
#include "quadrants/transforms/static_adstack_analysis.h"

namespace quadrants::lang {

class LaunchContextBuilder;
class Program;

// Adstack-specific caching state. Owns the per-task adstack-sizer metadata caches (SPIR-V + LLVM-GPU), the encoded
// SPIR-V bytecode cache, the per-launch SizeExpr-eval result cache, and the per-snode / per-DeviceAllocation generation
// counters that drive precise invalidation. Held by `Program` via a unique_ptr; all callers route through
// `program->adstack_cache().method(...)`. Lifecycle matches `Program`.
class AdStackCache {
 public:
  // One input read observed during a `evaluate_adstack_size_expr` walk. The cache entry records these so a subsequent
  // lookup re-reads the same inputs and compares to `observed_value`; a single mismatch forces a full re-walk.
  struct SizeExprReadObservation {
    enum Kind : uint8_t { FieldLoadObs, ExternalShapeObs, ExternalReadObs };
    Kind kind;
    int snode_id;
    std::vector<int> indices;
    std::vector<int> arg_id_path;
    int arg_shape_axis;
    int prim_dt;
    int64_t observed_value;
  };
  struct SizeExprCacheEntry {
    int64_t result;
    std::vector<SizeExprReadObservation> reads;
  };
  bool try_size_expr_cache_hit(Program *prog,
                               const SerializedSizeExpr *expr_key,
                               LaunchContextBuilder *ctx,
                               int64_t &out_result);
  void record_size_expr_eval(const SerializedSizeExpr *expr_key,
                             int64_t result,
                             std::vector<SizeExprReadObservation> reads);
  void invalidate_size_expr_cache() {
    size_expr_cache_.clear();
  }

  // Cache for encoded SPIR-V adstack-sizer bytecode. Same dep-tracking contract as `try_size_expr_cache_hit` but the
  // cached payload is the encoded bytes rather than an integer.
  struct SpirvBytecodeCacheEntry {
    std::vector<uint8_t> bytecode;
    std::vector<SizeExprReadObservation> reads;
  };
  bool try_spirv_bytecode_cache_hit(Program *prog,
                                    const void *attribs_key,
                                    LaunchContextBuilder *ctx,
                                    std::vector<uint8_t> &out_bytecode);
  void record_spirv_bytecode_eval(const void *attribs_key,
                                  std::vector<uint8_t> bytecode,
                                  std::vector<SizeExprReadObservation> reads);
  void invalidate_spirv_bytecode_cache() {
    spirv_bytecode_cache_.clear();
  }

  // Per-task adstack metadata output cache for the SPIR-V on-device sizer.
  struct PerTaskAdStackCacheEntry {
    std::vector<uint32_t> metadata;
    uint32_t stride_float{0};
    uint32_t stride_int{0};
    std::vector<std::pair<int, uint64_t>> snode_gens;
    std::vector<std::tuple<int, void *, uint64_t>> arg_gens;
  };
  bool try_per_task_ad_stack_cache_hit(const void *attribs_key,
                                       LaunchContextBuilder *ctx,
                                       PerTaskAdStackCacheEntry &out);
  void record_per_task_ad_stack(const void *attribs_key,
                                std::vector<uint32_t> metadata,
                                uint32_t stride_float,
                                uint32_t stride_int,
                                std::vector<std::pair<int, uint64_t>> snode_gens,
                                std::vector<std::tuple<int, void *, uint64_t>> arg_gens);
  void invalidate_per_task_ad_stack_cache() {
    per_task_ad_stack_cache_.clear();
  }

  // Per-task adstack metadata output cache for the LLVM-GPU on-device sizer (CUDA + AMDGPU).
  struct LlvmPerTaskAdStackCacheEntry {
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> max_sizes;
    uint64_t stride_combined{0};
    uint64_t stride_float{0};
    uint64_t stride_int{0};
    std::vector<std::pair<int, uint64_t>> snode_gens;
    std::vector<std::tuple<int, void *, uint64_t>> arg_gens;
  };
  bool try_llvm_per_task_ad_stack_cache_hit(const void *attribs_key,
                                            LaunchContextBuilder *ctx,
                                            LlvmPerTaskAdStackCacheEntry &out);
  void record_llvm_per_task_ad_stack(const void *attribs_key,
                                     std::vector<uint64_t> offsets,
                                     std::vector<uint64_t> max_sizes,
                                     uint64_t stride_combined,
                                     uint64_t stride_float,
                                     uint64_t stride_int,
                                     std::vector<std::pair<int, uint64_t>> snode_gens,
                                     std::vector<std::tuple<int, void *, uint64_t>> arg_gens);
  void invalidate_llvm_per_task_ad_stack_cache() {
    llvm_per_task_ad_stack_cache_.clear();
  }

  uint64_t snode_write_gen(int snode_id) const {
    auto it = snode_write_gen_.find(snode_id);
    return it == snode_write_gen_.end() ? 0u : it->second;
  }
  void bump_snode_write_gen(int snode_id) {
    ++snode_write_gen_[snode_id];
  }
  uint64_t ndarray_data_gen(void *devalloc_ptr) const {
    auto it = ndarray_data_gen_.find(devalloc_ptr);
    return it == ndarray_data_gen_.end() ? 0u : it->second;
  }
  void bump_ndarray_data_gen(void *devalloc_ptr) {
    ++ndarray_data_gen_[devalloc_ptr];
  }
  // Drop a per-DeviceAllocation entry. Called from `Ndarray::~Ndarray()` so the holder address can be reused by a
  // future allocation without inheriting the destroyed ndarray's stale generation. Leftover snapshots in
  // `per_task_ad_stack_cache_` / `llvm_per_task_ad_stack_cache_` referencing the dropped key fall back to gen=0
  // on the next lookup (their stored snapshot will not match), which forces a fresh sizer dispatch and self-heals.
  void erase_ndarray_data_gen(void *devalloc_ptr) {
    ndarray_data_gen_.erase(devalloc_ptr);
  }

 private:
  std::unordered_map<const SerializedSizeExpr *, SizeExprCacheEntry> size_expr_cache_;
  std::unordered_map<const void *, SpirvBytecodeCacheEntry> spirv_bytecode_cache_;
  std::unordered_map<const void *, PerTaskAdStackCacheEntry> per_task_ad_stack_cache_;
  std::unordered_map<const void *, LlvmPerTaskAdStackCacheEntry> llvm_per_task_ad_stack_cache_;
  std::unordered_map<int, uint64_t> snode_write_gen_;
  std::unordered_map<void *, uint64_t> ndarray_data_gen_;
};

// Evaluates a compile-time captured `SerializedSizeExpr` against the current field state of `prog` and the
// per-launch argument values in `ctx`, returning the concrete adstack capacity for this launch. Scalar i32/i64
// field loads are serviced by `SNodeRwAccessorsBank` (one reader-kernel dispatch each); ndarray-argument shapes
// are read from `ctx->get_struct_arg<int64>`; constants and arithmetic are folded in plain C++; `MaxOverRange`
// enumerates its range and takes the max of the body expression across the bound variable. Returns -1 when the
// expression is empty (no symbolic bound captured), signalling to the caller to use the compile-time fallback.
int64_t evaluate_adstack_size_expr(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx);

// RAII guard opening a thread-local read-cache scope. Every nested `evaluate_adstack_size_expr` running inside the
// scope shares one cache, so repeated `(snode_id, indices)` reads share a single reader-kernel dispatch. Place around
// any block that calls `evaluate_adstack_size_expr` more than once back-to-back.
class SizeExprLaunchScope {
 public:
  SizeExprLaunchScope();
  ~SizeExprLaunchScope();
  SizeExprLaunchScope(const SizeExprLaunchScope &) = delete;
  SizeExprLaunchScope &operator=(const SizeExprLaunchScope &) = delete;

 private:
  bool owns_;
};

// Flattens every alloca's `SerializedSizeExpr` tree into the device-readable bytecode defined in
// `quadrants/ir/adstack_size_expr_device.h` and returns the raw bytes ready to upload to a device scratch buffer.
// Two transforms happen at encoding time:
//
//   1. Pre-substitution of host-resolvable subtrees. Any subtree whose leaves consist only of `Const`,
//      `BoundVariable`, `FieldLoad`, and `ExternalTensorShape` nodes - i.e. nothing that requires an
//      on-device pointer dereference - is collapsed to a single `Const` node by running the existing host
//      evaluator over it. This routes `FieldLoad` through `SNodeRwAccessorsBank::read_int` (which itself
//      handles device-to-host via a tiny reader kernel on GPU) and `ExternalTensorShape` through the kernel
//      arg buffer that the host just wrote, so the device interpreter in `runtime.cpp` never has to walk
//      an SNode tree or index into `args_type` - it only has to handle arithmetic plus
//      `ExternalTensorRead`, which is the one leaf kind that actually needs device-resident memory.
//   2. `arg_buffer_offset` precomputation. Every surviving `ExternalTensorRead` carries the byte offset into
//      `RuntimeContext::arg_buffer` where the referenced ndarray's data pointer lives, resolved here against
//      `ctx->args_type->get_element_offset({arg_id, DATA_PTR_POS_IN_NDARRAY})`. The device interpreter does
//      a direct `*(void **)(arg_buffer + offset)` to fetch the ndarray pointer at launch time - no map
//      lookup, no `LaunchContextBuilder` touches from device code.
//
// Mixed subtrees that contain both an `ExternalTensorRead` and a `FieldLoad` are rejected with a hard error:
// the device interpreter does not support on-device SNode access, so a `FieldLoad` that cannot be lifted out
// to a host-resolvable `Const` has nowhere to run. The grammar today does not emit this combination and no
// user kernel has been observed to do so; the hard error pins the assumption so a future regression cannot
// slip past.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode(const AdStackSizingInfo &ad_stack,
                                                              Program *prog,
                                                              LaunchContextBuilder *ctx);

// SPIR-V-flavour encoder. Same transforms as the LLVM variant, but sources per-stack metadata from
// `TaskAttributes::AdStackSizingAttribs::allocas` (each entry has a `HeapKind` - `Float = 0`, `Int = 1` -
// that routes the stack onto the `AdStackHeapFloat` or `AdStackHeapInt` backing buffer on the host). The
// `heap_kind` field of each `AdStackSizeExprDeviceStackHeader` carries that selector into the shader; the
// shader splits the running-offset / stride computation into a float accumulator and an int accumulator so
// the output metadata buffer matches the layout the main kernel already reads today:
// `[stride_float, stride_int, (offset_i, max_size_i)*]`. The `entry_size_bytes` field is set to 1 on the
// SPIR-V path because the backing buffers are element-indexed (f32 / i32) rather than byte-indexed and the
// shader multiplies by `2` only for the `Float` heap (primal + adjoint interleaved) - see the running-offset
// arithmetic in `GfxRuntime::launch_kernel` for the convention this matches.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode_for_spirv(
    const spirv::TaskAttributes::AdStackSizingAttribs &ad_stack,
    Program *prog,
    LaunchContextBuilder *ctx);

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
// CPU launcher overload: per-task snode_writes / arr_writes are stored as separate parallel vectors on the launcher
// `Context` rather than as `OffloadedTask` clones, for legacy reasons documented in the CPU `Context` struct.
void bump_writes_for_kernel_llvm(Program *prog,
                                 LaunchContextBuilder *ctx,
                                 const std::vector<std::vector<int>> &snode_writes_per_task,
                                 const std::vector<std::vector<int>> &arr_writes_per_task);
void bump_writes_for_kernel_spirv(
    Program *prog,
    LaunchContextBuilder *ctx,
    const std::vector<spirv::TaskAttributes> &task_attribs,
    const std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>> &arr_access);

}  // namespace quadrants::lang
