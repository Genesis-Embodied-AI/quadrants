#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
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

// Adstack-specific state owned by `Program` and routed through `program->adstack_cache().method(...)`. Holds two
// orthogonal pieces:
//   1. The per-task adstack-sizer metadata caches (SPIR-V + LLVM-GPU), the encoded SPIR-V bytecode cache, the
//      per-launch SizeExpr-eval result cache, and the per-snode / per-DeviceAllocation generation counters that
//      drive precise invalidation.
//   2. The adstack-overflow identity registry + diagnostic classifier that the codegen-emitted overflow path
//      reads through (`Program::launch_kernel` populates `DiagnoseLaunchSnapshot`; the registry maps task ids
//      to kernel + offload-task identities + per-stack capacities, and `diagnose_adstack_overflow` runs the
//      synchronous sizer rerun against the captured snapshot to classify the failure mode).
// Both pieces are adstack-internal and lived in `Program` historically; consolidating them here keeps the
// `Program` surface focused on cross-feature program state.
class AdStackCache {
 public:
  // Back-reference to `Program` is used by the diagnose path to reach `evaluate_adstack_size_expr` /
  // `evaluate_adstack_size_expr_for_diagnose` (free functions that take `Program *`) and by the registry methods
  // to access `get_compute_device()` for `Device::map`-based ndarray reads. Stored as a raw pointer because
  // `AdStackCache` is owned by `Program` and shares its lifetime - the back-ref cannot dangle.
  explicit AdStackCache(Program *prog) : prog_(prog) {
  }

  // One input read observed during a `evaluate_adstack_size_expr` walk. The cache entry records these so a subsequent
  // lookup re-reads the same inputs and compares to `observed_value`; a single mismatch forces a full re-walk.
  // `observed_gen` snapshots `snode_write_gen` (FieldLoadObs) or `ndarray_data_gen` (ExternalReadObs) at record
  // time. The replay walk uses it as a fast-path short-circuit: if the gen counter has not advanced, the value
  // cannot have changed and the dispatch (reader kernel for SNode reads, device-pointer deref for ndarray reads)
  // is skipped. ExternalShapeObs reads the args buffer per launch (cheap host memory access), so it does not need
  // a gen and leaves this field at 0.
  struct SizeExprReadObservation {
    enum Kind : uint8_t { FieldLoadObs, ExternalShapeObs, ExternalReadObs };
    Kind kind;
    int snode_id;
    std::vector<int> indices;
    std::vector<int> arg_id_path;
    int arg_shape_axis;
    int prim_dt;
    int64_t observed_value;
    uint64_t observed_gen{0};
    void *observed_devalloc{nullptr};
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
  void invalidate_size_expr() {
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
  void invalidate_spirv_bytecode() {
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
  void invalidate_per_task_ad_stack() {
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
  void invalidate_llvm_per_task_ad_stack() {
    llvm_per_task_ad_stack_cache_.clear();
  }

  // Per-spec output cache for the max reducer. Keyed by `(registry_id, stack_id, mor_node_idx)` packed into a 64-bit
  // key (low 32 bits = `registry_id`, mid 16 bits = `stack_id`, high 16 bits = `mor_node_idx`). The recognizer caps
  // both `stack_id` and `mor_node_idx` well below 2^16 (per-task adstack count and per-stack node count are both
  // O(10s)), so the packing is collision-free. Same observation-walk dependency tracking as `try_size_expr_cache_hit`:
  // entries record the body's `ExternalTensorRead` reads plus the `begin` / `end` subtree's leaves; the next launch
  // re-walks observations and short-circuits on a generation match.
  struct MaxReducerCacheEntry {
    int64_t result;
    std::vector<SizeExprReadObservation> reads;
  };
  bool try_max_reducer_cache_hit(uint32_t registry_id,
                                 int32_t stack_id,
                                 int32_t mor_node_idx,
                                 LaunchContextBuilder *ctx,
                                 int64_t &out_result);
  void record_max_reducer_eval(uint32_t registry_id,
                               int32_t stack_id,
                               int32_t mor_node_idx,
                               int64_t result,
                               std::vector<SizeExprReadObservation> reads);
  void invalidate_max_reducer() {
    max_reducer_cache_.clear();
  }
  // Read-only accessor for the observations recorded for a captured spec. Returns `nullptr` when the spec is not
  // currently in the cache. Used by the bytecode encoder to thread the max-reducer body reads into the
  // `spirv_bytecode_cache_` entry's observation list, so a mutation to the gating ndarray invalidates the
  // bytecode cache (the encoder walks the post-substitution tree where the body is already a `Const` and would
  // otherwise miss the underlying ndarray dependency).
  const std::vector<SizeExprReadObservation> *lookup_max_reducer_reads(uint32_t registry_id,
                                                                       int32_t stack_id,
                                                                       int32_t mor_node_idx) const;
  // Monotone counter, incremented once per `record_max_reducer_eval` call. Reset only by the surrounding test
  // harness via `reset_max_reducer_dispatch_count`. Used by the regression tests to pin the cache short-circuit:
  // a second launch with unchanged inputs must not advance the counter, and a host mutation must.
  uint64_t max_reducer_dispatch_count() const {
    return max_reducer_dispatch_count_;
  }
  void reset_max_reducer_dispatch_count() {
    max_reducer_dispatch_count_ = 0;
  }

  // Bulk-invalidate just the per-task adstack metadata caches on the overflow raise path. The
  // `size_expr_cache_` and `spirv_bytecode_cache_` are intentionally NOT cleared: they self-validate via per-read
  // observation walks on the next lookup, so a DLPack-bypass mutation surfaces there as a normal observation
  // mismatch and triggers a fresh evaluation without explicit eviction. The per-task metadata caches need a
  // force-drop because their gen-counter snapshots match when the user's mutation bypassed our tracking.
  // Invalidation is bulk (every task) rather than targeted (just the offender) because a single shared DLPack /
  // torch view can back multiple tasks in the same kernel queue: targeted invalidation would let the next launch
  // hit a stale entry on a different task that reads the same now-mutated tensor and overflow again. Also evicts
  // the max-reducer cache so a stale-cache overflow auto-recovers across all four cache layers.
  void invalidate_all_per_task() {
    invalidate_per_task_ad_stack();
    invalidate_llvm_per_task_ad_stack();
    invalidate_max_reducer();
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

  // -----------------------------------------------------------------------------------------------------------
  // Adstack-overflow identity registry + diagnostic classifier
  // -----------------------------------------------------------------------------------------------------------
  // Codegen registers each `OffloadedTask::ad_stack` once per kernel compilation and bakes the assigned id as
  // an immediate into the lazy-claim overflow path; on overflow the codegen emits `cmpxchg(0, id)` against the
  // pinned-host task-id slot. The host raise site reads the slot and routes through
  // `diagnose_adstack_overflow_message(id)` to look up the kernel name, task index, and per-stack metadata for
  // an enriched error message. Pointer ownership stays with `OffloadedTask`; entries are added but not removed
  // - the registry size is bounded by the number of adstack-bearing tasks compiled in the program's lifetime,
  // typically dozens. The diagnose path NEVER dereferences `identity_key`; all size-expression data is stored
  // inline (`size_exprs`) so the entry is self-contained and immune to lifetime issues from the underlying
  // `AdStackSizingInfo` (LLVM) / `AdStackSizingAttribs` (SPIR-V) struct moves.
  struct AdStackSizingInfoEntry {
    const void *identity_key{nullptr};
    std::string kernel_name;
    int task_id_in_kernel{0};
    std::vector<int> allocated_max_sizes;
    std::vector<SerializedSizeExpr> size_exprs;
  };
  uint32_t register_adstack_sizing_info(const void *identity_key,
                                        const std::string &kernel_name,
                                        int task_id_in_kernel,
                                        std::vector<int> allocated_max_sizes,
                                        std::vector<SerializedSizeExpr> size_exprs);
  // Refresh just the `size_exprs` snapshot in an existing registry entry. Used by the LLVM launcher on the first
  // launch of a task whose codegen-time registration could not capture size_exprs (the codegen-time
  // `current_task->ad_stack` had not yet been finalized). No-op for `id == 0` and ids outside the registry range.
  void update_adstack_sizing_info_size_exprs(uint32_t id, std::vector<SerializedSizeExpr> size_exprs);
  // Returns a *copy* of the registry entry (not a pointer into the underlying vector) so the caller can safely
  // hold the data across operations that might trigger another `register_adstack_sizing_info` and grow / reallocate
  // the registry vector (e.g. `evaluate_adstack_size_expr` dispatching a reader kernel that compiles a fresh
  // task). Returns `std::nullopt` for the sentinel id `0` and for out-of-range ids.
  std::optional<AdStackSizingInfoEntry> lookup_adstack_sizing_info(uint32_t id) const;
  // Format a diagnostic message for an overflow signal. `task_id` is the value read from the pinned-host task-id
  // slot (0 if no thread overflowed; otherwise the registry id of the first overflowing task). The `message`
  // field is embedded into the `QuadrantsAssertionError` raised at the poll site. The `confirmed_invalid_cache`
  // field is true only when the synchronous sizer rerun classified the failure as a stale-cache /
  // DLPack-bypass case (`required > allocated` for at least one stack with every leaf resolved against the
  // captured launch snapshot); the caller (LLVM `check_adstack_overflow` / SPIR-V `GfxRuntime::synchronize`)
  // uses it to decide whether to bulk-invalidate the per-task metadata caches so the next launch auto-recovers.
  // We deliberately do NOT invalidate on Unknown / Quadrants-bug because invalidating would mask sizer bugs and
  // could let a never-confirmed cause silently retry against a possibly-broken cache.
  struct AdStackOverflowDiagnosis {
    std::string message;
    bool confirmed_invalid_cache{false};
  };
  AdStackOverflowDiagnosis diagnose_adstack_overflow(uint32_t task_id) const;
  // Convenience wrapper that returns just the message string; production code uses `diagnose_adstack_overflow`
  // to also act on the confirmed-cause signal.
  std::string diagnose_adstack_overflow_message(uint32_t task_id) const;

  // Snapshot of the most recent launch's context fields needed by `diagnose_adstack_overflow` to resolve
  // ndarray-bound `SizeExpr` leaves (`ExternalTensorRead` / `ExternalTensorShape`) at error time, when the
  // original `LaunchContextBuilder` is gone. Captured at the top of `Program::launch_kernel` BEFORE the
  // launcher rewrites `array_ptrs` (the CPU launcher's `set_host_accessible_ndarray_ptrs` overwrites the
  // `DeviceAllocation *` entry with a raw host pointer; capturing earlier keeps the original handle so the
  // diagnose path can use the unified `Device::map` API instead of trusting backend-specific semantics).
  //
  // Design choice (vs. re-dispatching the on-device sizer at diagnose time): `Device::map` is virtual on
  // every backend (CPU / CUDA / AMDGPU / Vulkan / Metal), so this snapshot-plus-map approach gets backend
  // parity for free without re-entering the launcher's pipeline-setup machinery (compute pipelines /
  // descriptor sets / command buffers / sync fences). The diagnose path stays out of the launch lifecycle.
  struct DiagnoseLaunchSnapshot {
    bool valid{false};
    // arg_id -> ctx->array_ptrs[(arg_id, DATA_PTR_POS_IN_NDARRAY)]. For `kNone` numpy passthrough this is a
    // raw host pointer. For `kNdarray` (qd.ndarray) this is a `DeviceAllocation *` handle the diagnose path
    // dereferences via `Device::map`. Captured before the CPU launcher's `set_host_accessible_ndarray_ptrs`
    // overwrite so the handle is uniform across backends.
    std::unordered_map<int, void *> data_ptrs;
    std::unordered_map<int, LaunchContextBuilder::DevAllocType> dev_alloc_types;
    // Pre-extracted ndarray shapes (`ctx->get_struct_arg_host<int32_t>({arg_id, SHAPE_POS, axis})`) so the
    // diagnose evaluator does not need a live `LaunchContextBuilder` to resolve `ExternalTensorShape` or
    // multi-axis `ExternalTensorRead` strides.
    std::unordered_map<int, std::vector<int32_t>> shapes;
  };
  // Capture the per-launch fields the diagnose evaluator needs (see `DiagnoseLaunchSnapshot`'s definition for
  // the design rationale and field-by-field semantics). Called eagerly from `Program::launch_kernel` only on
  // backends where the launch ctx is gone by the time overflow is detected (SPIR-V at `synchronize`); on LLVM
  // backends the per-launch overflow poll runs while ctx is still in scope, so we stash the ctx pointer with
  // `set_pending_launch_ctx` and let `diagnose_adstack_overflow` capture lazily on the (rare) overflow path.
  void capture_diagnose_snapshot(const LaunchContextBuilder &ctx);
  // Lazy-capture handoff: `Program::launch_kernel` on LLVM backends sets this to the in-scope ctx before
  // forwarding into the launcher and clears it after the per-launch overflow poll returns. If the poll fires,
  // `diagnose_adstack_overflow` reads the pointer and captures the snapshot just in time. Stored as a raw
  // pointer because it is transient per-launch and never outlives the call frame that set it.
  void set_pending_launch_ctx(const LaunchContextBuilder *ctx) {
    pending_launch_ctx_ = ctx;
  }
  // Read-only accessor for the latest snapshot, used by `diagnose_adstack_overflow` to resolve ndarray-bound
  // size_expr leaves. Returns `nullptr` when no launch has happened yet (e.g. a freshly constructed `Program`
  // hits `synchronize` during teardown without a prior kernel launch).
  const DiagnoseLaunchSnapshot *get_diagnose_snapshot() const;

 private:
  Program *prog_{nullptr};
  std::unordered_map<const SerializedSizeExpr *, SizeExprCacheEntry> size_expr_cache_;
  std::unordered_map<const void *, SpirvBytecodeCacheEntry> spirv_bytecode_cache_;
  std::unordered_map<const void *, PerTaskAdStackCacheEntry> per_task_ad_stack_cache_;
  std::unordered_map<const void *, LlvmPerTaskAdStackCacheEntry> llvm_per_task_ad_stack_cache_;
  // Max-reducer per-spec output cache. Key encoding: low 32 bits = `registry_id`, mid 16 bits =
  // `stack_id`, high 16 bits = `mor_node_idx`. See `try_max_reducer_cache_hit` for the contract and
  // `pack_max_reducer_key` in `adstack_size_expr_eval.cpp` for the packing helper.
  std::unordered_map<uint64_t, MaxReducerCacheEntry> max_reducer_cache_;
  // See `max_reducer_dispatch_count` for the contract. Bumped at every `record_max_reducer_eval` call (i.e. once
  // per cache miss that fired a real dispatch); cache hits do not bump it.
  uint64_t max_reducer_dispatch_count_{0};
  std::unordered_map<int, uint64_t> snode_write_gen_;
  std::unordered_map<void *, uint64_t> ndarray_data_gen_;

  // Adstack-overflow identity registry storage. Index 0 is reserved as the "no overflow" sentinel so the
  // codegen-emitted `cmpxchg(0, id)` cleanly distinguishes "task id recorded" from "slot still clean". The
  // reverse lookup map (keyed by `identity_key`) keeps `register_adstack_sizing_info` idempotent across
  // re-launches of the same kernel.
  std::vector<AdStackSizingInfoEntry> adstack_sizing_info_registry_{AdStackSizingInfoEntry{}};
  std::unordered_map<const void *, uint32_t> adstack_sizing_info_id_by_ptr_;
  mutable std::mutex adstack_sizing_info_registry_mutex_;
  // Latest captured launch context snapshot for the diagnose path's ndarray-bound leaf resolution. See
  // `DiagnoseLaunchSnapshot`'s comment above for why we capture in `Program::launch_kernel` before the launcher
  // forwards.
  // Single-threaded by construction: `capture_diagnose_snapshot` runs from `Program::launch_kernel` (Python
  // launcher thread) and `get_diagnose_snapshot` runs from `diagnose_adstack_overflow` on the same thread; no
  // mutex needed. The codegen-time identity registry above keeps its mutex because it is hit from compilation
  // worker threads.
  DiagnoseLaunchSnapshot diagnose_snapshot_;
  // Transient ctx handoff for the lazy LLVM capture path. See `set_pending_launch_ctx`.
  const LaunchContextBuilder *pending_launch_ctx_{nullptr};
};

// Evaluates a compile-time captured `SerializedSizeExpr` against the current field state of `prog` and the
// per-launch argument values in `ctx`, returning the concrete adstack capacity for this launch. Scalar i32/i64
// field loads are serviced by `SNodeRwAccessorsBank` (one reader-kernel dispatch each); ndarray-argument shapes
// are read from `ctx->get_struct_arg<int64>`; constants and arithmetic are folded in plain C++; `MaxOverRange`
// enumerates its range and takes the max of the body expression across the bound variable. Returns -1 when the
// expression is empty (no symbolic bound captured), signalling to the caller to use the compile-time fallback.
int64_t evaluate_adstack_size_expr(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx);
// Variant of `evaluate_adstack_size_expr` that bypasses `size_expr_cache_`. Used by the host-eval branch of the
// per-task sizer when feeding a stack-local substituted tree (the `size_expr_cache_` is keyed by `SerializedSizeExpr
// *`, so a transient stack address would alias unrelated cache entries across launches and return wrong cached values).
// Callers that need cache-warmed evaluation should use `evaluate_adstack_size_expr` with the original tree's stable
// pointer.
int64_t evaluate_adstack_size_expr_no_cache(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx);

// Sub-tree variant of `evaluate_adstack_size_expr`: evaluates the subtree rooted at `node_idx` instead of the full
// tree's root. Used by the max-reducer launcher to host-resolve a captured spec's `begin` / `end`
// subtrees against the live ctx (The recognizer grammar guarantees both subtrees are closed-form, so the recursive
// evaluator never re-enters a `MaxOverRange`). Returns -1 when `node_idx` is out of range; -1 from a deeper
// host-eval failure propagates the same way as in the full-tree variant.
int64_t evaluate_adstack_size_expr_at_node(const SerializedSizeExpr &expr,
                                           int32_t node_idx,
                                           Program *prog,
                                           LaunchContextBuilder *ctx);

// Diagnose-time variant that evaluates the same `SerializedSizeExpr` against the captured
// `AdStackCache::DiagnoseLaunchSnapshot` rather than a live `LaunchContextBuilder`. Used by
// `AdStackCache::diagnose_adstack_overflow` to resolve `ExternalTensorRead` / `ExternalTensorShape` leaves at
// error time against the live (potentially mutated) ndarray contents, without needing the launch ctx that is
// gone by sync time on async backends. The cross-backend `Device::map(*allocation, &host_ptr)` path is the
// design pivot - see `AdStackCache::DiagnoseLaunchSnapshot`'s comment for the rationale (vs. re-dispatching
// the on-device sizer). Returns -1 if any leaf cannot be resolved (e.g. an arg_id missing from the snapshot,
// or an allocation whose `Device::map` fails); callers fall back to the static dual-cause body in that case.
int64_t evaluate_adstack_size_expr_for_diagnose(const SerializedSizeExpr &expr, Program *prog);

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
// Type alias for the max-reducer result map. Keyed by `(registry_id, stack_id, mor_node_idx)` packed via
// the same `pack_max_reducer_key` encoding `AdStackCache::try_max_reducer_cache_hit` uses, so a single map shared
// between the dispatch path and the substitution helper avoids re-packing at every lookup.
using MaxReducerResultMap = std::unordered_map<uint64_t, int64_t>;

// Mixed subtrees that contain both an `ExternalTensorRead` and a `FieldLoad` are rejected with a hard error:
// the device interpreter does not support on-device SNode access, so a `FieldLoad` that cannot be lifted out
// to a host-resolvable `Const` has nowhere to run. The grammar today does not emit this combination and no
// user kernel has been observed to do so; the hard error pins the assumption so a future regression cannot
// slip past.
std::vector<uint8_t> encode_adstack_size_expr_device_bytecode(
    const AdStackSizingInfo &ad_stack,
    Program *prog,
    LaunchContextBuilder *ctx,
    const MaxReducerResultMap &max_reducer_results = MaxReducerResultMap{});

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
    LaunchContextBuilder *ctx,
    const MaxReducerResultMap &max_reducer_results = MaxReducerResultMap{});

// extract a captured `MaxOverRange`'s body subtree from
// `expr` and emit it as a flat `[AdStackSizeExprDeviceNode x body_node_count][int32 x indices_count]` bytecode blob
// plus a parallel `[uint8_t]` byte buffer ready to upload to a device storage buffer. Reachable nodes are walked in
// post-order from `body_node_idx` and renumbered to dense `[0, body_node_count)` indices; referenced indices entries
// from `expr.indices_table` (the `idx_raw, elem_stride` pairs `kExternalTensorRead` reads) are copied into the same
// flat buffer at `body_node_count * sizeof(AdStackSizeExprDeviceNode)`. Returns the raw bytes plus
// `body_node_count` and `indices_count` so the caller can populate the matching `AdStackMaxReducerParams` /
// `LlvmAdStackMaxReducerDeviceParams` fields. The recognizer grammar guarantees the body subtree contains no
// `kMaxOverRange` / `kFieldLoad`, so the body interpreter only needs the small grammar set the SPIR-V max-reducer
// shader and the LLVM `runtime_eval_adstack_max_reduce` runtime function both implement.
//
// `arg_buffer_offset_resolver` resolves `(arg_id_path) -> byte_offset_in_arg_buffer` for `kExternalTensorRead` leaves.
// On the gfx caller path this is a closure over `LaunchContextBuilder::args_type::get_element_offset` (same path the
// SizeExpr device-bytecode encoder uses). On the LLVM caller path the resolver mirrors the per-task adstack sizer's
// arg-buffer-offset precomputation. Returns `-1` on resolution failure (caller should hard-error or skip the spec).
struct EncodedMaxReducerBody {
  std::vector<uint8_t> bytes;
  uint32_t body_node_count{0};
  uint32_t indices_count{0};
  // Reads observed during encoding: one entry per body leaf (`kExternalTensorRead`) and per begin/end leaf the caller
  // resolved separately. Used by `AdStackCache::record_max_reducer_eval` so the next launch can short-circuit on a
  // generation match. Caller fills in the begin/end observations and appends body observations from this list.
  std::vector<AdStackCache::SizeExprReadObservation> body_reads;
};
EncodedMaxReducerBody encode_max_reducer_body_bytecode(
    const SerializedSizeExpr &expr,
    int32_t body_node_idx,
    int32_t bound_var_id,
    const std::function<int32_t(const std::vector<int32_t> &arg_id_path)> &arg_buffer_offset_resolver);

// Snapshot the live ndarray data pointer + generation counter into each `ExternalReadObs` record. The encoder emits
// the observation skeleton (kind / arg_id_path / prim_dt) but cannot fill in the runtime-resolved `data_ptr` /
// `observed_gen` / `observed_value` because it has no `LaunchContextBuilder`. This helper closes that gap right
// before the max-reducer dispatch site calls `AdStackCache::record_max_reducer_eval`, so the next launch's
// `try_max_reducer_cache_hit` replay can fast-skip on a matching `ndarray_data_gen`. `observed_value` is recorded
// as `INT64_MIN` so the replay's gen-mismatch dereference path returns a value strictly greater than the recorded
// sentinel and forces the cache to invalidate; the cached max itself is stored in `MaxReducerCacheEntry::result`,
// not in any per-leaf observation.
void populate_max_reducer_body_observations(std::vector<AdStackCache::SizeExprReadObservation> &reads,
                                            LaunchContextBuilder *ctx,
                                            AdStackCache *cache);

// walk every per-stack `SerializedSizeExpr` in `size_exprs`
// post-order and return the list of `MaxOverRange` nodes the runtime can reduce in parallel via a dedicated
// max-reducer dispatch. Each returned spec references its alloca by `stack_id` (index into `size_exprs`) and its
// `MaxOverRange` by `mor_node_idx` (index into `size_exprs[stack_id].nodes`). Specs are returned in dependency
// order: deeper `MaxOverRange` nodes first so the runtime can substitute their results before evaluating outer
// nodes that depend on them. Grammar:
//   * `body` subtree references only `Const`, `ExternalTensorRead(arg, [BoundVariable(this_var_id)])`, and
//     `Add` / `Sub` / `Mul` / `Max` of those. Single index axis. Integer dtype on every leaf.
//   * `begin` and `end` subtrees reference only `Const`, `ExternalTensorShape`, `Add` / `Sub` / `Mul` / `Max`,
//     or another `MaxOverRange` already captured deeper in the same tree (becomes a `Const` after substitution).
// Anything outside the grammar is skipped silently; that `MaxOverRange` continues to fall through to the existing
// capped path (host hard-error when `QD_DEBUG_ADSTACK=1`, silent truncation otherwise).
std::vector<StaticAdStackMaxReducerSpec> recognize_adstack_max_reducer_specs(
    const std::vector<SerializedSizeExpr> &size_exprs);

// walk `expr.nodes`, replace every captured `MaxOverRange`
// node whose `(registry_id, stack_id, mor_node_idx)` is in `results` with a `Const` carrying the dispatched value.
// Other nodes (and their `operand_a` / `operand_b` / `body_node_idx` references) are copied through verbatim. The
// returned `SerializedSizeExpr` has `nodes.size() == expr.nodes.size()` (in-place substitution); operand indices in
// non-substituted nodes remain valid because the count is unchanged.
//
// Empty-input fast path: when no captured spec matches this `(registry_id, stack_id)` (computed by checking
// `results` against every `MaxOverRange` node in `expr`), return `expr` unchanged (the caller's reference into the
// per-stack tree stays valid). Use `SerializedSizeExpr` by value as the return so the caller can transparently swap
// the reference depending on whether substitution fired.
SerializedSizeExpr substitute_precomputed_max_over_range(const SerializedSizeExpr &expr,
                                                         uint32_t registry_id,
                                                         int32_t stack_id,
                                                         const MaxReducerResultMap &results);

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
