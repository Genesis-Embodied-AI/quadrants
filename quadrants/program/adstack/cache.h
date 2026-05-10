#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

class LaunchContextBuilder;
class Program;
struct AdStackSizingInfo;
class OffloadedTask;

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
  // lookup checks whether each recorded input is still consistent; a single mismatch forces a full re-walk.
  // FieldLoadObs / ExternalReadObs use the per-buffer gen counter (`snode_write_gen` / `ndarray_data_gen`) snapshotted
  // in `observed_gen` as the sole staleness signal: a gen-counter advance unconditionally invalidates the cache, even
  // if the read cells happen to be untouched. The bump invariant is therefore required: every kernel write,
  // `Ndarray.write` / `fill`, and `SNodeRwAccessorsBank` writer kernel must bump the matching gen counter for the
  // cache to stay correct. `observed_value` is unused for these kinds. ExternalShapeObs has no gen counter (shapes
  // live in launch-arg metadata, not buffer content) and falls back to value comparison against `observed_value`.
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
  // `spirv_bytecode_cache_` entry's observation list, so a mutation to the gating ndarray invalidates the bytecode
  // cache (the encoder walks the post-substitution tree where the body is already a `Const` and would otherwise miss
  // the underlying ndarray dependency).
  const std::vector<SizeExprReadObservation> *lookup_max_reducer_reads(uint32_t registry_id,
                                                                       int32_t stack_id,
                                                                       int32_t mor_node_idx) const;

  // Per-kernel-handle launch cache for the max-reducer dispatcher. Skips the per-spec hash-lookup-and-observation
  // -replay loop in `dispatch_max_reducers_for_tasks` when the kernel handle's dependency snapshot is unchanged since
  // the previous launch. Key is the address of the launcher's stable per-handle vector
  // (`KernelLauncher::contexts_[i].offloaded_tasks` for CUDA / AMDGPU, `ad_stacks` for the CPU launcher); the address
  // is reused on every launch of the same kernel handle so it serves as a stable identity. Stores the dispatched result
  // map plus a deduplicated `(snode_id, gen)` / `(arg_id, devalloc, gen)` snapshot covering every spec's observation
  // deps. The fast path replays the snapshot in O(distinct deps) and short-circuits on full match; this collapses the
  // steady-state per-launch cost from O(specs) hash lookups + observation replays + result-map rebuilds (the dominant
  // chunk of the regression introduced when the dispatcher first landed) to a small dependency walk and a single map
  // copy. Slow path falls through to the per-spec cache.
  struct ArgGenObservation {
    int arg_id;
    void *devalloc;
    uint64_t gen;
  };
  // Cache entry stores the dispatched result map via shared ownership so the per-launch fast path can repoint
  // `LlvmRuntimeExecutor::current_max_reducer_results_` to it without copying, and so a snapshot taken by
  // `publish_adstack_metadata` survives a recursive snode-reader-kernel reentry that may rewrite the executor's
  // transient. The result map is logically `const` once recorded; mutation should happen by replacing the entry, not by
  // reaching through the `shared_ptr`.
  struct MaxReducerLaunchCacheEntry {
    std::shared_ptr<const std::unordered_map<uint64_t, int64_t>> result;
    std::vector<std::pair<int, uint64_t>> snode_gens;
    std::vector<ArgGenObservation> arg_gens;
  };
  // Replay the recorded dependency snapshot for `launch_cache_key`. Returns true (and populates `out_result` with the
  // cache entry's `shared_ptr`) when every recorded dep still matches the live `(snode_write_gen, ndarray_data_gen)`;
  // false otherwise. A miss leaves the entry in place so a subsequent `record_max_reducer_launch_cache` can overwrite
  // it.
  bool try_max_reducer_launch_cache_hit(const void *launch_cache_key,
                                        LaunchContextBuilder *ctx,
                                        std::shared_ptr<const std::unordered_map<uint64_t, int64_t>> &out_result);
  // Aggregate every spec's observation deps (resolved via `lookup_max_reducer_reads`) into a deduplicated snapshot and
  // store `result` under `launch_cache_key`. Called at the end of every successful run of
  // `dispatch_max_reducers_for_tasks` (both full-cache-hit and any-cache-miss paths) so the next launch's
  // `try_max_reducer_launch_cache_hit` short-circuits the per-spec walk. `ctx` must be non-null. `result` is moved into
  // the cache entry; the caller should `make_shared` it once and pass the same `shared_ptr` down to
  // `LlvmRuntimeExecutor::current_max_reducer_results_` to avoid materialising a second copy.
  void record_max_reducer_launch_cache(const void *launch_cache_key,
                                       const std::vector<const AdStackSizingInfo *> &ad_stacks,
                                       std::shared_ptr<const std::unordered_map<uint64_t, int64_t>> result,
                                       LaunchContextBuilder *ctx);
  void invalidate_max_reducer_launch() {
    max_reducer_launch_cache_.clear();
  }
  // Monotone counter, incremented once per `record_max_reducer_eval` call. Reset only by the surrounding test harness
  // via `reset_max_reducer_dispatch_count`. Used by the regression tests to pin the cache short-circuit: a second
  // launch with unchanged inputs must not advance the counter, and a host mutation must.
  uint64_t max_reducer_dispatch_count() const {
    return max_reducer_dispatch_count_;
  }
  void reset_max_reducer_dispatch_count() {
    max_reducer_dispatch_count_ = 0;
  }

  // Bulk-invalidate just the per-task adstack metadata caches on the overflow raise path. The `size_expr_cache_` and
  // `spirv_bytecode_cache_` are intentionally NOT cleared: they self-validate via per-read observation walks on the
  // next lookup, so a DLPack-bypass mutation surfaces there as a normal observation mismatch and triggers a fresh
  // evaluation without explicit eviction. The per-task metadata caches need a force-drop because their gen-counter
  // snapshots match when the user's mutation bypassed our tracking. Invalidation is bulk (every task) rather than
  // targeted (just the offender) because a single shared DLPack / torch view can back multiple tasks in the same kernel
  // queue: targeted invalidation would let the next launch hit a stale entry on a different task that reads the same
  // now-mutated tensor and overflow again. Also evicts the max-reducer cache so a stale-cache overflow auto-recovers
  // across all four cache layers.
  void invalidate_all_per_task() {
    invalidate_per_task_ad_stack();
    invalidate_llvm_per_task_ad_stack();
    invalidate_max_reducer();
    invalidate_max_reducer_launch();
  }

  uint64_t snode_write_gen(int snode_id) const {
    auto it = snode_write_gen_.find(snode_id);
    return it == snode_write_gen_.end() ? 0u : it->second;
  }
  void bump_snode_write_gen(int snode_id) {
    ++snode_write_gen_[snode_id];
  }
  // One-way switch flipped true the first time any `record_*` populates a cache. Read by the per-launch
  // `bump_writes_for_kernel_*` helpers to short-circuit the per-arg / per-snode gen-counter bumps when no sizer,
  // per-task, or max-reducer cache entry exists; forward-only programs never record and therefore pay zero per-launch
  // bump-writes overhead on the ndarray path where the bump loop would otherwise touch every kernel-bound arg slot.
  // Not reset by `invalidate_*` because the cost of a stale `true` (a few extra map inserts per launch) is dwarfed by
  // the cost of repeated invalidate-then-record cycles, which never happen in steady state.
  bool has_any_recordings() const {
    return any_recordings_;
  }
  // Per-id refinement of `has_any_recordings()`: was `snode_id` ever named in a recorded observation's read set or in
  // a per-task `snode_gens` snapshot? Bump-writes calls in mixed programs (some forward kernels, some autodiff ones)
  // use this to skip `snode_write_gen_` updates for snodes that no cached entry depends on; the gen-counter for an
  // unobserved snode stays at 0 and the very first `record_*` on that snode records `observed_gen = 0`, which the
  // subsequent first observed-write bumps to 1 and invalidates as expected.
  bool is_snode_observed(int snode_id) const {
    return observed_snode_ids_.find(snode_id) != observed_snode_ids_.end();
  }
  // Has any recorded entry observed a value through `ExternalTensorRead` (i.e. depends on an ndarray's data content)?
  // Bump-writes consults this to skip the entire `array_ptrs.find` + `bump_ndarray_data_gen` loops when no cached entry
  // could depend on an ndarray, even if `has_any_recordings()` is true because some snode-only entry already exists.
  bool any_external_read_observed() const {
    return any_external_read_observed_;
  }
  // Per-devalloc refinement of `any_external_read_observed()`: was this exact `DeviceAllocation *` ever named by a
  // recorded ExternalReadObs or by a per-task `arg_gens` snapshot? Bump-writes uses this after resolving each arg's
  // devalloc to decide whether the bump itself can be skipped: if no cached entry depends on this devalloc, the
  // `bump_ndarray_data_gen` update is dead work because nothing reads back the advanced counter. Stale pointers
  // (Ndarrays that were destroyed and whose holder was reclaimed by a new allocation) at most cost an extra bump on
  // the recycled address; the cache entry that originally observed the freed pointer is self-invalidated on its next
  // lookup since `data_ptr == observed_devalloc` will fail against the new binding.
  bool is_devalloc_observed(void *devalloc_ptr) const {
    return observed_devalloc_ptrs_.find(devalloc_ptr) != observed_devalloc_ptrs_.end();
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
  // Re-populate the per-`Program` registry for offline-cache-loaded LLVM kernels. `registry_id` is now content-stable
  // (`fnv1a32(kernel_name + ":" + task_id_in_kernel)`) and serialised, so the dispatcher and metadata-publish gates
  // already see the right id without us touching anything; this loop exists purely to seed the `Program`-level registry
  // so `diagnose_adstack_overflow_message` can resolve the cmpxchg-recorded id on the rare overflow path. Idempotent
  // (same hash inputs yield the same id, so re-registration on every launch is a single map lookup) and only walks
  // tasks whose `ad_stack.max_reducer_specs` is non-empty (forward-only and reverse-mode-without-recognized
  // -MaxOverRange tasks pay zero overhead). Mirrors the SPIR-V `adstack_sizer_launch.cpp` runtime re-registration loop.
  void ensure_runtime_registry_ids_for_max_reducer(std::vector<OffloadedTask> &tasks);
  // Returns a *copy* of the registry entry (not a pointer into the underlying vector) so the caller can safely hold the
  // data across operations that might trigger another `register_adstack_sizing_info` and grow / reallocate the registry
  // vector (e.g. `evaluate_adstack_size_expr` dispatching a reader kernel that compiles a fresh task). Returns
  // `std::nullopt` for the sentinel id `0` and for out-of-range ids.
  std::optional<AdStackSizingInfoEntry> lookup_adstack_sizing_info(uint32_t id) const;
  // Cheap lookup gate for the per-launch re-registration loop in `ensure_runtime_registry_ids_for_max_reducer`. Returns
  // true when `identity_key` already maps to a registry entry, so the caller can skip the vector-building work
  // `register_adstack_sizing_info` would otherwise pay on every steady-state launch.
  bool is_adstack_sizing_info_registered(const void *identity_key) const;
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
  // Register every `FieldLoadObs` and `ExternalReadObs` entry of `reads` into `observed_snode_ids_` /
  // `any_external_read_observed_` / `observed_devalloc_ptrs_`. Called from each `record_*` site so the per-id gate
  // consulted by bump-writes sees the latest observation footprint before the next launch.
  void note_observations(const std::vector<SizeExprReadObservation> &reads);
  // Per-task variant: snapshots from `record_per_task_ad_stack` / `record_llvm_per_task_ad_stack` ship `snode_gens` /
  // `arg_gens` lists directly rather than `SizeExprReadObservation`s, so we route them through this overload to keep
  // the observation footprint in sync no matter which cache layer holds the entry. The arg_gens tuple shape matches
  // both `PerTaskAdStackCacheEntry` and `LlvmPerTaskAdStackCacheEntry`; the `MaxReducerLaunchCacheEntry` overload
  // below adapts its `ArgGenObservation` struct.
  void note_per_task_dependencies(const std::vector<std::pair<int, uint64_t>> &snode_gens,
                                  const std::vector<std::tuple<int, void *, uint64_t>> &arg_gens);
  void note_per_task_dependencies(const std::vector<std::pair<int, uint64_t>> &snode_gens,
                                  const std::vector<ArgGenObservation> &arg_gens);

  Program *prog_{nullptr};
  std::unordered_map<const SerializedSizeExpr *, SizeExprCacheEntry> size_expr_cache_;
  std::unordered_map<const void *, SpirvBytecodeCacheEntry> spirv_bytecode_cache_;
  std::unordered_map<const void *, PerTaskAdStackCacheEntry> per_task_ad_stack_cache_;
  std::unordered_map<const void *, LlvmPerTaskAdStackCacheEntry> llvm_per_task_ad_stack_cache_;
  // Max-reducer per-spec output cache. Key encoding: low 32 bits = `registry_id`, mid 16 bits = `stack_id`, high 16
  // bits = `mor_node_idx`. See `try_max_reducer_cache_hit` for the contract and `pack_max_reducer_key` in
  // `adstack_size_expr_eval.cpp` for the packing helper.
  std::unordered_map<uint64_t, MaxReducerCacheEntry> max_reducer_cache_;
  // See `max_reducer_dispatch_count` for the contract. Bumped at every `record_max_reducer_eval` call (i.e. once per
  // cache miss that fired a real dispatch); cache hits do not bump it.
  uint64_t max_reducer_dispatch_count_{0};
  // Per-kernel-handle launch-level result cache; see `try_max_reducer_launch_cache_hit` for the contract.
  std::unordered_map<const void *, MaxReducerLaunchCacheEntry> max_reducer_launch_cache_;
  std::unordered_map<int, uint64_t> snode_write_gen_;
  std::unordered_map<void *, uint64_t> ndarray_data_gen_;

  // Adstack-overflow identity registry storage. Keyed by the content-stable hash id assigned by
  // `register_adstack_sizing_info` (`fnv1a32(kernel_name + ":" + task_id_in_kernel)`); the same id is baked into the
  // codegen-emitted `cmpxchg(0, id)` so a host lookup of the slot the cmpxchg wrote resolves to the matching entry
  // across re-compiles, `Program` lifetimes, and offline-cache reloads. Id `0` is reserved for "not registered" - the
  // codegen short-circuits the cmpxchg in that case. Map (rather than vector) because the hash is non-contiguous and
  // may collide; collisions are resolved by linear-probing within the map (see `register_adstack_sizing_info` for the
  // loop). The reverse lookup map (keyed by `identity_key`) keeps `register_adstack_sizing_info` idempotent across
  // re-launches of the same kernel.
  std::unordered_map<uint32_t, AdStackSizingInfoEntry> adstack_sizing_info_registry_;
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
  // Backing storage for `has_any_recordings()`. See accessor doc for behavior and reset rules.
  bool any_recordings_{false};
  // Backing storage for `is_snode_observed()` / `any_external_read_observed()` / `is_devalloc_observed()`. All grow
  // monotonically: every `record_*` call routes its observation list (and per-task snapshot lists) through the
  // `note_*` helpers in cache.cpp so a freshly captured snapshot is reflected before the next
  // `bump_writes_for_kernel_*` call queries them. The devalloc set may hold stale pointers across Ndarray destruction
  // / reuse; see `is_devalloc_observed` for why that is correctness-safe.
  std::unordered_set<int> observed_snode_ids_;
  std::unordered_set<void *> observed_devalloc_ptrs_;
  bool any_external_read_observed_{false};
};

// Snapshot the live ndarray data pointer + generation counter into each `ExternalReadObs` record. The encoder emits the
// observation skeleton (kind / arg_id_path / prim_dt) but cannot fill in the runtime-resolved `data_ptr` /
// `observed_gen` / `observed_value` because it has no `LaunchContextBuilder`. This helper closes that gap right before
// the max-reducer dispatch site calls `AdStackCache::record_max_reducer_eval`, so the next launch's
// `try_max_reducer_cache_hit` replay can fast-skip on a matching `ndarray_data_gen`. `observed_value` is recorded as
// `INT64_MIN` so the replay's gen-mismatch dereference path returns a value strictly greater than the recorded sentinel
// and forces the cache to invalidate; the cached max itself is stored in `MaxReducerCacheEntry::result`, not in any
// per-leaf observation.
void populate_max_reducer_body_observations(std::vector<AdStackCache::SizeExprReadObservation> &reads,
                                            LaunchContextBuilder *ctx,
                                            AdStackCache *cache);

}  // namespace quadrants::lang
