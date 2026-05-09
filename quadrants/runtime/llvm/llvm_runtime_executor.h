#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#ifdef QD_WITH_LLVM

#include "quadrants/rhi/llvm/llvm_device.h"
#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/runtime/llvm/llvm_offline_cache.h"
#include "quadrants/runtime/llvm/snode_tree_buffer_manager.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/struct/snode_tree.h"
#include "quadrants/program/compile_config.h"

#include "quadrants/system/threading.h"

#define QD_RUNTIME_HOST
#include "quadrants/program/context.h"
#undef QD_RUNTIME_HOST

namespace quadrants::lang {

class ProgramImpl;
class LaunchContextBuilder;

namespace cuda {
class CudaDevice;
}  // namespace cuda

namespace amdgpu {
class AmdgpuDevice;
}  // namespace amdgpu

namespace cpu {
class CpuDevice;
}  // namespace cpu

class LlvmRuntimeExecutor {
 public:
  LlvmRuntimeExecutor(CompileConfig &config, KernelProfilerBase *profiler, quadrants::lang::ProgramImpl *program_impl);
  virtual ~LlvmRuntimeExecutor();
  /**
   * Initializes the runtime system for LLVM based backends.
   */
  void materialize_runtime(KernelProfilerBase *profiler, uint64 **result_buffer_ptr);

  // SNodeTree Allocation
  void initialize_llvm_runtime_snodes(const LlvmOfflineCache::FieldCacheData &field_cache_data, uint64 *result_buffer);

  // Ndarray and ArgPack Allocation
  DeviceAllocation allocate_memory_on_device(std::size_t alloc_size, uint64 *result_buffer);

  void deallocate_memory_on_device(DeviceAllocation handle);

  void check_runtime_error(uint64 *result_buffer);

  // Poll the runtime's adstack-overflow flag and raise if set. Unlike check_runtime_error, this runs
  // unconditionally at every synchronize() (not gated on `compile_config.debug`) because adstack overflow silently
  // corrupts gradients and we do not want to hide it. Safe to call before materialize_runtime() -- no-op when the
  // cached result buffer is not yet populated.
  void check_adstack_overflow();

  uint64_t *get_device_alloc_info_ptr(const DeviceAllocation &alloc);

  const CompileConfig &get_config() const {
    return config_;
  }

  QuadrantsLLVMContext *get_llvm_context();

  JITModule *create_jit_module(std::unique_ptr<llvm::Module> module);

  JITModule *get_runtime_jit_module();

  LLVMRuntime *get_llvm_runtime();

  Device *get_compute_device();

  LlvmDevice *llvm_device();

  void synchronize();

  bool use_device_memory_pool() {
    return use_device_memory_pool_;
  }

  // Publish the per-task adstack metadata into `LLVMRuntime.adstack_{per_thread_stride,offsets,max_sizes}` and size
  // the heap for the launch. Returns the `per_thread_stride * num_threads` byte size the heap was grown to (zero if
  // the task has no adstacks). When `ad_stack.size_exprs` is populated (cache miss after the `determine_ad_stack_size`
  // pre-pass) each entry is evaluated against the live field state; otherwise each alloca's compile-time
  // `max_size_compile_time` is used (cache-hit path - symbolic tree is currently not serialized into the offline
  // cache, so the compile-time fallback is all the launcher has).
  //
  // `device_runtime_context_ptr` is the device-side pointer the sizer kernel should receive as its `ctx` argument when
  // the GPU cannot dereference the host `&ctx->get_context()` pointer directly: the sizer kernel reads
  // `ctx->arg_buffer` on device to resolve `ExternalTensorRead` leaves against the kernel arg buffer. AMDGPU/HIP has no
  // UVA fallback and always faults with `hipErrorIllegalAddress`, so the AMDGPU launcher always stages a device copy of
  // `RuntimeContext` and passes that pointer. CUDA UVA covers pinned / CUDA-managed memory only; the plain
  // `std::make_unique<RuntimeContext>()` backing is neither, so dereferencing the host pointer requires HMM (system-
  // allocated memory), which is a driver + kernel capability advertised via
  // `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`. The CUDA launcher stages the device copy only when that attribute
  // reports unsupported (Turing without HMM, Windows, pre-535 Linux drivers) and passes `nullptr` otherwise to keep the
  // HMM-equipped fast path zero-overhead. CPU ignores this argument and runs the host evaluator in-process.
  std::size_t publish_adstack_metadata(const AdStackSizingInfo &ad_stack,
                                       std::size_t num_threads,
                                       LaunchContextBuilder *ctx,
                                       void *device_runtime_context_ptr = nullptr);

  // Accessor for the per-program `Program *` used by the launchers to bump per-snode and per-
  // DeviceAllocation generation counters in `Program::snode_write_gen_` / `ndarray_data_gen_`. Used
  // exclusively by the LLVM-GPU per-task adstack metadata cache (`record_llvm_per_task_ad_stack`)
  // so a kernel write to a SNode or ndarray a downstream `size_expr::FieldLoad` /
  // `ExternalTensorRead` reads invalidates cached metadata on the next launch. Returns nullptr in
  // C++-only test contexts where `program_impl_` was constructed without a parent Program.
  Program *get_program() const;

  // Allocate-on-demand and clear the per-kernel lazy-claim arrays:
  //   `adstack_row_counters[num_tasks]` = 0  (codegen-emitted LCA-block atomic-rmw target; each task counts its own
  //                                           LCA-block-reaching threads in slot `task_codegen_id`)
  //   `adstack_bound_row_capacities[num_tasks]` = UINT32_MAX  (clamp value the codegen-emitted bounds check reads;
  //                                                            a reducer can override per-task with a tighter count,
  //                                                            otherwise the default keeps the clamp inert)
  // Called by every kernel launcher (CPU / CUDA / AMDGPU) before dispatching the first task in a kernel so each task
  // observes a clean counter slot. Idempotent for `num_tasks <= adstack_lazy_claim_capacity_`; grows the arrays on
  // amortised doubling otherwise. Publishes the array pointers into `runtime->adstack_row_counters` /
  // `adstack_bound_row_capacities` via the cached field addresses on first call (and after every grow).
  void publish_adstack_lazy_claim_buffers(std::size_t num_tasks);

  // Per-task host-side evaluation of the captured `StaticAdStackBoundExpr`. Handles both ndarray-backed and
  // SNode-backed sources: ndarray sources read through `ctx->array_ptrs[arg_id, DATA_PTR_POS_IN_NDARRAY]` (populated by
  // the launcher); SNode sources read through `runtime->roots[snode_root_id] + snode_byte_base_offset +
  // gid * snode_byte_cell_stride` (resolved via the `LLVMRuntime_get_roots` STRUCT_FIELD_ARRAY getter). Walks
  // `[0, length)` evaluating the captured comparison + polarity, returns the count of gate-passing threads. Writes
  // that count into `runtime->adstack_bound_row_capacities[task_index]` so the codegen-emitted bounds clamp at the
  // float LCA-block claim site activates for legitimate over-claim, and so the float heap can be sized at
  // `count * stride_float` instead of the dispatched-threads worst case. Returns `UINT32_MAX` (meaning "no capacity
  // known, leave the default") when the field source is neither ndarray nor SNode, when the ndarray data pointer is
  // null, or when the SNode root pointer is unavailable.
  uint32_t publish_per_task_bound_count_cpu(std::size_t task_index,
                                            const AdStackSizingInfo &ad_stack,
                                            std::size_t length,
                                            LaunchContextBuilder *ctx);

  // Per-arch device-side reducer counterpart for CUDA / AMDGPU. Packs the captured `StaticAdStackBoundExpr` into a
  // small device-resident params buffer (h2d on-demand, reused across tasks via a grow-on-demand allocation) and
  // invokes `runtime_eval_static_bound_count` via the runtime JIT module. The device function walks the gating field
  // on-device (single-threaded; the runtime function dispatches as a 1x1x1 kernel launch) - reading from the ndarray
  // arg buffer for ndarray sources, or from `runtime->roots[snode_root_id]` for SNode sources - counts gate-passing
  // threads, and writes the count into `runtime->adstack_bound_row_capacities[task_index]`. The codegen-emitted clamp
  // at the float LCA-block claim site reads that slot back. No-op on backends other than CUDA / AMDGPU (CPU goes
  // through `publish_per_task_bound_count_cpu`).
  void publish_per_task_bound_count_device(std::size_t task_index,
                                           const AdStackSizingInfo &ad_stack,
                                           std::size_t length,
                                           LaunchContextBuilder *ctx,
                                           void *device_runtime_context_ptr);

  // Grow `runtime->adstack_heap_buffer_float` to at least `needed_bytes` and publish the new pointer / size into the
  // runtime struct via the cached field addresses. Amortised-doubling growth + release-deferred-until-next-launch
  // semantics: the previous `DeviceAllocationGuard` is dropped only after the new pointer has been published, so any
  // in-flight kernel still holding the old base on AMDGPU (where `dealloc_memory` does not synchronise) keeps reading
  // valid storage; the cross-launch invariant comes from `amdgpu::KernelLauncher::launch_llvm_kernel`'s tail
  // `hipFree(context_pointer)` synchronising with all kernels launched during that call.
  void ensure_adstack_heap_float(std::size_t needed_bytes);

  // Mirror of `ensure_adstack_heap_float` for the int / u1 heap. Sized at `num_threads * stride_int` worst case (every
  // dispatched thread's int allocas - loop counters, branch flags - fit in the eager `linear_tid * stride_int + offset`
  // layout). Independent grow-on-demand from the float heap.
  void ensure_adstack_heap_int(std::size_t needed_bytes);

  // Read back the per-task gate-passing count the reducer wrote into `runtime->adstack_bound_row_capacities[
  // task_index]` and size `runtime->adstack_heap_buffer_float` to `count * per_thread_stride_float`. On CPU the
  // capacity slot is host memory so the readback is a direct load; on CUDA / AMDGPU it's a small DtoH per task. Falls
  // back to `num_threads * per_thread_stride_float` (the codegen worst case) when the slot still holds UINT32_MAX (no
  // reducer ran for this task) or the task did not capture a `bound_expr`. Called by every kernel launcher (CPU / CUDA
  // / AMDGPU) per task between `publish_per_task_bound_count_{cpu,device}` and the main task dispatch so the float heap
  // is sized exactly to the reducer's count instead of the dispatched-threads worst case.
  void ensure_per_task_float_heap_post_reducer(std::size_t task_index,
                                               const AdStackSizingInfo &ad_stack,
                                               std::size_t num_threads,
                                               LaunchContextBuilder *ctx);

  // Max-reducer dispatch on LLVM. For each captured `StaticAdStackMaxReducerSpec` across every task in `tasks`, hits
  // `AdStackCache::try_max_reducer_cache_hit` first; on miss h2d-copies the params blob + body bytecode and invokes
  // `runtime_eval_adstack_max_reduce` via the runtime JIT as a grid-strided launch with an `atomic_max_i64` reduction.
  // CUDA and AMDGPU only; on CPU the recognizer is skipped at codegen time so this path runs zero dispatches. The
  // result map (keyed by `(registry_id, stack_id, mor_node_idx)` packed via the same encoding the gfx variant uses)
  // lands in `current_max_reducer_results_` for the per-task `publish_adstack_metadata` loop to pick up. Caller invokes
  // this BEFORE the per-task `publish_adstack_metadata` loop. Returns `void` because callers consume the result through
  // `current_max_reducer_results_` rather than the return value; this keeps the post-cache-hit fast path free of
  // result-map copies.
  void dispatch_max_reducers_for_tasks(const std::vector<AdStackSizingInfo> &ad_stacks,
                                       LaunchContextBuilder *ctx,
                                       void *device_runtime_context_ptr);
  // Convenience overload that extracts each task's `ad_stack` and forwards to the primary entry point. Lets the CUDA /
  // AMDGPU per-arch launchers call into the dispatcher with the `OffloadedTask` list they already hold, without each
  // launcher copy-pasting the per-task `ad_stack` extraction loop. Forwards a pointer-view of the per-task `ad_stack`s
  // to `dispatch_max_reducers_impl` (no deep copy of the `AdStackSizingInfo`s).
  void dispatch_max_reducers_for_tasks(const std::vector<OffloadedTask> &tasks,
                                       LaunchContextBuilder *ctx,
                                       void *device_runtime_context_ptr);

  // Return (and lazily cache) the device pointer to `runtime->temporaries`, the global temporary buffer backing
  // `GlobalTemporaryStmt` loads and stores. GPU kernel launchers use this to read back dynamic range_for bounds (begin
  // / end i32 values at known byte offsets) via a host-side DtoH memcpy when sizing the adstack heap. Cached because
  // `runtime->temporaries` is assigned once during `runtime_initialize` and never rebound.
  void *get_runtime_temporaries_device_ptr();

 private:
  /* ----------------------- */
  /* ---- Adstack helpers --- */
  /* ----------------------- */
  // Shared implementation for both `dispatch_max_reducers_for_tasks` overloads. Takes a stable per-kernel-handle
  // `launch_cache_key` (the address of the caller's `tasks` / `ad_stacks` vector) and a non-owning pointer view of the
  // per-task `AdStackSizingInfo`s. Avoids the per-launch deep copy of `AdStackSizingInfo` the OffloadedTask overload
  // used to do, and short-circuits on a kernel-level dependency-fingerprint hit before walking specs (logic factored
  // into `AdStackCache::try_max_reducer_launch_cache_hit` / `record_max_reducer_launch_cache`). Result lands in
  // `current_max_reducer_results_`; returns `void` to keep the cache-hit fast path free of result-map copies.
  void dispatch_max_reducers_impl(const void *launch_cache_key,
                                  const std::vector<const AdStackSizingInfo *> &ad_stacks,
                                  LaunchContextBuilder *ctx,
                                  void *device_runtime_context_ptr);

  /* ----------------------- */
  /* ------ Allocation ----- */
  /* ----------------------- */
  template <typename T>
  T fetch_result(int i, uint64 *result_buffer) {
    return quadrants_union_cast_with_different_sizes<T>(fetch_result_uint64(i, result_buffer));
  }

  template <typename T>
  T fetch_result(char *result_buffer, int offset) {
    return *(T *)(result_buffer + offset);
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id);

  void fill_ndarray(const DeviceAllocation &alloc, std::size_t size, uint32_t data);

  void *preallocate_memory(std::size_t prealloc_size, DeviceAllocationUnique &devalloc);
  void preallocate_runtime_memory();

  /* ------------------------- */
  /* ---- Runtime Helpers ---- */
  /* ------------------------- */
  void print_list_manager_info(void *list_manager, uint64 *result_buffer);
  void print_memory_profiler_info(std::vector<std::unique_ptr<SNodeTree>> &snode_trees_, uint64 *result_buffer);

  template <typename T, typename... Args>
  T runtime_query(const std::string &key, uint64 *result_buffer, Args &&...args) {
    QD_ASSERT(arch_uses_llvm(config_.arch));

    auto runtime = get_runtime_jit_module();
    runtime->call<void *>("runtime_" + key, llvm_runtime_, std::forward<Args>(args)...);
    return quadrants_union_cast_with_different_sizes<T>(
        fetch_result_uint64(quadrants_result_buffer_runtime_query_id, result_buffer));
  }

  /* -------------------------- */
  /* ------ Member Access ----- */
  /* -------------------------- */
  void finalize();

  uint64 fetch_result_uint64(int i, uint64 *result_buffer);
  void destroy_snode_tree(SNodeTree *snode_tree);
  std::size_t get_snode_num_dynamically_allocated(SNode *snode, uint64 *result_buffer);

  void init_runtime_jit_module(std::unique_ptr<llvm::Module> module);

 private:
  CompileConfig &config_;

  std::unique_ptr<QuadrantsLLVMContext> llvm_context_{nullptr};
  std::unique_ptr<JITSession> jit_session_{nullptr};
  JITModule *runtime_jit_module_{nullptr};
  void *llvm_runtime_{nullptr};
  // Non-owning cache of the Program-owned result buffer so internal polls (adstack overflow, etc.) can be
  // invoked from `synchronize()` without threading the pointer through the public API. Ownership stays with
  // `Program` for its lifetime; reallocating or repointing `Program::result_buffer` mid-run would invalidate
  // this cache, so avoid that.
  uint64 *result_buffer_cache_{nullptr};

  // Pinned host slot for the adstack overflow flag. The kernel-side `stack_push` writes through the runtime's
  // device-mapped address (`adstack_overflow_flag_dev_ptr_` published into `LLVMRuntime` at materialise time);
  // the host polls `adstack_overflow_flag_host_ptr_` directly with a relaxed atomic exchange - no DtoH, no JIT call,
  // no sync drain. Backends:
  //   - CPU LLVM: plain malloc; host and device addresses are identical.
  //   - CUDA: `cuMemAllocHost_v2` plus `cuMemHostGetDevicePointer` for the device-mapped address.
  //   - AMDGPU: `hipHostMalloc(0)` plus the HIP equivalent.
  // Required hardware envelope (Pascal+ / GFX9+) matches what the existing pinned-host H2D-async pattern in
  // `llvm_adstack_lazy_claim.cpp` already needs.
  int64_t *adstack_overflow_flag_host_ptr_{nullptr};
  void *adstack_overflow_flag_dev_ptr_{nullptr};

  // Companion task-id slot. Codegen emits `cmpxchg(0, id)` from the lazy-claim overflow path; the first
  // overflowing thread's `Program::adstack_sizing_info_registry_` id sticks. Host reads the slot at the
  // raise site for diagnostic message generation. Same allocation strategy as the flag above; on a fresh
  // page so neither write contends with the flag's atomic OR.
  int64_t *adstack_overflow_task_id_host_ptr_{nullptr};
  void *adstack_overflow_task_id_dev_ptr_{nullptr};

  std::unique_ptr<ThreadPool> thread_pool_{nullptr};
  std::shared_ptr<Device> device_{nullptr};

  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager_{nullptr};
  std::unordered_map<int, DeviceAllocation> snode_tree_allocs_;
  DeviceAllocationUnique preallocated_runtime_objects_allocs_ = nullptr;
  DeviceAllocationUnique preallocated_runtime_memory_allocs_ = nullptr;
  std::unordered_map<DeviceAllocationId, DeviceAllocation> allocated_runtime_memory_allocs_;

  // Split-layout float heap: dedicated slab holding only the f32 adstack rows for tasks that captured a `bound_expr`.
  // Sized by the launcher at `min(num_threads, max_bound_capacity) * max_stride_float` instead of the
  // dispatched-threads worst case, so workloads where the gating predicate matches few threads (sparse-grid MPM, masked
  // update kernels) shrink the float storage proportionally. Independent grow-on-demand from the combined heap; the
  // codegen-emitted `heap_float + row_id_var * stride_float + offset` formula reads from
  // `runtime->adstack_heap_buffer_float` (and `_size_float`) which the host writes via the cached field addresses
  // below.
  DeviceAllocationUnique adstack_heap_alloc_float_ = nullptr;
  std::size_t adstack_heap_size_float_{0};

  // Mirror of `adstack_heap_alloc_float_` for the int / u1 heap. Sized at `num_threads * stride_int` worst case. All
  // int allocas address through `runtime->adstack_heap_buffer_int + linear_tid * stride_int + int_offset` regardless of
  // whether the task captured a `bound_expr`; the int allocas are autodiff-emitted unconditionally at the offload root
  // (loop-index recovery, branch flags) so the lazy float row claim does not apply to them.
  DeviceAllocationUnique adstack_heap_alloc_int_ = nullptr;
  std::size_t adstack_heap_size_int_{0};

  // Cached device pointer to `runtime->temporaries`, populated lazily by `get_runtime_temporaries_device_ptr()`.
  void *runtime_temporaries_cache_{nullptr};

  // Cached field-of-LLVMRuntime addresses for the split float / int heap layout. Resolved by a single one-shot
  // `runtime_get_adstack_split_heap_field_ptrs` kernel the first time `ensure_adstack_heap_float` /
  // `ensure_adstack_heap_int` needs to publish a new buffer (returns float-buffer-ptr, float-size, int-buffer-ptr,
  // int-size in fixed slot order). Subsequent publishes are plain host->device memcpys onto these addresses, so no
  // kernel launch is required per grow.
  void *runtime_adstack_heap_buffer_float_field_ptr_{nullptr};
  void *runtime_adstack_heap_size_float_field_ptr_{nullptr};
  void *runtime_adstack_heap_buffer_int_field_ptr_{nullptr};
  void *runtime_adstack_heap_size_int_field_ptr_{nullptr};

  // Cached device pointers to the per-launch metadata fields `runtime->{adstack_per_thread_stride, adstack_offsets,
  // adstack_max_sizes}`. Populated lazily on the first `publish_adstack_metadata` call via a one-shot
  // `runtime_get_adstack_metadata_field_ptrs` kernel and reused for every subsequent launch.
  void *runtime_adstack_stride_field_ptr_{nullptr};
  // Cached field-of-LLVMRuntime addresses for the split per-thread strides (`adstack_per_thread_stride_float` /
  // `_int`). Returned by `runtime_get_adstack_metadata_field_ptrs` in slots 0 and 1; the legacy combined
  // `adstack_per_thread_stride` field is no longer present (the combined value is computed host-side as `float + int`
  // and written into the legacy cache for code paths that have not yet migrated to the split layout).
  void *runtime_adstack_stride_float_field_ptr_{nullptr};
  void *runtime_adstack_stride_int_field_ptr_{nullptr};
  void *runtime_adstack_offsets_field_ptr_{nullptr};
  void *runtime_adstack_max_sizes_field_ptr_{nullptr};
  // Cached field-of-LLVMRuntime addresses for the per-task lazy-claim counter array and bound row capacity array.
  // Resolved by `runtime_get_adstack_lazy_claim_field_ptrs`; the executor publishes the two array pointers via
  // `memcpy_host_to_device` to these cached addresses whenever the per-task slot count grows beyond the prior
  // allocation.
  void *runtime_adstack_row_counters_field_ptr_{nullptr};
  void *runtime_adstack_bound_row_capacities_field_ptr_{nullptr};
  // Cached address of `LLVMRuntime::adstack_max_reducer_outputs` (a `i64 *` field). Resolved once per program lifetime
  // via `runtime_get_adstack_max_reducer_field_ptr`; the per-launch dispatch writes the (possibly grown) device buffer
  // pointer to this address so `runtime_eval_adstack_max_reduce` deref's the live allocation.
  void *runtime_adstack_max_reducer_outputs_field_ptr_{nullptr};

  // Per-launch transient: shared, read-only view of the `MaxReducerResultMap` populated by
  // `dispatch_max_reducers_for_tasks` and read by `publish_adstack_metadata`. Held as a `shared_ptr<const map>` so the
  // per-launch fast path can repoint this field to the cached entry's map without copying it (refcount bump only), and
  // so a `publish_adstack_metadata` snapshot survives a recursive snode-reader-kernel reentry that may rewrite this
  // field. Reset to a fresh empty map at the top of every `dispatch_max_reducers_for_tasks` call so a kernel without
  // captured specs sees an empty view.
  std::shared_ptr<const std::unordered_map<uint64_t, int64_t>> current_max_reducer_results_;

  // Host-owned storage for the per-kernel lazy-claim arrays: `adstack_row_counters_alloc_`: u32[num_tasks] atomic
  // counter the codegen-emitted LCA-block row claim atomic-rmws
  //                                into; cleared host-side at the start of each kernel-launch so each task's claims
  //                                accumulate in its own slot from zero.
  // `adstack_bound_row_capacities_alloc_`: u32[num_tasks] capacity each task's claim is clamped against; the host
  //                                        writes UINT32_MAX into every slot by default so the clamp is inert when no
  //                                        reducer count is published.
  // Both buffers are sized at `max(num_tasks_observed)` and grown on demand; the pointers we publish into the runtime
  // stay stable across launches unless we actually grow.
  DeviceAllocationUnique adstack_row_counters_alloc_ = nullptr;
  DeviceAllocationUnique adstack_bound_row_capacities_alloc_ = nullptr;
  std::size_t adstack_lazy_claim_capacity_{0};

  // Host-owned storage for the two per-launch adstack metadata arrays. We reuse these buffers across launches so
  // the device pointers we publish remain stable; they are grown (never shrunk) when a larger task is hit.
  DeviceAllocationUnique adstack_offsets_alloc_ = nullptr;
  DeviceAllocationUnique adstack_max_sizes_alloc_ = nullptr;
  std::size_t adstack_metadata_capacity_{0};
  // Last device-pointer values published into `runtime->adstack_offsets` / `adstack_max_sizes`. The pointed-to scratch
  // allocations stay stable across launches (only grown via `grow_to`), so we skip the per-launch h2d when the pointer
  // we would write matches what we already wrote on a prior launch.
  void *adstack_offsets_dev_ptr_published_{nullptr};
  void *adstack_max_sizes_dev_ptr_published_{nullptr};
  // The float stride (in bytes) the most recent `publish_adstack_metadata` wrote to
  // `runtime->adstack_per_thread_stride_float`. Read by `ensure_per_task_float_heap_post_reducer` for the matching
  // task to size the float heap; we keep the value host-side so we do not need to DtoH-readback the stride field on
  // every bound_expr task. The launcher pairs publish + reducer + post-reducer per task with no intervening publish,
  // so the value remains accurate at the consumer call site.
  std::size_t last_published_stride_float_bytes_{0};

  // Per-launch scratch buffer used on GPU arches (CUDA / AMDGPU) to ship the `LlvmAdStackBoundReducerDeviceParams` blob
  // into for `runtime_eval_static_bound_count`. Allocated on demand on the first bound_expr task in a kernel, reused
  // across tasks within the same kernel and across kernels for the runtime's lifetime, grown amortised-doubling when a
  // future struct expansion would need more bytes (the struct is currently a fixed 32-byte POD). Unused on CPU, which
  // evaluates the predicate host-side via `publish_per_task_bound_count_cpu`.
  DeviceAllocationUnique adstack_bound_reducer_params_alloc_ = nullptr;
  std::size_t adstack_bound_reducer_params_capacity_{0};

  // Per-launch scratch buffer used on GPU arches (CUDA / AMDGPU) to ship the encoded adstack SizeExpr bytecode consumed
  // by `runtime_eval_adstack_size_expr`. Amortised-doubling growth, reused across launches. Unused on CPU where the
  // host evaluator runs directly without a device round-trip. See `encode_adstack_size_expr_device_bytecode` for the
  // byte layout.
  DeviceAllocationUnique adstack_sizer_bytecode_alloc_ = nullptr;
  std::size_t adstack_sizer_bytecode_capacity_{0};

  // Per-launch scratch buffers for the max-reducer dispatch. One holds a single `LlvmAdStackMaxReducerDeviceParams`
  // blob per call (the runtime function is dispatched per spec); the other holds the body bytecode (concatenated
  // `AdStackSizeExprDeviceNode` array followed by indices). Both grow amortised-doubling and are reused across specs
  // within a launch and across launches. Unused on CPU when the runtime function is invoked directly host-side without
  // staging.
  DeviceAllocationUnique adstack_max_reducer_params_alloc_ = nullptr;
  std::size_t adstack_max_reducer_params_capacity_{0};
  DeviceAllocationUnique adstack_max_reducer_bytecode_alloc_ = nullptr;
  std::size_t adstack_max_reducer_bytecode_capacity_{0};
  // Per-launch output buffer the runtime function writes into (`runtime->adstack_max_reducer_outputs[output_slot]` =
  // i64 dispatched value). Sized to fit the kernel's spec count; grown amortised-doubling. Backed by the runtime
  // module's `adstack_max_reducer_outputs` field via `runtime_LLVMRuntime_set_adstack_max_reducer_outputs`.
  DeviceAllocationUnique adstack_max_reducer_outputs_alloc_ = nullptr;
  std::size_t adstack_max_reducer_outputs_capacity_{0};

  // Pinned (page-locked) host scratch + completion event used by the host-eval branch of `publish_adstack_metadata` on
  // CUDA / AMDGPU to issue the per-launch adstack metadata writes asynchronously. With pageable host memory
  // `cuMemcpyHtoDAsync` synchronises on the staging copy, so the source MUST be pinned for the async copy to be truly
  // non-blocking; with pinned host memory the host returns immediately after queuing the copy on the default stream and
  // the GPU pulls the bytes via DMA before the subsequent main-kernel launch (which is also queued on the default
  // stream and stream-orders after the copies). One pinned scratch is reused across launches; `pinned_metadata_event_`
  // is recorded after the last copy of each launch, and the next launch's `publish_adstack_metadata` waits on it before
  // overwriting the scratch so the in-flight copy cannot read torn bytes. Allocated lazily on the first call that
  // exercises this branch and grown via `cuMemAllocHost_v2` / `hipHostMalloc`; freed in the executor destructor.
  // Capacity is in bytes and tracks the largest `(2 + 2 * n_stacks) * sizeof(uint64)` payload published so far.
  void *pinned_metadata_scratch_{nullptr};
  std::size_t pinned_metadata_scratch_capacity_{0};
  void *pinned_metadata_event_{nullptr};
  bool pinned_metadata_event_pending_{false};

  // good buddy
  friend LlvmProgramImpl;
  friend SNodeTreeBufferManager;

  bool use_device_memory_pool_ = false;
  bool finalized_{false};
  KernelProfilerBase *profiler_ = nullptr;
  ProgramImpl *program_impl_;
};

}  // namespace quadrants::lang

#endif  // QD_WITH_LLVM
