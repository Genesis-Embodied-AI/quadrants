#pragma once

#include <cstddef>
#include <memory>

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

  // Host-managed per-runtime adstack heap. Each kernel launcher calls this before dispatching a task whose
  // `OffloadedTask::ad_stack.per_thread_stride > 0`; `needed_bytes` is `per_thread_stride * num_threads` computed
  // per the resolution rule in `AdStackSizingInfo`. Growth is amortized via `max(needed, 2 * current)` doubling,
  // old slabs are returned to the driver memory pool (no leak), and the new pointer/size are published into the
  // runtime struct at `runtime->{adstack_heap_buffer, adstack_heap_size}` without a per-grow kernel launch: a
  // one-shot `runtime_get_adstack_heap_field_ptrs` kernel caches the device addresses of the two fields on the
  // first grow, and subsequent publishes are `memcpy_host_to_device` (CUDA / AMDGPU) or plain pointer stores
  // (CPU) against those cached addresses.
  void ensure_adstack_heap(std::size_t needed_bytes);

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

  // Return (and lazily cache) the device pointer to `runtime->temporaries`, the global temporary buffer backing
  // `GlobalTemporaryStmt` loads and stores. GPU kernel launchers use this to read back dynamic range_for bounds
  // (begin / end i32 values at known byte offsets) via a host-side DtoH memcpy when sizing the adstack heap.
  // Cached because `runtime->temporaries` is assigned once during `runtime_initialize` and never rebound.
  void *get_runtime_temporaries_device_ptr();

 private:
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

  std::unique_ptr<ThreadPool> thread_pool_{nullptr};
  std::shared_ptr<Device> device_{nullptr};

  std::unique_ptr<SNodeTreeBufferManager> snode_tree_buffer_manager_{nullptr};
  std::unordered_map<int, DeviceAllocation> snode_tree_allocs_;
  DeviceAllocationUnique preallocated_runtime_objects_allocs_ = nullptr;
  DeviceAllocationUnique preallocated_runtime_memory_allocs_ = nullptr;
  std::unordered_map<DeviceAllocationId, DeviceAllocation> allocated_runtime_memory_allocs_;

  // Per-runtime adstack heap slab, owned here. `ensure_adstack_heap` grows via the driver allocator and
  // publishes the new pointer/size into the LLVMRuntime struct; replacing `adstack_heap_alloc_` releases the
  // previous allocation via `DeviceAllocationGuard`, which calls `llvm_device()->dealloc_memory`. Safety of
  // releasing the old slab while a prior-launch kernel may still hold its base pointer depends on the backend:
  // on CPU the release is a host `std::free` (trivially safe); on CUDA `cuMemFree_v2` synchronizes with
  // pending device work before returning; on AMDGPU `dealloc_memory` routes through
  // `DeviceMemoryPool::release(release_raw=false)` -> `CachingAllocator::release`, which pools the allocation
  // *without* calling `hipFree` and *without* synchronizing - so on AMDGPU the cross-launch invariant instead
  // comes from `amdgpu::KernelLauncher::launch_llvm_kernel` ending with a synchronous `hipFree(context_pointer)`
  // before the next launch reaches `ensure_adstack_heap`. See the detailed block comment in
  // `LlvmRuntimeExecutor::ensure_adstack_heap` for the full derivation; do not remove the launcher-tail
  // `hipFree(context_pointer)` without simultaneously fixing the AMDGPU release path.
  DeviceAllocationUnique adstack_heap_alloc_ = nullptr;
  std::size_t adstack_heap_size_{0};

  // Cached device pointer to `runtime->temporaries`, populated lazily by `get_runtime_temporaries_device_ptr()`.
  void *runtime_temporaries_cache_{nullptr};

  // Cached device pointers to `runtime->adstack_heap_buffer` and `runtime->adstack_heap_size`, populated by a
  // single one-shot `runtime_get_adstack_heap_field_ptrs` kernel the first time `ensure_adstack_heap` needs to
  // publish a new buffer. Subsequent publishes are plain host->device memcpys onto these addresses, so no kernel
  // launch is required per grow.
  void *runtime_adstack_heap_buffer_field_ptr_{nullptr};
  void *runtime_adstack_heap_size_field_ptr_{nullptr};

  // Cached device pointers to the per-launch metadata fields
  // `runtime->{adstack_per_thread_stride, adstack_offsets, adstack_max_sizes}`. Populated lazily on the first
  // `publish_adstack_metadata` call via a one-shot `runtime_get_adstack_metadata_field_ptrs` kernel and reused
  // for every subsequent launch.
  void *runtime_adstack_stride_field_ptr_{nullptr};
  void *runtime_adstack_offsets_field_ptr_{nullptr};
  void *runtime_adstack_max_sizes_field_ptr_{nullptr};

  // Host-owned storage for the two per-launch adstack metadata arrays. We reuse these buffers across launches so
  // the device pointers we publish remain stable; they are grown (never shrunk) when a larger task is hit.
  DeviceAllocationUnique adstack_offsets_alloc_ = nullptr;
  DeviceAllocationUnique adstack_max_sizes_alloc_ = nullptr;
  std::size_t adstack_metadata_capacity_{0};

  // Per-launch scratch buffer used on GPU arches (CUDA / AMDGPU) to ship the encoded adstack SizeExpr bytecode
  // consumed by `runtime_eval_adstack_size_expr`. Amortised-doubling growth, reused across launches. Unused on
  // CPU where the host evaluator runs directly without a device round-trip. See
  // `encode_adstack_size_expr_device_bytecode` for the byte layout.
  DeviceAllocationUnique adstack_sizer_bytecode_alloc_ = nullptr;
  std::size_t adstack_sizer_bytecode_capacity_{0};

  // Pinned (page-locked) host scratch + completion event used by the host-eval branch of `publish_adstack_metadata`
  // on CUDA / AMDGPU to issue the per-launch adstack metadata writes asynchronously. With pageable host memory
  // `cuMemcpyHtoDAsync` synchronises on the staging copy, so the source MUST be pinned for the async copy to be
  // truly non-blocking; with pinned host memory the host returns immediately after queuing the copy on the
  // default stream and the GPU pulls the bytes via DMA before the subsequent main-kernel launch (which is also
  // queued on the default stream and stream-orders after the copies). One pinned scratch is reused across
  // launches; `pinned_metadata_event_` is recorded after the last copy of each launch, and the next launch's
  // `publish_adstack_metadata` waits on it before overwriting the scratch so the in-flight copy cannot read torn
  // bytes. Allocated lazily on the first call that exercises this branch and grown via
  // `cuMemAllocHost_v2` / `hipHostMalloc`; freed in the executor destructor. Capacity is in bytes and tracks the
  // largest `(2 + 2 * n_stacks) * sizeof(uint64)` payload published so far.
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
