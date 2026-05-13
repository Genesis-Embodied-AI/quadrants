// Stage C of the LLVM sparse-adstack-heap lazy-claim pipeline: heap-allocation lifecycle. See
// `adstack_lazy_claim/heap_grow.h` for the stage-level documentation.

#include "quadrants/runtime/llvm/adstack_lazy_claim/heap_grow.h"

#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/rhi/llvm/llvm_device.h"

#include "quadrants/platform/cuda/detect_cuda.h"
#include "quadrants/rhi/cuda/cuda_driver.h"

#include "quadrants/platform/amdgpu/detect_amdgpu.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

namespace quadrants::lang {

void LlvmRuntimeExecutor::ensure_adstack_heap_int(std::size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= adstack_heap_size_int_) {
    return;
  }
  std::size_t new_size = std::max(needed_bytes, std::size_t(2) * adstack_heap_size_int_);

  Device::AllocParams params{};
  params.size = new_size;
  params.host_read = false;
  params.host_write = false;
  params.export_sharing = false;
  params.usage = AllocUsage::Storage;
  DeviceAllocation new_alloc;
  RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
  QD_ERROR_IF(res != RhiResult::success,
              "Failed to allocate {} bytes for the adstack int heap (err: {}). Consider lowering "
              "`ad_stack_size` or the per-kernel reverse-mode adstack count.",
              new_size, int(res));
  void *new_ptr = get_device_alloc_info_ptr(new_alloc);
  auto new_guard = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));

  // The split-heap field-of-LLVMRuntime addresses are cached together by `ensure_adstack_heap_float` on its first grow
  // (the same `runtime_get_adstack_split_heap_field_ptrs` getter returns all four addresses - float-buffer, float-size,
  // int-buffer, int-size - in fixed slot order). On a fresh executor where this is the very first split-heap call,
  // resolve the addresses here so we can publish independently of the float heap path.
  if (runtime_adstack_heap_buffer_int_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_split_heap_field_ptrs", llvm_runtime_);
    runtime_adstack_heap_buffer_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_heap_size_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_heap_buffer_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_heap_size_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
  }
  uint64 size_u64 = static_cast<uint64>(new_size);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_int_field_ptr_, &new_ptr,
                                                     sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_int_field_ptr_, &size_u64,
                                                     sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_int_field_ptr_, &new_ptr,
                                                       sizeof(void *));
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_int_field_ptr_, &size_u64,
                                                       sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    *reinterpret_cast<void **>(runtime_adstack_heap_buffer_int_field_ptr_) = new_ptr;
    *reinterpret_cast<uint64 *>(runtime_adstack_heap_size_int_field_ptr_) = size_u64;
  }

  adstack_heap_alloc_int_ = std::move(new_guard);
  adstack_heap_size_int_ = new_size;
}

void LlvmRuntimeExecutor::ensure_per_task_float_heap_post_reducer(std::size_t task_index,
                                                                  const AdStackSizingInfo &ad_stack,
                                                                  std::size_t num_threads,
                                                                  LaunchContextBuilder *ctx) {
  // Skip when the task has no float heap need (no f32 allocas, or analysis didn't capture a gate so we wouldn't have
  // routed it through the lazy float path on the codegen side).
  if (!ad_stack.bound_expr.has_value() || ad_stack.per_thread_stride_float == 0) {
    return;
  }

  // Read the per-task count the reducer published. On CPU the capacity buffer is host-resident; on CUDA / AMDGPU it's
  // device memory and the read is a small (4-byte) DtoH per task. Cost is dominated by the actual main kernel.
  uint32_t count = std::numeric_limits<uint32_t>::max();
  if (adstack_bound_row_capacities_alloc_) {
    void *capacities_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_row_capacities_alloc_);
    char *slot_ptr = static_cast<char *>(capacities_dev_ptr) + task_index * sizeof(uint32_t);
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_device_to_host(&count, slot_ptr, sizeof(uint32_t));
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_device_to_host(&count, slot_ptr, sizeof(uint32_t));
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      count = *reinterpret_cast<const uint32_t *>(slot_ptr);
    }
  }

  // Floor at 1 row when the captured count is zero (no thread passed the gate this launch). The codegen-emitted bounds
  // clamp keeps `claimed_row` in [0, count-1] so threads that miss the gate never reach the LCA-block claim - the heap
  // row stays unused. A 1-row allocation is cheap and keeps the heap pointer non-null. Clip by the captured
  // compile-time loop trip count when known: each iteration claims at most one row at the LCA-block (one `atomic_add`
  // per gating iteration), so the heap needs at most `loop_iter_static` rows regardless of how many cells of an
  // oversized gating SNode the reducer counted. The analyzer leaves `loop_iter_static == 0` for runtime-bounded loops
  // and for CPU LLVM tasks whose `[begin_value, end_value)` is a post-chunking subrange (the unclipped reducer count is
  // the right upper bound there).
  std::size_t effective_rows =
      (count == std::numeric_limits<uint32_t>::max()) ? num_threads : std::max<std::size_t>(count, 1);
  if (count != std::numeric_limits<uint32_t>::max() && ad_stack.bound_expr.has_value()) {
    // Shared with the SPIR-V launcher: see `clip_effective_rows_by_loop_trip_count` in
    // `program/adstack_size_expr_eval.cpp`. LLVM dispatches one thread per loop iteration without the
    // SPIR-V dispatch-cap-driven serialisation, so pass `numeric_limits::max()` to disable the
    // dispatched-threads ceiling - any positive trip-count value is a sound upper bound on row claims
    // here. `numeric_limits<size_t>::max()` is the ceiling sentinel `clip_effective_rows_by_loop_trip_count`
    // documents.
    Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
    clip_effective_rows_by_loop_trip_count(effective_rows, *ad_stack.bound_expr,
                                           std::numeric_limits<std::size_t>::max(), prog, ctx);
  }
  // The per-thread float stride (in bytes) was just published into `runtime->adstack_per_thread_stride_float` by the
  // matching `publish_adstack_metadata` call earlier in this task's per-task block. We stash the value host-side so
  // we can read it directly here instead of paying a sync DtoH on every bound_expr task. The launcher pairs publish
  // + reducer + post-reducer per task with no intervening publish for another task, so the stash is accurate at this
  // call site. `AdStackSizingInfo::per_thread_stride_float` from the analysis pre-pass is in entry-count units
  // (`2 * max_size`), not bytes, and would massively undersize the heap.
  uint64_t stride_float_bytes_u64 = static_cast<uint64_t>(last_published_stride_float_bytes_);
  const std::size_t needed_bytes = effective_rows * static_cast<std::size_t>(stride_float_bytes_u64);
  // `QD_DEBUG_ADSTACK=1` opt-in diagnostic. Persistent so memory regressions can be debugged without re-instrumenting.
  if (std::getenv("QD_DEBUG_ADSTACK")) {
    const char *src = (count == std::numeric_limits<uint32_t>::max())
                          ? "worst_case_num_threads"
                          : (count == 0 ? "reducer_zero_floored" : "reducer_count");
    std::fprintf(stderr,
                 "[adstack_heap] arch=llvm task_idx=%zu kind=F src=%s effective_rows=%zu stride=%llu "
                 "required_bytes=%zu (%.2f MB)\n",
                 task_index, src, effective_rows, static_cast<unsigned long long>(stride_float_bytes_u64), needed_bytes,
                 double(needed_bytes) / (1024.0 * 1024.0));
    std::fflush(stderr);
  }
  ensure_adstack_heap_float(needed_bytes);
}

void LlvmRuntimeExecutor::ensure_adstack_heap_float(std::size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= adstack_heap_size_float_) {
    return;
  }
  // Mirror `ensure_adstack_heap`'s amortised-doubling growth and grow-on-demand semantics. The float heap is allocated
  // independently from the combined heap so a kernel with bound_expr tasks can shrink the combined slice to int-only
  // while still backing float allocas at `row_id_var * stride_float + float_offset`.
  std::size_t new_size = std::max(needed_bytes, std::size_t(2) * adstack_heap_size_float_);

  Device::AllocParams params{};
  params.size = new_size;
  params.host_read = false;
  params.host_write = false;
  params.export_sharing = false;
  params.usage = AllocUsage::Storage;
  DeviceAllocation new_alloc;
  RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
  QD_ERROR_IF(res != RhiResult::success,
              "Failed to allocate {} bytes for the adstack float heap (err: {}). Consider lowering "
              "`ad_stack_size` or the per-kernel reverse-mode adstack count.",
              new_size, int(res));
  void *new_ptr = get_device_alloc_info_ptr(new_alloc);
  auto new_guard = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));

  // Resolve and cache the field-of-LLVMRuntime addresses for the split-heap fields on first grow. The
  // `runtime_get_adstack_split_heap_field_ptrs` helper returns four addresses in fixed slot order: float-buffer-ptr,
  // float-size, int-buffer-ptr, int-size. We only consume the float pair here; the int half is reserved for a future
  // symmetric `ensure_adstack_heap_int` if it becomes useful (today the int allocas in bound_expr tasks ride the
  // combined heap with a smaller stride).
  if (runtime_adstack_heap_buffer_float_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_split_heap_field_ptrs", llvm_runtime_);
    runtime_adstack_heap_buffer_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_heap_size_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_heap_buffer_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_heap_size_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
  }
  uint64 size_u64 = static_cast<uint64>(new_size);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_float_field_ptr_, &new_ptr,
                                                     sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_float_field_ptr_, &size_u64,
                                                     sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_float_field_ptr_, &new_ptr,
                                                       sizeof(void *));
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_float_field_ptr_, &size_u64,
                                                       sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    *reinterpret_cast<void **>(runtime_adstack_heap_buffer_float_field_ptr_) = new_ptr;
    *reinterpret_cast<uint64 *>(runtime_adstack_heap_size_float_field_ptr_) = size_u64;
  }

  adstack_heap_alloc_float_ = std::move(new_guard);
  adstack_heap_size_float_ = new_size;
}

void LlvmRuntimeExecutor::check_adstack_overflow() {
  // Called from `synchronize_and_assert()` on every qd.sync(), plus per-launch from `Program::launch_kernel`. The
  // flag lives in pinned host memory (allocated at `materialize_runtime`); polling is a relaxed atomic load/exchange
  // on the cached host pointer via `std::atomic<int64_t>` reinterpret_cast - no DtoH, no JIT call, no sync drain.
  // Available on all backends because the pinned-host memory is in the host process address space regardless of
  // where the kernel that wrote it ran. The reinterpret_cast is portable because `std::atomic<int64_t>` is
  // layout-compatible with `int64_t` on every target (verified by the static_assert below); see also Itanium ABI /
  // MSVC ABI lock-free guarantees.
  //
  // Returns early when the slot has not been allocated yet (e.g. a C++ test that constructs Program without
  // materializing the runtime and then triggers `Program::finalize -> synchronize`).
  static_assert(std::atomic<int64_t>::is_always_lock_free,
                "std::atomic<int64_t> must be lock-free for the reinterpret_cast pattern below to be portable");
  if (adstack_overflow_flag_host_ptr_ == nullptr) {
    return;
  }
  // Peek first: a relaxed load is cheaper than an exchange and avoids consuming the flag when the companion task_id
  // slot has not yet been flushed from the device.  The per-launch call site does NOT synchronize before polling, so
  // the device's two atomic writes (flag OR, then task_id cmpxchg) may arrive at the host out of order.  If we
  // consumed the flag here but the task_id hadn't landed, the diagnostic would lack the kernel name and the later
  // qd.sync() would see both slots clean - losing the identity forever.
  int64_t flag =
      reinterpret_cast<std::atomic<int64_t> *>(adstack_overflow_flag_host_ptr_)->load(std::memory_order_relaxed);
  if (flag == 0) {
    return;
  }
  // Flag is set - drain the default stream so that the companion task_id write is guaranteed to be host-visible
  // before we read it.  This sync only fires on the rare overflow path, so it has zero cost on the fast path.
  synchronize();
  // Now consume both slots.  Both cleared so the next overflow records a fresh identity.  `task_id == 0` means the
  // kernel that overflowed pre-dates the registry wiring or its `ad_stack.registry_id` was unset for any reason
  // (e.g. a deserialised offline-cache task that has not yet been re-registered); the diagnose helper falls through
  // to the generic dual-cause message in that case.
  reinterpret_cast<std::atomic<int64_t> *>(adstack_overflow_flag_host_ptr_)->store(0, std::memory_order_relaxed);
  uint32_t task_id = 0;
  if (adstack_overflow_task_id_host_ptr_ != nullptr) {
    int64_t recorded = reinterpret_cast<std::atomic<int64_t> *>(adstack_overflow_task_id_host_ptr_)
                           ->exchange(0, std::memory_order_relaxed);
    task_id = static_cast<uint32_t>(recorded);
  }
  Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
  std::string diagnostic;
  if (prog != nullptr) {
    auto diag = prog->adstack_cache().diagnose_adstack_overflow(task_id);
    diagnostic = std::move(diag.message);
    // Auto-invalidate the per-task metadata caches when the synchronous sizer rerun confirmed the cache is stale
    // (DLPack-bypass cause). The current run is corrupted (we are about to raise), but the next launch's sizer
    // reruns from scratch against the live (mutated) state and the kernel runs to completion without further
    // user intervention. Unknown / Quadrants-bug cases skip the invalidation so a real sizer bug is not masked
    // by silent recompute.
    if (diag.confirmed_invalid_cache) {
      prog->adstack_cache().invalidate_all_per_task();
    }
  } else {
    diagnostic =
        "Adstack overflow: a reverse-mode autodiff kernel pushed more elements than the adstack capacity "
        "allows.";
  }
  throw QuadrantsAssertionError(
      "Adstack overflow: a reverse-mode autodiff kernel pushed more elements "
      "than the adstack capacity allows. Raised at the next Quadrants Python "
      "entry rather than at the offending kernel launch.\n" +
      diagnostic);
}

}  // namespace quadrants::lang
