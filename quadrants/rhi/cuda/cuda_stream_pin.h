#pragma once

// RAII guard that pins `CUDAContext::stream_` to the legacy default stream (nullptr) for the duration of the scope,
// then restores the prior `stream_` on destruction. Pinning makes the consumer + producer streams match unconditionally
// so the cross-stream-visibility break that CUDA's stream-ordered allocator (`cuMemAllocAsync` pool) shows on
// pre-Ampere hardware (Turing T4 faults at `cuLaunchKernel` with `illegal-address`) is closed by construction. The
// destructor does NOT call `stream_synchronize(nullptr)`: the helpers using this guard always issue a synchronous DtoH
// inside the scope which already drains the null stream, and the main launcher's kernel-only fast path (no host-side
// sync DtoH) deliberately stays asynchronous so kernel launches do not pay a forced per-launch host-GPU round-trip.
// Restoring the prior stream while null-stream work is still in flight is safe: subsequent operations on the prior
// stream that depend on the null-stream work will surface the standard cross-stream visibility constraints anyway, and
// the next launch through this guard re-pins to null and serialises with prior null-stream work via legacy default
// stream semantics. Used at the entry of every per-launch CUDA host path that issues HtoD or kernel work that needs
// the same-stream invariant: `KernelLauncher::launch_llvm_kernel`, the adstack helpers
// `dispatch_max_reducers_for_tasks` / `publish_adstack_metadata` / `publish_per_task_bound_count_device`. The launcher
// engages the guard only on the default-stream fast path (entry `stream_ == nullptr` with all tasks on
// `stream_parallel_group_id == 0`); on the explicit-stream path the streams feature requires the kernel to inherit
// the user-supplied `stream_` and the guard stays disengaged. A symmetric `AmdgpuDefaultStreamPinGuard` lives in
// `rhi/amdgpu/amdgpu_stream_pin.h` for consistency on AMDGPU - the streams plumbing now passes `stream_` through to
// `hipModuleLaunchKernel`, so the same same-stream-invariant rationale applies, even though the cross-stream pool
// fault that motivated the CUDA guard has not been observed on AMDGPU hardware.

#include "quadrants/platform/cuda/detect_cuda.h"

#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#endif

namespace quadrants::lang {

#if defined(QD_WITH_CUDA)
struct CudaDefaultStreamPinGuard {
  bool engaged{false};
  void *prev_stream{nullptr};
  explicit CudaDefaultStreamPinGuard(bool engage) : engaged(engage) {
    if (engaged) {
      prev_stream = CUDAContext::get_instance().get_stream();
      CUDAContext::get_instance().set_stream(nullptr);
    }
  }
  ~CudaDefaultStreamPinGuard() {
    if (engaged) {
      CUDAContext::get_instance().set_stream(prev_stream);
    }
  }
};
#endif

}  // namespace quadrants::lang
