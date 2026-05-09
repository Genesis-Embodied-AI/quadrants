#pragma once

// RAII guard that pins `AMDGPUContext::stream_` to the legacy default stream (nullptr) for the duration of the scope,
// then restores the prior `stream_` on destruction. Symmetric counterpart to `CudaDefaultStreamPinGuard` in
// `rhi/cuda/cuda_stream_pin.h`; engaged at the same launcher-scope callsite (`KernelLauncher::launch_llvm_kernel` on
// AMDGPU) on the default-stream fast path so the HtoD / kernel / DtoH chain lands on a single stream without an
// explicit `stream_synchronize` between phases. AMDGPU has not been observed to hit the cross-stream pool-visibility
// fault that motivated the CUDA guard on pre-Ampere Turing T4, but `AMDGPUContext::launch` now forwards `stream_` to
// `hipModuleLaunchKernel` (post-streams plumbing), so the same same-stream-invariant rationale applies and keeping the
// guard symmetric with CUDA hardens against future driver / hardware combos that tighten cross-stream pool semantics.
// The launcher engages this guard only when entry `stream_ == nullptr` and all tasks have
// `stream_parallel_group_id == 0`; on the explicit-stream path the streams feature requires the kernel to inherit
// the user-supplied `stream_` and the guard stays disengaged.

#include "quadrants/platform/amdgpu/detect_amdgpu.h"

#if defined(QD_WITH_AMDGPU)
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#endif

namespace quadrants::lang {

#if defined(QD_WITH_AMDGPU)
struct AmdgpuDefaultStreamPinGuard {
  bool engaged{false};
  void *prev_stream{nullptr};
  explicit AmdgpuDefaultStreamPinGuard(bool engage) : engaged(engage) {
    if (engaged) {
      prev_stream = AMDGPUContext::get_instance().get_stream();
      AMDGPUContext::get_instance().set_stream(nullptr);
    }
  }
  ~AmdgpuDefaultStreamPinGuard() {
    if (engaged) {
      AMDGPUContext::get_instance().set_stream(prev_stream);
    }
  }
};
#endif

}  // namespace quadrants::lang
