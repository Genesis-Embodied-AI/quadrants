#pragma once

// RAII guard that pins `CUDAContext::stream_` to the legacy default stream (nullptr) for the duration of the scope,
// then synchronises that stream and restores the prior `stream_` on destruction. Used at the entry of every
// per-launch adstack helper (`dispatch_max_reducers_for_tasks`, `publish_adstack_metadata`,
// `publish_per_task_bound_count_device`) so the helper kernel - launched via `runtime_jit->call/launch` and reading
// ndarray data through `ctx->arg_buffer` - lands on the same stream the ndarray pool's `malloc_async` used. CUDA's
// stream-ordered allocator denies cross-stream visibility of pool allocations on pre-Ampere hardware (Turing T4
// faults the helper at `cuLaunchKernel` with an illegal-address); pinning makes the consumer + producer streams
// match unconditionally. Trailing `stream_synchronize(nullptr)` is paid for already by the synchronous DtoH calls
// inside the helpers, so there is no measurable cost on Ampere+ where cross-stream access works. AMDGPU is not
// pinned because HIP's allocator appears more permissive about cross-stream visibility today; if that changes, an
// equivalent guard can be added against `AMDGPUContext`.

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
      CUDADriver::get_instance().stream_synchronize(nullptr);
      CUDAContext::get_instance().set_stream(prev_stream);
    }
  }
};
#endif

}  // namespace quadrants::lang
