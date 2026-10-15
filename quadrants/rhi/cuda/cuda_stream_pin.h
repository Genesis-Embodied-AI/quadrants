#pragma once

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
