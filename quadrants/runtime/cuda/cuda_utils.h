#pragma once

#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace cuda {

bool on_cuda_device(void *ptr);

// Unwraps a DeviceAllocation handle to a raw device pointer.
inline void *resolve_device_alloc_ptr(LlvmRuntimeExecutor *executor,
                                      void *alloc) {
  return executor->get_device_alloc_info_ptr(
      *static_cast<DeviceAllocation *>(alloc));
}

}  // namespace cuda
}  // namespace quadrants::lang
