#pragma once

#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/llvm/gpu_device.h"

namespace quadrants::lang {
class DeviceMemoryPool;

namespace cuda {

// Preserve the existing backend policies while sharing the implementation of every device operation.
struct CudaDeviceBackend {
  static constexpr const char *name = "CUDA";
  static constexpr bool reject_runtime_host_access = false;
  static constexpr bool release_raw_allocations = true;

  static CUDADriver &driver();
  static CUDAContext &context();
  static DeviceMemoryPool &memory_pool();
  static void free_pooled_memory(void *ptr);
};

using CudaPipeline = GpuPipeline;
using CudaCommandList = GpuCommandList;
using CudaStream = GpuStream;

class CudaDevice : public GpuDevice<CudaDeviceBackend> {};

}  // namespace cuda

extern template class GpuDevice<cuda::CudaDeviceBackend>;

}  // namespace quadrants::lang
