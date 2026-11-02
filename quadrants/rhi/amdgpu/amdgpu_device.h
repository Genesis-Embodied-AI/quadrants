#pragma once

#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/rhi/llvm/gpu_device.h"

namespace quadrants::lang {
class DeviceMemoryPool;

namespace amdgpu {

// Preserve the existing backend policies while sharing the implementation of every device operation.
struct AmdgpuDeviceBackend {
  static constexpr const char *name = "AMDGPU";
  static constexpr bool reject_runtime_host_access = true;
  static constexpr bool release_raw_allocations = false;

  static AMDGPUDriver &driver();
  static AMDGPUContext &context();
  static DeviceMemoryPool &memory_pool();
  static void free_pooled_memory(void *ptr);
};

using AmdgpuCommandList = GpuCommandList;
using AmdgpuStream = GpuStream;

class AmdgpuDevice : public GpuDevice<AmdgpuDeviceBackend> {};

}  // namespace amdgpu

extern template class GpuDevice<amdgpu::AmdgpuDeviceBackend>;

}  // namespace quadrants::lang
