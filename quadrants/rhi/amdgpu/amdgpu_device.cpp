#include "quadrants/rhi/amdgpu/amdgpu_device.h"
#include "quadrants/rhi/llvm/gpu_device_impl.h"

namespace quadrants::lang {

template class GpuDevice<amdgpu::AmdgpuDeviceBackend>;

namespace amdgpu {

AMDGPUDriver &AmdgpuDeviceBackend::driver() {
  return AMDGPUDriver::get_instance();
}

AMDGPUContext &AmdgpuDeviceBackend::context() {
  return AMDGPUContext::get_instance();
}

DeviceMemoryPool &AmdgpuDeviceBackend::memory_pool() {
  return DeviceMemoryPool::get_instance(Arch::amdgpu, /*merge_upon_release=*/false);
}

void AmdgpuDeviceBackend::free_pooled_memory(void *ptr) {
  driver().mem_free(ptr);
}

}  // namespace amdgpu
}  // namespace quadrants::lang
