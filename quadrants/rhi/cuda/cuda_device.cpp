#include "quadrants/rhi/cuda/cuda_device.h"
#include "quadrants/rhi/llvm/gpu_device_impl.h"

namespace quadrants::lang {

template class GpuDevice<cuda::CudaDeviceBackend>;

namespace cuda {

CUDADriver &CudaDeviceBackend::driver() {
  return CUDADriver::get_instance();
}

CUDAContext &CudaDeviceBackend::context() {
  return CUDAContext::get_instance();
}

DeviceMemoryPool &CudaDeviceBackend::memory_pool() {
  return DeviceMemoryPool::get_instance(Arch::cuda, /*merge_upon_release=*/true);
}

void CudaDeviceBackend::free_pooled_memory(void *ptr) {
  driver().mem_free_async(ptr, nullptr);
}

}  // namespace cuda
}  // namespace quadrants::lang
