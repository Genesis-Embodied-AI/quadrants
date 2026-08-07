#include "quadrants/rhi/llvm/device_memory_pool.h"

#include <cstdint>
#include <memory>

#include "quadrants/math/arithmetic.h"

#ifdef QD_WITH_AMDGPU
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#endif

#ifdef QD_WITH_CUDA
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/cuda/cuda_device.h"
#endif

#if defined(QD_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include "quadrants/platform/windows/windows.h"
#endif

namespace quadrants::lang {

DeviceMemoryPool::DeviceMemoryPool(Arch arch, bool merge_upon_release)
    : merge_upon_release_(merge_upon_release), arch_(arch) {
  allocator_ = std::make_unique<CachingAllocator>(merge_upon_release);
}

void *DeviceMemoryPool::allocate_with_cache(LlvmDevice *device, const LlvmDevice::LlvmRuntimeAllocParams &params) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  return allocator_->allocate(device, params);
}

void *DeviceMemoryPool::allocate(std::size_t size, std::size_t alignment, bool managed) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  return allocate_raw_memory(size, alignment, managed);
}

void DeviceMemoryPool::release(std::size_t size, void *ptr, bool release_raw) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  if (release_raw) {
    deallocate_raw_memory(ptr);
  } else {
    allocator_->release(size, (uint64_t *)ptr);
  }
}

void *DeviceMemoryPool::allocate_raw_memory(std::size_t size, std::size_t alignment, bool managed) {
  /*
    Be aware that this methods is not protected by the mutex.

    allocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */
  QD_ASSERT(alignment > 0);

  // cuMemAlloc/hipMalloc only promise an alignment suitable for any built-in type (256 bytes in practice), and on a
  // driver heap fragmented by sub-page allocations they do return sub-page-aligned bases. Callers depend on the
  // alignment they ask for: materialize_runtime sizes the runtime-objects chunk as an exact sum of page-rounded
  // blocks, while the device-side bump allocator in runtime_initialize charges alignment padding against that same
  // budget, so a misaligned base overruns the chunk. Over-allocate and align up instead of trusting the driver.
  const std::size_t raw_size = size + alignment - 1;
  void *ptr = nullptr;

  if (arch_ == Arch::cuda) {
#if QD_WITH_CUDA
    if (!managed) {
      CUDADriver::get_instance().malloc(&ptr, raw_size);
    } else {
      CUDADriver::get_instance().malloc_managed(&ptr, raw_size, CU_MEM_ATTACH_GLOBAL);
    }
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (arch_ == Arch::amdgpu) {
#if QD_WITH_AMDGPU
    if (!managed) {
      AMDGPUDriver::get_instance().malloc(&ptr, raw_size);
    } else {
      AMDGPUDriver::get_instance().malloc_managed(&ptr, raw_size, HIP_MEM_ATTACH_GLOBAL);
    }
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    QD_NOT_IMPLEMENTED;
  }

  if (ptr == nullptr) {
    QD_ERROR("Device memory allocation ({} B) failed.", raw_size);
  }

  void *aligned_ptr = reinterpret_cast<void *>(quadrants::iroundup(reinterpret_cast<std::uintptr_t>(ptr), alignment));
  if (raw_memory_chunks_.count(aligned_ptr)) {
    QD_ERROR("Memory address ({:}) is already allocated", aligned_ptr);
  }

  raw_memory_chunks_[aligned_ptr] = RawMemoryChunk{ptr, raw_size};
  return aligned_ptr;
}

void DeviceMemoryPool::deallocate_raw_memory(void *ptr) {
  /*
    Be aware that this methods is not protected by the mutex.

    deallocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */
  auto chunk = raw_memory_chunks_.find(ptr);
  if (chunk == raw_memory_chunks_.end()) {
    QD_ERROR("Memory address ({:}) is not allocated", ptr);
  }

  // Only the driver's own base is valid to free; `ptr` may sit inside the block to satisfy the requested alignment.
  void *base = chunk->second.base;
  if (arch_ == Arch::cuda) {
#if QD_WITH_CUDA
    CUDADriver::get_instance().mem_free(base);
    raw_memory_chunks_.erase(chunk);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (arch_ == Arch::amdgpu) {
#if QD_WITH_AMDGPU
    AMDGPUDriver::get_instance().mem_free(base);
    raw_memory_chunks_.erase(chunk);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    QD_NOT_IMPLEMENTED;
  }
}

void DeviceMemoryPool::reset() {
  std::lock_guard<std::mutex> _(mut_allocation_);

  const auto ptr_map_copied = raw_memory_chunks_;
  for (auto &ptr : ptr_map_copied) {
    deallocate_raw_memory(ptr.first);
  }
  allocator_ = std::make_unique<CachingAllocator>(merge_upon_release_);
}

DeviceMemoryPool::~DeviceMemoryPool() {
  reset();
}

const size_t DeviceMemoryPool::page_size{1 << 12};  // 4 KB page size by default

DeviceMemoryPool &DeviceMemoryPool::get_instance(Arch arch, bool merge_upon_release) {
  static DeviceMemoryPool *cuda_memory_pool = new DeviceMemoryPool(arch, merge_upon_release);
  assert(cuda_memory_pool->arch_ == arch);
  return *cuda_memory_pool;
}

}  // namespace quadrants::lang
