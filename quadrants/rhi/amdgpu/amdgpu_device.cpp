#include "quadrants/rhi/amdgpu/amdgpu_device.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/llvm/device_memory_pool.h"

#include "quadrants/jit/jit_module.h"

namespace quadrants {
namespace lang {

namespace amdgpu {

AmdgpuDevice::AmdgpuDevice() {
  // Initialize the device memory pool
  DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/);
}

AmdgpuDevice::AllocInfo AmdgpuDevice::get_alloc_info(const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

RhiResult AmdgpuDevice::allocate_memory(const AllocParams &params, DeviceAllocation *out_devalloc) {
  AllocInfo info;
  auto &mem_pool = DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/);

  bool managed = params.host_read || params.host_write;
  void *ptr = mem_pool.allocate(params.size, DeviceMemoryPool::page_size, managed);
  if (ptr == nullptr) {
    return RhiResult::out_of_memory;
  }

  info.ptr = ptr;
  info.size = params.size;
  info.is_imported = false;
  info.use_cached = false;
  info.use_preallocated = false;

  if (info.ptr == nullptr) {
    return RhiResult::out_of_memory;
  }

  AMDGPUDriver::get_instance().memset((void *)info.ptr, 0, info.size);

  *out_devalloc = DeviceAllocation{};
  out_devalloc->alloc_id = allocations_.size();
  out_devalloc->device = this;

  allocations_.push_back(info);
  return RhiResult::success;
}

DeviceAllocation AmdgpuDevice::allocate_memory_runtime(const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = quadrants::iroundup(params.size, quadrants_page_size);
  if (params.host_read || params.host_write) {
    QD_NOT_IMPLEMENTED
  } else if (info.size == 0) {
    info.ptr = nullptr;
  } else if (params.use_memory_pool) {
    // Grow-on-demand path (HIP memory pool). Matches CudaDevice::allocate_memory_runtime; the fixed-size
    // device_memory_GB preallocation is skipped entirely upstream when the device supports pools.
    AMDGPUDriver::get_instance().malloc_async((void **)&info.ptr, info.size, nullptr);
  } else {
    info.ptr =
        DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/).allocate_with_cache(this, params);

    if (!info.ptr) {
      DeviceAllocation fail_alloc;
      fail_alloc.alloc_id = kDeviceAllocationFailed;
      fail_alloc.device = this;

      return fail_alloc;
    }
  }

  if (info.ptr)
    AMDGPUDriver::get_instance().memset((void *)info.ptr, 0, info.size);

  info.is_imported = false;
  info.use_cached = true;
  info.use_preallocated = true;
  info.use_memory_pool = params.use_memory_pool;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

uint64_t *AmdgpuDevice::allocate_llvm_runtime_memory_jit(const LlvmRuntimeAllocParams &params) {
  // The device-side runtime_memory_allocate_aligned fires quadrants_assert_runtime on pool exhaustion, which stops
  // the kernel without writing to *result. To detect that here, zero the slot first so a null readback unambiguously
  // means "allocation failed" and we can surface a helpful host-side message instead of letting the downstream
  // hipMemset trip on the stale pointer with a cryptic hipErrorInvalidValue.
  void *active_stream = AMDGPUContext::get_instance().get_stream();
  uint64 zero = 0;
  AMDGPUDriver::get_instance().memcpy_host_to_device_async(params.result_buffer, &zero, sizeof(uint64), active_stream);
  params.runtime_jit->call<void *, std::size_t, std::size_t>("runtime_memory_allocate_aligned", params.runtime,
                                                             params.size, quadrants_page_size, params.result_buffer);
  AMDGPUDriver::get_instance().stream_synchronize(active_stream);
  uint64 *ret{nullptr};
  AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, params.result_buffer, sizeof(uint64));
  QD_ERROR_IF(ret == nullptr,
              "Out of AMDGPU pre-allocated memory. Consider using qd.init(device_memory_fraction=0.9) or "
              "qd.init(device_memory_GB=N) to allocate more GPU memory.");
  return ret;
}

void AmdgpuDevice::dealloc_memory(DeviceAllocation handle) {
  // After reset, all allocations are invalid
  if (allocations_.empty()) {
    return;
  }

  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];

  if (info.size == 0) {
    return;
  }
  if (info.ptr == nullptr) {
    QD_ERROR("the DeviceAllocation is already deallocated");
  }
  QD_ASSERT(!info.is_imported);
  if (info.use_memory_pool) {
    AMDGPUDriver::get_instance().mem_free(info.ptr);
  } else if (info.use_cached) {
    DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/)
        .release(info.size, (uint64_t *)info.ptr, false);
  } else if (!info.use_preallocated) {
    DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/).release(info.size, info.ptr);
  }
  info.ptr = nullptr;
}

RhiResult AmdgpuDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  size_t size = info.size;
  info.mapped = new char[size];
  // FIXME: there should be a better way to do this...
  AMDGPUDriver::get_instance().memcpy_device_to_host(info.mapped, info.ptr, size);
  *mapped_ptr = info.mapped;
  return RhiResult::success;
}

void AmdgpuDevice::unmap(DeviceAllocation alloc) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  AMDGPUDriver::get_instance().memcpy_host_to_device(info.ptr, info.mapped, info.size);
  delete[] static_cast<char *>(info.mapped);
  return;
}

void AmdgpuDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  void *dst_ptr = static_cast<char *>(allocations_[dst.alloc_id].ptr) + dst.offset;
  void *src_ptr = static_cast<char *>(allocations_[src.alloc_id].ptr) + src.offset;
  AMDGPUDriver::get_instance().memcpy_device_to_device(dst_ptr, src_ptr, size);
}

DeviceAllocation AmdgpuDevice::import_memory(void *ptr, size_t size) {
  AllocInfo info;
  info.ptr = ptr;
  info.size = size;
  info.is_imported = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

}  // namespace amdgpu
}  // namespace lang
}  // namespace quadrants
