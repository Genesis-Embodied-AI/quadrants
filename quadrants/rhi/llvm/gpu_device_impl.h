#pragma once

#include "quadrants/rhi/llvm/gpu_device.h"
#include "quadrants/rhi/llvm/device_memory_pool.h"
#include "quadrants/jit/jit_module.h"

namespace quadrants::lang {

template <typename Backend>
GpuDevice<Backend>::GpuDevice() {
  // Initialize the device memory pool
  Backend::memory_pool();
}

template <typename Backend>
typename GpuDevice<Backend>::AllocInfo GpuDevice<Backend>::get_alloc_info(const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

template <typename Backend>
RhiResult GpuDevice<Backend>::allocate_memory(const AllocParams &params, DeviceAllocation *out_devalloc) {
  AllocInfo info;

  auto &mem_pool = Backend::memory_pool();

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

  Backend::driver().memset((void *)info.ptr, 0, info.size);

  *out_devalloc = DeviceAllocation{};
  out_devalloc->alloc_id = allocations_.size();
  out_devalloc->device = this;

  allocations_.push_back(info);
  return RhiResult::success;
}

template <typename Backend>
DeviceAllocation GpuDevice<Backend>::allocate_memory_runtime(const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = quadrants::iroundup(params.size, quadrants_page_size);
  if constexpr (Backend::reject_runtime_host_access) {
    if (params.host_read || params.host_write) {
      QD_NOT_IMPLEMENTED;
    }
  }
  if (info.size == 0) {
    info.ptr = nullptr;
  } else if (params.use_memory_pool) {
    Backend::driver().malloc_async((void **)&info.ptr, info.size, nullptr);
  } else {
    info.ptr =
        Backend::memory_pool().allocate_with_cache(this, params);

    if (!info.ptr) {
      DeviceAllocation fail_alloc;
      fail_alloc.alloc_id = kDeviceAllocationFailed;
      fail_alloc.device = this;

      return fail_alloc;
    }
  }

  if (info.ptr)
    Backend::driver().memset((void *)info.ptr, 0, info.size);

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

template <typename Backend>
uint64_t *GpuDevice<Backend>::allocate_llvm_runtime_memory_jit(const LlvmRuntimeAllocParams &params) {
  // An aborted allocation can leave the result slot untouched. Clear it first so the host detects failure instead
  // of using a stale pointer. Clear and wait on the allocation kernel's stream; a device memset needs no host buffer.
  void *active_stream = Backend::context().get_stream();
  Backend::driver().memset_async(params.result_buffer, 0, sizeof(uint64), active_stream);
  params.runtime_jit->call<void *, std::size_t, std::size_t>("runtime_memory_allocate_aligned", params.runtime,
                                                             params.size, quadrants_page_size, params.result_buffer);
  Backend::driver().stream_synchronize(active_stream);
  uint64 *ret{nullptr};
  Backend::driver().memcpy_device_to_host(&ret, params.result_buffer, sizeof(uint64));
  QD_ERROR_IF(ret == nullptr,
              "Out of {} pre-allocated memory. Consider using qd.init(device_memory_fraction=0.9) or "
              "qd.init(device_memory_GB=N) to allocate more GPU memory.", Backend::name);
  return ret;
}

template <typename Backend>
void GpuDevice<Backend>::dealloc_memory(DeviceAllocation handle) {
  // After reset, all allocations are invalid
  if (allocations_.empty())
    return;

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
    Backend::free_pooled_memory(info.ptr);
  } else if (info.use_cached) {
    Backend::memory_pool()
        .release(info.size, (uint64_t *)info.ptr, false);
  } else if (!info.use_preallocated) {
    auto &mem_pool = Backend::memory_pool();
    mem_pool.release(info.size, info.ptr, Backend::release_raw_allocations);
  }
  info.ptr = nullptr;
}

template <typename Backend>
RhiResult GpuDevice<Backend>::upload_data(DevicePtr *device_ptr, const void **data, size_t *size, int num_alloc) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }

    AllocInfo &info = allocations_[device_ptr[i].alloc_id];
    Backend::driver().memcpy_host_to_device((uint8_t *)info.ptr + device_ptr[i].offset, (void *)data[i],
                                                     size[i]);
  }

  return RhiResult::success;
}

template <typename Backend>
RhiResult GpuDevice<Backend>::readback_data(DevicePtr *device_ptr,
                                    void **data,
                                    size_t *size,
                                    int num_alloc,
                                    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }

    AllocInfo &info = allocations_[device_ptr[i].alloc_id];
    Backend::driver().memcpy_device_to_host(data[i], (uint8_t *)info.ptr + device_ptr[i].offset, size[i]);
  }

  return RhiResult::success;
}

template <typename Backend>
RhiResult GpuDevice<Backend>::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  size_t size = info.size;
  info.mapped = new char[size];
  // FIXME: there should be a better way to do this...
  Backend::driver().memcpy_device_to_host(info.mapped, info.ptr, size);
  *mapped_ptr = info.mapped;
  return RhiResult::success;
}

template <typename Backend>
void GpuDevice<Backend>::unmap(DeviceAllocation alloc) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  Backend::driver().memcpy_host_to_device(info.ptr, info.mapped, info.size);
  delete[] static_cast<char *>(info.mapped);
  return;
}

template <typename Backend>
void GpuDevice<Backend>::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  void *dst_ptr = static_cast<char *>(allocations_[dst.alloc_id].ptr) + dst.offset;
  void *src_ptr = static_cast<char *>(allocations_[src.alloc_id].ptr) + src.offset;
  Backend::driver().memcpy_device_to_device(dst_ptr, src_ptr, size);
}

template <typename Backend>
DeviceAllocation GpuDevice<Backend>::import_memory(void *ptr, size_t size) {
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

}  // namespace quadrants::lang
