#pragma once
#include <vector>
#include <set>

#include "quadrants/common/core.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/llvm/llvm_device.h"
#include "quadrants/rhi/llvm/allocator.h"

namespace quadrants {
namespace lang {

namespace amdgpu {

class AmdgpuCommandList : public CommandList {
 public:
  ~AmdgpuCommandList() override {
  }

  void bind_pipeline(Pipeline *p) noexcept final { QD_NOT_IMPLEMENTED };
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  RhiResult bind_raster_resources(RasterResources *res) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  void buffer_barrier(DevicePtr ptr, size_t size) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  void buffer_barrier(DeviceAllocation alloc) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  void memory_barrier() noexcept final { QD_NOT_IMPLEMENTED };
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  RhiResult dispatch(uint32_t x,
                     uint32_t y = 1,
                     uint32_t z = 1) noexcept override {
    QD_NOT_IMPLEMENTED
  };
};

class AmdgpuStream : public Stream {
 public:
  ~AmdgpuStream() override {};

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final {
    QD_NOT_IMPLEMENTED
  };
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override {
    QD_NOT_IMPLEMENTED
  };
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override {
    QD_NOT_IMPLEMENTED
  };

  void command_sync() override { QD_NOT_IMPLEMENTED };
};

class AmdgpuDevice : public LlvmDevice {
 public:
  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
    bool use_preallocated{true};
    bool use_cached{false};
    void *mapped{nullptr};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);

  AmdgpuDevice();
  ~AmdgpuDevice() override {};

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) override;

  ShaderResourceSet *create_resource_set() final { QD_NOT_IMPLEMENTED };

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final {
    QD_NOT_IMPLEMENTED;
  }

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final {
    QD_NOT_IMPLEMENTED;
  }
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) override { QD_NOT_IMPLEMENTED };
  void unmap(DeviceAllocation alloc) override;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  DeviceAllocation import_memory(void *ptr, size_t size) override;

  void *get_memory_addr(DeviceAllocation devalloc) override {
    return get_alloc_info(devalloc).ptr;
  }

  std::size_t get_total_memory() override {
    return AMDGPUContext::get_instance().get_total_memory();
  }

  Stream *get_compute_stream() override { QD_NOT_IMPLEMENTED };

  void wait_idle() override { QD_NOT_IMPLEMENTED };

  void clear() override {
    allocations_.clear();
  }

 private:
  std::vector<AllocInfo> allocations_;
  void validate_device_alloc(const DeviceAllocation alloc) {
    if (allocations_.size() <= alloc.alloc_id) {
      QD_ERROR("invalid DeviceAllocation");
    }
  }
};

}  // namespace amdgpu

}  // namespace lang

}  // namespace quadrants
