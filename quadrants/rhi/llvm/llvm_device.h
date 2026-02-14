#pragma once

#include "quadrants/rhi/device.h"

namespace quadrants::lang {

class JITModule;
struct LLVMRuntime;

class LlvmDevice : public Device {
 public:
  struct LlvmRuntimeAllocParams : AllocParams {
    JITModule *runtime_jit{nullptr};
    LLVMRuntime *runtime{nullptr};
    uint64 *result_buffer{nullptr};
    bool use_memory_pool{false};
  };

  Arch arch() const override {
    QD_NOT_IMPLEMENTED
  }

  template <typename DEVICE>
  DEVICE *as() {
    auto *device = dynamic_cast<DEVICE *>(this);
    QD_ASSERT(device != nullptr);
    return device;
  }

  virtual void *get_memory_addr(DeviceAllocation devalloc) {
    QD_NOT_IMPLEMENTED
  }

  virtual std::size_t get_total_memory() {
    QD_NOT_IMPLEMENTED
  }

  virtual DeviceAllocation import_memory(void *ptr, size_t size) {
    QD_NOT_IMPLEMENTED
  }

  virtual DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) {
    QD_NOT_IMPLEMENTED;
  }

  virtual void clear() {
    QD_NOT_IMPLEMENTED;
  }

  virtual uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) {
    QD_NOT_IMPLEMENTED;
  }
};

}  // namespace quadrants::lang
