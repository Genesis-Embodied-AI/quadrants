#pragma once
#include "quadrants/common/core.h"
#include "quadrants/rhi/device.h"
#include "quadrants/rhi/llvm/llvm_device.h"
#include "quadrants/rhi/llvm/allocator.h"
#include "quadrants/rhi/arch.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace quadrants::lang {

// A memory pool that runs on the host

class QD_DLL_EXPORT DeviceMemoryPool {
 public:
  std::unique_ptr<CachingAllocator> allocator_{nullptr};
  static const size_t page_size;

  static DeviceMemoryPool &get_instance(Arch arch, bool merge_upon_release = true);

  void *allocate_with_cache(LlvmDevice *device, const LlvmDevice::LlvmRuntimeAllocParams &params);
  void *allocate(std::size_t size, std::size_t alignment, bool managed = false);
  void release(std::size_t size, void *ptr, bool release_raw = false);
  void reset();
  explicit DeviceMemoryPool(Arch arch, bool merge_upon_release);
  ~DeviceMemoryPool();

 protected:
  void *allocate_raw_memory(std::size_t size, std::size_t alignment, bool managed = false);
  void deallocate_raw_memory(void *ptr);

  // The driver only guarantees an alignment suitable for any built-in type, so satisfying a larger requested
  // alignment means handing out a pointer inside the driver's block rather than its base. `base` is the only address
  // the driver will accept back, so it has to be kept alongside.
  struct RawMemoryChunk {
    void *base = nullptr;
    std::size_t size = 0;
  };

  // All the raw memory allocated from OS/Driver, keyed by the aligned pointer handed to callers.
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, RawMemoryChunk> raw_memory_chunks_;

  std::mutex mut_allocation_;
  bool merge_upon_release_ = true;
  Arch arch_;
};

}  // namespace quadrants::lang
