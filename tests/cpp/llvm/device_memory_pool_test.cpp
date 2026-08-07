#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

#include "quadrants/math/arithmetic.h"
#include "quadrants/rhi/llvm/device_memory_pool.h"

namespace quadrants::lang {
namespace {

constexpr std::size_t kChunkBytes = 1024 * 1024;

// cuMemAlloc/hipMalloc only promise an alignment suitable for any built-in type (256 bytes in practice), which is
// weaker than the page alignment their callers ask for. A handful of sub-page allocations is enough to walk the driver
// heap onto a sub-page-aligned base, which is the state a third-party CUDA library leaves behind when it initialises
// before quadrants in the same process. Whether a real driver actually drifts depends on its internal heap state, so
// the drift is stubbed here instead: the pool has to close the gap regardless of what the driver hands back.
class DriftingDriverPool : public DeviceMemoryPool {
 public:
  explicit DriftingDriverPool(std::size_t driver_offset)
      : DeviceMemoryPool(Arch::cuda, true /*merge_upon_release*/), driver_offset_(driver_offset) {
  }

  ~DriftingDriverPool() override {
    // The base destructor also resets, but by then the overrides below are gone and the base would try to hand these
    // host pointers to a device driver.
    reset();
  }

 protected:
  void *allocate_driver_memory(std::size_t size, bool managed) override {
    const std::size_t slack = DeviceMemoryPool::page_size + driver_offset_;
    char *raw = static_cast<char *>(std::malloc(size + slack));
    if (raw == nullptr) {
      return nullptr;
    }
    const std::uintptr_t page = iroundup(reinterpret_cast<std::uintptr_t>(raw), DeviceMemoryPool::page_size);
    char *base = reinterpret_cast<char *>(page) + driver_offset_;
    malloc_bases_[base] = raw;
    return base;
  }

  void deallocate_driver_memory(void *ptr) override {
    auto it = malloc_bases_.find(ptr);
    ASSERT_NE(it, malloc_bases_.end()) << "pool freed " << ptr << ", which is not a base this driver returned";
    std::free(it->second);
    malloc_bases_.erase(it);
  }

 private:
  std::size_t driver_offset_;
  std::map<void *, char *> malloc_bases_;
};

bool is_aligned(void *ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

}  // namespace

// CudaDevice::allocate_memory asks the pool for page-aligned memory, and LlvmRuntimeExecutor::materialize_runtime
// depends on getting it: the runtime-objects chunk is sized as an exact sum of page-rounded blocks, while the
// device-side bump allocator in runtime_initialize charges alignment padding against that same budget. A
// sub-page-aligned base therefore overruns the chunk and trips the in-kernel __assertfail, which is sticky and kills
// the process's CUDA context for every subsequent launch.
TEST(DeviceMemoryPool, AllocateHonorsRequestedAlignmentWhenDriverDoesNot) {
  // 256 is the alignment cuMemAlloc actually promises; the others are arbitrary sub-page drifts.
  for (std::size_t driver_offset : {std::size_t(256), std::size_t(512), std::size_t(1024), std::size_t(3840)}) {
    for (std::size_t alignment : {DeviceMemoryPool::page_size, std::size_t(64 * 1024)}) {
      DriftingDriverPool pool(driver_offset);

      void *ptr = pool.allocate(kChunkBytes, alignment);
      ASSERT_NE(ptr, nullptr);
      EXPECT_TRUE(is_aligned(ptr, alignment))
          << "driver base offset " << driver_offset << " leaked into the pointer handed out: " << ptr
          << " is off by " << reinterpret_cast<std::uintptr_t>(ptr) % alignment << " bytes of the requested "
          << alignment << "-byte alignment";
    }
  }
}

// Honouring the alignment means the pointer handed out may differ from the base the driver returned, and only the
// latter is valid to free. Releasing would fault or leak if the pool lost track of the base.
TEST(DeviceMemoryPool, AlignedAllocationsAreFreedAtTheDriverBase) {
  DriftingDriverPool pool(512);

  for (int i = 0; i < 8; i++) {
    void *ptr = pool.allocate(kChunkBytes, DeviceMemoryPool::page_size);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, DeviceMemoryPool::page_size));
    pool.release(kChunkBytes, ptr, true /*release_raw*/);
  }
}

}  // namespace quadrants::lang
