#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

#include "quadrants/platform/cuda/detect_cuda.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/llvm/device_memory_pool.h"

namespace quadrants::lang {
namespace {

// cuMemAlloc only promises an alignment suitable for any built-in type (256 bytes in practice), which is weaker than
// the page alignment its callers ask for. A handful of sub-page allocations is enough to walk the driver heap onto a
// sub-page-aligned base for the next allocation, which is the state a third-party CUDA library leaves behind when it
// initialises before quadrants in the same process.
constexpr std::size_t kFragmentBytes = 512;
constexpr std::size_t kChunkBytes = 1024 * 1024;
constexpr int kNumAllocations = 8;

bool is_aligned(void *ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

}  // namespace

// CudaDevice::allocate_memory asks the pool for page-aligned memory, and LlvmRuntimeExecutor::materialize_runtime
// depends on getting it: the runtime-objects chunk is sized as an exact sum of page-rounded blocks, while the
// device-side bump allocator in runtime_initialize charges alignment padding against that same budget. A
// sub-page-aligned base therefore overruns the chunk and trips the in-kernel __assertfail, which is sticky and kills
// the process's CUDA context for every subsequent launch.
TEST(DeviceMemoryPool, AllocateHonorsRequestedAlignment) {
  if (!is_cuda_api_available()) {
    GTEST_SKIP() << "CUDA driver not available";
  }

  DeviceMemoryPool pool(Arch::cuda, true /*merge_upon_release*/);
  std::vector<void *> fragments;

  // 64 KiB is checked alongside the page size because the driver satisfies it by chance far less often, so the test
  // still discriminates on machines whose driver happens to hand out page-aligned bases even when fragmented.
  for (std::size_t alignment : {DeviceMemoryPool::page_size, std::size_t(64 * 1024)}) {
    for (int i = 0; i < kNumAllocations; i++) {
      void *fragment = nullptr;
      CUDADriver::get_instance().malloc(&fragment, kFragmentBytes);
      fragments.push_back(fragment);

      void *ptr = pool.allocate(kChunkBytes, alignment);
      ASSERT_NE(ptr, nullptr);
      EXPECT_TRUE(is_aligned(ptr, alignment))
          << "allocation " << i << " returned " << ptr << ", which is not aligned to " << alignment << " bytes (offset "
          << reinterpret_cast<std::uintptr_t>(ptr) % alignment << ")";
    }
  }

  for (void *fragment : fragments) {
    CUDADriver::get_instance().mem_free(fragment);
  }
}

// Honouring the alignment means the pointer handed out may differ from the base the driver returned, and only the
// latter is valid to pass to cuMemFree. Releasing repeatedly would fault or leak if the pool lost track of the base.
TEST(DeviceMemoryPool, AlignedAllocationsCanBeReleased) {
  if (!is_cuda_api_available()) {
    GTEST_SKIP() << "CUDA driver not available";
  }

  DeviceMemoryPool pool(Arch::cuda, true /*merge_upon_release*/);
  std::vector<void *> fragments;

  for (int i = 0; i < kNumAllocations; i++) {
    void *fragment = nullptr;
    CUDADriver::get_instance().malloc(&fragment, kFragmentBytes);
    fragments.push_back(fragment);

    void *ptr = pool.allocate(kChunkBytes, DeviceMemoryPool::page_size);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, DeviceMemoryPool::page_size));
    pool.release(kChunkBytes, ptr, true /*release_raw*/);
  }

  for (void *fragment : fragments) {
    CUDADriver::get_instance().mem_free(fragment);
  }
}

}  // namespace quadrants::lang
