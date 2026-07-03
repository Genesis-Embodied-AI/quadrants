#include "gtest/gtest.h"

#include "quadrants/rhi/common/host_memory_pool.h"

namespace quadrants::lang {

class HostMemoryPoolTestHelper {
 public:
  static void setDefaultAllocatorSize(std::size_t size) {
    UnifiedAllocator::default_allocator_size = size;
  }
  static size_t getDefaultAllocatorSize() {
    return UnifiedAllocator::default_allocator_size;
  }
};

TEST(HostMemoryPool, AllocateMemory) {
  auto oldAllocatorSize = HostMemoryPoolTestHelper::getDefaultAllocatorSize();
  HostMemoryPoolTestHelper::setDefaultAllocatorSize(102400);  // 100KB

  HostMemoryPool pool;

  void *ptr1 = pool.allocate(1024, 16);
  void *ptr2 = pool.allocate(1024, 16);
  void *ptr3 = pool.allocate(1024, 16);

  EXPECT_NE(ptr1, ptr2);
  EXPECT_NE(ptr1, ptr3);
  EXPECT_NE(ptr2, ptr3);

  EXPECT_EQ((std::size_t)ptr2, (std::size_t)ptr1 + 1024);
  EXPECT_EQ((std::size_t)ptr3, (std::size_t)ptr2 + 1024);

  HostMemoryPoolTestHelper::setDefaultAllocatorSize(oldAllocatorSize);
}

TEST(HostMemoryPool, ChunkTailMatchesAllocationSize) {
  auto oldAllocatorSize = HostMemoryPoolTestHelper::getDefaultAllocatorSize();
  HostMemoryPoolTestHelper::setDefaultAllocatorSize(102400);  // 100KB

  HostMemoryPool pool;

  // The first allocation creates a 100KB chunk. The second does not fit in the remaining 40KB, so
  // it must open a new chunk instead of being placed past the real end of the first one.
  void *ptr1 = pool.allocate(61440, 16);
  void *ptr2 = pool.allocate(81920, 16);

  EXPECT_NE((std::size_t)ptr2, (std::size_t)ptr1 + 61440);

  HostMemoryPoolTestHelper::setDefaultAllocatorSize(oldAllocatorSize);
}

}  // namespace quadrants::lang
