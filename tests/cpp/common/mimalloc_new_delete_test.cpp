#include "gtest/gtest.h"

#if defined(QD_USE_MIMALLOC)

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#include <malloc.h>
#include <mimalloc.h>

namespace quadrants {

namespace {
struct alignas(128) OverAligned {
  char data[8];
};
}  // namespace

TEST(MimallocNewDelete, NewUsesMimalloc) {
  auto scalar = std::make_unique<int>(42);
  EXPECT_TRUE(mi_is_in_heap_region(scalar.get()));

  std::vector<std::uint64_t> vec(1000, 7);
  EXPECT_TRUE(mi_is_in_heap_region(vec.data()));

  auto array = std::make_unique<char[]>(3);
  EXPECT_TRUE(mi_is_in_heap_region(array.get()));

  auto nothrow = std::unique_ptr<int>(new (std::nothrow) int(1));
  EXPECT_TRUE(mi_is_in_heap_region(nothrow.get()));
}

TEST(MimallocNewDelete, OverAlignedNew) {
  auto p = std::make_unique<OverAligned>();
  EXPECT_TRUE(mi_is_in_heap_region(p.get()));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p.get()) % alignof(OverAligned), 0u);

  std::vector<OverAligned> vec(17);
  EXPECT_TRUE(mi_is_in_heap_region(vec.data()));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(vec.data()) % alignof(OverAligned), 0u);
}

// Memory from the CRT heap (e.g. allocated by another module and handed over) must be returned to the CRT.
TEST(MimallocNewDelete, DeleteForeignPointer) {
  void *p = std::malloc(64);
  ASSERT_NE(p, nullptr);
  EXPECT_FALSE(mi_is_in_heap_region(p));
  ::operator delete(p);

  p = std::malloc(64);
  ASSERT_NE(p, nullptr);
  ::operator delete(p, std::size_t{64});

  void *aligned = _aligned_malloc(256, 128);
  ASSERT_NE(aligned, nullptr);
  EXPECT_FALSE(mi_is_in_heap_region(aligned));
  ::operator delete(aligned, std::align_val_t{128});

  aligned = _aligned_malloc(256, 128);
  ASSERT_NE(aligned, nullptr);
  ::operator delete(aligned, std::size_t{256}, std::align_val_t{128});
}

TEST(MimallocNewDelete, DeleteNull) {
  ::operator delete(nullptr);
  ::operator delete[](nullptr);
  ::operator delete(nullptr, std::align_val_t{64});
}

}  // namespace quadrants

#endif  // defined(QD_USE_MIMALLOC)
