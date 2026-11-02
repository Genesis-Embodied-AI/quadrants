#include "gtest/gtest.h"

#include <cstdint>

#include "quadrants/runtime/llvm/runtime_module/allocation_result.h"

namespace quadrants::lang {
namespace {

struct AllocationInterrupted {};

TEST(AllocationResult, ClearsStalePointerBeforeInterruptedAllocation) {
  std::uint64_t result = 0x1234;
  bool called = false;
  auto allocate = [&]() -> std::uint64_t {
    called = true;
    EXPECT_EQ(result, 0u);
    // Simulate a device-side failure that never returns to the result assignment. Exceptions are used only in this
    // host test; the production runtime still compiles with -fno-exceptions.
    throw AllocationInterrupted{};
  };

  EXPECT_THROW(store_allocation_result(&result, allocate), AllocationInterrupted);
  EXPECT_TRUE(called);
  EXPECT_EQ(result, 0u);
}

TEST(AllocationResult, ReplacesStalePointerWhenAllocationReturnsNull) {
  std::uint64_t result = 0x1234;
  store_allocation_result(&result, []() -> std::uint64_t { return 0; });
  EXPECT_EQ(result, 0u);
}

TEST(AllocationResult, PublishesSuccessfulAllocationAfterClearingStalePointer) {
  std::uint64_t result = 0x1234;
  store_allocation_result(&result, [&]() -> std::uint64_t {
    EXPECT_EQ(result, 0u);
    return 0x5678;
  });
  EXPECT_EQ(result, 0x5678u);
}

}  // namespace
}  // namespace quadrants::lang
