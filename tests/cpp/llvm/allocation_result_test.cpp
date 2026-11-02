#include "gtest/gtest.h"

#include <cstdint>

#include "quadrants/runtime/llvm/runtime_module/allocation_result.h"

namespace quadrants::lang {
namespace {

struct AllocationInterrupted {};

// Exercise the same clear-and-store helper used by the production runtime. The fake allocator throws before it
// returns, so the helper's final assignment never runs. Exceptions are confined to this host test; the production
// runtime still compiles with -fno-exceptions.
TEST(AllocationResult, ClearsStalePointerWhenAllocationIsInterrupted) {
  constexpr std::uint64_t MOCK_STALE_ALLOCATION = 0x0badbeef;

  std::uint64_t result = MOCK_STALE_ALLOCATION;
  int calls = 0;
  EXPECT_THROW(store_allocation_result(&result,
                                       [&]() -> std::uint64_t {
                                         ++calls;
                                         throw AllocationInterrupted{};
                                       }),
               AllocationInterrupted);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(result, 0u);
}

TEST(AllocationResult, StoresReturnedAllocationResult) {
  constexpr std::uint64_t MOCK_SUCCESSFUL_ALLOCATION = 1024;
  constexpr std::uint64_t MOCK_STALE_ALLOCATION = 0x0badbeef;

  std::uint64_t allocation_test_cases[] = {0, MOCK_SUCCESSFUL_ALLOCATION};
  for (std::uint64_t returned_value : allocation_test_cases) {
    SCOPED_TRACE(returned_value);
    std::uint64_t result = MOCK_STALE_ALLOCATION;
    store_allocation_result(&result, [&]() -> std::uint64_t { return returned_value; });
    EXPECT_EQ(result, returned_value);
  }
}

}  // namespace
}  // namespace quadrants::lang
