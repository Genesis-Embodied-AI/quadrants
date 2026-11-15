// The runtime source uses variable-length arrays that the Windows MSVC compiler cannot compile.
#if !defined(_WIN32)

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>

// Compile the unchanged production runtime as native CPU code in this test, with exceptions enabled. This lets the
// mock throw through runtime_memory_allocate_aligned and LLVMRuntime::allocate_aligned, so we can inspect the result
// after interrupted allocation. Production runtime bitcode still builds separately with -fno-exceptions.
#include "quadrants/runtime/llvm/runtime_module/runtime.cpp"

namespace quadrants::lang {
namespace {

struct AllocationInterrupted {};

TEST(AllocationResult, RuntimeClearsStalePointerWhenAllocationIsInterrupted) {
  void *const MOCK_STALE_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{0x0badbeef});

  std::uint64_t result = reinterpret_cast<std::uintptr_t>(MOCK_STALE_ALLOCATION);
  std::unique_ptr<::LLVMRuntime> runtime = std::make_unique<::LLVMRuntime>();
  struct MockMemoryPool {
    // Record the allocator's arguments to verify that production code forwards them unchanged.
    std::size_t size = 0;
    std::size_t alignment = 0;
    int calls = 0;
  } mock_memory_pool;
  // Substitute observation data for the real memory pool. The mock never allocates or inspects the result buffer.
  runtime->memory_pool = reinterpret_cast<::Ptr>(&mock_memory_pool);
  // Replace only the host allocator callback. Throwing prevents the final assignment in runtime_memory_allocate_aligned
  // from running. No device or memory exhaustion is needed.
  runtime->host_allocator = [](void *context, std::size_t size, std::size_t alignment) -> void * {
    MockMemoryPool &mock_memory_pool = *static_cast<MockMemoryPool *>(context);
    mock_memory_pool.size = size;
    mock_memory_pool.alignment = alignment;
    ++mock_memory_pool.calls;
    throw AllocationInterrupted{};
  };

  EXPECT_THROW(runtime_memory_allocate_aligned(runtime.get(), std::size_t(1), std::size_t(1), &result),
               AllocationInterrupted);
  EXPECT_EQ(mock_memory_pool.calls, 1);
  EXPECT_EQ(mock_memory_pool.size, 1u);
  EXPECT_EQ(mock_memory_pool.alignment, 1u);
  EXPECT_EQ(result, 0u);
}

TEST(AllocationResult, RuntimeStoresReturnedAllocationResult) {
  void *const MOCK_SUCCESSFUL_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{1024});
  void *const MOCK_STALE_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{0x0badbeef});

  std::unique_ptr<::LLVMRuntime> runtime = std::make_unique<::LLVMRuntime>();
  void *allocation_test_cases[] = {nullptr, MOCK_SUCCESSFUL_ALLOCATION};
  for (void *returned_pointer : allocation_test_cases) {
    SCOPED_TRACE(returned_pointer);
    std::uint64_t result = reinterpret_cast<std::uintptr_t>(MOCK_STALE_ALLOCATION);
    struct MockMemoryPool {
      // Return this pointer to simulate an ordinary failed or successful allocation without allocating memory.
      void *returned_pointer;
    } mock_memory_pool{returned_pointer};
    // Substitute the selected test case for the real memory pool.
    runtime->memory_pool = reinterpret_cast<::Ptr>(&mock_memory_pool);
    // Substitute a host allocator that returns the pointer supplied by this test case.
    runtime->host_allocator = [](void *context, std::size_t, std::size_t) -> void * {
      return static_cast<MockMemoryPool *>(context)->returned_pointer;
    };

    runtime_memory_allocate_aligned(runtime.get(), std::size_t(1), std::size_t(1), &result);
    EXPECT_EQ(result, reinterpret_cast<std::uintptr_t>(returned_pointer));
  }
}

}  // namespace
}  // namespace quadrants::lang

#endif
