#include "gtest/gtest.h"

#include <cstdint>
#include <memory>

#include "quadrants/jit/jit_module.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/runtime/llvm/runtime_module/llvm_runtime.h"

namespace quadrants::lang {
namespace {

// Call the production runtime_memory_allocate_aligned and LLVMRuntime::allocate_aligned through the CPU JIT.
// On a test-owned LLVMRuntime, replace only the host_allocator callback with a fake that records the result before
// returning null or a known pointer. Its memory_pool field carries test observation data instead of a real memory pool.
// The result clearing, callback invocation, and final result assignment all execute unchanged production code.
TEST(AllocationResult, RuntimeClearsStalePointerBeforeCallingAllocator) {
  void *const MOCK_SUCCESSFUL_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{1024});
  void *const MOCK_STALE_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{0x0badbeef});

  CompileConfig config;
  config.arch = host_arch();
  config.cpu_max_num_threads = 1;
  LlvmRuntimeExecutor executor(config, nullptr, nullptr);
  JITModule *jit = executor.get_runtime_jit_module();

  // Use a separate runtime so the fake allocator cannot affect the executor's own allocations or cleanup.
  std::unique_ptr<::LLVMRuntime> runtime = std::make_unique<::LLVMRuntime>();
  void *allocation_test_cases[] = {nullptr, MOCK_SUCCESSFUL_ALLOCATION};
  for (void *returned_pointer : allocation_test_cases) {
    SCOPED_TRACE(returned_pointer == nullptr ? "failed allocation" : "successful allocation");
    std::uint64_t result = reinterpret_cast<std::uintptr_t>(MOCK_STALE_ALLOCATION);
    struct MockMemoryPool {
      std::uint64_t *result;
      void *returned_pointer;
      std::uint64_t result_at_call = 1;
      // Record the allocator's arguments so the test can verify that production code forwards them unchanged.
      std::size_t size = 0;
      std::size_t alignment = 0;
      int calls = 0;
    } mock_memory_pool{&result, returned_pointer};
    // Substitute observation data for the real memory pool. The fake allocator interprets this pointer as
    // MockMemoryPool; no HostMemoryPool object is created or used for this request.
    runtime->memory_pool = reinterpret_cast<::Ptr>(&mock_memory_pool);
    // Substitute a fake host allocator that records the result on entry and returns the selected pointer without
    // allocating memory. The production runtime code must clear the result before invoking this callback.
    runtime->host_allocator = [](void *context, std::size_t size, std::size_t alignment) -> void * {
      MockMemoryPool &mock_memory_pool = *static_cast<MockMemoryPool *>(context);
      mock_memory_pool.result_at_call = *mock_memory_pool.result;
      mock_memory_pool.size = size;
      mock_memory_pool.alignment = alignment;
      ++mock_memory_pool.calls;
      return mock_memory_pool.returned_pointer;
    };

    // Inspect the result inside the allocator: checking only the final null would miss #451, where an interrupted
    // allocation left a stale pointer because the result was not cleared before entering the allocator.
    jit->call("runtime_memory_allocate_aligned", runtime.get(), std::size_t(1), std::size_t(1), &result);
    EXPECT_EQ(mock_memory_pool.calls, 1);
    EXPECT_EQ(mock_memory_pool.result_at_call, 0u);
    EXPECT_EQ(mock_memory_pool.size, 1u);
    EXPECT_EQ(mock_memory_pool.alignment, 1u);
    EXPECT_EQ(result, reinterpret_cast<std::uintptr_t>(returned_pointer));
  }
}

}  // namespace
}  // namespace quadrants::lang
