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
  CompileConfig config;
  config.arch = host_arch();
  config.cpu_max_num_threads = 1;
  LlvmRuntimeExecutor executor(config, nullptr, nullptr);
  JITModule *jit = executor.get_runtime_jit_module();

  // Use a separate runtime so the fake allocator cannot affect the executor's own allocations or cleanup.
  std::unique_ptr<::LLVMRuntime> runtime = std::make_unique<::LLVMRuntime>();
  std::uint8_t storage = 0;
  void *returned_pointers[] = {nullptr, &storage};
  for (void *returned_pointer : returned_pointers) {
    SCOPED_TRACE(returned_pointer == nullptr ? "failed allocation" : "successful allocation");
    std::uint64_t result = reinterpret_cast<std::uintptr_t>(&storage);
    struct AllocationState {
      std::uint64_t *result;
      void *returned_pointer;
      std::uint64_t result_at_call = 1;
      std::size_t size = 0;
      std::size_t alignment = 0;
      int calls = 0;
    } state{&result, returned_pointer};
    runtime->memory_pool = reinterpret_cast<::Ptr>(&state);
    runtime->host_allocator = [](void *context, std::size_t size, std::size_t alignment) -> void * {
      AllocationState &state = *static_cast<AllocationState *>(context);
      state.result_at_call = *state.result;
      state.size = size;
      state.alignment = alignment;
      ++state.calls;
      return state.returned_pointer;
    };

    // Inspect the result inside the allocator: checking only the final null would miss #451, where an interrupted
    // allocation left a stale pointer because the result was not cleared before entering the allocator.
    jit->call("runtime_memory_allocate_aligned", runtime.get(), std::size_t(1), std::size_t(1), &result);
    EXPECT_EQ(state.calls, 1);
    EXPECT_EQ(state.result_at_call, 0u);
    EXPECT_EQ(state.size, 1u);
    EXPECT_EQ(state.alignment, 1u);
    EXPECT_EQ(result, reinterpret_cast<std::uintptr_t>(returned_pointer));
  }
}

}  // namespace
}  // namespace quadrants::lang
