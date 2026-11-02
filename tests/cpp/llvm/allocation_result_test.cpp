#include "gtest/gtest.h"

#include <cstdint>
#include <memory>

#include "quadrants/jit/jit_module.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/runtime/llvm/runtime_module/llvm_runtime.h"

namespace quadrants::lang {
namespace {

TEST(AllocationResult, RuntimeClearsStalePointerBeforeCallingAllocator) {
  CompileConfig config;
  config.arch = host_arch();
  config.cpu_max_num_threads = 1;
  LlvmRuntimeExecutor executor(config, nullptr, nullptr);
  auto *jit = executor.get_runtime_jit_module();

  // Use a separate runtime so the fake allocator cannot affect the executor's own allocations or cleanup.
  auto runtime = std::make_unique<::LLVMRuntime>();
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
      auto &state = *static_cast<AllocationState *>(context);
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
