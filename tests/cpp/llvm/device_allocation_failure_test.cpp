#include "gtest/gtest.h"

#include <cstring>
#include <string>
#include <vector>

#include "quadrants/jit/jit_module.h"
#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_device.h"
#endif
#if defined(QD_WITH_AMDGPU)
#include "quadrants/rhi/amdgpu/amdgpu_device.h"
#endif

namespace quadrants::lang {
namespace {

// Keep the production device's allocation method, including result clearing, readback, and error handling.
// Replace only driver operations with CPU memory copies and observations; no GPU context is initialized.
template <typename Device>
class MockGpuDevice : public Device {
 public:
  void *stream = nullptr;
  std::vector<std::string> operations;

 protected:
  void *allocation_stream() override {
    return stream;
  }

  void allocation_copy_to_device(void *dst, void *src, std::size_t size, void *copy_stream) override {
    EXPECT_EQ(copy_stream, stream);
    operations.push_back("clear");
    std::memcpy(dst, src, size);
  }

  void allocation_synchronize(void *sync_stream) override {
    EXPECT_EQ(sync_stream, stream);
    operations.push_back("synchronize");
  }

  void allocation_copy_to_host(void *dst, void *src, std::size_t size) override {
    operations.push_back("readback");
    std::memcpy(dst, src, size);
  }
};

// Replace the GPU kernel launch. A failed allocation leaves the result slot untouched, as an aborted kernel can.
// A successful allocation writes a synthetic pointer. The production host code must handle both outcomes.
class MockAllocationJit : public JITModule {
 public:
  explicit MockAllocationJit(std::vector<std::string> &operations) : operations_(operations) {
  }

  uint64 *result_buffer = nullptr;
  void *returned_pointer = nullptr;

  bool direct_dispatch() const override {
    return false;
  }

  void *lookup_function(const std::string &) override {
    ADD_FAILURE() << "The GPU allocation path must dispatch a kernel, not call a CPU function.";
    return nullptr;
  }

  void call(const std::string &name,
            const std::vector<void *> &arg_pointers,
            const std::vector<int> &arg_sizes) override {
    operations_.push_back("launch");
    EXPECT_EQ(name, "runtime_memory_allocate_aligned");
    ASSERT_EQ(arg_pointers.size(), 4u);
    EXPECT_EQ(arg_sizes, (std::vector<int>{sizeof(void *), sizeof(std::size_t), sizeof(std::size_t), sizeof(uint64 *)}));
    EXPECT_EQ(*static_cast<void **>(arg_pointers[0]), nullptr);
    EXPECT_EQ(*static_cast<std::size_t *>(arg_pointers[1]), 1u);
    EXPECT_EQ(*static_cast<std::size_t *>(arg_pointers[2]), quadrants_page_size);
    EXPECT_EQ(*static_cast<uint64 **>(arg_pointers[3]), result_buffer);
    if (returned_pointer != nullptr) {
      *result_buffer = reinterpret_cast<std::uintptr_t>(returned_pointer);
    }
  }

 private:
  std::vector<std::string> &operations_;
};

template <typename Device>
void check_allocation_failure_and_success(const std::string &backend, void *stream) {
  void *const MOCK_SUCCESSFUL_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{1024});
  void *const MOCK_STALE_ALLOCATION = reinterpret_cast<void *>(std::uintptr_t{0x0badbeef});

  MockGpuDevice<Device> device;
  device.stream = stream;
  MockAllocationJit mock_jit(device.operations);
  // This local slot stands in for GPU memory. Only the mocked copies and kernel launch access it.
  uint64 mock_device_result = 0;
  mock_jit.result_buffer = &mock_device_result;
  typename Device::LlvmRuntimeAllocParams params;
  params.size = 1;
  params.runtime_jit = &mock_jit;
  params.result_buffer = &mock_device_result;
  void *allocation_test_cases[] = {nullptr, MOCK_SUCCESSFUL_ALLOCATION};
  for (void *returned_pointer : allocation_test_cases) {
    SCOPED_TRACE(returned_pointer);
    mock_device_result = reinterpret_cast<std::uintptr_t>(MOCK_STALE_ALLOCATION);
    mock_jit.returned_pointer = returned_pointer;
    device.operations.clear();
    if (returned_pointer == nullptr) {
      try {
        device.allocate_llvm_runtime_memory_jit(params);
        ADD_FAILURE() << "An interrupted allocation must report exhaustion instead of returning a stale pointer.";
      } catch (const std::string &error) {
        // QD_ERROR throws a C++ string; the Python binding translates it into RuntimeError.
        EXPECT_NE(error.find("Out of " + backend + " pre-allocated memory"), std::string::npos);
      }
    } else {
      EXPECT_EQ(device.allocate_llvm_runtime_memory_jit(params), returned_pointer);
    }
    EXPECT_EQ(device.operations, (std::vector<std::string>{"clear", "launch", "synchronize", "readback"}));
  }
}

#if defined(QD_WITH_CUDA)
TEST(DeviceAllocationFailure, CudaReportsInterruptedAllocation) {
  check_allocation_failure_and_success<cuda::CudaDevice>("CUDA", nullptr);
}
#endif

#if defined(QD_WITH_AMDGPU)
TEST(DeviceAllocationFailure, AmdgpuReportsInterruptedAllocationOnActiveStream) {
  void *const MOCK_ACTIVE_STREAM = reinterpret_cast<void *>(std::uintptr_t{2048});

  check_allocation_failure_and_success<amdgpu::AmdgpuDevice>("AMDGPU", MOCK_ACTIVE_STREAM);
}
#endif

}  // namespace
}  // namespace quadrants::lang
