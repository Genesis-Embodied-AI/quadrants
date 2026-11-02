#include "gtest/gtest.h"

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "quadrants/rhi/llvm/gpu_device_impl.h"

namespace quadrants::lang {
namespace {

// Host storage substitutes for GPU memory. Record driver and pool operations without loading either GPU driver.
struct MockDriver {
  std::array<uint8_t, 8192> memory{};
  std::vector<std::string> operations;
  void *expected_stream = nullptr;
  int pooled_allocations = 0;

  void memset(void *ptr, uint8_t value, std::size_t size) {
    std::memset(ptr, value, size);
  }
  void memset_async(void *ptr, uint8_t value, std::size_t size, void *stream) {
    EXPECT_EQ(stream, expected_stream);
    operations.push_back("clear");
    std::memset(ptr, value, size);
  }
  void malloc_async(void **ptr, std::size_t size, void *stream) {
    EXPECT_LE(size, memory.size());
    EXPECT_EQ(stream, nullptr);
    ++pooled_allocations;
    *ptr = memory.data();
  }
  void stream_synchronize(void *stream) {
    EXPECT_EQ(stream, expected_stream);
    operations.push_back("synchronize");
  }
  void memcpy_host_to_device(void *dst, void *src, std::size_t size) {
    std::memcpy(dst, src, size);
  }
  void memcpy_device_to_host(void *dst, void *src, std::size_t size) {
    operations.push_back("readback");
    std::memcpy(dst, src, size);
  }
  void memcpy_device_to_device(void *dst, void *src, std::size_t size) {
    std::memcpy(dst, src, size);
  }
};

struct MockContext {
  void *stream = nullptr;
  void *get_stream() {
    return stream;
  }
  std::size_t get_total_memory() {
    return 8192;
  }
};

struct MockMemoryPool {
  std::array<uint8_t, 8192> memory{};
  bool fail = false;
  bool managed = false;
  int raw_allocations = 0;
  int cached_allocations = 0;
  std::vector<bool> raw_releases;

  void *allocate(std::size_t size, std::size_t alignment, bool use_managed) {
    EXPECT_LE(size, memory.size());
    EXPECT_EQ(alignment, DeviceMemoryPool::page_size);
    ++raw_allocations;
    managed = use_managed;
    return fail ? nullptr : memory.data();
  }
  uint64 *allocate_with_cache(LlvmDevice *, const LlvmDevice::LlvmRuntimeAllocParams &params) {
    EXPECT_LE(params.size, memory.size());
    ++cached_allocations;
    return fail ? nullptr : reinterpret_cast<uint64 *>(memory.data());
  }
  void release(std::size_t, void *ptr, bool raw) {
    EXPECT_EQ(ptr, memory.data());
    raw_releases.push_back(raw);
  }
};

struct MockState {
  MockDriver driver;
  MockContext context;
  MockMemoryPool pool;
  std::string pooled_free;
} mock_state;

// These adapters select the same policy combinations as the real CUDA and AMDGPU adapters. Every operation below
// runs the shared production GpuDevice implementation; only driver/context/pool dependencies are mocked.
struct MockCudaBackend {
  static constexpr const char *name = "CUDA";
  static constexpr bool reject_runtime_host_access = false;
  static constexpr bool release_raw_allocations = true;
  static MockDriver &driver() {
    return mock_state.driver;
  }
  static MockContext &context() {
    return mock_state.context;
  }
  static MockMemoryPool &memory_pool() {
    return mock_state.pool;
  }
  static void free_pooled_memory(void *) {
    mock_state.pooled_free = "async";
  }
};

struct MockAmdgpuBackend : MockCudaBackend {
  static constexpr const char *name = "AMDGPU";
  static constexpr bool reject_runtime_host_access = true;
  static constexpr bool release_raw_allocations = false;
  static void free_pooled_memory(void *) {
    mock_state.pooled_free = "sync";
  }
};

template <typename Backend>
class GpuDeviceTest : public testing::Test {
 protected:
  void SetUp() override {
    mock_state = MockState{};
    device = std::make_unique<GpuDevice<Backend>>();
  }
  std::unique_ptr<GpuDevice<Backend>> device;
};
using Backends = testing::Types<MockCudaBackend, MockAmdgpuBackend>;
TYPED_TEST_SUITE(GpuDeviceTest, Backends);

TYPED_TEST(GpuDeviceTest, RawAllocationIsZeroedAndReleasedWithBackendPolicy) {
  mock_state.pool.memory.fill(0x7f);
  Device::AllocParams params{};
  params.size = 16;
  params.host_read = true;
  DeviceAllocation allocation;
  ASSERT_EQ(this->device->allocate_memory(params, &allocation), RhiResult::success);
  EXPECT_EQ(allocation.device, this->device.get());
  EXPECT_EQ(this->device->get_memory_addr(allocation), mock_state.pool.memory.data());
  EXPECT_TRUE(mock_state.pool.managed);
  EXPECT_EQ(mock_state.pool.memory[0], 0);
  EXPECT_EQ(mock_state.pool.memory[15], 0);
  this->device->dealloc_memory(allocation);
  EXPECT_EQ(mock_state.pool.raw_releases, std::vector<bool>{TypeParam::release_raw_allocations});
  EXPECT_EQ(this->device->get_alloc_info(allocation).ptr, nullptr);
}

TYPED_TEST(GpuDeviceTest, RuntimeAllocationUsesCacheOrDriverPool) {
  for (bool use_memory_pool : {false, true}) {
    SCOPED_TRACE(use_memory_pool);
    LlvmDevice::LlvmRuntimeAllocParams params;
    params.size = 17;
    params.use_memory_pool = use_memory_pool;
    DeviceAllocation allocation = this->device->allocate_memory_runtime(params);
    ASSERT_TRUE(allocation.is_valid());
    EXPECT_EQ(this->device->get_alloc_info(allocation).size, quadrants_page_size);
    this->device->dealloc_memory(allocation);
  }
  EXPECT_EQ(mock_state.pool.cached_allocations, 1);
  EXPECT_EQ(mock_state.driver.pooled_allocations, 1);
  EXPECT_EQ(mock_state.pool.raw_releases, std::vector<bool>{false});
  EXPECT_EQ(mock_state.pooled_free, TypeParam::reject_runtime_host_access ? "sync" : "async");
}

TYPED_TEST(GpuDeviceTest, ZeroSizeAndAllocationFailures) {
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 0;
  DeviceAllocation empty = this->device->allocate_memory_runtime(params);
  EXPECT_EQ(this->device->get_memory_addr(empty), nullptr);
  this->device->dealloc_memory(empty);
  EXPECT_EQ(mock_state.pool.cached_allocations, 0);
  EXPECT_EQ(mock_state.driver.pooled_allocations, 0);

  mock_state.pool.fail = true;
  params.size = 16;
  EXPECT_FALSE(this->device->allocate_memory_runtime(params).is_valid());
  DeviceAllocation allocation;
  EXPECT_EQ(this->device->allocate_memory(params, &allocation), RhiResult::out_of_memory);
}

TYPED_TEST(GpuDeviceTest, RuntimeHostAccessPreservesBackendPolicy) {
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 16;
  params.host_read = true;
  if constexpr (TypeParam::reject_runtime_host_access) {
    EXPECT_THROW(this->device->allocate_memory_runtime(params), std::string);
  } else {
    EXPECT_TRUE(this->device->allocate_memory_runtime(params).is_valid());
  }
}

TYPED_TEST(GpuDeviceTest, UploadReadbackMapAndCopyUseOffsets) {
  std::array<uint8_t, 8> first{}, second{};
  DeviceAllocation a = this->device->import_memory(first.data(), first.size());
  DeviceAllocation b = this->device->import_memory(second.data(), second.size());
  DevicePtr dst{a};
  dst.offset = 2;
  uint8_t input[] = {41, 42};
  const void *source = input;
  std::size_t size = sizeof(input);
  ASSERT_EQ(this->device->upload_data(&dst, &source, &size), RhiResult::success);
  EXPECT_EQ(first[2], 41);
  EXPECT_EQ(first[3], 42);
  uint8_t output[2]{};
  void *target = output;
  ASSERT_EQ(this->device->readback_data(&dst, &target, &size), RhiResult::success);
  EXPECT_EQ(output[0], 41);
  EXPECT_EQ(output[1], 42);
  DevicePtr other{b};
  other.offset = 1;
  this->device->memcpy_internal(other, dst, size);
  EXPECT_EQ(second[1], 41);
  EXPECT_EQ(second[2], 42);
  void *mapped = nullptr;
  ASSERT_EQ(this->device->map(a, &mapped), RhiResult::success);
  static_cast<uint8_t *>(mapped)[0] = 99;
  this->device->unmap(a);
  EXPECT_EQ(first[0], 99);
  EXPECT_EQ(this->device->get_total_memory(), 8192u);
  EXPECT_EQ(this->device->upload_data(nullptr, &source, &size), RhiResult::invalid_usage);
  EXPECT_EQ(this->device->readback_data(&dst, nullptr, &size), RhiResult::invalid_usage);
  this->device->clear();
  EXPECT_NO_THROW(this->device->dealloc_memory(a));
  EXPECT_THROW(this->device->get_alloc_info(a), std::string);
}

// A failed GPU launch leaves the result untouched. A successful launch writes a known pointer without allocating.
class MockAllocationJit : public JITModule {
 public:
  uint64 *result = nullptr;
  void *returned_pointer = nullptr;
  bool direct_dispatch() const override {
    return false;
  }
  void *lookup_function(const std::string &) override {
    return nullptr;
  }
  void call(const std::string &name, const std::vector<void *> &, const std::vector<int> &) override {
    EXPECT_EQ(name, "runtime_memory_allocate_aligned");
    mock_state.driver.operations.push_back("launch");
    if (returned_pointer) {
      *result = reinterpret_cast<std::uintptr_t>(returned_pointer);
    }
  }
};

TYPED_TEST(GpuDeviceTest, RuntimeAllocationClearsAndWaitsOnActiveStream) {
  constexpr uint64 MOCK_STALE_ALLOCATION = 0x0badbeef;
  void *const MOCK_ACTIVE_STREAM = reinterpret_cast<void *>(std::uintptr_t{1024});

  void *streams[] = {nullptr, MOCK_ACTIVE_STREAM};
  for (void *stream : streams) {
    SCOPED_TRACE(stream);
    mock_state.context.stream = stream;
    mock_state.driver.expected_stream = stream;
    uint64 result = MOCK_STALE_ALLOCATION;
    MockAllocationJit jit;
    jit.result = &result;
    LlvmDevice::LlvmRuntimeAllocParams params;
    params.result_buffer = &result;
    params.runtime_jit = &jit;
    mock_state.driver.operations.clear();
    EXPECT_THROW(this->device->allocate_llvm_runtime_memory_jit(params), std::string);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(mock_state.driver.operations, (std::vector<std::string>{"clear", "launch", "synchronize", "readback"}));
    jit.returned_pointer = mock_state.pool.memory.data();
    EXPECT_EQ(this->device->allocate_llvm_runtime_memory_jit(params), jit.returned_pointer);
  }
}

}  // namespace
}  // namespace quadrants::lang
