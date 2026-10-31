#include "gtest/gtest.h"

#ifdef QD_WITH_AMDGPU
#include "gmock/gmock.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/runtime/amdgpu/amdgpu_utils.h"

namespace quadrants::lang {

// Check that a real HIP device allocation is classified as GPU memory.
TEST(AMDGPUInterop, DevicePointerClassification) {
  auto &driver = AMDGPUDriver::get_instance();
  void *device_ptr = nullptr;
  driver.malloc(&device_ptr, 65536);
  EXPECT_TRUE(amdgpu::on_amdgpu_device(device_ptr));
  driver.mem_free(device_ptr);
}

// Check that registered CPU memory is rejected as GPU memory even when its attribute query succeeds.
TEST(AMDGPUInterop, RegisteredHostPointerIsNotDeviceMemory) {
  // Query registered memory, not an ordinary malloc pointer whose attribute query could simply fail.
  auto &driver = AMDGPUDriver::get_instance();
  DynamicLoader hip("libamdhip64.so");
  uint32 (*host_register)(void *, std::size_t, uint32);
  uint32 (*host_unregister)(void *);
  hip.load_function("hipHostRegister", host_register);
  hip.load_function("hipHostUnregister", host_unregister);
  alignas(4096) char host_memory[4096] = {};
  ASSERT_EQ(host_register(host_memory, sizeof(host_memory), 0), HIP_SUCCESS);
  unsigned int attributes[8] = {};
  EXPECT_EQ(driver.mem_get_attributes.call(attributes, host_memory), HIP_SUCCESS);
  EXPECT_EQ(attributes[0], 1u);  // hipMemoryTypeHost in HIP 6+.
  EXPECT_FALSE(amdgpu::on_amdgpu_device(host_memory));
  EXPECT_EQ(host_unregister(host_memory), HIP_SUCCESS);
}

// Check that a reported HIP 5 runtime is rejected before HIP initialization.
TEST(AMDGPUInterop, RejectOldRuntimeBeforeInitialization) {
  auto &driver = AMDGPUDriver::get_instance_without_context();
  auto original = driver.runtime_get_version;
  driver.runtime_get_version = [](int *version) -> uint32 {
    *version = 50700000;
    return HIP_SUCCESS;
  };
  try {
    AMDGPUContext context;
    ADD_FAILURE() << "HIP 5 must be rejected";
  } catch (const std::string &error) {
    EXPECT_THAT(error, ::testing::HasSubstr("requires HIP 6.0 or newer"));
  } catch (...) {
    driver.runtime_get_version = original;
    throw;
  }
  driver.runtime_get_version = original;
}

// Check that a failed version query is rejected even when it writes a supported version number.
TEST(AMDGPUInterop, RejectFailedRuntimeVersionQuery) {
  auto &driver = AMDGPUDriver::get_instance_without_context();
  auto original = driver.runtime_get_version;
  driver.runtime_get_version = [](int *version) -> uint32 {
    // Even a plausible output must not be trusted when the API reports failure.
    *version = 60000000;
    return 1;
  };
  try {
    AMDGPUContext context;
    ADD_FAILURE() << "A failed HIP version query must be rejected";
  } catch (const std::string &error) {
    EXPECT_THAT(error, ::testing::HasSubstr("Cannot query the loaded HIP runtime version"));
  } catch (...) {
    driver.runtime_get_version = original;
    throw;
  }
  driver.runtime_get_version = original;
}

// Check that reporting exactly HIP 6.0 allows context initialization on the installed runtime.
TEST(AMDGPUInterop, AcceptMinimumRuntimeVersion) {
  auto &driver = AMDGPUDriver::get_instance_without_context();
  auto original = driver.runtime_get_version;
  driver.runtime_get_version = [](int *version) -> uint32 {
    *version = 60000000;
    return HIP_SUCCESS;
  };
  EXPECT_NO_THROW({
    AMDGPUContext context;
    EXPECT_TRUE(context.detected());
  });
  driver.runtime_get_version = original;
}

}  // namespace quadrants::lang
#endif
