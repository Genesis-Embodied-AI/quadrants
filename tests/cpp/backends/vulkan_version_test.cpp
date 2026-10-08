#ifdef QD_WITH_VULKAN

#include <stdexcept>

#include "gtest/gtest.h"
#include "quadrants/common/logging.h"
#include "quadrants/rhi/vulkan/vulkan_device_creator.h"

namespace quadrants::lang::vulkan {

TEST(VulkanVersion, RejectsOlderExplicitVersionsBeforeLoadingDriver) {
  for (uint32_t version : {0u, VK_API_VERSION_1_0}) {
    VulkanDeviceCreator::Params params;
    params.api_version = version;
    EXPECT_THROW({ VulkanDeviceCreator creator(params); }, std::runtime_error);
  }
}

}  // namespace quadrants::lang::vulkan

#endif
