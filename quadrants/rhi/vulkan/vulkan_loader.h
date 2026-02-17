#pragma once

#include <thread>
#include <mutex>

#include "quadrants/rhi/vulkan/vulkan_common.h"
#include "quadrants/common/dynamic_loader.h"

namespace quadrants::lang {
namespace vulkan {

class QD_DLL_EXPORT VulkanLoader {
 public:
  static VulkanLoader &instance() {
    static VulkanLoader instance;
    return instance;
  }

 public:
  VulkanLoader(VulkanLoader const &) = delete;
  void operator=(VulkanLoader const &) = delete;

  bool check_vulkan_device();

  void load_instance(VkInstance instance_);
  void load_device(VkDevice device_);
  bool init(PFN_vkGetInstanceProcAddr get_proc_addr = nullptr);
  PFN_vkVoidFunction load_function(const char *name);
  VkInstance get_instance() {
    return vulkan_instance_;
  }
  std::string visible_device_id;

 private:
  std::once_flag init_flag_;
  bool initialized_{false};

  VulkanLoader();

#if defined(__APPLE__)
  std::unique_ptr<DynamicLoader> vulkan_rt_{nullptr};
#endif

  VkInstance vulkan_instance_{VK_NULL_HANDLE};
  VkDevice vulkan_device_{VK_NULL_HANDLE};
};

QD_DLL_EXPORT bool is_vulkan_api_available();

QD_DLL_EXPORT void set_vulkan_visible_device(std::string id);

}  // namespace vulkan
}  // namespace quadrants::lang
