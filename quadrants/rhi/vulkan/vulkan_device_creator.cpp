#include "quadrants/rhi/vulkan/vulkan_device_creator.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "quadrants/rhi/vulkan/vulkan_common.h"
#include "quadrants/rhi/vulkan/vulkan_loader.h"
#include "quadrants/rhi/vulkan/vulkan_device.h"
#include "quadrants/common/utils.h"

namespace quadrants::lang {
namespace vulkan {

namespace {

const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

bool check_validation_layer_support() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  std::unordered_set<std::string> available_layer_names;
  for (const auto &layer_props : available_layers) {
    available_layer_names.insert(layer_props.layerName);
  }
  for (const char *name : kValidationLayers) {
    if (available_layer_names.count(std::string(name)) == 0) {
      return false;
    }
  }
  return true;
}

static const std::unordered_set<std::string> ignored_messages = {
    "VVL-DEBUG-PRINTF",
    "VUID_Undefined",
    // (penguinliong): Attempting to map a non-host-visible piece of memory.
    // `VulkanDevice::map()` returns `RhiResult::invalid_usage` in this case.
    "VUID-vkMapMemory-memory-00682",
};

[[maybe_unused]] bool vk_ignore_validation_warning(const std::string &msg_name) {
  if (ignored_messages.count(msg_name) > 0) {
    return true;
  }

  return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
                                                 void *p_user_data) {
  if (message_type == VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT &&
      message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT && p_callback_data->pMessageIdName != nullptr &&
      strstr(p_callback_data->pMessageIdName, "DEBUG-PRINTF") != nullptr) {
    std::cout << p_callback_data->pMessage << std::flush;
  }

  if (message_severity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    char msg_buf[4096];
    snprintf(msg_buf, sizeof(msg_buf), "Vulkan validation layer: %d, %s", message_type, p_callback_data->pMessage);

    if (is_ci()) {
      auto msg_name = std::string(p_callback_data->pMessageIdName);
      if (!vk_ignore_validation_warning(msg_name))
        QD_ERROR(msg_buf);
    } else {
      RHI_LOG_ERROR(msg_buf);
    }
  }

  return VK_FALSE;
}

void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT *create_info) {
  *create_info = {};
  create_info->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info->pfnUserCallback = vk_debug_callback;
  create_info->pUserData = nullptr;
}

VkResult create_debug_utils_messenger_ext(VkInstance instance,
                                          const VkDebugUtilsMessengerCreateInfoEXT *p_create_info,
                                          const VkAllocationCallbacks *p_allocator,
                                          VkDebugUtilsMessengerEXT *p_debug_messenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, p_create_info, p_allocator, p_debug_messenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void destroy_debug_utils_messenger_ext(VkInstance instance,
                                       VkDebugUtilsMessengerEXT debug_messenger,
                                       const VkAllocationCallbacks *p_allocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debug_messenger, p_allocator);
  }
}

std::vector<const char *> get_required_extensions(bool enable_validation) {
  std::vector<const char *> extensions;
  if (enable_validation) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

VulkanQueueFamilyIndices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) {
  VulkanQueueFamilyIndices indices;

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
  // TODO: What the heck is this?
  constexpr VkQueueFlags kFlagMask = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT));

  // first try and find a queue that has just the compute bit set
  for (int i = 0; i < (int)queue_family_count; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if ((masked_flags & VK_QUEUE_COMPUTE_BIT) && !(masked_flags & VK_QUEUE_GRAPHICS_BIT)) {
      indices.compute_family = i;
    }
    if (masked_flags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }

    if (surface != VK_NULL_HANDLE) {
      VkBool32 present_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
      char msg_buf[128];
      RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf), "Queue %d %s support for presenting", i,
                         present_support ? "has" : "does NOT have");
      RHI_LOG_DEBUG(msg_buf);

      if (present_support) {
        indices.present_family = i;
      }
    }

    if (indices.is_complete() && indices.is_complete_for_ui()) {
      char msg_buf[128];
      RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf), "Found async compute queue %d, graphics queue %d",
                         indices.compute_family.value(), indices.graphics_family.value());
      RHI_LOG_DEBUG(msg_buf);
      return indices;
    }
  }

  // lastly get any queue that will work
  for (int i = 0; i < (int)queue_family_count; ++i) {
    const VkQueueFlags masked_flags = kFlagMask & queue_families[i].queueFlags;
    if (masked_flags & VK_QUEUE_COMPUTE_BIT) {
      indices.compute_family = i;
    }
    if (indices.is_complete()) {
      return indices;
    }
  }
  return indices;
}

size_t get_device_score(VkPhysicalDevice device, VkSurfaceKHR surface) {
  auto indices = find_queue_families(device, surface);
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(device, &features);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device, &properties);

  size_t score = 0;

  if (surface != VK_NULL_HANDLE) {
    // this means we need ui
    score = size_t(indices.is_complete_for_ui()) * 1000;
  } else {
    score = size_t(indices.is_complete()) * 1000;
  }

  score += features.wideLines * 100;
  score += size_t(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) * 500;
  score += size_t(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) * 1000;
  score += VK_API_VERSION_MINOR(properties.apiVersion) * 100;

  return score;
}

}  // namespace

VulkanDeviceCreator::VulkanDeviceCreator(const VulkanDeviceCreator::Params &params) : params_(params) {
  if (!VulkanLoader::instance().init()) {
    throw std::runtime_error("Error loading vulkan");
  }

  ti_device_ = std::make_unique<VulkanDevice>();
  uint32_t vk_api_version;
  bool manual_create;
  if (params_.api_version.has_value()) {
    // The version client specified to use
    //
    // If the user provided an API version then the device creation process is
    // totally directed by the information provided externally.
    vk_api_version = params_.api_version.value();
    manual_create = true;
  } else {
    // The highest version designed to use
    vk_api_version = VulkanEnvSettings::k_api_version();
    manual_create = false;
  }

  create_instance(vk_api_version, manual_create);
  setup_debug_messenger();
  VkSurfaceKHR test_surface = VK_NULL_HANDLE;
  if (params_.is_for_ui) {
    test_surface = params_.surface_creator(instance_);
    RHI_ASSERT((test_surface != VK_NULL_HANDLE) && "failed to create window surface!");
  }
  pick_physical_device(test_surface);
  create_logical_device(manual_create);

  {
    VulkanDevice::Params params;
    params.instance = instance_;
    params.physical_device = physical_device_;
    params.device = device_;
    params.compute_queue = compute_queue_;
    params.compute_queue_family_index = queue_family_indices_.compute_family.value();
    params.graphics_queue = graphics_queue_;
    params.graphics_queue_family_index = queue_family_indices_.graphics_family.value();
    ti_device_->init_vulkan_structs(params);
  }

  if (params_.is_for_ui) {
    vkDestroySurfaceKHR(instance_, test_surface, kNoVkAllocCallbacks);
  }
}

VulkanDeviceCreator::~VulkanDeviceCreator() {
  ti_device_.reset();
  if (params_.enable_validation_layer) {
    destroy_debug_utils_messenger_ext(instance_, debug_messenger_, kNoVkAllocCallbacks);
  }
  vkDestroyDevice(device_, kNoVkAllocCallbacks);
  // VkInstance is intentionally kept alive in VulkanLoader (process-lifetime).
  // Repeated vkDestroyInstance/vkCreateInstance triggers an NVIDIA driver bug
  // that corrupts SubgroupLocalInvocationId after ~11 cycles.
}

// Create (or reuse) a VkInstance and populate ti_device_ capability flags.
//
// Phase 1 — Capability discovery.  Enumerates instance extensions and sets
//   `surface` and `physical_device_features2` on ti_device_->vk_caps().
//   This runs every cycle because ti_device_ is freshly constructed (all caps
//   default to false).  The `physical_device_features2` flag gates whether
//   create_logical_device() will query and enable f16, i8, atomic float,
//   variable pointers, shader clock, buffer device address, etc.
//
// Phase 2 — VkInstance reuse.  If the VulkanLoader singleton already holds a
//   live VkInstance (2nd+ qd.init() cycle), copies it into instance_ and
//   returns early.  This avoids an NVIDIA driver bug that corrupts
//   SubgroupLocalInvocationId after ~11 vkDestroyInstance/vkCreateInstance
//   cycles in the same process.
//
// Phase 3 — First-time VkInstance creation (first qd.init() only).  Builds
//   VkInstanceCreateInfo with app info, optional validation layers and debug
//   printf, collects required + supported instance extensions, calls
//   vkCreateInstance (with a Vulkan 1.0 fallback on
//   VK_ERROR_INCOMPATIBLE_DRIVER), and stores the new instance in the
//   VulkanLoader singleton for future reuse.
void VulkanDeviceCreator::create_instance(uint32_t vk_api_version, bool manual_create) {
  // Discover instance extensions and set capability flags on ti_device_.
  // This must run every cycle because ti_device_ is freshly created.
  uint32_t num_instance_extensions = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &num_instance_extensions, nullptr);
  std::vector<VkExtensionProperties> supported_extensions(num_instance_extensions);
  vkEnumerateInstanceExtensionProperties(nullptr, &num_instance_extensions, supported_extensions.data());

  for (auto &ext : supported_extensions) {
    std::string name = ext.extensionName;
    if (name == VK_KHR_SURFACE_EXTENSION_NAME) {
      ti_device_->vk_caps().surface = true;
    } else if (name == VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) {
      ti_device_->vk_caps().physical_device_features2 = true;
    }
  }

  // Normalize `params_.enable_validation_layer` against the host's actual layer availability BEFORE
  // the instance-reuse short-circuit below: downstream code (device-extension enumeration at
  // `create_logical_device`, the `VK_KHR_SHADER_NON_SEMANTIC_INFO` cap gate, the device layer arrays
  // at the tail of `create_logical_device`) reads this flag on every cycle, and `create_logical_device`
  // runs on every re-init. If we let the request pass through unchanged on the reuse path, the cap is
  // set to `true` every subsequent cycle even when the layer is not actually loaded on the cached
  // instance (the only cycle where the layer-load flip happens is the very first, which is also the
  // only cycle where a fresh `vkCreateInstance` runs). Users observed this as `test_overflow.py`
  // passing on the first parametrization and failing on every subsequent one in the same pytest
  // session: `DebugPrintf`-ext-imported shaders compile fine, but the validation layer that would
  // route those messages to stdout was never loaded, so the overflow-detected strings never appear in
  // `capfd`. Running the check here keeps the flag consistent across re-inits; re-running on every
  // call is cheap (it enumerates instance layers, ~microseconds).
  if (params_.enable_validation_layer && !check_validation_layer_support()) {
    RHI_LOG_ERROR(
        "Validation layers requested but not available, turning off... "
        "Please make sure Vulkan SDK from https://vulkan.lunarg.com/sdk/home "
        "is installed.");
    params_.enable_validation_layer = false;
  }

  // Reuse the VkInstance from a previous init/reset cycle if available.
  // Repeated vkDestroyInstance/vkCreateInstance triggers an NVIDIA driver bug
  // that corrupts SubgroupLocalInvocationId after ~11 cycles.
  VkInstance existing = VulkanLoader::instance().get_instance();
  if (existing != VK_NULL_HANDLE) {
    instance_ = existing;
    ti_device_->vk_caps().vk_api_version = vk_api_version;
    return;
  }
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Quadrants Vulkan Backend";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VulkanEnvSettings::k_api_version();

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  // `params_.enable_validation_layer` has already been normalized against the host's actual layer
  // availability above, before the cached-instance short-circuit. No second check needed here.

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  if (params_.enable_validation_layer) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();

    populate_debug_messenger_create_info(&debug_create_info);
    create_info.pNext = &debug_create_info;
  } else {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  // Response to `DebugPrintf`.
  std::array<VkValidationFeatureEnableEXT, 1> vfes = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  VkValidationFeaturesEXT vf = {};
  if (params_.enable_validation_layer) {
    vf.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    vf.pNext = create_info.pNext;
    vf.enabledValidationFeatureCount = vfes.size();
    vf.pEnabledValidationFeatures = vfes.data();
    create_info.pNext = &vf;
  }

  std::unordered_set<std::string> extensions;
  for (auto &ext : get_required_extensions(params_.enable_validation_layer)) {
    extensions.insert(std::string(ext));
  }
  for (auto &ext : params_.additional_instance_extensions) {
    extensions.insert(std::string(ext));
  }

  for (auto &ext : supported_extensions) {
    std::string name = ext.extensionName;
    if (name == VK_KHR_SURFACE_EXTENSION_NAME || name == VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME ||
        name == VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME ||
        name == VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME || name == VK_EXT_DEBUG_UTILS_EXTENSION_NAME) {
      extensions.insert(name);
    }
  }

  std::vector<const char *> confirmed_extensions;
  confirmed_extensions.reserve(extensions.size());
  for (auto &ext : extensions) {
    confirmed_extensions.push_back(ext.data());
  }

  create_info.enabledExtensionCount = (uint32_t)confirmed_extensions.size();
  create_info.ppEnabledExtensionNames = confirmed_extensions.data();

  VkResult res = vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_);

  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkApplicationInfo.html
    // Vulkan 1.0 implementation will return this when api version is not 1.0
    // Vulkan 1.1+ implementation will work with maximum version set
    ti_device_->vk_caps().vk_api_version = VK_API_VERSION_1_0;
    app_info.apiVersion = VK_API_VERSION_1_0;

    res = vkCreateInstance(&create_info, kNoVkAllocCallbacks, &instance_);
  } else {
    ti_device_->vk_caps().vk_api_version = vk_api_version;
  }

  if (res != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance");
  }

  VulkanLoader::instance().load_instance(instance_);
}

void VulkanDeviceCreator::setup_debug_messenger() {
  if (!params_.enable_validation_layer) {
    return;
  }
  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  populate_debug_messenger_create_info(&create_info);

  BAIL_ON_VK_BAD_RESULT_NO_RETURN(
      create_debug_utils_messenger_ext(instance_, &create_info, kNoVkAllocCallbacks, &debug_messenger_),
      "failed to set up debug messenger");
}

void VulkanDeviceCreator::pick_physical_device(VkSurfaceKHR test_surface) {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  RHI_ASSERT(device_count > 0 && "failed to find GPUs with Vulkan support");

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
  physical_device_ = VK_NULL_HANDLE;

  for (int i = 0; i < device_count; i++) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(devices[i], &properties);

    char msg_buf[128];
    RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf), "Found Vulkan Device %d (%s)", i, properties.deviceName);
    RHI_LOG_DEBUG(msg_buf);
  }

  auto device_id = VulkanLoader::instance().visible_device_id;
  bool has_visible_device{false};
  if (!device_id.empty()) {
    int id = std::stoi(device_id);
    if (id < 0 || id >= device_count) {
      char msg_buf[128];
      snprintf(msg_buf, sizeof(msg_buf), "QD_VISIBLE_DEVICE=%d is not valid, found %d devices available", id,
               device_count);
      RHI_LOG_ERROR(msg_buf);
    } else if (get_device_score(devices[id], test_surface)) {
      physical_device_ = devices[id];
      has_visible_device = true;
    }
  }

  if (!has_visible_device) {
    // could not find a user defined visible device, use the first one suitable
    size_t max_score = 0;
    for (const auto &device : devices) {
      size_t score = get_device_score(device, test_surface);
      if (score > max_score) {
        physical_device_ = device;
        max_score = score;
      }
    }
  }
  RHI_ASSERT(physical_device_ != VK_NULL_HANDLE && "failed to find a suitable GPU");

  queue_family_indices_ = find_queue_families(physical_device_, test_surface);
}

void VulkanDeviceCreator::create_logical_device(bool manual_create) {
  DeviceCapabilityConfig caps{};

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::unordered_set<uint32_t> unique_families;

  if (queue_family_indices_.compute_family.has_value()) {
    unique_families.insert(queue_family_indices_.compute_family.value());
  }
  if (queue_family_indices_.graphics_family.has_value()) {
    unique_families.insert(queue_family_indices_.graphics_family.value());
  }

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_families) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queue_family;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queueCreateInfo);
  }

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.queueCreateInfoCount = queue_create_infos.size();

  // Get device properties
  VkPhysicalDeviceProperties physical_device_properties{};
  vkGetPhysicalDeviceProperties(physical_device_, &physical_device_properties);

  {
    char msg_buf[256];
    RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf), "Vulkan Device \"%s\" supports Vulkan %d version %d.%d.%d",
                       physical_device_properties.deviceName,
                       VK_API_VERSION_VARIANT(physical_device_properties.apiVersion),
                       VK_API_VERSION_MAJOR(physical_device_properties.apiVersion),
                       VK_API_VERSION_MINOR(physical_device_properties.apiVersion),
                       VK_API_VERSION_PATCH(physical_device_properties.apiVersion));
    RHI_LOG_DEBUG(msg_buf);
  }

  // (penguinliong) The actual logical device is created with lastest version of
  // Vulkan but we use the device like it has a lower version (if the user
  // wanted a lower version device).
  uint32_t vk_api_version = physical_device_properties.apiVersion;
  ti_device_->vk_caps().vk_api_version = vk_api_version;
  if (vk_api_version >= VK_API_VERSION_1_3) {
    caps.set(DeviceCapability::spirv_version, 0x10500);
  } else if (vk_api_version >= VK_API_VERSION_1_2) {
    caps.set(DeviceCapability::spirv_version, 0x10500);
  } else if (vk_api_version >= VK_API_VERSION_1_1) {
    caps.set(DeviceCapability::spirv_version, 0x10300);
  } else {
    caps.set(DeviceCapability::spirv_version, 0x10000);
  }

  // Detect extensions
  std::vector<const char *> enabled_extensions;

  uint32_t extension_count = 0;
  // FIXME: (penguinliong) This was NOT called when `manual_create` is true.
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extension_properties.data());

  bool has_swapchain = false;

  [[maybe_unused]] bool portability_subset_enabled = false;

  for (auto &ext : extension_properties) {
    char msg_buf[256];
    RHI_DEBUG_SNPRINTF(msg_buf, sizeof(msg_buf), "Vulkan device extension {%s} (%x)", ext.extensionName,
                       ext.specVersion);
    RHI_LOG_DEBUG(msg_buf);

    std::string name = std::string(ext.extensionName);

    if (name == "VK_KHR_portability_subset") {
      RHI_LOG_ERROR(
          "Potential non-conformant Vulkan implementation, enabling "
          "VK_KHR_portability_subset");
      portability_subset_enabled = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
      has_swapchain = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SPIRV_1_4_EXTENSION_NAME) {
      if (caps.get(DeviceCapability::spirv_version) < 0x10400) {
        caps.set(DeviceCapability::spirv_version, 0x10400);
        enabled_extensions.push_back(ext.extensionName);
      }
    } else if (name == VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME ||
               name == VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) {
      ti_device_->vk_caps().external_memory = true;
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_BIND_MEMORY_2_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME && params_.enable_validation_layer) {
#if !defined(__APPLE__)
      // VK_KHR_shader_non_semantic_info isn't fully supported on MoltenVK: the extension enumerates as
      // available on the LunarG-SDK-sourced build, the device accepts `OpExtInstImport "NonSemantic.DebugPrintf"`
      // and the downstream `OpExtInst` call sites at SPIR-V validation time, but the SPIRV-Cross -> MSL
      // translator inside MoltenVK emits unconditional `debugPrintfEXT(...)` calls that Metal's MSL compiler
      // rejects with `use of undeclared identifier 'debugPrintfEXT'`. Since the only Vulkan implementation on
      // Apple platforms is MoltenVK, drop the advertisement here rather than at each SPIR-V codegen site.
      // Tracking issue: https://github.com/KhronosGroup/MoltenVK/issues/1214
      caps.set(DeviceCapability::spirv_has_non_semantic_info, true);
      enabled_extensions.push_back(ext.extensionName);
#endif
    } else if (name == VK_KHR_8BIT_STORAGE_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_16BIT_STORAGE_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (name == VK_KHR_SHADER_CLOCK_EXTENSION_NAME) {
      enabled_extensions.push_back(ext.extensionName);
    } else if (std::find(params_.additional_device_extensions.begin(), params_.additional_device_extensions.end(),
                         name) != params_.additional_device_extensions.end()) {
      enabled_extensions.push_back(ext.extensionName);
    }
    // Vulkan doesn't seem to support SPV_KHR_no_integer_wrap_decoration at all.
  }

  if (has_swapchain) {
    ti_device_->vk_caps().present = true;
  }

  VkPhysicalDeviceFeatures device_features{};

  VkPhysicalDeviceFeatures device_supported_features;
  vkGetPhysicalDeviceFeatures(physical_device_, &device_supported_features);

  if (device_supported_features.shaderInt16) {
    device_features.shaderInt16 = true;
    caps.set(DeviceCapability::spirv_has_int16, true);
  }
  if (device_supported_features.shaderInt64) {
    device_features.shaderInt64 = true;
    caps.set(DeviceCapability::spirv_has_int64, true);
  }
  if (device_supported_features.shaderFloat64) {
    device_features.shaderFloat64 = true;
    caps.set(DeviceCapability::spirv_has_float64, true);
  }
  if (device_supported_features.wideLines) {
    device_features.wideLines = true;
    ti_device_->vk_caps().wide_line = true;
  }

  if (ti_device_->vk_caps().vk_api_version >= VK_API_VERSION_1_1) {
    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_properties.pNext = nullptr;

    VkPhysicalDeviceProperties2 physical_device_properties{};
    physical_device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical_device_properties.pNext = &subgroup_properties;

    vkGetPhysicalDeviceProperties2(physical_device_, &physical_device_properties);

    // Vulkan exposes `subgroupSize` as a per-device property that's uniform across every dispatch we submit (the
    // `VK_EXT_subgroup_size_control` extension can vary it per-pipeline, but Quadrants doesn't opt in). Stash it in the
    // device cap so `Program::subgroup_size()` and the fe-ll cache key can see it on the SPIR-V backends.
    caps.set(DeviceCapability::spirv_subgroup_size, subgroup_properties.subgroupSize);

    if (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_basic, true);
    }
    if (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_vote, true);
    }
    if (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_arithmetic, true);
    }
    if (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) {
      caps.set(DeviceCapability::spirv_has_subgroup_ballot, true);
    }
  }

  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount = enabled_extensions.size();
  create_info.ppEnabledExtensionNames = enabled_extensions.data();

  void **pNextEnd = (void **)&create_info.pNext;

  // Use physicalDeviceFeatures2 to features enabled by extensions
  VkPhysicalDeviceVariablePointersFeaturesKHR variable_ptr_feature{};
  variable_ptr_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR;
  VkPhysicalDeviceShaderAtomicInt64Features shader_atomic_int64_feature{};
  shader_atomic_int64_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
  // VK_KHR_shader_subgroup_extended_types (promoted to Vulkan 1.2 core). Enables OpGroupNonUniform*
  // ops on 8/16/64-bit integer and 16-bit float types. Required by the `qd.simt.subgroup.*` and
  // `qd.simt.block.*` reductions when invoked with `qd.i64` (and 8/16-bit types), otherwise SPIR-V
  // validation rejects the shader with VUID-RuntimeSpirv-None-06275 and the dispatch returns
  // uninitialised lanes.
  VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures shader_subgroup_extended_types_feature{};
  shader_subgroup_extended_types_feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES;
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shader_atomic_float_feature{};
  shader_atomic_float_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
  VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT shader_atomic_float_2_feature{};
  shader_atomic_float_2_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
  VkPhysicalDeviceFloat16Int8FeaturesKHR shader_f16_i8_feature{};
  shader_f16_i8_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;

  VkPhysicalDevice8BitStorageFeatures shader_8bit_storage_feature{};
  shader_8bit_storage_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
  VkPhysicalDevice16BitStorageFeatures shader_16bit_storage_feature{};
  shader_16bit_storage_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR buffer_device_address_feature{};
  buffer_device_address_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature{};
  dynamic_rendering_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;

  if (ti_device_->vk_caps().physical_device_features2) {
    VkPhysicalDeviceFeatures2KHR features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

#define CHECK_EXTENSION(ext)                                         \
  std::find_if(enabled_extensions.begin(), enabled_extensions.end(), \
               [=](const char *o) { return strcmp(ext, o) == 0; }) != enabled_extensions.end()

    uint32_t vk_api_version = ti_device_->vk_caps().vk_api_version;
#define CHECK_VERSION(major, minor) vk_api_version >= VK_MAKE_API_VERSION(0, major, minor, 0)

    // Variable ptr
    if (CHECK_VERSION(1, 1) || CHECK_EXTENSION(VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME)) {
      features2.pNext = &variable_ptr_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (variable_ptr_feature.variablePointers && variable_ptr_feature.variablePointersStorageBuffer) {
        caps.set(DeviceCapability::spirv_has_variable_ptr, true);
      }
      *pNextEnd = &variable_ptr_feature;
      pNextEnd = &variable_ptr_feature.pNext;
    }

    // Atomic int64 (promoted to Vulkan 1.2 core)
    if (CHECK_VERSION(1, 2) || CHECK_EXTENSION(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME)) {
      features2.pNext = &shader_atomic_int64_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);
      if (shader_atomic_int64_feature.shaderBufferInt64Atomics ||
          shader_atomic_int64_feature.shaderSharedInt64Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_int64, true);
      }
      *pNextEnd = &shader_atomic_int64_feature;
      pNextEnd = &shader_atomic_int64_feature.pNext;
    }

    // Subgroup extended types (promoted to Vulkan 1.2 core). Required for OpGroupNonUniform* ops on
    // 8/16/64-bit integers and 16-bit floats — the `qd.simt.subgroup.*` and `qd.simt.block.*`
    // reductions emit these when the lane dtype is `qd.i64` (etc.). The feature has no
    // DeviceCapability bit because the SPIR-V codegen already gates wide-int dispatch on
    // `spirv_has_int64` / `spirv_has_int{8,16}`; the device must merely accept the SPIR-V at
    // pipeline creation time.
    if (CHECK_VERSION(1, 2) || CHECK_EXTENSION(VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME)) {
      features2.pNext = &shader_subgroup_extended_types_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);
      if (shader_subgroup_extended_types_feature.shaderSubgroupExtendedTypes) {
        *pNextEnd = &shader_subgroup_extended_types_feature;
        pNextEnd = &shader_subgroup_extended_types_feature.pNext;
      }
    }

    // Atomic float
    if (CHECK_EXTENSION(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
      features2.pNext = &shader_atomic_float_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);
      if (shader_atomic_float_feature.shaderBufferFloat32AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_atomic_float_add, true);
      }
      if (shader_atomic_float_feature.shaderBufferFloat64AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_atomic_float64_add, true);
      }
      if (shader_atomic_float_feature.shaderBufferFloat32Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_float, true);
      }
      if (shader_atomic_float_feature.shaderBufferFloat64Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_float64, true);
      }
#if !defined(__APPLE__)
      // Shared (threadgroup) float atomics are not actually usable through MoltenVK: the underlying Metal
      // Shading Language rejects `atomic_fetch_add_explicit` on `threadgroup atomic_float*` with
      // `cannot pass pointer to address space 'threadgroup' as a pointer to address space 'device'`, so
      // MSL translation of any SPIR-V that emits `OpAtomicFAdd` against the Workgroup storage class fails
      // at pipeline creation. The `shaderSharedFloatN AtomicAdd` feature bit is advertised by MoltenVK
      // anyway; advertising the cap back up would route `has_native_float_atomic_add(..., is_shared=true)`
      // to the native path and make every shared-atomic-float kernel unusable on Apple. Dropping the cap
      // here falls back to the CAS-emulated path in `atomic_operation_widened`, which targets uint
      // atomics and works on every backend. Companion gate to the `spirv_has_non_semantic_info` drop a
      // few branches up; see that comment for the overall MoltenVK cap-sanitisation rationale.
      if (shader_atomic_float_feature.shaderSharedFloat32AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_shared_atomic_float_add, true);
      }
      if (shader_atomic_float_feature.shaderSharedFloat64AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_shared_atomic_float64_add, true);
      }
#endif
      *pNextEnd = &shader_atomic_float_feature;
      pNextEnd = &shader_atomic_float_feature.pNext;
    }

    // Atomic float 2
    if (CHECK_EXTENSION(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME)) {
      features2.pNext = &shader_atomic_float_2_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);
      if (shader_atomic_float_2_feature.shaderBufferFloat16AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_atomic_float_add, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat16AtomicMinMax) {
        caps.set(DeviceCapability::spirv_has_atomic_float16_minmax, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat16Atomics) {
        caps.set(DeviceCapability::spirv_has_atomic_float16, true);
      }
#if !defined(__APPLE__)
      // Same MoltenVK limitation as `shader_atomic_float_feature.shaderSharedFloat32AtomicAdd` above:
      // the feature bit is advertised but the MSL translator cannot emit a valid threadgroup
      // `atomic_fetch_add_explicit` for it. Drop the cap on Apple and let the CAS-emulated fallback
      // handle f16 shared atomics.
      if (shader_atomic_float_2_feature.shaderSharedFloat16AtomicAdd) {
        caps.set(DeviceCapability::spirv_has_shared_atomic_float16_add, true);
      }
#endif
      if (shader_atomic_float_2_feature.shaderBufferFloat32AtomicMinMax) {
        caps.set(DeviceCapability::spirv_has_atomic_float_minmax, true);
      }
      if (shader_atomic_float_2_feature.shaderBufferFloat64AtomicMinMax) {
        caps.set(DeviceCapability::spirv_has_atomic_float64_minmax, true);
      }
      *pNextEnd = &shader_atomic_float_2_feature;
      pNextEnd = &shader_atomic_float_2_feature.pNext;
    }

    // F16 / I8
    if (CHECK_VERSION(1, 2) || CHECK_EXTENSION(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
      features2.pNext = &shader_f16_i8_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (shader_f16_i8_feature.shaderFloat16) {
        caps.set(DeviceCapability::spirv_has_float16, true);
      }
      if (shader_f16_i8_feature.shaderInt8) {
        caps.set(DeviceCapability::spirv_has_int8, true);
      }
      *pNextEnd = &shader_f16_i8_feature;
      pNextEnd = &shader_f16_i8_feature.pNext;
    }

    if (CHECK_VERSION(1, 1) || CHECK_EXTENSION(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
      features2.pNext = &shader_8bit_storage_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      // Gate the SPIR-V `CapabilityStorageBuffer8BitAccess` emission strictly on the queried
      // feature bit. `VK_KHR_8bit_storage` promoted into Vulkan 1.2 core doesn't imply the feature
      // is actually supported -- implementations can still expose `storageBuffer8BitAccess = FALSE`
      // -- so the SPIR-V cap may only be emitted when this feature is true, otherwise strict
      // validation rejects shaders that declare the capability.
      if (shader_8bit_storage_feature.storageBuffer8BitAccess) {
        caps.set(DeviceCapability::spirv_has_storage_buffer_8bit_access, true);
      }

      *pNextEnd = &shader_8bit_storage_feature;
      pNextEnd = &shader_8bit_storage_feature.pNext;
    }
    if (CHECK_VERSION(1, 1) || CHECK_EXTENSION(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
      features2.pNext = &shader_16bit_storage_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (shader_16bit_storage_feature.storageBuffer16BitAccess) {
        caps.set(DeviceCapability::spirv_has_storage_buffer_16bit_access, true);
      }

      *pNextEnd = &shader_16bit_storage_feature;
      pNextEnd = &shader_16bit_storage_feature.pNext;
    }

    // Buffer Device Address
    if (CHECK_VERSION(1, 2) || CHECK_EXTENSION(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
      features2.pNext = &buffer_device_address_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      // Gate strictly on the queried feature bit. Vulkan 1.3 *promotes* `VK_KHR_buffer_device_address`
      // into core but still lets the implementation expose `bufferDeviceAddress = VK_FALSE`; the previous
      // `CHECK_VERSION(1, 3) || feature_bit` condition therefore treated 1.3 devices as PSB-capable even
      // when they were not, which would drive `vkGetBufferDeviceAddressKHR` / BDA usage flags on hardware
      // that didn't advertise the feature. The `shaderInt64` guard stays because the sizer shader holds
      // PSB pointers as `i64` SSA values.
      if (buffer_device_address_feature.bufferDeviceAddress && device_supported_features.shaderInt64) {
        // The prior `#if !defined(__APPLE__) && false` kill-switch referenced Taichi issue #6295 (broken
        // BDA on the 2022-era MoltenVK pinned in `quadrants/rhi/CMakeLists.txt`). That pin has since been
        // replaced with a LunarG-SDK-sourced MoltenVK (>= 1.3, sporting working `vkGetBufferDeviceAddress`),
        // so there is no reason to hard-disable PSB on Apple (or anywhere) - the feature query above now
        // correctly reflects the device's actual capability.
        caps.set(DeviceCapability::spirv_has_physical_storage_buffer, true);
      }
      *pNextEnd = &buffer_device_address_feature;
      pNextEnd = &buffer_device_address_feature.pNext;
    }

    // Dynamic rendering
    // TODO: Figure out how to integrate this correctly with ImGui,
    //       and then figure out the layout & barrier stuff
    /*
    if (CHECK_EXTENSION(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
      features2.pNext = &dynamic_rendering_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (dynamic_rendering_feature.dynamicRendering) {
        ti_device_->vk_caps().dynamic_rendering = true;
      }

      *pNextEnd = &dynamic_rendering_feature;
      pNextEnd = &dynamic_rendering_feature.pNext;
    }
    */

    // Shader clock
    if (CHECK_EXTENSION(VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
      VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_feature{};
      shader_clock_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
      features2.pNext = &shader_clock_feature;
      vkGetPhysicalDeviceFeatures2KHR(physical_device_, &features2);

      if (shader_clock_feature.shaderDeviceClock) {
        caps.set(DeviceCapability::spirv_has_shader_clock, true);
      }
      *pNextEnd = &shader_clock_feature;
      pNextEnd = &shader_clock_feature.pNext;
    }

    // TODO: add atomic min/max feature
  }

  if (params_.enable_validation_layer) {
    create_info.enabledLayerCount = (uint32_t)kValidationLayers.size();
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(vkCreateDevice(physical_device_, &create_info, kNoVkAllocCallbacks, &device_),
                                  "failed to create logical device");
  VulkanLoader::instance().load_device(device_);

  if (queue_family_indices_.compute_family.has_value()) {
    vkGetDeviceQueue(device_, queue_family_indices_.compute_family.value(), 0, &compute_queue_);
  }
  if (queue_family_indices_.graphics_family.has_value()) {
    vkGetDeviceQueue(device_, queue_family_indices_.graphics_family.value(), 0, &graphics_queue_);
  }

  // Dump capabilities
  caps.dbg_print_all();
  ti_device_->set_caps(std::move(caps));
}

}  // namespace vulkan
}  // namespace quadrants::lang
