/* Two VkBuffers in one VkDeviceMemory. A compute shader reads them through buffer device addresses.
   Pass 2 overwrites the ints and reads again. Run once with MoltenVK's default heaps and once with
   MVK_CONFIG_USE_MTLHEAP=0, set before this process loads the dylib.
   A host readback copy checks that the bytes were stored. VERDICT=MISMATCH means the address load
   disagreed with that copy. */
#define VK_NO_PROTOTYPES
#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

static void die(const char *message) {
    std::fprintf(stderr, "setup failed: %s\nVERDICT=SETUP_FAILED\n", message);
    std::exit(2);
}

static void check(VkResult result, const char *what) {
    if (result == VK_SUCCESS) return;
    std::fprintf(stderr, "%s failed: %d\n", what, (int)result);
    if (result == VK_ERROR_DEVICE_LOST) {
        std::fprintf(stderr, "VERDICT=DEVICE_LOST\n");
        std::exit(1);
    }
    die(what);
}

struct Api {
    PFN_vkGetInstanceProcAddr get_instance;
    PFN_vkCreateInstance create_instance;
    PFN_vkDestroyInstance destroy_instance;
    PFN_vkEnumeratePhysicalDevices enumerate_physical_devices;
    PFN_vkGetPhysicalDeviceProperties get_properties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_families;
    PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties;
    PFN_vkCreateDevice create_device;
    PFN_vkDestroyDevice destroy_device;
    PFN_vkGetDeviceQueue get_device_queue;
    PFN_vkCreateBuffer create_buffer;
    PFN_vkDestroyBuffer destroy_buffer;
    PFN_vkGetBufferMemoryRequirements get_buffer_requirements;
    PFN_vkAllocateMemory allocate_memory;
    PFN_vkFreeMemory free_memory;
    PFN_vkBindBufferMemory bind_buffer_memory;
    PFN_vkMapMemory map_memory;
    PFN_vkUnmapMemory unmap_memory;
    PFN_vkGetBufferDeviceAddress get_buffer_device_address;
    PFN_vkCreateCommandPool create_command_pool;
    PFN_vkDestroyCommandPool destroy_command_pool;
    PFN_vkAllocateCommandBuffers allocate_command_buffers;
    PFN_vkBeginCommandBuffer begin_command_buffer;
    PFN_vkEndCommandBuffer end_command_buffer;
    PFN_vkCmdCopyBuffer cmd_copy_buffer;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier;
    PFN_vkCmdBindPipeline cmd_bind_pipeline;
    PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets;
    PFN_vkCmdDispatch cmd_dispatch;
    PFN_vkCreateShaderModule create_shader_module;
    PFN_vkDestroyShaderModule destroy_shader_module;
    PFN_vkCreateDescriptorSetLayout create_descriptor_set_layout;
    PFN_vkDestroyDescriptorSetLayout destroy_descriptor_set_layout;
    PFN_vkCreatePipelineLayout create_pipeline_layout;
    PFN_vkDestroyPipelineLayout destroy_pipeline_layout;
    PFN_vkCreateComputePipelines create_compute_pipelines;
    PFN_vkDestroyPipeline destroy_pipeline;
    PFN_vkCreateDescriptorPool create_descriptor_pool;
    PFN_vkDestroyDescriptorPool destroy_descriptor_pool;
    PFN_vkAllocateDescriptorSets allocate_descriptor_sets;
    PFN_vkUpdateDescriptorSets update_descriptor_sets;
    PFN_vkQueueSubmit queue_submit;
    PFN_vkQueueWaitIdle queue_wait_idle;
};

static Api g;

template <typename T>
static T load(VkInstance instance, const char *name) {
    T fn = reinterpret_cast<T>(g.get_instance(instance, name));
    if (!fn) die(name);
    return fn;
}

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize offset = 0;
    bool owns_memory = false;
};

static Buffer make_buffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer out;
    out.size = size;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(g.create_buffer(device, &info, nullptr, &out.buffer), "vkCreateBuffer");
    return out;
}

static void allocate_and_bind(VkDevice device, const VkPhysicalDeviceMemoryProperties &props, Buffer *buffers, int count,
                              VkMemoryPropertyFlags required, VkMemoryPropertyFlags excluded, bool device_address) {
    VkMemoryRequirements requirements[2]{};
    if (count > 2) die("only two suballocations are set up");
    uint32_t type_bits = 0xffffffffu;
    VkDeviceSize cursor = 0;
    for (int i = 0; i < count; i++) {
        g.get_buffer_requirements(device, buffers[i].buffer, &requirements[i]);
        type_bits &= requirements[i].memoryTypeBits;
        cursor = align_up(cursor, requirements[i].alignment);
        buffers[i].offset = cursor;
        cursor += requirements[i].size;
    }
    uint32_t type_index = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = props.memoryTypes[i].propertyFlags;
        if ((type_bits & (1u << i)) && (flags & required) == required && (flags & excluded) == 0) {
            type_index = i;
            break;
        }
    }
    if (type_index == UINT32_MAX) die("no matching memory type");
    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = device_address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &flags;
    alloc.allocationSize = cursor;
    alloc.memoryTypeIndex = type_index;
    check(g.allocate_memory(device, &alloc, nullptr, &buffers[0].memory), "vkAllocateMemory");
    buffers[0].owns_memory = true;
    for (int i = 0; i < count; i++) {
        buffers[i].memory = buffers[0].memory;
        check(g.bind_buffer_memory(device, buffers[i].buffer, buffers[0].memory, buffers[i].offset),
              "vkBindBufferMemory");
    }
    std::printf("shared_memory bytes=%llu\n", (unsigned long long)cursor);
    for (int i = 0; i < count; i++) {
        std::printf("suballocation[%d] offset=%llu size=%llu align=%llu\n", i,
                    (unsigned long long)buffers[i].offset, (unsigned long long)requirements[i].size,
                    (unsigned long long)requirements[i].alignment);
    }
}

static void allocate_dedicated(VkDevice device, const VkPhysicalDeviceMemoryProperties &props, Buffer *buffer,
                               VkMemoryPropertyFlags required, VkMemoryPropertyFlags excluded, bool device_address) {
    allocate_and_bind(device, props, buffer, 1, required, excluded, device_address);
}

static void barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size, VkPipelineStageFlags src_stage,
                    VkAccessFlags src_access, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
    VkBufferMemoryBarrier info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.srcAccessMask = src_access;
    info.dstAccessMask = dst_access;
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.buffer = buffer;
    info.size = size;
    g.cmd_pipeline_barrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 1, &info, 0, nullptr);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s libMoltenVK.dylib read.spv default|0\n", argv[0]);
        return 64;
    }
    unsetenv("MVK_CONFIG_USE_MTLHEAP");
    if (std::strcmp(argv[3], "0") == 0) {
        if (setenv("MVK_CONFIG_USE_MTLHEAP", "0", 1) != 0) die("setenv");
    } else if (std::strcmp(argv[3], "default") != 0) {
        die("heap mode must be default or 0");
    }
    std::printf("MVK_CONFIG_USE_MTLHEAP=%s\n", std::getenv("MVK_CONFIG_USE_MTLHEAP")
                                                   ? std::getenv("MVK_CONFIG_USE_MTLHEAP")
                                                   : "(unset)");

    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) die(dlerror());
    g.get_instance = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
    if (!g.get_instance) die("vkGetInstanceProcAddr");
    g.create_instance = load<PFN_vkCreateInstance>(VK_NULL_HANDLE, "vkCreateInstance");

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "shared-buffer-read";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    check(g.create_instance(&instance_info, nullptr, &instance), "vkCreateInstance");

    g.destroy_instance = load<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
    g.enumerate_physical_devices = load<PFN_vkEnumeratePhysicalDevices>(instance, "vkEnumeratePhysicalDevices");
    g.get_properties = load<PFN_vkGetPhysicalDeviceProperties>(instance, "vkGetPhysicalDeviceProperties");
    g.get_queue_families = load<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    g.get_memory_properties = load<PFN_vkGetPhysicalDeviceMemoryProperties>(instance, "vkGetPhysicalDeviceMemoryProperties");
    g.create_device = load<PFN_vkCreateDevice>(instance, "vkCreateDevice");
    g.destroy_device = load<PFN_vkDestroyDevice>(instance, "vkDestroyDevice");
    g.get_device_queue = load<PFN_vkGetDeviceQueue>(instance, "vkGetDeviceQueue");
    g.create_buffer = load<PFN_vkCreateBuffer>(instance, "vkCreateBuffer");
    g.destroy_buffer = load<PFN_vkDestroyBuffer>(instance, "vkDestroyBuffer");
    g.get_buffer_requirements = load<PFN_vkGetBufferMemoryRequirements>(instance, "vkGetBufferMemoryRequirements");
    g.allocate_memory = load<PFN_vkAllocateMemory>(instance, "vkAllocateMemory");
    g.free_memory = load<PFN_vkFreeMemory>(instance, "vkFreeMemory");
    g.bind_buffer_memory = load<PFN_vkBindBufferMemory>(instance, "vkBindBufferMemory");
    g.map_memory = load<PFN_vkMapMemory>(instance, "vkMapMemory");
    g.unmap_memory = load<PFN_vkUnmapMemory>(instance, "vkUnmapMemory");
    g.get_buffer_device_address = load<PFN_vkGetBufferDeviceAddress>(instance, "vkGetBufferDeviceAddress");
    g.create_command_pool = load<PFN_vkCreateCommandPool>(instance, "vkCreateCommandPool");
    g.destroy_command_pool = load<PFN_vkDestroyCommandPool>(instance, "vkDestroyCommandPool");
    g.allocate_command_buffers = load<PFN_vkAllocateCommandBuffers>(instance, "vkAllocateCommandBuffers");
    g.begin_command_buffer = load<PFN_vkBeginCommandBuffer>(instance, "vkBeginCommandBuffer");
    g.end_command_buffer = load<PFN_vkEndCommandBuffer>(instance, "vkEndCommandBuffer");
    g.cmd_copy_buffer = load<PFN_vkCmdCopyBuffer>(instance, "vkCmdCopyBuffer");
    g.cmd_pipeline_barrier = load<PFN_vkCmdPipelineBarrier>(instance, "vkCmdPipelineBarrier");
    g.cmd_bind_pipeline = load<PFN_vkCmdBindPipeline>(instance, "vkCmdBindPipeline");
    g.cmd_bind_descriptor_sets = load<PFN_vkCmdBindDescriptorSets>(instance, "vkCmdBindDescriptorSets");
    g.cmd_dispatch = load<PFN_vkCmdDispatch>(instance, "vkCmdDispatch");
    g.create_shader_module = load<PFN_vkCreateShaderModule>(instance, "vkCreateShaderModule");
    g.destroy_shader_module = load<PFN_vkDestroyShaderModule>(instance, "vkDestroyShaderModule");
    g.create_descriptor_set_layout = load<PFN_vkCreateDescriptorSetLayout>(instance, "vkCreateDescriptorSetLayout");
    g.destroy_descriptor_set_layout = load<PFN_vkDestroyDescriptorSetLayout>(instance, "vkDestroyDescriptorSetLayout");
    g.create_pipeline_layout = load<PFN_vkCreatePipelineLayout>(instance, "vkCreatePipelineLayout");
    g.destroy_pipeline_layout = load<PFN_vkDestroyPipelineLayout>(instance, "vkDestroyPipelineLayout");
    g.create_compute_pipelines = load<PFN_vkCreateComputePipelines>(instance, "vkCreateComputePipelines");
    g.destroy_pipeline = load<PFN_vkDestroyPipeline>(instance, "vkDestroyPipeline");
    g.create_descriptor_pool = load<PFN_vkCreateDescriptorPool>(instance, "vkCreateDescriptorPool");
    g.destroy_descriptor_pool = load<PFN_vkDestroyDescriptorPool>(instance, "vkDestroyDescriptorPool");
    g.allocate_descriptor_sets = load<PFN_vkAllocateDescriptorSets>(instance, "vkAllocateDescriptorSets");
    g.update_descriptor_sets = load<PFN_vkUpdateDescriptorSets>(instance, "vkUpdateDescriptorSets");
    g.queue_submit = load<PFN_vkQueueSubmit>(instance, "vkQueueSubmit");
    g.queue_wait_idle = load<PFN_vkQueueWaitIdle>(instance, "vkQueueWaitIdle");

    uint32_t device_count = 0;
    check(g.enumerate_physical_devices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices");
    if (device_count == 0) die("no physical device");
    std::vector<VkPhysicalDevice> physical(device_count);
    check(g.enumerate_physical_devices(instance, &device_count, physical.data()), "vkEnumeratePhysicalDevices");
    VkPhysicalDeviceProperties properties{};
    g.get_properties(physical[0], &properties);
    std::printf("device=%s api=%u.%u.%u driver=%u.%u.%u\n", properties.deviceName,
                VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion),
                VK_VERSION_PATCH(properties.apiVersion), VK_VERSION_MAJOR(properties.driverVersion),
                VK_VERSION_MINOR(properties.driverVersion), VK_VERSION_PATCH(properties.driverVersion));

    uint32_t family_count = 0;
    g.get_queue_families(physical[0], &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    g.get_queue_families(physical[0], &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = i;
            break;
        }
    }
    if (family == UINT32_MAX) die("no compute queue");

    VkPhysicalDeviceBufferDeviceAddressFeatures address_features{};
    address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    address_features.bufferDeviceAddress = VK_TRUE;
    const char *device_extensions[] = {VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME};
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &address_features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = device_extensions;
    VkDevice device = VK_NULL_HANDLE;
    check(g.create_device(physical[0], &device_info, nullptr, &device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    g.get_device_queue(device, family, 0, &queue);

    VkPhysicalDeviceMemoryProperties memory_properties{};
    g.get_memory_properties(physical[0], &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        std::printf("memory_type[%u]=0x%x\n", i, memory_properties.memoryTypes[i].propertyFlags);
    }
    VkBufferUsageFlags data_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    Buffer data[2] = {make_buffer(device, 256, data_usage), make_buffer(device, 256, data_usage)};
    allocate_and_bind(device, memory_properties, data, 2, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, true);

    VkBufferDeviceAddressInfo address_info{};
    address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    address_info.buffer = data[0].buffer;
    VkDeviceAddress address_a = g.get_buffer_device_address(device, &address_info);
    address_info.buffer = data[1].buffer;
    VkDeviceAddress address_b = g.get_buffer_device_address(device, &address_info);
    std::printf("address_a=0x%llx address_b=0x%llx delta=%lld\n", (unsigned long long)address_a,
                (unsigned long long)address_b, (long long)address_b - (long long)address_a);
    if (address_a == 0 || address_b == 0 || (address_a % 16) != 0 || (address_b % 16) != 0) die("unaligned device address");

    Buffer staging = make_buffer(device, 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer readback = make_buffer(device, 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer addresses = make_buffer(device, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output = make_buffer(device, 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    allocate_dedicated(device, memory_properties, &staging, host, 0, false);
    allocate_dedicated(device, memory_properties, &readback, host, 0, false);
    allocate_dedicated(device, memory_properties, &addresses, host, 0, false);
    allocate_dedicated(device, memory_properties, &output, host, 0, false);

    void *mapped = nullptr;
    check(g.map_memory(device, addresses.memory, 0, 16, 0, &mapped), "vkMapMemory(addresses)");
    VkDeviceAddress *stored = static_cast<VkDeviceAddress *>(mapped);
    stored[0] = address_a;
    stored[1] = address_b;
    g.unmap_memory(device, addresses.memory);

    std::FILE *shader_file = std::fopen(argv[2], "rb");
    if (!shader_file) die("open spir-v");
    std::fseek(shader_file, 0, SEEK_END);
    long shader_bytes = std::ftell(shader_file);
    std::rewind(shader_file);
    std::vector<uint32_t> shader((size_t)shader_bytes / 4);
    if (shader_bytes <= 0 || shader_bytes % 4 != 0 ||
        std::fread(shader.data(), 1, (size_t)shader_bytes, shader_file) != (size_t)shader_bytes) {
        die("read spir-v");
    }
    std::fclose(shader_file);
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = (size_t)shader_bytes;
    shader_info.pCode = shader.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check(g.create_shader_module(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1] = bindings[0];
    bindings[1].binding = 1;
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    check(g.create_descriptor_set_layout(device, &layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    check(g.create_pipeline_layout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader_module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(g.create_compute_pipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
          "vkCreateComputePipelines");

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 2;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    check(g.create_descriptor_pool(device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    check(g.allocate_descriptor_sets(device, &set_info, &set), "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo descriptor_buffers[2]{};
    descriptor_buffers[0].buffer = addresses.buffer;
    descriptor_buffers[0].range = 16;
    descriptor_buffers[1].buffer = output.buffer;
    descriptor_buffers[1].range = 8;
    VkWriteDescriptorSet writes[2]{};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &descriptor_buffers[i];
    }
    g.update_descriptor_sets(device, 2, writes, 0, nullptr);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = family;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(g.create_command_pool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(g.allocate_command_buffers(device, &command_info, &command), "vkAllocateCommandBuffers");

    const uint32_t expected[2][2] = {{0x11111111u, 0x22222222u}, {0x33333333u, 0x44444444u}};
    bool mismatch = false;
    for (int pass = 0; pass < 2; pass++) {
        check(g.map_memory(device, staging.memory, 0, 8, 0, &mapped), "vkMapMemory(staging)");
        std::memcpy(mapped, expected[pass], 8);
        g.unmap_memory(device, staging.memory);
        check(g.map_memory(device, output.memory, 0, 8, 0, &mapped), "vkMapMemory(output)");
        uint32_t poison[2] = {0xdeadbeefu, 0xdeadbeefu};
        std::memcpy(mapped, poison, 8);
        g.unmap_memory(device, output.memory);

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check(g.begin_command_buffer(command, &begin), "vkBeginCommandBuffer");
        VkBufferCopy copies[2]{};
        copies[0].srcOffset = 0;
        copies[0].size = 4;
        copies[1].srcOffset = 4;
        copies[1].size = 4;
        g.cmd_copy_buffer(command, staging.buffer, data[0].buffer, 1, &copies[0]);
        g.cmd_copy_buffer(command, staging.buffer, data[1].buffer, 1, &copies[1]);
        barrier(command, data[0].buffer, 4, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        barrier(command, data[1].buffer, 4, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        g.cmd_bind_pipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        g.cmd_bind_descriptor_sets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        g.cmd_dispatch(command, 1, 1, 1);
        barrier(command, data[0].buffer, 4, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(command, data[1].buffer, 4, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(command, output.buffer, 8, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        g.cmd_copy_buffer(command, data[0].buffer, readback.buffer, 1, &copies[0]);
        VkBufferCopy back_b{};
        back_b.srcOffset = 0;
        back_b.dstOffset = 4;
        back_b.size = 4;
        g.cmd_copy_buffer(command, data[1].buffer, readback.buffer, 1, &back_b);
        barrier(command, readback.buffer, 8, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        check(g.end_command_buffer(command), "vkEndCommandBuffer");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(g.queue_submit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        check(g.queue_wait_idle(queue), "vkQueueWaitIdle");

        uint32_t loaded[2]{}, copied[2]{};
        check(g.map_memory(device, output.memory, 0, 8, 0, &mapped), "vkMapMemory(result)");
        std::memcpy(loaded, mapped, 8);
        g.unmap_memory(device, output.memory);
        check(g.map_memory(device, readback.memory, 0, 8, 0, &mapped), "vkMapMemory(readback)");
        std::memcpy(copied, mapped, 8);
        g.unmap_memory(device, readback.memory);
        bool pass_ok = loaded[0] == expected[pass][0] && loaded[1] == expected[pass][1] &&
                       copied[0] == expected[pass][0] && copied[1] == expected[pass][1];
        mismatch = mismatch || !pass_ok;
        std::printf("pass=%d expected=%08x,%08x address_load=%08x,%08x readback=%08x,%08x %s\n", pass + 1,
                    expected[pass][0], expected[pass][1], loaded[0], loaded[1], copied[0], copied[1],
                    pass_ok ? "ok" : "MISMATCH");
    }

    g.destroy_pipeline(device, pipeline, nullptr);
    g.destroy_pipeline_layout(device, pipeline_layout, nullptr);
    g.destroy_descriptor_pool(device, pool, nullptr);
    g.destroy_descriptor_set_layout(device, set_layout, nullptr);
    g.destroy_shader_module(device, shader_module, nullptr);
    g.destroy_command_pool(device, command_pool, nullptr);
    Buffer *items[] = {&data[0], &data[1], &staging, &readback, &addresses, &output};
    for (Buffer *item : items) g.destroy_buffer(device, item->buffer, nullptr);
    for (Buffer *item : items) {
        if (item->owns_memory) g.free_memory(device, item->memory, nullptr);
    }
    g.destroy_device(device, nullptr);
    g.destroy_instance(instance, nullptr);
    dlclose(library);
    std::printf("VERDICT=%s\n", mismatch ? "MISMATCH" : "MATCH");
    return mismatch ? 1 : 0;
}
