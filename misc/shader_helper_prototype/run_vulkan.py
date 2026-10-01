"""Run a standalone SPIR-V kernel using the Python Vulkan bindings."""

import struct
from contextlib import ExitStack

import vulkan as vk


def check_kernel(path):
    # Four local invocations per workgroup, with a 7 x 3 x 2 workgroup grid.
    expected = []
    for z in range(2):
        for y in range(3):
            for x in range(28):
                block = (x // 4, y, z)
                expected.extend((*block, block[x % 3]))
    size = len(expected) * 4
    with ExitStack() as cleanup:
        app = vk.VkApplicationInfo(pApplicationName="shader-helper-prototype", apiVersion=vk.VK_MAKE_VERSION(1, 1, 0))
        instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
        cleanup.callback(vk.vkDestroyInstance, instance, None)
        devices = vk.vkEnumeratePhysicalDevices(instance)
        # Require real GPU hardware so a software Vulkan driver cannot produce a false GPU success.
        physical = next(
            d
            for d in devices
            if vk.vkGetPhysicalDeviceProperties(d).deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        )
        print("Vulkan device:", vk.vkGetPhysicalDeviceProperties(physical).deviceName, flush=True)
        family = next(
            i
            for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical))
            if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
        )
        queue_info = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=family, queueCount=1, pQueuePriorities=[1.0])
        device = vk.vkCreateDevice(
            physical, vk.VkDeviceCreateInfo(queueCreateInfoCount=1, pQueueCreateInfos=[queue_info]), None
        )
        cleanup.callback(vk.vkDestroyDevice, device, None)
        queue = vk.vkGetDeviceQueue(device, family, 0)
        buffer = vk.vkCreateBuffer(
            device,
            vk.VkBufferCreateInfo(
                size=size, usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(device, buffer)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(physical)
        flags = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        memory_type = next(
            i
            for i in range(properties.memoryTypeCount)
            if requirements.memoryTypeBits & (1 << i) and properties.memoryTypes[i].propertyFlags & flags == flags
        )
        memory = vk.vkAllocateMemory(
            device, vk.VkMemoryAllocateInfo(allocationSize=requirements.size, memoryTypeIndex=memory_type), None
        )
        cleanup.callback(vk.vkFreeMemory, device, memory, None)
        cleanup.callback(vk.vkDestroyBuffer, device, buffer, None)
        vk.vkBindBufferMemory(device, buffer, memory, 0)
        mapped = vk.vkMapMemory(device, memory, 0, size, 0)
        mapped[:] = b"\xff" * size
        vk.vkUnmapMemory(device, memory)
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        layout = vk.vkCreateDescriptorSetLayout(
            device, vk.VkDescriptorSetLayoutCreateInfo(bindingCount=1, pBindings=[binding]), None
        )
        cleanup.callback(vk.vkDestroyDescriptorSetLayout, device, layout, None)
        pool_size = vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1)
        pool = vk.vkCreateDescriptorPool(
            device, vk.VkDescriptorPoolCreateInfo(maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size]), None
        )
        cleanup.callback(vk.vkDestroyDescriptorPool, device, pool, None)
        descriptor = vk.vkAllocateDescriptorSets(
            device, vk.VkDescriptorSetAllocateInfo(descriptorPool=pool, descriptorSetCount=1, pSetLayouts=[layout])
        )[0]
        info = vk.VkDescriptorBufferInfo(buffer=buffer, offset=0, range=size)
        write = vk.VkWriteDescriptorSet(
            dstSet=descriptor,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[info],
        )
        vk.vkUpdateDescriptorSets(device, 1, [write], 0, None)
        pipeline_layout = vk.vkCreatePipelineLayout(
            device, vk.VkPipelineLayoutCreateInfo(setLayoutCount=1, pSetLayouts=[layout]), None
        )
        cleanup.callback(vk.vkDestroyPipelineLayout, device, pipeline_layout, None)
        code = path.read_bytes()
        module = vk.vkCreateShaderModule(device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        cleanup.callback(vk.vkDestroyShaderModule, device, module, None)
        stage = vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main")
        pipeline = vk.vkCreateComputePipelines(
            device, vk.VK_NULL_HANDLE, 1, [vk.VkComputePipelineCreateInfo(stage=stage, layout=pipeline_layout)], None
        )[0]
        cleanup.callback(vk.vkDestroyPipeline, device, pipeline, None)
        command_pool = vk.vkCreateCommandPool(device, vk.VkCommandPoolCreateInfo(queueFamilyIndex=family), None)
        cleanup.callback(vk.vkDestroyCommandPool, device, command_pool, None)
        command = vk.vkAllocateCommandBuffers(
            device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=command_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1
            ),
        )[0]
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo())
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, [descriptor], 0, None
        )
        vk.vkCmdDispatch(command, 7, 3, 2)
        barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT, dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT
        )
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            [barrier],
            0,
            None,
            0,
            None,
        )
        vk.vkEndCommandBuffer(command)
        vk.vkQueueSubmit(
            queue, 1, [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[command])], vk.VK_NULL_HANDLE
        )
        vk.vkQueueWaitIdle(queue)
        mapped = vk.vkMapMemory(device, memory, 0, size, 0)
        actual = list(struct.unpack(f"{len(expected)}I", bytes(mapped)))
        vk.vkUnmapMemory(device, memory)
        if actual != expected:
            index = next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
            raise AssertionError(f"{path.name}: output[{index}]={actual[index]}, expected {expected[index]}")
        print(f"PASS {path.name}: {len(expected)} values across 168 invocations / 42 workgroups", flush=True)
