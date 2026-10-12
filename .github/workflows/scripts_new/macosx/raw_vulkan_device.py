"""Direct MoltenVK device creation/destruction, without Quadrants runtime allocations or shaders.

Vulkan 1.0 ABI layouts follow Khronos Vulkan-Headers v1.3.290/include/vulkan/vulkan_core.h.
One instance/physical device survives the loop. Each iteration creates one compute queue/device, optionally allocates two device-local buffers and fills them,
waits idle, and destroys its resources/device. Host modes bind two four-byte buffers into one shared allocation,
map/write/unmap, fill on GPU, use a transfer-to-host barrier, wait, and verify mapped readback. The shared allocation
is 32MiB in host mode and at least one 16KiB page in host-small. Private/private-small use the same sizes and
bindings with device-local memory, GPU fills, and no host mapping or readback. No application shader or pipeline is created. The caller owns the surrounding retained-graph/GC diagnostic and closes the instance.
"""

import ctypes as C
import hashlib

U32, Ptr, Strings = C.c_uint32, C.c_void_p, C.POINTER(C.c_char_p)


class Application(C.Structure):
    _fields_ = [
        ("sType", U32),
        ("pNext", Ptr),
        ("pApplicationName", C.c_char_p),
        ("applicationVersion", U32),
        ("pEngineName", C.c_char_p),
        ("engineVersion", U32),
        ("apiVersion", U32),
    ]


class InstanceInfo(C.Structure):
    _fields_ = [
        ("sType", U32),
        ("pNext", Ptr),
        ("flags", U32),
        ("pApplicationInfo", C.POINTER(Application)),
        ("enabledLayerCount", U32),
        ("ppEnabledLayerNames", Strings),
        ("enabledExtensionCount", U32),
        ("ppEnabledExtensionNames", Strings),
    ]


class QueueInfo(C.Structure):
    _fields_ = [
        ("sType", U32),
        ("pNext", Ptr),
        ("flags", U32),
        ("queueFamilyIndex", U32),
        ("queueCount", U32),
        ("pQueuePriorities", C.POINTER(C.c_float)),
    ]


class DeviceInfo(C.Structure):
    _fields_ = [
        ("sType", U32),
        ("pNext", Ptr),
        ("flags", U32),
        ("queueCreateInfoCount", U32),
        ("pQueueCreateInfos", C.POINTER(QueueInfo)),
        ("enabledLayerCount", U32),
        ("ppEnabledLayerNames", Strings),
        ("enabledExtensionCount", U32),
        ("ppEnabledExtensionNames", Strings),
        ("pEnabledFeatures", Ptr),
    ]


class Extension(C.Structure):
    _fields_ = [("extensionName", C.c_char * 256), ("specVersion", U32)]


class QueueProperties(C.Structure):
    _fields_ = [
        ("queueFlags", U32),
        ("queueCount", U32),
        ("timestampValidBits", U32),
        ("minImageTransferGranularity", U32 * 3),
    ]


U64 = C.c_uint64


class MemoryType(C.Structure):
    _fields_ = [("propertyFlags", U32), ("heapIndex", U32)]


class MemoryHeap(C.Structure):
    _fields_ = [("size", U64), ("flags", U32)]


class MemoryProperties(C.Structure):
    _fields_ = [
        ("memoryTypeCount", U32),
        ("memoryTypes", MemoryType * 32),
        ("memoryHeapCount", U32),
        ("memoryHeaps", MemoryHeap * 16),
    ]


class BufferInfo(C.Structure):
    _fields_ = [
        ("sType", U32),
        ("pNext", Ptr),
        ("flags", U32),
        ("size", U64),
        ("usage", U32),
        ("sharingMode", U32),
        ("queueFamilyIndexCount", U32),
        ("pQueueFamilyIndices", C.POINTER(U32)),
    ]


class MemoryRequirements(C.Structure):
    _fields_ = [("size", U64), ("alignment", U64), ("memoryTypeBits", U32)]


class AllocateInfo(C.Structure):
    _fields_ = [("sType", U32), ("pNext", Ptr), ("allocationSize", U64), ("memoryTypeIndex", U32)]


class PoolInfo(C.Structure):
    _fields_ = [("sType", U32), ("pNext", Ptr), ("flags", U32), ("queueFamilyIndex", U32)]


class CommandAllocateInfo(C.Structure):
    _fields_ = [("sType", U32), ("pNext", Ptr), ("commandPool", U64), ("level", U32), ("commandBufferCount", U32)]


class BeginInfo(C.Structure):
    _fields_ = [("sType", U32), ("pNext", Ptr), ("flags", U32), ("pInheritanceInfo", Ptr)]


class MemoryBarrier(C.Structure):
    _fields_ = [("sType", U32), ("pNext", Ptr), ("srcAccessMask", U32), ("dstAccessMask", U32)]


class SubmitInfo(C.Structure):
    _fields_ = [
        ("sType", U32),
        ("pNext", Ptr),
        ("waitSemaphoreCount", U32),
        ("pWaitSemaphores", C.POINTER(U64)),
        ("pWaitDstStageMask", C.POINTER(U32)),
        ("commandBufferCount", U32),
        ("pCommandBuffers", C.POINTER(Ptr)),
        ("signalSemaphoreCount", U32),
        ("pSignalSemaphores", C.POINTER(U64)),
    ]


def check(result, name):
    if result != 0:
        raise RuntimeError(f"{name}: VkResult {result}")


class RawVulkanDevice:
    def __init__(self, library_path, work="device"):
        assert work in ("device", "buffers", "fill", "host", "host-small", "private", "private-small")
        self.work = work
        self.buffer_records = []
        assert C.sizeof(Ptr) == 8
        self.library = C.CDLL(str(library_path))
        self.get_proc = self.library.vkGetInstanceProcAddr
        self.get_proc.argtypes, self.get_proc.restype = [Ptr, C.c_char_p], Ptr
        self.instance = Ptr()
        self.destroy_instance = None
        try:
            enumerate_extensions = self.bind(
                "vkEnumerateInstanceExtensionProperties", C.c_int32, C.c_char_p, C.POINTER(U32), C.POINTER(Extension)
            )
            names = self.extension_names(enumerate_extensions, None)
            portability = b"VK_KHR_portability_enumeration"
            enabled = (C.c_char_p * 1)(portability) if portability in names else None
            application = Application(sType=0, pApplicationName=b"raw-device-cycle-control", apiVersion=1 << 22)
            info = InstanceInfo(
                sType=1,
                flags=1 if enabled is not None else 0,
                pApplicationInfo=C.pointer(application),
                enabledExtensionCount=1 if enabled is not None else 0,
                ppEnabledExtensionNames=enabled,
            )
            create_instance = self.bind("vkCreateInstance", C.c_int32, C.POINTER(InstanceInfo), Ptr, C.POINTER(Ptr))
            check(create_instance(C.byref(info), None, C.byref(self.instance)), "vkCreateInstance")
            self.destroy_instance = self.bind("vkDestroyInstance", None, Ptr, Ptr)
            enumerate_physical = self.bind("vkEnumeratePhysicalDevices", C.c_int32, Ptr, C.POINTER(U32), C.POINTER(Ptr))
            count = U32()
            check(enumerate_physical(self.instance, C.byref(count), None), "vkEnumeratePhysicalDevices(count)")
            assert count.value > 0
            devices = (Ptr * count.value)()
            check(enumerate_physical(self.instance, C.byref(count), devices), "vkEnumeratePhysicalDevices")
            self.physical = devices[0]
            physical_count = count.value
            get_queues = self.bind(
                "vkGetPhysicalDeviceQueueFamilyProperties", None, Ptr, C.POINTER(U32), C.POINTER(QueueProperties)
            )
            get_queues(self.physical, C.byref(count), None)
            queues = (QueueProperties * count.value)()
            get_queues(self.physical, C.byref(count), queues)
            self.queue_index = next(i for i, q in enumerate(queues) if q.queueCount and q.queueFlags & 2)
            enumerate_device_extensions = self.bind(
                "vkEnumerateDeviceExtensionProperties", C.c_int32, Ptr, C.c_char_p, C.POINTER(U32), C.POINTER(Extension)
            )
            names = self.extension_names(enumerate_device_extensions, self.physical, None)
            portability = b"VK_KHR_portability_subset"
            self.extensions = (C.c_char_p * 1)(portability) if portability in names else None
            self.create_device = self.bind("vkCreateDevice", C.c_int32, Ptr, C.POINTER(DeviceInfo), Ptr, C.POINTER(Ptr))
            self.destroy_device = self.bind("vkDestroyDevice", None, Ptr, Ptr)
            self.wait_idle = self.bind("vkDeviceWaitIdle", C.c_int32, Ptr)
            if self.work != "device":
                self.bind_buffer_functions()
            self.metadata = dict(
                library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
                physical_devices=physical_count,
                queue_family=self.queue_index,
                instance_portability_enabled=enabled is not None,
                device_portability_enabled=self.extensions is not None,
                api_version="1.0",
                work=self.work,
                application_buffer_bytes=(
                    []
                    if self.work == "device"
                    else [4, 4] if self.work.startswith(("host", "private")) else [1 << 20, 32 << 20]
                ),
                application_shaders=0,
            )
        except BaseException:
            self.close()
            raise

    def bind(self, name, result, *arguments):
        address = self.get_proc(self.instance, name.encode())
        if not address:
            raise RuntimeError(f"Missing Vulkan entry point: {name}")
        return C.CFUNCTYPE(result, *arguments)(address)

    @staticmethod
    def extension_names(function, *prefix):
        count = U32()
        check(function(*prefix, C.byref(count), None), "enumerate extensions(count)")
        extensions = (Extension * count.value)()
        check(function(*prefix, C.byref(count), extensions), "enumerate extensions")
        return {extension.extensionName for extension in extensions[: count.value]}

    def cycle(self):
        priority = C.c_float(1)
        queue = QueueInfo(
            sType=2, queueFamilyIndex=self.queue_index, queueCount=1, pQueuePriorities=C.pointer(priority)
        )
        info = DeviceInfo(
            sType=3,
            queueCreateInfoCount=1,
            pQueueCreateInfos=C.pointer(queue),
            enabledExtensionCount=1 if self.extensions is not None else 0,
            ppEnabledExtensionNames=self.extensions,
        )
        device = Ptr()
        check(self.create_device(self.physical, C.byref(info), None, C.byref(device)), "vkCreateDevice")
        try:
            if self.work.startswith(("host", "private")):
                self.shared_cycle(device)
            elif self.work != "device":
                self.buffer_cycle(device)
            check(self.wait_idle(device), "vkDeviceWaitIdle")
        finally:
            self.destroy_device(device, None)

    def bind_buffer_functions(self):
        self.memory = MemoryProperties()
        self.bind("vkGetPhysicalDeviceMemoryProperties", None, Ptr, C.POINTER(MemoryProperties))(
            self.physical, C.byref(self.memory)
        )
        assert 0 < self.memory.memoryTypeCount <= 32 and 0 < self.memory.memoryHeapCount <= 16
        self.create_buffer = self.bind("vkCreateBuffer", C.c_int32, Ptr, C.POINTER(BufferInfo), Ptr, C.POINTER(U64))
        self.destroy_buffer = self.bind("vkDestroyBuffer", None, Ptr, U64, Ptr)
        self.requirements = self.bind("vkGetBufferMemoryRequirements", None, Ptr, U64, C.POINTER(MemoryRequirements))
        self.allocate_memory = self.bind(
            "vkAllocateMemory", C.c_int32, Ptr, C.POINTER(AllocateInfo), Ptr, C.POINTER(U64)
        )
        self.free_memory = self.bind("vkFreeMemory", None, Ptr, U64, Ptr)
        self.bind_memory = self.bind("vkBindBufferMemory", C.c_int32, Ptr, U64, U64, U64)
        if self.work in ("fill", "host", "host-small", "private", "private-small"):
            self.create_pool = self.bind(
                "vkCreateCommandPool", C.c_int32, Ptr, C.POINTER(PoolInfo), Ptr, C.POINTER(U64)
            )
            self.destroy_pool = self.bind("vkDestroyCommandPool", None, Ptr, U64, Ptr)
            self.allocate_commands = self.bind(
                "vkAllocateCommandBuffers", C.c_int32, Ptr, C.POINTER(CommandAllocateInfo), C.POINTER(Ptr)
            )
            self.begin_commands = self.bind("vkBeginCommandBuffer", C.c_int32, Ptr, C.POINTER(BeginInfo))
            self.end_commands = self.bind("vkEndCommandBuffer", C.c_int32, Ptr)
            self.fill_buffer = self.bind("vkCmdFillBuffer", None, Ptr, U64, U64, U64, U32)
            self.get_queue = self.bind("vkGetDeviceQueue", None, Ptr, U32, U32, C.POINTER(Ptr))
            self.submit = self.bind("vkQueueSubmit", C.c_int32, Ptr, U32, C.POINTER(SubmitInfo), U64)
            self.queue_wait = self.bind("vkQueueWaitIdle", C.c_int32, Ptr)

        if self.work.startswith("host"):
            self.map_memory = self.bind("vkMapMemory", C.c_int32, Ptr, U64, U64, U64, U32, C.POINTER(Ptr))
            self.unmap_memory = self.bind("vkUnmapMemory", None, Ptr, U64)
            self.barrier = self.bind(
                "vkCmdPipelineBarrier",
                None,
                Ptr,
                U32,
                U32,
                U32,
                U32,
                C.POINTER(MemoryBarrier),
                U32,
                Ptr,
                U32,
                Ptr,
            )

    def shared_cycle(self, device):
        buffers, requirements = [], []
        memory, mapped = U64(), Ptr()
        host_access = self.work.startswith("host")
        required_flags = 15 if host_access else 1
        try:
            for _ in range(2):
                buffer = U64()
                info = BufferInfo(sType=12, size=4, usage=0x33)
                check(self.create_buffer(device, C.byref(info), None, C.byref(buffer)), "vkCreateBuffer(shared)")
                buffers.append(buffer)
                requirement = MemoryRequirements()
                self.requirements(device, buffer, C.byref(requirement))
                requirements.append(requirement)
            assert requirements[1].alignment > 0
            alignment = requirements[1].alignment
            offset = ((requirements[0].size + alignment - 1) // alignment) * alignment
            minimum_size = offset + requirements[1].size
            allocation_size = max(minimum_size, 16384) if self.work.endswith("small") else 32 << 20
            assert minimum_size <= allocation_size
            bits = requirements[0].memoryTypeBits & requirements[1].memoryTypeBits
            memory_type = next(
                i
                for i in range(self.memory.memoryTypeCount)
                if bits & (1 << i) and self.memory.memoryTypes[i].propertyFlags & required_flags == required_flags
            )
            info = AllocateInfo(sType=5, allocationSize=allocation_size, memoryTypeIndex=memory_type)
            check(self.allocate_memory(device, C.byref(info), None, C.byref(memory)), "vkAllocateMemory(shared)")
            offsets = (0, offset)
            for buffer, binding_offset in zip(buffers, offsets):
                check(self.bind_memory(device, buffer, memory, binding_offset), "vkBindBufferMemory(shared)")
            if not self.buffer_records:
                self.buffer_records.extend(
                    dict(
                        requested_bytes=4,
                        allocation_bytes=allocation_size,
                        binding_offset=binding_offset,
                        shared_allocation=True,
                        memory_type_bits=bits,
                        memory_type_index=memory_type,
                        property_flags=self.memory.memoryTypes[memory_type].propertyFlags,
                    )
                    for binding_offset in offsets
                )
            if host_access:
                check(self.map_memory(device, memory, 0, (1 << 64) - 1, 0, C.byref(mapped)), "vkMapMemory(write)")
                assert mapped.value
                for binding_offset in offsets:
                    C.c_uint32.from_address(mapped.value + binding_offset).value = 0xDEADBEEF
                self.unmap_memory(device, memory)
                mapped = Ptr()
            self.fill_cycle(device, [(buffer, memory) for buffer in buffers])
            if host_access:
                check(self.map_memory(device, memory, 0, (1 << 64) - 1, 0, C.byref(mapped)), "vkMapMemory(read)")
                assert mapped.value
                for binding_offset in offsets:
                    assert (
                        C.c_uint32.from_address(mapped.value + binding_offset).value == 0
                    ), "GPU fill/readback mismatch"
        finally:
            if mapped.value:
                self.unmap_memory(device, memory)
            for buffer in reversed(buffers):
                self.destroy_buffer(device, buffer, None)
            if memory.value:
                self.free_memory(device, memory, None)

    def buffer_cycle(self, device):
        allocations = []
        try:
            for size in (1 << 20, 32 << 20):
                buffer, memory = U64(), U64()
                info = BufferInfo(sType=12, size=size, usage=0x22)
                check(self.create_buffer(device, C.byref(info), None, C.byref(buffer)), "vkCreateBuffer")
                allocations.append((buffer, memory))
                requirements = MemoryRequirements()
                self.requirements(device, buffer, C.byref(requirements))
                memory_type = next(
                    i
                    for i in range(self.memory.memoryTypeCount)
                    if requirements.memoryTypeBits & (1 << i) and self.memory.memoryTypes[i].propertyFlags & 1
                )
                if len(self.buffer_records) < 2:
                    self.buffer_records.append(
                        dict(
                            requested_bytes=size,
                            allocation_bytes=requirements.size,
                            memory_type_bits=requirements.memoryTypeBits,
                            memory_type_index=memory_type,
                            property_flags=self.memory.memoryTypes[memory_type].propertyFlags,
                        )
                    )
                allocation = AllocateInfo(sType=5, allocationSize=requirements.size, memoryTypeIndex=memory_type)
                check(self.allocate_memory(device, C.byref(allocation), None, C.byref(memory)), "vkAllocateMemory")
                check(self.bind_memory(device, buffer, memory, 0), "vkBindBufferMemory")
            if self.work == "fill":
                self.fill_cycle(device, allocations)
        finally:
            for buffer, memory in reversed(allocations):
                self.destroy_buffer(device, buffer, None)
                if memory.value:
                    self.free_memory(device, memory, None)

    def fill_cycle(self, device, allocations):
        pool = U64()
        info = PoolInfo(sType=39, flags=1, queueFamilyIndex=self.queue_index)
        check(self.create_pool(device, C.byref(info), None, C.byref(pool)), "vkCreateCommandPool")
        try:
            command = Ptr()
            allocate = CommandAllocateInfo(sType=40, commandPool=pool.value, level=0, commandBufferCount=1)
            check(self.allocate_commands(device, C.byref(allocate), C.byref(command)), "vkAllocateCommandBuffers")
            begin = BeginInfo(sType=42, flags=1)
            check(self.begin_commands(command, C.byref(begin)), "vkBeginCommandBuffer")
            for buffer, _ in allocations:
                self.fill_buffer(command, buffer, 0, (1 << 64) - 1, 0)
            if self.work.startswith("host"):
                barrier = MemoryBarrier(sType=46, srcAccessMask=0x1000, dstAccessMask=0x2000)
                self.barrier(command, 0x1000, 0x4000, 0, 1, C.byref(barrier), 0, None, 0, None)
            check(self.end_commands(command), "vkEndCommandBuffer")
            queue = Ptr()
            self.get_queue(device, self.queue_index, 0, C.byref(queue))
            assert queue.value
            submit = SubmitInfo(sType=4, commandBufferCount=1, pCommandBuffers=C.pointer(command))
            check(self.submit(queue, 1, C.byref(submit), 0), "vkQueueSubmit")
            check(self.queue_wait(queue), "vkQueueWaitIdle")
        finally:
            self.destroy_pool(device, pool, None)

    def close(self):
        if self.instance and self.destroy_instance:
            self.destroy_instance(self.instance, None)
            self.instance = Ptr()
