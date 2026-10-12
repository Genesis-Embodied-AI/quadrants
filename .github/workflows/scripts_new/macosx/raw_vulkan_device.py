"""Direct MoltenVK device creation/destruction, without Quadrants runtime allocations or shaders.

Vulkan 1.0 ABI layouts follow Khronos Vulkan-Headers v1.3.290/include/vulkan/vulkan_core.h.
One instance/physical device survives the loop. Each iteration creates one compute queue/device, waits idle,
and destroys the device. The caller owns the surrounding retained-graph/GC diagnostic and closes the instance.
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


def check(result, name):
    if result != 0:
        raise RuntimeError(f"{name}: VkResult {result}")


class RawVulkanDevice:
    def __init__(self, library_path):
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
            self.metadata = dict(
                library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
                physical_devices=physical_count,
                queue_family=self.queue_index,
                instance_portability_enabled=enabled is not None,
                device_portability_enabled=self.extensions is not None,
                api_version="1.0",
                application_buffers=0,
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
            check(self.wait_idle(device), "vkDeviceWaitIdle")
        finally:
            self.destroy_device(device, None)

    def close(self):
        if self.instance and self.destroy_instance:
            self.destroy_instance(self.instance, None)
            self.instance = Ptr()
