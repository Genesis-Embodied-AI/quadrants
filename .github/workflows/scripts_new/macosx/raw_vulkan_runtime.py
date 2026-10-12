"""Direct API replay of runtime buffers, seed upload, and optional empty root; no VMA or shaders.

Uses one resettable command pool/buffer, simultaneous-use recording, per-submit fence/signal semaphore, and
queue-idle synchronization as in the archived runtime. Buffers/memory are released before the command pool.
"""

import ctypes as C

from raw_vulkan_device import (
    U32,
    U64,
    AllocateFlags,
    AllocateInfo,
    BeginInfo,
    BufferAddressInfo,
    BufferInfo,
    CommandAllocateInfo,
    MemoryRequirements,
    PoolInfo,
    Ptr,
    SubmitInfo,
    check,
)


class CopyRegion(C.Structure):
    _fields_ = [("srcOffset", U64), ("dstOffset", U64), ("size", U64)]


class SyncInfo(C.Structure):
    _fields_ = [("sType", U32), ("pNext", Ptr), ("flags", U32)]


def runtime_cycle(raw, device):
    map_memory = raw.bind("vkMapMemory", C.c_int32, Ptr, U64, U64, U64, U32, C.POINTER(Ptr))
    unmap_memory = raw.bind("vkUnmapMemory", None, Ptr, U64)
    copy_buffer = raw.bind("vkCmdCopyBuffer", None, Ptr, U64, U64, U32, C.POINTER(CopyRegion))
    create_fence = raw.bind("vkCreateFence", C.c_int32, Ptr, C.POINTER(SyncInfo), Ptr, C.POINTER(U64))
    destroy_fence = raw.bind("vkDestroyFence", None, Ptr, U64, Ptr)
    create_semaphore = raw.bind("vkCreateSemaphore", C.c_int32, Ptr, C.POINTER(SyncInfo), Ptr, C.POINTER(U64))
    destroy_semaphore = raw.bind("vkDestroySemaphore", None, Ptr, U64, Ptr)
    pool, command, queue = U64(), Ptr(), Ptr()
    resources, records = [], []

    def allocate(size, allocation_size, memory_type, usage):
        buffer, memory = U64(), U64()
        item = [buffer, memory]
        resources.append(item)
        info = BufferInfo(sType=12, size=size, usage=usage)
        check(raw.create_buffer(device, C.byref(info), None, C.byref(buffer)), "runtime vkCreateBuffer")
        req = MemoryRequirements()
        raw.requirements(device, buffer, C.byref(req))
        assert req.memoryTypeBits & (1 << memory_type) and req.size <= allocation_size
        flags = AllocateFlags(sType=1000060000, flags=2)
        allocation = AllocateInfo(
            sType=5, pNext=C.cast(C.pointer(flags), Ptr), allocationSize=allocation_size, memoryTypeIndex=memory_type
        )
        check(raw.allocate_memory(device, C.byref(allocation), None, C.byref(memory)), "runtime vkAllocateMemory")
        check(raw.bind_memory(device, buffer, memory, 0), "runtime vkBindBufferMemory")
        if usage & 0x20000:
            address = BufferAddressInfo(sType=1000244001, buffer=buffer)
            assert raw.buffer_address(device, C.byref(address))
        records.append(
            dict(
                buffer_bytes=size,
                allocation_bytes=allocation_size,
                memory_type=memory_type,
                property_flags=raw.memory.memoryTypes[memory_type].propertyFlags,
                usage=usage,
                flags=2,
                requirements_bytes=req.size,
                alignment=req.alignment,
                offset=0,
            )
        )
        return item

    def submit(fills=(), copy=None):
        begin = BeginInfo(sType=42, flags=4)
        check(raw.begin_commands(command, C.byref(begin)), "runtime vkBeginCommandBuffer")
        for buffer in fills:
            raw.fill_buffer(command, buffer, 0, (1 << 64) - 1, 0)
        if copy:
            region = CopyRegion(srcOffset=0, dstOffset=4096, size=4)
            copy_buffer(command, copy[0], copy[1], 1, C.byref(region))
        check(raw.end_commands(command), "runtime vkEndCommandBuffer")
        fence, semaphore = U64(), U64()
        try:
            fence_info, semaphore_info = SyncInfo(sType=8), SyncInfo(sType=9)
            check(create_fence(device, C.byref(fence_info), None, C.byref(fence)), "runtime vkCreateFence")
            check(
                create_semaphore(device, C.byref(semaphore_info), None, C.byref(semaphore)), "runtime vkCreateSemaphore"
            )
            info = SubmitInfo(
                sType=4,
                commandBufferCount=1,
                pCommandBuffers=C.pointer(command),
                signalSemaphoreCount=1,
                pSignalSemaphores=C.pointer(semaphore),
            )
            check(raw.submit(queue, 1, C.byref(info), fence), "runtime vkQueueSubmit")
            check(raw.queue_wait(queue), "runtime vkQueueWaitIdle")
        finally:
            if fence.value:
                destroy_fence(device, fence, None)
            if semaphore.value:
                destroy_semaphore(device, semaphore, None)

    try:
        # Match the recorded types, and fail rather than silently change storage on a different device.
        assert raw.memory.memoryTypeCount >= 2
        assert raw.memory.memoryTypes[0].propertyFlags == 1 and raw.memory.memoryTypes[1].propertyFlags == 15
        gtmp = allocate(1 << 20, 1 << 20, 0, 0x20023)
        listgen = allocate(32 << 20, 32 << 20, 0, 0x20023)
        info = PoolInfo(sType=39, flags=2, queueFamilyIndex=raw.queue_index)
        check(raw.create_pool(device, C.byref(info), None, C.byref(pool)), "runtime vkCreateCommandPool")
        info = CommandAllocateInfo(sType=40, commandPool=pool.value, level=0, commandBufferCount=1)
        check(raw.allocate_commands(device, C.byref(info), C.byref(command)), "runtime vkAllocateCommandBuffers")
        raw.get_queue(device, raw.queue_index, 0, C.byref(queue))
        assert queue.value
        submit(fills=(gtmp[0], listgen[0]))
        seed = allocate(4, 32 << 20, 1, 3)
        mapped = Ptr()
        check(map_memory(device, seed[1], 0, (1 << 64) - 1, 0, C.byref(mapped)), "runtime vkMapMemory")
        assert mapped.value
        C.c_uint32.from_address(mapped.value).value = 0
        unmap_memory(device, seed[1])
        submit(copy=(seed[0], gtmp[0]))
        raw.destroy_buffer(device, seed[0], None)
        seed[0] = U64()
        if raw.runtime_root:
            root = allocate(4, 32 << 20, 0, 0x20023)
            submit(fills=(root[0],))
        if not raw.buffer_records:
            raw.buffer_records = records
    finally:
        # The unpooled runtime buffers free their memory immediately. Host/root pools free after their buffers.
        for index, (buffer, memory) in enumerate(resources):
            if buffer.value:
                raw.destroy_buffer(device, buffer, None)
            if index < 2 and memory.value:
                raw.free_memory(device, memory, None)
        for _, memory in resources[2:]:
            if memory.value:
                raw.free_memory(device, memory, None)
        if pool.value:
            raw.destroy_pool(device, pool, None)
