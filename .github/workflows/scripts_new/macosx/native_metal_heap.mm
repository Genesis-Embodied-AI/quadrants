// Direct Metal allocation/blit control. No Vulkan, VMA, shader source, or Quadrants API calls.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>

struct ProbeStats {
  uint64_t allocation_bytes;
  uint64_t alignment;
  uint64_t second_offset;
  uint64_t registry_id;
  uint64_t device_object;
  uint64_t command_status;
};

extern "C" int native_metal_cycle(uint64_t bytes, int use_heap, ProbeStats *stats) {
  @autoreleasepool {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (devices.count == 0)
      return 1;
    id<MTLDevice> device = devices[0];
    stats->registry_id = device.registryID;
    stats->device_object = reinterpret_cast<uintptr_t>((__bridge void *)device);
    MTLResourceOptions options = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
    MTLSizeAndAlign requirement = [device heapBufferSizeAndAlignWithLength:4 options:options];
    if (requirement.align == 0)
      return 2;
    uint64_t offset = ((requirement.size + requirement.align - 1) / requirement.align) * requirement.align;
    if (offset + requirement.size > bytes)
      return 3;
    stats->alignment = requirement.align;
    stats->second_offset = offset;
    id<MTLHeap> heap = nil;
    id<MTLBuffer> first = nil;
    id<MTLBuffer> second = nil;
    if (use_heap) {
      MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
      descriptor.type = MTLHeapTypePlacement;
      descriptor.storageMode = MTLStorageModePrivate;
      descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
      descriptor.size = bytes;
      heap = [device newHeapWithDescriptor:descriptor];
      if (!heap)
        return 4;
      first = [heap newBufferWithLength:4 options:options offset:0];
      second = [heap newBufferWithLength:4 options:options offset:offset];
      stats->allocation_bytes = heap.size;
    } else {
      first = [device newBufferWithLength:bytes options:options];
      second = first;
      stats->allocation_bytes = first.length;
    }
    if (!first || !second || first.gpuAddress == 0 || second.gpuAddress == 0)
      return 5;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue)
      return 6;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    if (!command || !encoder)
      return 7;
    [encoder fillBuffer:first range:NSMakeRange(0, 4) value:0];
    [encoder fillBuffer:second range:NSMakeRange(use_heap ? 0 : offset, 4) value:0];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    stats->command_status = command.status;
    if (command.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "native Metal command failed: %s\n", command.error.description.UTF8String);
      return 8;
    }
    // ARC and this per-call autorelease pool release the queue, completed command, buffers, heap and device.
    return 0;
  }
}

#include "native_metal_runtime.h"
