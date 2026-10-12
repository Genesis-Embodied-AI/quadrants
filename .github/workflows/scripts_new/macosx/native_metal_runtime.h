// Selected MoltenVK runtime resource pattern expressed directly in Metal, without Vulkan or VMA.
// Included by native_metal_heap.mm; ARC and one autorelease pool per cycle own the resources.

struct RuntimeBuffer {
  id<MTLHeap> heap;
  id<MTLBuffer> buffer;
};

static RuntimeBuffer runtime_buffer(id<MTLDevice> device, uint64_t bytes, uint64_t allocation_bytes, bool heaps) {
  RuntimeBuffer result{};
  MTLResourceOptions options = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
  if (heaps) {
    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    descriptor.size = allocation_bytes;
    result.heap = [device newHeapWithDescriptor:descriptor];
    result.buffer = [result.heap newBufferWithLength:bytes options:options offset:0];
  } else {
    result.buffer = [device newBufferWithLength:allocation_bytes options:options];
  }
  return result;
}

static bool runtime_complete(id<MTLCommandBuffer> command) {
  if (!command)
    return false;
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::fprintf(stderr, "runtime Metal command failed: %s\n", command.error.description.UTF8String);
    return false;
  }
  return true;
}

static void runtime_fill(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> buffer, NSUInteger bytes) {
  uint32_t zero = 0;
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:buffer offset:0 atIndex:0];
  [encoder setBytes:&zero length:sizeof(zero) atIndex:1];
  NSUInteger words = bytes / 4;
  NSUInteger width = std::min(pipeline.maxTotalThreadsPerThreadgroup, pipeline.device.maxThreadsPerThreadgroup.width);
  if (words >= width)
    [encoder dispatchThreadgroups:MTLSizeMake(words / width, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  if (words % width) {
    [encoder setBufferOffset:(words / width) * width * 4 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(words % width, 1, 1)];
  }
}

extern "C" int native_metal_runtime_cycle(int use_heaps, ProbeStats *stats) {
  @autoreleasepool {
    // Retain only the device/library across cycles, like the retained physical device. Recreate pipeline and queue.
    static id<MTLDevice> device = MTLCopyAllDevices().firstObject;
    static id<MTLLibrary> library = [device
        newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
                              "kernel void fill_words(device uint* dst [[buffer(0)]], "
                              "constant uint& value [[buffer(1)]], uint i [[thread_position_in_grid]]) {dst[i]=value;}"
                     options:nil
                       error:nullptr];
    if (!device || !library)
      return 20;
    id<MTLFunction> function = [library newFunctionWithName:@"fill_words"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:nullptr];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!pipeline || !queue)
      return 21;
    auto gtmp = runtime_buffer(device, 1 << 20, 1 << 20, use_heaps);
    auto listgen = runtime_buffer(device, 32 << 20, 32 << 20, use_heaps);
    if (!gtmp.buffer || !listgen.buffer || !gtmp.buffer.gpuAddress || !listgen.buffer.gpuAddress)
      return 22;
    id<MTLCommandBuffer> initial = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [initial computeCommandEncoder];
    if (!encoder)
      return 23;
    runtime_fill(encoder, pipeline, gtmp.buffer, 1 << 20);
    runtime_fill(encoder, pipeline, listgen.buffer, 32 << 20);
    [encoder endEncoding];
    if (!runtime_complete(initial))
      return 24;

    id<MTLHeap> host_heap = nil;
    id<MTLBuffer> mapping = nil;
    id<MTLBuffer> seed = nil;
    MTLResourceOptions shared = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    if (use_heaps) {
      MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
      descriptor.type = MTLHeapTypePlacement;
      descriptor.storageMode = MTLStorageModeShared;
      descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
      descriptor.size = 32 << 20;
      host_heap = [device newHeapWithDescriptor:descriptor];
      mapping = [host_heap newBufferWithLength:32 << 20 options:shared offset:0];
      if (!mapping)
        return 25;
      [mapping makeAliasable];
      *static_cast<uint32_t *>(mapping.contents) = 0;
      seed = [host_heap newBufferWithLength:4 options:shared offset:0];
    } else {
      mapping = [device newBufferWithLength:32 << 20 options:shared];
      if (!mapping)
        return 26;
      *static_cast<uint32_t *>(mapping.contents) = 0;
      seed = mapping;
    }
    if (!seed)
      return 27;
    id<MTLCommandBuffer> upload = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [upload blitCommandEncoder];
    if (!blit)
      return 28;
    [blit copyFromBuffer:seed sourceOffset:0 toBuffer:gtmp.buffer destinationOffset:4096 size:4];
    [blit endEncoding];
    if (!runtime_complete(upload))
      return 29;
    // All commands have completed. Drop the upload command and retain the host mapping and heap.
    blit = nil;
    upload = nil;
    seed = nil;

    auto root = runtime_buffer(device, 4, 32 << 20, use_heaps);
    if (!root.buffer || !root.buffer.gpuAddress)
      return 30;
    id<MTLCommandBuffer> clear_root = [queue commandBuffer];
    encoder = [clear_root computeCommandEncoder];
    if (!encoder)
      return 31;
    runtime_fill(encoder, pipeline, root.buffer, 4);
    [encoder endEncoding];
    if (!runtime_complete(clear_root))
      return 32;
    stats->allocation_bytes = use_heaps ? root.heap.size : root.buffer.length;
    stats->alignment = [device heapBufferSizeAndAlignWithLength:4 options:MTLResourceStorageModePrivate].align;
    stats->second_offset = 0;
    stats->registry_id = device.registryID;
    stats->device_object = reinterpret_cast<uintptr_t>((__bridge void *)device);
    stats->command_status = clear_root.status;
    return 0;
  }
}
