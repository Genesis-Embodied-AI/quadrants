/* Minimum Metal placement-heap churn. No Vulkan, MoltenVK, or Quadrants.
   mode 0: one ordinary 128 MiB private buffer, two 4-byte blit fills.
   mode 1: one 128 MiB placement heap, two 4-byte buffers inside it, two blit fills.
   Each cycle releases every Metal object before the next one is created. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static int one_cycle(int mode) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (devices.count == 0) return 1;
        id<MTLDevice> device = devices[0];
        MTLResourceOptions options = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
        const uint64_t bytes = 128ull << 20;
        MTLSizeAndAlign requirement = [device heapBufferSizeAndAlignWithLength:4 options:options];
        if (requirement.align == 0) return 2;
        uint64_t offset = ((requirement.size + requirement.align - 1) / requirement.align) * requirement.align;
        if (offset + requirement.size > bytes) return 3;

        id<MTLHeap> heap = nil;
        id<MTLBuffer> first = nil;
        id<MTLBuffer> second = nil;
        if (mode == 1) {
            MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
            descriptor.type = MTLHeapTypePlacement;
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
            descriptor.size = bytes;
            heap = [device newHeapWithDescriptor:descriptor];
            if (!heap) return 4;
            first = [heap newBufferWithLength:4 options:options offset:0];
            second = [heap newBufferWithLength:4 options:options offset:offset];
        } else {
            first = [device newBufferWithLength:bytes options:options];
            second = first;
        }
        if (!first || !second || first.gpuAddress == 0 || second.gpuAddress == 0) return 5;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return 6;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (!command || !encoder) return 7;
        [encoder fillBuffer:first range:NSMakeRange(0, 4) value:0];
        [encoder fillBuffer:second range:NSMakeRange(mode == 1 ? 0 : offset, 4) value:0];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "command failed: %s\n", command.error.description.UTF8String);
            return 8;
        }
        return 0;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 64;
    int mode = atoi(argv[1]);
    int cycles = atoi(argv[2]);
    if ((mode != 0 && mode != 1) || cycles < 1) return 64;
    std::printf("worker pid=%d mode=%d cycles=%d bytes=%llu\n", getpid(), mode, cycles,
                (unsigned long long)(128ull << 20));
    std::fflush(stdout);
    for (int i = 0; i < cycles; i++) {
        int result = one_cycle(mode);
        if (result != 0) {
            std::fprintf(stderr, "failed cycle=%d result=%d\n", i, result);
            return 66;
        }
        if (i % 25 == 0 || i + 1 == cycles) {
            std::printf("completed=%d pid=%d\n", i + 1, getpid());
            std::fflush(stdout);
        }
        usleep(50000);
    }
    std::printf("worker done pid=%d\n", getpid());
    return 0;
}
