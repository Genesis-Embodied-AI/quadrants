#pragma once

#include "quadrants/rhi/device.h"

namespace quadrants::lang {

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size);

}  // namespace quadrants::lang
