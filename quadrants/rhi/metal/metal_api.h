#pragma once
#include "quadrants/rhi/device.h"

namespace quadrants::lang {
namespace metal {

bool is_metal_api_available();

std::shared_ptr<Device> create_metal_device();

}  // namespace metal
}  // namespace quadrants::lang
