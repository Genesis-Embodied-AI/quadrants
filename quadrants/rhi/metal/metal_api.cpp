#include "quadrants/rhi/metal/metal_api.h"
#include "quadrants/rhi/metal/metal_device.h"

namespace quadrants::lang {
namespace metal {

bool is_metal_api_available() {
#if defined(__APPLE__) && defined(QD_WITH_METAL)
  return true;
#else
  return false;
#endif  // defined(__APPLE__) && defined(QD_WITH_METAL)
}

std::shared_ptr<Device> create_metal_device() {
#if defined(__APPLE__) && defined(QD_WITH_METAL)
  return std::shared_ptr<Device>(metal::MetalDevice::create());
#else
  return nullptr;
#endif  // defined(__APPLE__) && defined(QD_WITH_METAL)
}

}  // namespace metal
}  // namespace quadrants::lang
