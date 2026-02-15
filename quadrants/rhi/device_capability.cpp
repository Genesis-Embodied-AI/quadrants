#include "quadrants/common/core.h"
#include "quadrants/rhi/device_capability.h"

namespace quadrants::lang {

DeviceCapability str2devcap(const std::string_view &name) {
#define PER_DEVICE_CAPABILITY(x) \
  if (#x == name)                \
    return DeviceCapability::x;
#include "quadrants/inc/rhi_constants.inc.h"
#undef PER_DEVICE_CAPABILITY
  QD_ERROR("unexpected device capability name {}", name);
}

const std::string to_string(DeviceCapability c) {
#define PER_DEVICE_CAPABILITY(name) \
  case DeviceCapability::name:      \
    return #name;                   \
    break;
  switch (c) {
#include "quadrants/inc/rhi_constants.inc.h"
    default:
      return "Unknown";
      break;
  }
#undef PER_DEVICE_CAPABILITY
}

uint32_t DeviceCapabilityConfig::contains(DeviceCapability cap) const {
  auto it = devcaps.find(cap);
  return it != devcaps.end();
}
uint32_t DeviceCapabilityConfig::get(DeviceCapability cap) const {
  auto it = devcaps.find(cap);
  if (it != devcaps.end()) {
    return it->second;
  }
  return 0;
}
void DeviceCapabilityConfig::set(DeviceCapability cap, uint32_t level) {
  devcaps[cap] = level;
}

void DeviceCapabilityConfig::dbg_print_all() const {
  for (auto &pair : devcaps) {
    QD_TRACE("DeviceCapability::{} ({}) = {}", to_string(pair.first),
             int(pair.first), pair.second);
  }
}

const std::map<DeviceCapability, uint32_t> &DeviceCapabilityConfig::to_inner()
    const {
  return devcaps;
}

}  // namespace quadrants::lang
