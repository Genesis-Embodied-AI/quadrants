#include "quadrants/runtime/amdgpu/amdgpu_utils.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

namespace quadrants::lang {
namespace amdgpu {

bool on_amdgpu_device(void *ptr) {
  unsigned int attr_val[8];
  // `mem_get_attribute` is unreliable on ROCm; `mem_get_attributes` (plural) returns the attribute block and we read
  // the `memoryType` slot from it.
  uint32_t ret_code = AMDGPUDriver::get_instance().mem_get_attributes.call(attr_val, ptr);
  return ret_code == HIP_SUCCESS && attr_val[0] == HIP_MEMORYTYPE_DEVICE;
}

}  // namespace amdgpu
}  // namespace quadrants::lang
