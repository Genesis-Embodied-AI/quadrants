#include "quadrants/platform/amdgpu/detect_amdgpu.h"

#if defined(QD_WITH_AMDGPU)
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#endif

namespace quadrants {

bool is_rocm_api_available() {
#if defined(QD_WITH_AMDGPU)
  return lang::AMDGPUDriver::get_instance_without_context().detected();
#else
  return false;
#endif
}

}  // namespace quadrants
