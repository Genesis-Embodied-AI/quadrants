#include "quadrants/platform/cuda/detect_cuda.h"

#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_driver.h"
#endif

namespace quadrants {

bool is_cuda_api_available() {
#if defined(QD_WITH_CUDA)
  try {
    auto &instance = lang::CUDADriver::get_instance_without_context();
    // Check driver availability without querying the device count here; device enumeration can break CUDA detection
    // in manylinux wheels.
    return instance.detected();
  } catch (const std::exception &e) {
    std::cerr << "Error occurred while checking CUDA availability: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown error occurred while checking CUDA availability." << std::endl;
    return false;
  }
#else
  return false;
#endif
}

}  // namespace quadrants
