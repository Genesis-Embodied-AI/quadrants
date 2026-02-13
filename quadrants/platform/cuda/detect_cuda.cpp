#include "quadrants/platform/cuda/detect_cuda.h"

#if defined(TI_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_driver.h"
#endif

namespace quadrants {

bool is_cuda_api_available() {
#if defined(TI_WITH_CUDA)
  try {
    auto &instance = lang::CUDADriver::get_instance_without_context();
    // Note that one must be careful when detecting if CUDA is available.
    // See
    // https://github.com/Genesis-Embodied-AI/quadrants/pull/139#discussion_r2291124463.
    // More detail:
    // When trying to run C-API tests (now removed), trying to use cuda would
    // segfault when no cuda devices available. I tried counting the number of
    // devices at this point of the code, which fixed that segfault, but broke
    // cuda in wheels built in manylinux. Strangely, cuda worked on wheels built
    // on Ubuntu 24.04, but not on manylinux.
    return instance.detected();
  } catch (const std::exception &e) {
    std::cerr << "Error occurred while checking CUDA availability: " << e.what()
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown error occurred while checking CUDA availability."
              << std::endl;
    return false;
  }
#else
  return false;
#endif
}

}  // namespace quadrants
