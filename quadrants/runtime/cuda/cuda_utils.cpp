#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"

namespace quadrants::lang {
namespace cuda {

bool on_cuda_device(void *ptr) {
  unsigned int attr_val = 0;
  uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
      &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (void *)ptr);
  return ret_code == CUDA_SUCCESS && attr_val == CU_MEMORYTYPE_DEVICE;
}

}  // namespace cuda
}  // namespace quadrants::lang
