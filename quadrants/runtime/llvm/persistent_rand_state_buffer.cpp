#include "quadrants/runtime/llvm/persistent_rand_state_buffer.h"

#include "quadrants/util/lang_util.h"

#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_driver.h"
#endif

#if defined(QD_WITH_AMDGPU)
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#endif

namespace quadrants::lang {

PersistentRandStateBuffer &PersistentRandStateBuffer::get_instance() {
  // Heap-allocated and intentionally leaked: the singleton owns a raw HIP/CUDA driver allocation, and freeing it at
  // static-destructor time would race the driver context's own teardown. Same pattern used by the AMDGPU
  // persistent-runtime-JIT cache.
  static auto *instance = new PersistentRandStateBuffer();
  return *instance;
}

void *PersistentRandStateBuffer::get_or_grow(Arch arch, std::size_t bytes_required) {
  std::lock_guard<std::mutex> guard(mu_);

  if (buffer_ != nullptr && arch_ != arch) {
    QD_ERROR("PersistentRandStateBuffer was initialised for arch={} but is now requested for arch={}", arch_name(arch_),
             arch_name(arch));
  }

  if (buffer_ != nullptr && size_ >= bytes_required) {
    return buffer_;
  }

  // Free the prior (smaller) allocation before growing. This is the only free path; on process exit the singleton is
  // leaked deliberately.
  if (buffer_ != nullptr) {
    if (arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().mem_free(buffer_);
#endif
    } else if (arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().mem_free(buffer_);
#endif
    }
    buffer_ = nullptr;
    size_ = 0;
  }

  void *ptr = nullptr;
  if (arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().malloc(&ptr, bytes_required);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().malloc(&ptr, bytes_required);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    QD_ERROR("PersistentRandStateBuffer is only valid for CUDA / AMDGPU");
  }

  if (ptr == nullptr) {
    QD_ERROR("PersistentRandStateBuffer allocation of {} B failed", bytes_required);
  }

  buffer_ = ptr;
  size_ = bytes_required;
  arch_ = arch;
  return buffer_;
}

}  // namespace quadrants::lang
