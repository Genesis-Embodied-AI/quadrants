// StreamManager implementation and Program delegation.

#include "program_stream.h"

#ifdef QD_WITH_CUDA
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#endif

#ifdef QD_WITH_AMDGPU
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#endif

namespace quadrants::lang {

// ---------------------------------------------------------------------------
// StreamManager
// ---------------------------------------------------------------------------

uint64 StreamManager::create_stream() {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    void *stream = nullptr;
    CUDADriver::get_instance().stream_create(&stream, 0x1 /*CU_STREAM_NON_BLOCKING*/);
    return reinterpret_cast<uint64>(stream);
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu) {
    AMDGPUContext::get_instance().make_current();
    void *stream = nullptr;
    AMDGPUDriver::get_instance().stream_create(&stream, 0x1 /*HIP_STREAM_NON_BLOCKING*/);
    return reinterpret_cast<uint64>(stream);
  }
#endif
  return 0;
}

void StreamManager::destroy_stream(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && stream_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_destroy(reinterpret_cast<void *>(stream_handle));
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu && stream_handle != 0) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUDriver::get_instance().stream_destroy(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void StreamManager::synchronize_stream(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_synchronize(reinterpret_cast<void *>(stream_handle));
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUDriver::get_instance().stream_synchronize(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void StreamManager::set_current_stream(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    CUDAContext::get_instance().set_stream(reinterpret_cast<void *>(stream_handle));
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUContext::get_instance().set_stream(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

uint64 StreamManager::create_event() {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    void *event = nullptr;
    CUDADriver::get_instance().event_create(&event, 0x02 /*CU_EVENT_DISABLE_TIMING*/);
    return reinterpret_cast<uint64>(event);
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu) {
    AMDGPUContext::get_instance().make_current();
    void *event = nullptr;
    AMDGPUDriver::get_instance().event_create(&event, 0x02 /*hipEventDisableTiming*/);
    return reinterpret_cast<uint64>(event);
  }
#endif
  return 0;
}

void StreamManager::destroy_event(uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_destroy(reinterpret_cast<void *>(event_handle));
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu && event_handle != 0) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUDriver::get_instance().event_destroy(reinterpret_cast<void *>(event_handle));
  }
#endif
}

void StreamManager::record_event(uint64 event_handle, uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_record(reinterpret_cast<void *>(event_handle),
                                            reinterpret_cast<void *>(stream_handle));
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu && event_handle != 0) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUDriver::get_instance().event_record(reinterpret_cast<void *>(event_handle),
                                              reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void StreamManager::synchronize_event(uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_synchronize(reinterpret_cast<void *>(event_handle));
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu && event_handle != 0) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUDriver::get_instance().event_synchronize(reinterpret_cast<void *>(event_handle));
  }
#endif
}

void StreamManager::stream_wait_event(uint64 stream_handle, uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_wait_event(reinterpret_cast<void *>(stream_handle),
                                                 reinterpret_cast<void *>(event_handle), 0 /*flags*/);
  }
#endif
#ifdef QD_WITH_AMDGPU
  if (arch_ == Arch::amdgpu && event_handle != 0) {
    AMDGPUContext::get_instance().make_current();
    AMDGPUDriver::get_instance().stream_wait_event(reinterpret_cast<void *>(stream_handle),
                                                   reinterpret_cast<void *>(event_handle), 0 /*flags*/);
  }
#endif
}

}  // namespace quadrants::lang
