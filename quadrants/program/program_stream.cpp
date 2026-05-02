// StreamManager implementation and Program delegation.

#include "program_stream.h"
#include "program.h"

#ifdef QD_WITH_CUDA
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/cuda/cuda_context.h"
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
  return 0;
}

void StreamManager::destroy_stream(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && stream_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_destroy(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void StreamManager::synchronize_stream(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && stream_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_synchronize(reinterpret_cast<void *>(stream_handle));
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
  return 0;
}

void StreamManager::destroy_event(uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_destroy(reinterpret_cast<void *>(event_handle));
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
}

void StreamManager::synchronize_event(uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (arch_ == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_synchronize(reinterpret_cast<void *>(event_handle));
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
}

// ---------------------------------------------------------------------------
// Program delegation — keeps the pybind / Python API unchanged.
// ---------------------------------------------------------------------------

uint64 Program::stream_create() {
  return stream_manager_.create_stream();
}
void Program::stream_destroy(uint64 h) {
  stream_manager_.destroy_stream(h);
}
void Program::stream_synchronize(uint64 h) {
  stream_manager_.synchronize_stream(h);
}
void Program::set_current_cuda_stream(uint64 h) {
  stream_manager_.set_current_stream(h);
}
uint64 Program::event_create() {
  return stream_manager_.create_event();
}
void Program::event_destroy(uint64 h) {
  stream_manager_.destroy_event(h);
}
void Program::event_record(uint64 eh, uint64 sh) {
  stream_manager_.record_event(eh, sh);
}
void Program::event_synchronize(uint64 h) {
  stream_manager_.synchronize_event(h);
}
void Program::stream_wait_event(uint64 sh, uint64 eh) {
  stream_manager_.stream_wait_event(sh, eh);
}

}  // namespace quadrants::lang
