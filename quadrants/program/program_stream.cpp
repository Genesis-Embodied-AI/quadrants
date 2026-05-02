// Stream and event operations for the Program class.
// Extracted from program.cpp to keep backend-specific GPU stream/event
// lifecycle code separate from the core Program logic.

#include "program.h"

#ifdef QD_WITH_CUDA
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#endif

namespace quadrants::lang {

uint64 Program::stream_create() {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    void *stream = nullptr;
    CUDADriver::get_instance().stream_create(&stream, 0x1 /*CU_STREAM_NON_BLOCKING*/);
    return reinterpret_cast<uint64>(stream);
  }
#endif
  return 0;
}

void Program::stream_destroy(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda && stream_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_destroy(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void Program::stream_synchronize(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda && stream_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_synchronize(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void Program::set_current_cuda_stream(uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    CUDAContext::get_instance().set_stream(reinterpret_cast<void *>(stream_handle));
  }
#endif
}

uint64 Program::event_create() {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    CUDAContext::get_instance().make_current();
    void *event = nullptr;
    CUDADriver::get_instance().event_create(&event, 0x02 /*CU_EVENT_DISABLE_TIMING*/);
    return reinterpret_cast<uint64>(event);
  }
#endif
  return 0;
}

void Program::event_destroy(uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_destroy(reinterpret_cast<void *>(event_handle));
  }
#endif
}

void Program::event_record(uint64 event_handle, uint64 stream_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_record(reinterpret_cast<void *>(event_handle),
                                            reinterpret_cast<void *>(stream_handle));
  }
#endif
}

void Program::event_synchronize(uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().event_synchronize(reinterpret_cast<void *>(event_handle));
  }
#endif
}

void Program::stream_wait_event(uint64 stream_handle, uint64 event_handle) {
#ifdef QD_WITH_CUDA
  if (compile_config().arch == Arch::cuda && event_handle != 0) {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_wait_event(reinterpret_cast<void *>(stream_handle),
                                                 reinterpret_cast<void *>(event_handle), 0 /*flags*/);
  }
#endif
}

}  // namespace quadrants::lang
