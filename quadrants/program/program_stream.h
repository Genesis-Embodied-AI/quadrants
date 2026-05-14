// StreamManager — manages CUDA stream and event lifecycle, isolated from Program so that backend-specific GPU
// plumbing does not pollute the core Program interface.

#pragma once

#include "quadrants/common/core.h"
#include "quadrants/util/lang_util.h"

namespace quadrants::lang {

class StreamManager {
 public:
  explicit StreamManager(Arch arch) : arch_(arch) {
  }

  uint64 create_stream();
  void destroy_stream(uint64 stream_handle);
  void synchronize_stream(uint64 stream_handle);
  void set_current_stream(uint64 stream_handle);

  uint64 create_event();
  void destroy_event(uint64 event_handle);
  void record_event(uint64 event_handle, uint64 stream_handle);
  void synchronize_event(uint64 event_handle);
  void stream_wait_event(uint64 stream_handle, uint64 event_handle);

 private:
  Arch arch_;
};

}  // namespace quadrants::lang
