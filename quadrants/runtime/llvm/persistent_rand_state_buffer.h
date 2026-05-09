// Process-lifetime device-memory buffer for the LLVM runtime's per-thread random-state array.
//
// Background: prior to this, every `qd.init()` -> `materialize_runtime` preallocated `sizeof(RandState) *
// saturating_grid_dim * max_block_dim` bytes of device memory (~480 MiB at default config) as part of the
// runtime-objects preallocation, and freed it on `qd.reset()`. Under multi-process pytest-xdist contention this
// realloc churn was a major contributor to `HSA_STATUS_ERROR_OUT_OF_RESOURCES` on AMDGPU. Bisection on the C++ omnibus
// reproducer (see `experiments/launch_oor_repro/`) showed that removing the per-cycle ~400 MiB realloc was sufficient
// to push the failure threshold past the workload that real `pytest -n 8` exercises. RandState contents depend only
// on `(num_states, starting_seed)`, both of which are typically stable across `qd.init`/`qd.reset` cycles in a worker,
// so reusing the buffer changes no observable behavior.
//
// This module owns one device allocation per process. It grows monotonically (never shrinks, never frees on its own)
// and is intentionally leaked on process exit, the same shape used by the AMDGPU persistent-runtime-JIT cache: freeing
// driver memory at static-destructor time would race the GPU driver context's own teardown.

#pragma once

#include <cstddef>
#include <mutex>

#include "quadrants/rhi/arch.h"

namespace quadrants::lang {

class PersistentRandStateBuffer {
 public:
  static PersistentRandStateBuffer &get_instance();

  // Returns a device pointer with at least `bytes_required` bytes available. Grows monotonically: subsequent calls
  // with a smaller `bytes_required` return the same (larger) buffer. Allocation goes directly through the CUDA /
  // AMDGPU driver (`cuMemAlloc` / `hipMalloc`), bypassing `LlvmDevice` and `DeviceMemoryPool` so the lifetime is
  // independent of any `LlvmRuntimeExecutor`.
  //
  // Thread-safe.
  void *get_or_grow(Arch arch, std::size_t bytes_required);

 private:
  PersistentRandStateBuffer() = default;
  PersistentRandStateBuffer(const PersistentRandStateBuffer &) = delete;
  PersistentRandStateBuffer &operator=(const PersistentRandStateBuffer &) = delete;

  std::mutex mu_;
  void *buffer_{nullptr};
  std::size_t size_{0};
  Arch arch_{Arch::x64};
};

}  // namespace quadrants::lang
