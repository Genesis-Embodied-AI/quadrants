#pragma once

#include <cstddef>
#include <utility>

#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

namespace quadrants::lang {
namespace amdgpu {

// RAII owner of a device-side scratch buffer allocated via hipMallocAsync.
//
// Motivation
// ----------
// hipMallocAsync / hipFreeAsync on ROCm are NOT as cheap as the
// stream-ordered pool API is meant to be. Even on a fully-cached pool hit
// (i.e. when no actual device-side allocation work is required), each call
// still pays:
//
//   * a mutex acquire on the pool data structure,
//   * stream-ordering metadata bookkeeping in the CLR (rocclr) layer,
//   * a hop into the runtime's worker thread for HSA-level scheduling.
//
// We cannot eliminate that cost from inside the driver. We have already
// raised HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD (see amdgpu_context.cpp) so
// that the pool keeps freed memory cached and never round-trips to the OS,
// but that knob only addresses the "actual allocation" branch, not the
// per-call overhead above. Empirically, eliminating the per-launch
// hipMallocAsync / hipFreeAsync calls in the kernel launcher hot path
// yields a uniform throughput improvement; nothing at the driver/pool
// layer can substitute for it.
//
// On NVIDIA the equivalent cuMemAllocAsync is fast enough that the CUDA
// backend doesn't bother caching. This is a ROCm-specific workaround at
// the application layer for a ROCm-specific cost.
//
// This class encapsulates the workaround: each kernel handle owns one of
// these per scratch buffer (arg buffer, result buffer), allocates it
// lazily on first use, reuses it across launches, grows it on demand, and
// frees it deterministically when the owning launcher is destroyed.
//
// Semantics
// ---------
// - Allocates lazily on the first ensure() that requires a non-zero size.
// - Grows monotonically: ensure(N) only re-allocates if N > current capacity;
//   the buffer is never shrunk.
// - Frees its allocation in the destructor (and on grow, when discarding the
//   smaller backing buffer).
// - Move-only. Copying would alias the same device pointer with two owners
//   and double-free in the destructor.
//
// Stream affinity
// ---------------
// Each DeviceScratchBuffer is bound to one HIP stream for the lifetime
// of its allocation: the constructor accepts an optional `stream`
// argument, which is used for every malloc_async / mem_free_async issued
// by ensure() and the destructor. Defaults to nullptr, i.e. the default
// (NULL) stream, which is what the launcher uses today.
//
// Single-stream affinity is intentional. Stream-ordered allocators are
// safe to free on a different stream than they were allocated on, but
// only if the caller threads the appropriate event dependencies; tying
// the buffer to one stream makes the grow-cycle (mem_free_async old,
// malloc_async new) trivially ordered and lets the destructor free
// without any extra synchronisation. Callers that want a buffer on a
// different stream should construct a separate DeviceScratchBuffer with
// that stream rather than mutating an existing one.
//
// Reuse across back-to-back launches on the same stream is safe because
// the stream serialises H2D, kernel, D2H, and any subsequent grow-cycle
// free in the order they are enqueued. Callers performing async H2D
// into the buffer followed by a kernel that reads it (the typical
// kernel-launcher path) get correct ordering for free.
//
// Thread-safety
// -------------
// Not thread-safe. Concurrent ensure() calls on the same instance race on
// ptr_ / capacity_. Callers must serialise access. (Genesis launches kernels
// from a single host thread, so this is not a practical limitation.)
//
// Teardown safety
// ---------------
// In the normal flow the launcher (and therefore every DeviceScratchBuffer
// it owns) is destroyed during gs.destroy() while the AMDGPU context is
// still alive, so mem_free_async returns the memory to the pool cleanly.
// In the at-exit path, where the AMDGPU context may have already been torn
// down, the underlying driver call is a best-effort no-op; the OS reclaims
// device memory at process exit regardless.
class DeviceScratchBuffer {
 public:
  // Constructs an empty buffer (no allocation) bound to `stream`. The
  // stream is opaque from this class's point of view (typed void* to
  // match the AMDGPUDriver API surface, which mirrors hipStream_t).
  // Pass nullptr (the default) for the default HIP stream, which is what
  // the kernel launcher uses today; callers that introduce a non-default
  // stream in the future can plumb it through here without touching this
  // class.
  explicit DeviceScratchBuffer(void *stream = nullptr) noexcept
      : stream_(stream) {}
  ~DeviceScratchBuffer() noexcept { release(); }

  DeviceScratchBuffer(const DeviceScratchBuffer &) = delete;
  DeviceScratchBuffer &operator=(const DeviceScratchBuffer &) = delete;

  // Move preserves stream affinity: the destination operates on the same
  // stream the source was bound to. The source is left in a valid empty
  // state on its (unchanged) stream.
  DeviceScratchBuffer(DeviceScratchBuffer &&other) noexcept
      : stream_(other.stream_),
        ptr_(other.ptr_),
        capacity_(other.capacity_) {
    other.ptr_ = nullptr;
    other.capacity_ = 0;
  }

  DeviceScratchBuffer &operator=(DeviceScratchBuffer &&other) noexcept {
    if (this != &other) {
      release();
      stream_ = other.stream_;
      ptr_ = other.ptr_;
      capacity_ = other.capacity_;
      other.ptr_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  // Ensures the backing allocation is at least `min_bytes`, growing it
  // on this buffer's bound stream if necessary. Returns the device
  // pointer.
  //
  // The returned pointer is invalidated by any subsequent ensure() call
  // that grows the buffer; callers must not retain it across launches.
  char *ensure(std::size_t min_bytes) {
    if (capacity_ < min_bytes) {
      if (ptr_ != nullptr) {
        AMDGPUDriver::get_instance().mem_free_async(ptr_, stream_);
        ptr_ = nullptr;
      }
      AMDGPUDriver::get_instance().malloc_async(
          reinterpret_cast<void **>(&ptr_), min_bytes, stream_);
      capacity_ = min_bytes;
    }
    return ptr_;
  }

  char *get() const noexcept { return ptr_; }
  std::size_t capacity() const noexcept { return capacity_; }
  void *stream() const noexcept { return stream_; }

  // Rebinds the buffer to a new stream. Only legal on an empty buffer
  // (capacity == 0); rebinding while holding an allocation would risk
  // freeing on a stream that doesn't have a valid happens-before edge
  // to the buffer's last use. Callers that want to migrate a populated
  // buffer to a different stream should release() first, set_stream(),
  // then ensure() again.
  void set_stream(void *stream) noexcept {
    QD_ASSERT(capacity_ == 0);
    stream_ = stream;
  }

  // Eagerly releases the underlying allocation on this buffer's bound
  // stream. Idempotent. Invoked automatically by the destructor;
  // explicit invocation is rarely needed.
  void release() noexcept {
    if (ptr_ != nullptr) {
      AMDGPUDriver::get_instance().mem_free_async(ptr_, stream_);
      ptr_ = nullptr;
      capacity_ = 0;
    }
  }

 private:
  void *stream_{nullptr};
  char *ptr_{nullptr};
  std::size_t capacity_{0};
};

}  // namespace amdgpu
}  // namespace quadrants::lang

