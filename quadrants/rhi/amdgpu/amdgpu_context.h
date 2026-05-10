#pragma once

#include <mutex>
#include <unordered_map>
#include <thread>
#include <vector>

#include "quadrants/program/kernel_profiler.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

namespace quadrants {
namespace lang {

class AMDGPUDriver;

class AMDGPUContext {
 private:
  void *device_{nullptr};
  void *context_{nullptr};
  int dev_count_{0};
  int compute_capability_{0};
  std::string mcpu_;
  std::mutex lock_;
  KernelProfilerBase *profiler_{nullptr};
  AMDGPUDriver &driver_;
  bool debug_{false};
  bool supports_mem_pool_{false};
  static thread_local void *stream_;
  std::vector<void *> stream_pool_;

 public:
  AMDGPUContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();
  std::string get_device_name();

  bool detected() const {
    return dev_count_ != 0;
  }

  void pack_args(std::vector<void *> arg_pointers, std::vector<int> arg_sizes, char *arg_packed);

  int get_args_byte(std::vector<int> arg_sizes);

  void set_profiler(KernelProfilerBase *profiler) {
    profiler_ = profiler;
  }

  void launch(void *func,
              const std::string &task_name,
              const std::vector<void *> &arg_pointers,
              const std::vector<int> &arg_sizes,
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t dynamic_shared_mem_bytes);

  void set_debug(bool debug) {
    debug_ = debug;
  }

  std::string get_mcpu() const {
    return mcpu_;
  }

  void *get_context() {
    return context_;
  }

  void make_current() {
    driver_.context_set_current(context_);
  }

  int get_compute_capability() const {
    return compute_capability_;
  }

  bool supports_mem_pool() const {
    return supports_mem_pool_;
  }

  // Force the default device memory pool to release every cached page back to the driver. Called by
  // `LlvmRuntimeExecutor::finalize` (i.e. `qd.reset()`) to align actual driver-visible free VRAM with the user-facing
  // contract that `qd.reset()` releases everything Quadrants allocated. Without this, up to the configured release
  // threshold (128 MiB at construction) of freed pages stays cached in the pool and shows up as "used" to other
  // processes on the same VF, materially raising the chance of multi-process `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
  // failures across pytest-xdist workers. No-op if the device does not advertise mempool support.
  void trim_default_mem_pool();

  ~AMDGPUContext();

  class ContextGuard {
   private:
    // Both fields store HIP driver context handles (opaque `hipCtx_t` aliased as `void *`), NOT
    // the enclosing `AMDGPUContext *` wrapper pointer. Storing the wrapper address made the
    // `old_ctx_ != new_ctx_` compare at construction and destruction always evaluate true (the
    // wrapper address and the HIP handle live in disjoint value spaces), so the guard called
    // `make_current()` unconditionally on entry and `context_set_current(old_ctx_)` unconditionally
    // on exit even when the target context was already active. Always-equal semantics are
    // preserved by storing `new_ctx->get_context()` here.
    void *old_ctx_;
    void *new_ctx_;

   public:
    explicit ContextGuard(AMDGPUContext *new_ctx) : old_ctx_(nullptr), new_ctx_(new_ctx->get_context()) {
      AMDGPUDriver::get_instance().context_get_current(&old_ctx_);
      if (old_ctx_ != new_ctx_) {
        new_ctx->make_current();
      }
    }

    ~ContextGuard() {
      if (old_ctx_ != new_ctx_) {
        AMDGPUDriver::get_instance().context_set_current(old_ctx_);
      }
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::unique_lock<std::mutex> get_lock_guard() {
    return std::unique_lock<std::mutex>(lock_);
  }

  void set_stream(void *stream) {
    stream_ = stream;
  }

  void *get_stream() const {
    return stream_;
  }

  void *acquire_stream() {
    std::lock_guard<std::mutex> _(lock_);
    if (!stream_pool_.empty()) {
      auto s = stream_pool_.back();
      stream_pool_.pop_back();
      return s;
    }
    void *s = nullptr;
    AMDGPUDriver::get_instance().stream_create(&s, 0x1 /*HIP_STREAM_NON_BLOCKING*/);
    return s;
  }

  void release_stream(void *s) {
    std::lock_guard<std::mutex> _(lock_);
    stream_pool_.push_back(s);
  }

  static AMDGPUContext &get_instance();
};

}  // namespace lang
}  // namespace quadrants
