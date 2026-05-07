#pragma once

#include <mutex>
#include <unordered_map>
#include <thread>

#include "quadrants/program/kernel_profiler.h"
#include "quadrants/rhi/cuda/cuda_driver.h"

namespace quadrants::lang {

// Note:
// It would be ideal to create a CUDA context per Quadrants program, yet CUDA
// context creation takes time. Therefore we use a shared context to accelerate
// cases such as unit testing where many Quadrants programs are
// created/destroyed.

class CUDADriver;

class CUDAContext {
 private:
  void *device_;
  void *context_;
  int dev_count_;
  int compute_capability_;
  std::string mcpu_;
  std::mutex lock_;
  KernelProfilerBase *profiler_;
  CUDADriver &driver_;
  int max_shared_memory_bytes_;
  bool debug_;
  bool supports_mem_pool_;
  bool supports_pageable_memory_access_;
  bool uses_host_page_tables_;
  void *stream_;

 public:
  CUDAContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();
  std::string get_device_name();

  bool detected() const {
    return dev_count_ != 0;
  }

  void launch(void *func,
              const std::string &task_name,
              std::vector<void *> arg_pointers,
              std::vector<int> arg_sizes,
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t dynamic_shared_mem_bytes);

  void set_profiler(KernelProfilerBase *profiler) {
    profiler_ = profiler;
  }

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

  int get_max_shared_memory_bytes() const {
    return max_shared_memory_bytes_;
  }

  int64_t get_clock_rate_khz() const;

  bool supports_mem_pool() const {
    return supports_mem_pool_;
  }

  // True when the device can coherently dereference plain host pointers (`malloc` / `new`) from kernel code via HMM /
  // system-allocated memory. Maps `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` directly - 1 on Linux with an
  // HMM-capable driver + kernel (open-source nvidia module or 535+ with HMM enabled), 0 on Turing and older parts,
  // Windows, and any Linux host without HMM. Used by the adstack sizer launcher to decide whether to stage a device
  // copy of `RuntimeContext` before each launch.
  bool supports_pageable_memory_access() const {
    return supports_pageable_memory_access_;
  }

  // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES (= 100). 1 when the device walks the host's page
  // tables directly via ATS / HPT (Ampere and newer with HMM-equipped drivers, including Blackwell). 0 on pre-Ampere
  // HMM, where pageable-memory access goes through the legacy fault-and-migrate path. The legacy path is sensitive to
  // the timing of host writes vs. kernel reads under stream-ordered allocations and to multi-process pressure; the
  // adstack runtime helper kernels (`runtime_eval_adstack_size_expr`, `runtime_eval_adstack_max_reduce`,
  // `runtime_eval_static_bound_count`) hit those issues empirically on Turing T4 (sm_75) and are gated off there.
  bool uses_host_page_tables() const {
    return uses_host_page_tables_;
  }

  ~CUDAContext();

  class ContextGuard {
   private:
    void *old_ctx_;
    void *new_ctx_;

   public:
    explicit ContextGuard(CUDAContext *new_ctx) : old_ctx_(nullptr), new_ctx_(new_ctx->context_) {
      CUDADriver::get_instance().context_get_current(&old_ctx_);
      if (old_ctx_ != new_ctx_)
        new_ctx->make_current();
    }

    ~ContextGuard() {
      if (old_ctx_ != new_ctx_) {
        CUDADriver::get_instance().context_set_current(old_ctx_);
      }
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::unique_lock<std::mutex> get_lock_guard() {
    return std::unique_lock<std::mutex>(lock_);
  }

  static CUDAContext &get_instance();

  void set_stream(void *stream) {
    stream_ = stream;
  }

  void *get_stream() const {
    return stream_;
  }
};

}  // namespace quadrants::lang
