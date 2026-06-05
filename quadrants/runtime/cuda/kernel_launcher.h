#pragma once

#include <deque>
#include <string>
#include <vector>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/cuda/graph_manager.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace cuda {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    std::vector<OffloadedTask> offloaded_tasks;
    // Per-kernel-handle persistent scratch. See the matching AMDGPU `Context` struct in
    // `runtime/amdgpu/kernel_launcher.h` for the rationale: `publish_adstack_metadata`'s host-eval branch
    // recursively launches `snode_reader_*` kernels via this same launcher, and a launcher-global persistent
    // buffer would let those recursive launches overwrite the parent's `arg_buffer` / `RuntimeContext` before
    // the parent kernel reads them. Per-handle isolation prevents the cross-clobber.
    void *arg_buffer_dev_ptr{nullptr};
    std::size_t arg_buffer_capacity{0};
    void *runtime_context_dev_ptr{nullptr};
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;
  std::size_t get_graph_cache_size() const override {
    return graph_manager_.cache_size();
  }
  bool get_graph_cache_used_on_last_call() const override {
    return graph_manager_.used_on_last_call();
  }
  std::size_t get_graph_num_nodes_on_last_call() const override {
    return graph_manager_.num_nodes_on_last_call();
  }
  std::size_t get_graph_num_checkpoints_on_last_call() const override {
    return graph_manager_.num_checkpoints_on_last_call();
  }
  int get_graph_last_yield_cp_id_on_last_call() const override {
    return graph_manager_.last_yield_cp_id_on_last_call();
  }
  std::size_t get_graph_total_builds() const override {
    return graph_manager_.total_builds();
  }

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx,
                              JITModule *cuda_module,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              void *device_context_ptr);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                            JITModule *cuda_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            void *device_context_ptr);

  // std::deque (not std::vector): `publish_adstack_metadata`'s host-eval branch recursively registers snode-reader
  // kernels via this same launcher, calling `contexts_.resize()` while a parent `launch_llvm_kernel` frame still
  // holds a reference into the container.  std::deque never invalidates references on push_back / resize, so the
  // parent's `launcher_ctx` reference survives the child's registration.
  std::deque<Context> contexts_;
  GraphManager graph_manager_;
  // `result_buffer` stays launcher-global: kernels write to it, the host reads it back synchronously before any
  // other kernel runs as a reader, so recursive snode-reader launches that reuse the buffer cannot smuggle stale
  // bytes into the parent's read path. Grown amortised-doubling. See `runtime/amdgpu/kernel_launcher.h` for the
  // matching scheme and the recursive-launch rationale this whole layout is designed around.
  void *persistent_result_buffer_dev_ptr_{nullptr};
  std::size_t persistent_result_buffer_capacity_{0};

 public:
  ~KernelLauncher() override;
};

}  // namespace cuda
}  // namespace quadrants::lang
