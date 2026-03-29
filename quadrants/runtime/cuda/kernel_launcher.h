#pragma once

#include <string>
#include <vector>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/cuda/gpu_graph_manager.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace cuda {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    std::vector<OffloadedTask> offloaded_tasks;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;
  std::size_t get_gpu_graph_cache_size() const override {
    return graph_manager_.cache_size();
  }
  bool get_gpu_graph_cache_used_on_last_call() const override {
    return graph_manager_.used_on_last_call();
  }
  std::size_t get_gpu_graph_num_nodes_on_last_call() const override {
    return graph_manager_.num_nodes_on_last_call();
  }

 private:
  void launch_offloaded_tasks(
      LaunchContextBuilder &ctx,
      JITModule *cuda_module,
      const std::vector<OffloadedTask> &offloaded_tasks);
  void launch_offloaded_tasks_with_do_while(
      LaunchContextBuilder &ctx,
      JITModule *cuda_module,
      const std::vector<OffloadedTask> &offloaded_tasks);

  std::vector<Context> contexts_;
  GpuGraphManager graph_manager_;
};

}  // namespace cuda
}  // namespace quadrants::lang
