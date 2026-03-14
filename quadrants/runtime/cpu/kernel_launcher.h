#pragma once

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace cpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  using TaskFunc = int32 (*)(void *);

  struct Context {
    std::vector<TaskFunc> task_funcs;
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(
      LaunchContextBuilder &ctx,
      const std::vector<TaskFunc> &task_funcs);
  void launch_offloaded_tasks_with_do_while(
      LaunchContextBuilder &ctx,
      const std::vector<TaskFunc> &task_funcs);

  std::vector<Context> contexts_;
};

}  // namespace cpu
}  // namespace quadrants::lang
