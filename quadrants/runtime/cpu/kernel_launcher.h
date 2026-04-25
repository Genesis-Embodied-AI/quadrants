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
    // Parallel vector to `task_funcs`: `ad_stack_needed_bytes[i]` is the exact adstack heap size required
    // before dispatching task i on this kernel. Precomputed at `register_llvm_kernel` time from the
    // `OffloadedTask::ad_stack` sizing info. 0 means the task has no adstack and the launcher skips the
    // ensure call. CPU sizing is always `static_num_threads` (set by codegen to `num_cpu_threads` for
    // non-serial tasks, 1 for serial) so no launch-time gtmps resolution is needed on this backend.
    std::vector<std::size_t> ad_stack_needed_bytes;
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx,
                              const std::vector<TaskFunc> &task_funcs,
                              const std::vector<std::size_t> &ad_stack_needed_bytes);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                            const std::vector<TaskFunc> &task_funcs,
                                            const std::vector<std::size_t> &ad_stack_needed_bytes);

  std::vector<Context> contexts_;
};

}  // namespace cpu
}  // namespace quadrants::lang
