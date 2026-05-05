#pragma once

#include <deque>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace cpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  using TaskFunc = int32 (*)(void *);

  struct Context {
    std::vector<TaskFunc> task_funcs;
    // Parallel vectors to `task_funcs`: `ad_stacks[i]` points into the owning `OffloadedTask::ad_stack` (stable
    // for the kernel's lifetime) and `num_threads_per_task[i]` is the thread count used to size the heap. CPU
    // sizing is always `static_num_threads` (set by codegen to `num_cpu_threads` for non-serial tasks, 1 for
    // serial), so no launch-time gtmp resolution is needed on this backend.
    std::vector<AdStackSizingInfo> ad_stacks;
    std::vector<std::size_t> num_threads_per_task;
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx,
                              const std::vector<TaskFunc> &task_funcs,
                              const std::vector<AdStackSizingInfo> &ad_stacks,
                              const std::vector<std::size_t> &num_threads_per_task);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                            const std::vector<TaskFunc> &task_funcs,
                                            const std::vector<AdStackSizingInfo> &ad_stacks,
                                            const std::vector<std::size_t> &num_threads_per_task);

  // `std::deque` so references to existing entries survive an `emplace_back` from a nested launch.
  std::deque<Context> contexts_;
};

}  // namespace cpu
}  // namespace quadrants::lang
