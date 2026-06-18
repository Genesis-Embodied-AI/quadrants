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
    // Per-task snode-write set / arg-write set / arg-read set, copied off `OffloadedTask::snode_writes` /
    // `arr_writes` / `arr_reads` at register time. Used by `launch_llvm_kernel` to bump
    // `Program::snode_write_gen_` / `ndarray_data_gen_` before each launch so the per-task adstack metadata cache
    // invalidates when this kernel mutates a SNode / ndarray a downstream `size_expr` reads, plus the read-only
    // `kNone` host-array case where the data pointer stays stable across launches but the user's content can change.
    std::vector<std::vector<int>> snode_writes_per_task;
    std::vector<std::vector<int>> arr_writes_per_task;
    std::vector<std::vector<int>> arr_reads_per_task;
    // Per-task innermost graph_do_while level id (-1 if outside all loops), copied off
    // `OffloadedTask::graph_do_while_level_id`. Used by the nested graph_do_while host driver.
    std::vector<int> graph_do_while_level_per_task;
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    // arg_ids of the array-typed entries in `parameters`, precomputed at register time.
    std::vector<int> array_arg_ids;
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
                                            const std::vector<std::size_t> &num_threads_per_task,
                                            const std::vector<int> &graph_do_while_level_per_task);
  // Once-per-launch adstack setup (lazy-claim buffers + max-reducer dispatch), shared by the flat and nested
  // graph_do_while paths.
  void prepare_offloaded_tasks(LaunchContextBuilder &ctx,
                               const std::vector<TaskFunc> &task_funcs,
                               const std::vector<AdStackSizingInfo> &ad_stacks);
  // Run a single offloaded task (per-task adstack publish + invoke). Returns false if a device-side assert fired
  // (kernel should stop). `launch_scope` must span the whole launch.
  bool run_one_offloaded_task(LaunchContextBuilder &ctx,
                              std::size_t i,
                              const std::vector<TaskFunc> &task_funcs,
                              const std::vector<AdStackSizingInfo> &ad_stacks,
                              const std::vector<std::size_t> &num_threads_per_task);

  // `std::deque` so references to existing entries survive an `emplace_back` from a nested launch.
  std::deque<Context> contexts_;
};

}  // namespace cpu
}  // namespace quadrants::lang
