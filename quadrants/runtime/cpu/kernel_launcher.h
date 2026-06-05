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
    // Per-task nested-child markers (parallel to `task_funcs`). For a launch_child task, `task_funcs[i]` is null and
    // `child_call_index_per_task[i]` selects the embedded child to launch sequentially at that position (CPU has no
    // graph; nested qd.kernel calls fall back to an in-order recursive launch). Non-child tasks store -1.
    std::vector<char> is_launch_child_per_task;
    std::vector<int> child_call_index_per_task;
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    // arg_ids of the array-typed entries in `parameters`, precomputed at register time.
    std::vector<int> array_arg_ids;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx, const Context &launcher_ctx);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx, const Context &launcher_ctx);
  // Sequentially launch the embedded child selected by `child_call_index` (looked up in `ctx.child_launches`). Used
  // for the CPU nested qd.kernel fallback in place of a subgraph.
  void launch_child(LaunchContextBuilder &ctx, int child_call_index);

  // `std::deque` so references to existing entries survive an `emplace_back` from a nested launch.
  std::deque<Context> contexts_;
};

}  // namespace cpu
}  // namespace quadrants::lang
