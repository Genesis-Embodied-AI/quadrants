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
    // Parallel vector to `task_funcs`: cp_id of the `qd.checkpoint(...)` block the task lives in, or `-1` if the task
    // is outside every checkpoint. Populated at register time off `OffloadedTask::checkpoint_id` (which the CPU codegen
    // now propagates from `OffloadedStmt::checkpoint_id`, slice 6 change in codegen_cpu.cpp). Consumed by the
    // launcher's host-branch gating to skip checkpoint bodies whose cp_id sits below the current resume point or that
    // follow a yielding checkpoint within the same launch.
    std::vector<int32_t> checkpoint_id_per_task;
    // Parallel vector to `task_funcs`: `true` iff this is the last task in a contiguous run of same-cp_id tasks for a
    // checkpoint that declared `yield_on=`. Used by the host-branch gating to know exactly which task boundary to read
    // the user's yield_on flag at, instead of paying the flag-read after every task in the body.
    std::vector<bool> is_last_task_of_yielding_checkpoint;
    // Precomputed at register time off `checkpoint_id_per_task`: `true` iff at least one task in this kernel sits
    // inside a `qd.checkpoint(...)` block. Used to bypass the per-task `run_gated_task` wrapper for kernels that have
    // no checkpoints at all -- those skip straight to a tight `run_one_offloaded_task` loop with no per-task gating
    // overhead, matching the pre-`qd.checkpoint` hot path byte-for-byte.
    bool has_any_checkpoint{false};
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

  // Slice 6: cp_id of the checkpoint that wrote a non-zero `yield_on=` flag on the most recent launch (or `-1` if none
  // yielded / the kernel had no `yield_on=` checkpoints / the kernel ran without graph). Read by
  // `Program::get_graph_last_yield_cp_id_on_last_call` which feeds the host-facing `qd.GraphStatus`. Mirrors
  // `cuda::GraphManager::last_yield_cp_id_on_last_call`.
  int get_graph_last_yield_cp_id_on_last_call() const override {
    return last_yield_cp_id_on_last_call_;
  }

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx, const Context &launcher_ctx);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx, const Context &launcher_ctx);
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
  // Run a single offloaded task with host-branch checkpoint gating + yield observation, sharing the `resume_point` /
  // `yield_signal` host scalars across a whole graph_do_while drive. Returns false only when a device-side assert fired
  // (the caller should stop).
  bool run_gated_task(LaunchContextBuilder &ctx,
                      std::size_t i,
                      const Context &launcher_ctx,
                      int32_t &resume_point,
                      int32_t &yield_signal);

  // `std::deque` so references to existing entries survive an `emplace_back` from a nested launch.
  std::deque<Context> contexts_;
  // Reset to `-1` at the start of every launch; the host-branch gating loop sets it to the first cp_id whose
  // `yield_on=` flag was non-zero. Read once per launch by the proxy above.
  int last_yield_cp_id_on_last_call_{-1};
};

}  // namespace cpu
}  // namespace quadrants::lang
