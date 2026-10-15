#pragma once

#include <memory>
#include <optional>
#include <unordered_set>

#include "llvm/IR/Module.h"
#include "quadrants/common/serialization.h"

namespace quadrants::lang {

// Stack storage is the measured capacity times these bytes per worker.
struct AdStackSizingInfo {
  std::size_t per_thread_stride{0};
  std::size_t static_num_threads{0};
  QD_IO_DEF(per_thread_stride, static_num_threads);
};

class OffloadedTask {
 public:
  std::string name;
  int block_dim{0};
  int grid_dim{0};
  int dynamic_shared_array_bytes{0};
  int stream_parallel_group_id{0};
  // Per-kernel `qd.graph_parallel_context()` region id (0 outside any region). Populated by the CUDA LLVM codegen from
  // `OffloadedStmt::graph_parallel_region_id`. The GraphManager pairs it with `stream_parallel_group_id` so two
  // back-to-back regions are built as separate fork/join groups (each with its own join) instead of one merged group.
  int graph_parallel_region_id{0};
  // `cp_id` of the enclosing `qd.checkpoint(...)` block for this task (`-1` outside any checkpoint). Populated by the
  // CUDA / AMDGPU LLVM codegen from `OffloadedStmt::checkpoint_id` (set by the offload pass from
  // `RangeForStmt::checkpoint_id` / `StructForStmt::checkpoint_id`). The GraphManager will consume this in slice 1c to
  // group consecutive same-cp_id tasks under a CUDA IF conditional node. Default `-1` keeps non-checkpoint kernels
  // (every existing user) on the unchanged code path. Serialised into the offline-cache via the `QD_IO_DEF` list below.
  int checkpoint_id{-1};
  // Innermost enclosing `graph_do_while` level id (-1 if none). Set from the OffloadedStmt at codegen time and consumed
  // at launch to reconstruct nested graph_do_while loops (CUDA native conditional nodes / host fallback). See
  // docs/source/user_guide/graph.md.
  int graph_do_while_level_id{-1};
  AdStackSizingInfo ad_stack{};

  explicit OffloadedTask(const std::string &name = "",
                         int block_dim = 0,
                         int grid_dim = 0,
                         int dynamic_shared_array_bytes = 0,
                         int stream_parallel_group_id = 0,
                         int checkpoint_id = -1)
      : name(name),
        block_dim(block_dim),
        grid_dim(grid_dim),
        dynamic_shared_array_bytes(dynamic_shared_array_bytes),
        stream_parallel_group_id(stream_parallel_group_id),
        checkpoint_id(checkpoint_id) {};
  QD_IO_DEF(name,
            block_dim,
            grid_dim,
            dynamic_shared_array_bytes,
            stream_parallel_group_id,
            graph_parallel_region_id,
            checkpoint_id,
            graph_do_while_level_id,
            ad_stack);
};

struct LLVMCompiledTask {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> struct_for_tls_sizes;
  LLVMCompiledTask() = default;
  LLVMCompiledTask(LLVMCompiledTask &&) = default;
  LLVMCompiledTask &operator=(LLVMCompiledTask &&) = default;
  LLVMCompiledTask(std::vector<OffloadedTask> tasks,
                   std::unique_ptr<llvm::Module> module,
                   std::unordered_set<int> used_tree_ids,
                   std::unordered_set<int> struct_for_tls_sizes)
      : tasks(std::move(tasks)),
        module(std::move(module)),
        used_tree_ids(std::move(used_tree_ids)),
        struct_for_tls_sizes(std::move(struct_for_tls_sizes)) {
  }
  LLVMCompiledTask clone() const;
  QD_IO_DEF(tasks);
};

struct LLVMCompiledKernel {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  LLVMCompiledKernel() = default;
  LLVMCompiledKernel(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel &operator=(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel(std::vector<OffloadedTask> tasks, std::unique_ptr<llvm::Module> module)
      : tasks(std::move(tasks)), module(std::move(module)) {
  }
  LLVMCompiledKernel clone() const;
  QD_IO_DEF(tasks);
};

// The exclusive end of the maximal run of stream-parallel tasks starting at `start` that belong to the SAME
// qd.graph_parallel_context() fork/join group as `tasks[start]`: same graph_do_while level, same checkpoint bucket,
// same graph_parallel_region_id, all with `stream_parallel_group_id != 0`. `limit` bounds the scan (a level's task
// range end, or `tasks.size()`).
//
// This is the SINGLE definition of a fork/join "run" boundary. It is shared by `GraphManager::build_level` (the CUDA
// graph path) and the CUDA/AMDGPU streaming launchers (`launch_offloaded_tasks`) so the region / level / checkpoint
// boundary cannot drift between those code paths -- the class of bug that let two back-to-back regions merge on the
// streaming path (their joins dropped) while the graph path kept them apart. graph_do_while and
// qd.graph_parallel_context emit no IR of their own, so these four tags are the only record of the grouping; a task
// belongs to a run only if all four match the run's first task. Precondition: `start < limit` and
// `tasks[start].stream_parallel_group_id != 0`.
inline std::size_t next_stream_parallel_run(const std::vector<OffloadedTask> &tasks,
                                            std::size_t start,
                                            std::size_t limit) {
  const int level = tasks[start].graph_do_while_level_id;
  const int region = tasks[start].graph_parallel_region_id;
  const int checkpoint = tasks[start].checkpoint_id;
  std::size_t run_end = start;
  while (run_end < limit && tasks[run_end].stream_parallel_group_id != 0 &&
         tasks[run_end].graph_do_while_level_id == level && tasks[run_end].graph_parallel_region_id == region &&
         tasks[run_end].checkpoint_id == checkpoint) {
    ++run_end;
  }
  return run_end;
}

}  // namespace quadrants::lang
