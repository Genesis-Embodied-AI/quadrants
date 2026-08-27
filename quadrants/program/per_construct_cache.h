#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace quadrants::lang {

// Program-scoped record of the most recent FRONTEND split (last_stats) and per-task artifact-cache probe
// (last_task_stats), per kernel. Written by the split and LLVM codegen drivers; read back by
// KernelCompilationManager::load_or_compile on a fresh compile for Python's PerOffloadCacheObservations.
// last_task_stats is CUDA-only (reported as -1 elsewhere). Holds only ints (no IR, no SNode*), so unlike the dropped
// in-memory frontend cache it needs no wipe on destroy_snode_tree.
struct PerConstructCache {
  struct Stats {
    int total = 0;
    int hit = 0;
    int recompiled = 0;
  };

  std::mutex mu;
  std::unordered_map<std::string, Stats> last_stats;       // kernel name -> most-recent frontend split
  std::unordered_map<std::string, Stats> last_task_stats;  // kernel name -> most-recent per-task artifact probe
};

}  // namespace quadrants::lang
