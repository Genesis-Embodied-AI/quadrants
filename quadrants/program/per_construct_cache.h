#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace quadrants::lang {

// Program-scoped record of the most recent per-construct FRONTEND split and per-task artifact-cache probe, per kernel.
// The split driver (`split_frontend_per_construct`) records the construct {total, hit, recompiled} in `last_stats`; the
// LLVM codegen driver records the per-task equivalent in `last_task_stats`. The compilation manager
// (`KernelCompilationManager::load_or_compile`) reads both back on a fresh compile so Python
// (`PerOffloadCacheObservations`) can assert reuse. Reading in the compilation manager -- rather than a backend codegen
// driver -- keeps the construct half backend-agnostic (LLVM and SPIR-V alike) and lets a cache hit report the no-split
// sentinel uniformly. `last_task_stats` is only ever filled on CUDA (the sole artifact-cache backend) and stays empty
// (reported as -1) elsewhere. Holds only ints -- no IR and no `SNode *` -- so, unlike the dropped in-memory frontend
// cache, it needs no wipe on `destroy_snode_tree`.
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
