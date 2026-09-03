#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace quadrants::lang {

// Program-scoped record of the most recent FRONTEND split (last_stats) and per-task artifact-cache probe
// (last_task_stats) per kernel, written by the split / codegen drivers and read back by the compilation manager for
// Python's PerOffloadCacheObservations. Ints only (no IR, no SNode*), so unlike the old in-memory cache it needs no
// wipe on destroy_snode_tree.
struct PerConstructCache {
  struct Stats {
    int total = 0;
    int hit = 0;
    int recompiled = 0;
    // Frontend split only: flattened [a0,b0,a1,b1,...] of the arg-slot pairs whose cross-parameter ndarray
    // disjointness the split relied on to clear condition (3). Unsound if a caller binds the same ndarray to both
    // slots of a pair, so the launch guard falls back only when one of these pairs actually aliases.
    std::vector<int> assumed_disjoint_pairs;
  };

  std::mutex mu;
  std::unordered_map<std::string, Stats> last_stats;       // kernel name -> most-recent frontend split
  std::unordered_map<std::string, Stats> last_task_stats;  // kernel name -> most-recent per-task artifact probe
};

}  // namespace quadrants::lang
