#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace quadrants::lang {

// Program-scoped record of the most recent per-construct FRONTEND split, per kernel. The split driver
// (`split_frontend_per_construct`) records {total, hit, recompiled} here keyed by kernel name; the compilation manager
// (`KernelCompilationManager::load_or_compile`) reads it back on a fresh compile so Python
// (`PerOffloadCacheObservations`) can assert the split enumerated the expected constructs. Reading it in the
// compilation manager -- rather than in a backend codegen driver -- makes the observability backend-agnostic (LLVM and
// SPIR-V alike) and lets a cache hit report the no-split sentinel uniformly. This PR ships the split with NO reuse
// tier, so `hit` is always 0 and `recompiled == total`; the cross-process manifest PR adds the reuse (and the manifest
// half of this header). Holds only ints -- no IR and no `SNode *` -- so, unlike the dropped in-memory frontend cache,
// it needs no wipe on `destroy_snode_tree`. The type keeps the name `PerConstructCache` for that continuation.
struct PerConstructCache {
  struct Stats {
    int total = 0;
    int hit = 0;
    int recompiled = 0;
  };

  std::mutex mu;
  std::unordered_map<std::string, Stats> last_stats;  // kernel name -> most-recent split
};

}  // namespace quadrants::lang
