#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "quadrants/codegen/llvm/llvm_compiled_data.h"

namespace quadrants::lang {

// In-memory per-offloaded-task codegen cache, scoped to a single `Program` (owned by `LlvmProgramImpl`). Keyed on the
// A1 per-task IR key (`get_hashed_per_task_cache_key`), which folds compile config, device caps, touched-SNode layout,
// the re-id'd task body IR, and autodiff mode. It skips `compile_task` (the CHI -> LLVM lowering, the dominant per-task
// codegen cost) for an unchanged task on a warm compile -- the payoff for "edit one offload in a many-offload kernel".
//
// Correctness: a key collision requires byte-identical task IR + SNode layout + config + caps, and `compile_task` is a
// pure function of exactly those, so a colliding entry compiles to the same module. The cached `used_tree_ids` are
// SNode-tree ids, which are only stable within one `Program`; scoping the cache to the `Program` is what keeps those
// ids valid on a hit (a fresh `Program` from `qd.init` starts with an empty cache).
//
// Threading: cached modules live in the dedicated `ctx`; every access to it (bitcode clone in/out) is serialized by
// `mu`. The expensive `compile_task` on a miss runs outside `mu`, so worker parallelism is preserved.
//
// `entries` is declared after `ctx` so it is destroyed first: the cached modules must outlive nothing and must be torn
// down before the context that owns them. Unbounded for now (prototype); a deployment wants an LRU + disk tier (see
// perso_hugh per-construct compilation design doc, S2 / slice-1b).
struct PerTaskModuleCache {
  struct Entry {
    std::unique_ptr<llvm::Module> module;  // lives in `ctx`
    std::vector<OffloadedTask> tasks;
    std::unordered_set<int> used_tree_ids;
    std::unordered_set<int> struct_for_tls_sizes;
  };
  std::mutex mu;
  llvm::LLVMContext ctx;
  std::unordered_map<std::string, Entry> entries;
};

}  // namespace quadrants::lang
