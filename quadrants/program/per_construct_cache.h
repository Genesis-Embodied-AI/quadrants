#pragma once

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "quadrants/common/serialization.h"

namespace quadrants::lang {

// The ordered per-task artifact keys one top-level construct produced, so a warm process reproduces the construct
// WITHOUT running its frontend (merge_global_ptrs / full_simplify / offload). Each key names a `PerTaskArtifact` that
// already holds the task's code + launch metadata. Keyed by the CROSS-PROCESS construct key
// (`get_hashed_per_construct_disk_key`). Reusable only if the construct lands at the same task offset -- each key's
// `#<index>` is baked into the compiled entry-fn / shared-array symbol names, so the loader treats a shift as a miss.
struct ConstructManifest {
  std::vector<std::string> task_keys;
  QD_IO_DEF(task_keys);
};

class ConstructManifestCache {
 public:
  explicit ConstructManifestCache(std::string dir) : dir_(std::move(dir)) {
  }

  bool try_load(const std::string &ckey, ConstructManifest *out) const {
    if (dir_.empty() || out == nullptr) {
      return false;
    }
    const auto p = path_for(ckey);
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
      return false;
    }
    return read_from_binary_file(*out, p);
  }

  void store(const std::string &ckey, const ConstructManifest &m) const {
    if (dir_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    const auto p = path_for(ckey);
    // temp+rename; the temp must keep the `.qdb` suffix the serializer requires
    const auto tmp =
        p + ".tmp" +
        std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()) + ".qdb";
    write_to_binary_file(m, tmp);
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
      std::filesystem::remove(tmp, ec);
    }
  }

 private:
  std::string path_for(const std::string &ckey) const {
    std::string safe = ckey;
    for (char &c : safe) {
      if (c == '#' || c == '/') {
        c = '_';
      }
    }
    return (std::filesystem::path(dir_) / (safe + ".qdb")).string();
  }

  std::string dir_;
};

inline std::string construct_manifest_dir_for(const std::string &offline_cache_file_path) {
  if (offline_cache_file_path.empty()) {
    return std::string("/tmp/qd_construct_manifests");
  }
  return offline_cache_file_path + "/construct_manifests";
}

// Side channel between the frontend split and the codegen driver, per kernel, indexed by task position. The split
// decides per construct whether its frontend can be skipped, but the per-task keys that record a manifest are only
// formed later in codegen (they embed the task's kernel-wide index), so the two phases talk through this table rather
// than threading fields through the IR (which would perturb the printed-IR-derived per-task key).
//   artifact_key_by_task[i] non-empty  => task i is a PLACEHOLDER: a manifest hit already named its artifact, so
//                                         codegen must load it, not compile it.
//   construct_key_by_task[i] non-empty => task i came from that construct; codegen groups the keys it computes by
//                                         this and writes the construct's manifest for next time.
struct ConstructDiskPlan {
  std::vector<std::string> construct_key_by_task;
  std::vector<std::string> artifact_key_by_task;
};

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
  std::unordered_map<std::string, Stats> last_stats;              // kernel name -> most-recent split
  std::unordered_map<std::string, ConstructDiskPlan> disk_plans;  // kernel name -> cross-process plan
};

}  // namespace quadrants::lang
