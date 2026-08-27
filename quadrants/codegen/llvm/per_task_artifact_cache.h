#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/common/serialization.h"

namespace quadrants::lang {

// One offloaded task's launch-ready artifact, persisted so a fresh process can launch it with no compilation (no
// CHI->LLVM, no link, no optimize, no PTX/ptxas). The code alone is not enough: the launcher and graph builder consume
// the `OffloadedTask` metadata (entry-fn name, dims, graph_do_while / parallel-group / region / checkpoint ids,
// adstack, snode/arg read-write sets) and the runtime needs `used_tree_ids`. `code` is the backend payload -- PTX on
// CUDA under Option B. `used_tree_ids` / `struct_for_tls_sizes` are stored sorted so the on-disk bytes are
// deterministic (their live form is an unordered_set).
struct PerTaskArtifact {
  std::vector<OffloadedTask> tasks;
  std::vector<int> used_tree_ids;
  std::vector<int> struct_for_tls_sizes;
  std::vector<char> code;
  QD_IO_DEF(tasks, used_tree_ids, struct_for_tls_sizes, code);
};

// Content-addressed on-disk store of `PerTaskArtifact`, shared by the codegen driver (probe) and the CUDA JIT (fill).
// Both derive the path from the same (dir, ir_key), so they must resolve `dir` identically (from the compile config).
class PerTaskArtifactCache {
 public:
  explicit PerTaskArtifactCache(std::string dir) : dir_(std::move(dir)) {
  }

  const std::string &dir() const {
    return dir_;
  }

  bool try_load(const std::string &ir_key, PerTaskArtifact *out) const {
    if (dir_.empty() || out == nullptr) {
      return false;
    }
    const auto p = path_for(ir_key);
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
      return false;
    }
    // A truncated/corrupt record (e.g. a crash mid-write) must degrade to a miss, not a hard failure.
    return read_from_binary_file(*out, p);
  }

  void store(const std::string &ir_key, const PerTaskArtifact &rec) const {
    if (dir_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    const auto p = path_for(ir_key);
    // Temp-write then rename so concurrent workers/processes never observe a partial record. The temp must keep the
    // `.qdb` suffix -- the serializer rejects any other extension.
    const auto tmp =
        p + ".tmp" +
        std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()) + ".qdb";
    write_to_binary_file(rec, tmp);
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
      std::filesystem::remove(tmp, ec);
    }
  }

 private:
  // The IR key is a hex digest plus a `#<index>` suffix; '#' is legal but shell-awkward, so map it (and '/') to '_'.
  // The `.qdb` extension is mandatory -- the binary serializer refuses any other suffix.
  std::string path_for(const std::string &ir_key) const {
    std::string safe = ir_key;
    for (char &c : safe) {
      if (c == '#' || c == '/') {
        c = '_';
      }
    }
    return (std::filesystem::path(dir_) / (safe + ".qdb")).string();
  }

  std::string dir_;
};

inline std::string pertask_artifact_dir_for(const std::string &offline_cache_file_path) {
  if (offline_cache_file_path.empty()) {
    return std::string("/tmp/qd_pertask_artifacts");
  }
  return offline_cache_file_path + "/pertask_artifacts";
}

// `CompiledKernelData::load_impl` needs the artifact dir but sits far from any `CompileConfig`; artifacts always live
// beside the `.qdc` files, so resolve it once when the LLVM program is constructed (before any kernel loads/compiles).
inline std::string &pertask_artifact_dir_ref() {
  static std::string dir;
  return dir;
}

inline void set_pertask_artifact_dir_from_offline_cache(const std::string &offline_cache_file_path) {
  pertask_artifact_dir_ref() = pertask_artifact_dir_for(offline_cache_file_path);
}

inline std::string resolved_pertask_artifact_dir() {
  const auto &d = pertask_artifact_dir_ref();
  return d.empty() ? pertask_artifact_dir_for(std::string()) : d;
}

}  // namespace quadrants::lang
