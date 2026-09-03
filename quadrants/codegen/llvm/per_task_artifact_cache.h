#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/common/serialization.h"

namespace quadrants::lang {

// One offloaded task's launch-ready artifact, persisted so a fresh process launches it with no compilation. `code` is
// the backend payload (PTX on CUDA); the launch metadata / used_tree_ids are needed too, so code alone isn't enough.
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
    // Degrade a truncated/corrupt record to a miss, not a crash: read_from_binary_file trusts the on-disk length
    // header and can over-read, so decode via the length-checked read_from_binary (size guard covers a file shorter
    // than the header itself).
    const std::vector<uint8_t> bytes = read_data_from_file(p);
    if (bytes.size() < sizeof(std::size_t)) {
      return false;
    }
    if (!read_from_binary(*out, bytes.data(), bytes.size())) {
      return false;
    }
    // A record can decode yet be semantically empty (corruption that leaves a valid length header). A genuine artifact
    // always has >=1 task and non-empty code; treat empty as a miss so the caller can't build a broken task.
    if (out->tasks.empty() || out->code.empty()) {
      return false;
    }
    return true;
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
    const auto tmp = p + ".tmp" +
                     std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()) +
                     ".qdb";
    write_to_binary_file(rec, tmp);
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
      // POSIX rename replaces an existing `p` atomically, but Windows fails when `p` exists (e.g. a corrupt record
      // try_load declined), leaving it to miss forever. Drop the stale destination and retry so it self-heals; only if
      // that also fails, remove the temp. (POSIX takes the first rename, so this path is Windows-only.)
      std::error_code repair_ec;
      std::filesystem::remove(p, repair_ec);
      std::filesystem::rename(tmp, p, ec);
      if (ec) {
        std::filesystem::remove(tmp, repair_ec);
      }
    }
  }

  // Drop a cached record so a later process refills it. try_load rejects framing-level corruption, but a malformed
  // payload is only detectable by the backend loader.
  void erase(const std::string &ir_key) const {
    if (dir_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove(path_for(ir_key), ec);
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
  return offline_cache_file_path + "/pertask_artifacts";
}

// Process-global artifact directory, set once at LLVM-program construction. The probe (codegen.cpp), fill
// (jit_cuda.cpp) and `.qdc` load (compiled_kernel_data.cpp) have no CompileConfig at hand, so they read it here.
// Sound as a global because Quadrants allows only one live Program (its ctor asserts num_instances_ == 0). Empty
// disables the tier (the single off switch).
inline std::string &pertask_artifact_dir_ref() {
  static std::string dir;
  return dir;
}

}  // namespace quadrants::lang
