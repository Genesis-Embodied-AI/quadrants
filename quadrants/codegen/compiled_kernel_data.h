#pragma once

#include <string>
#include <memory>
#include <optional>
#include <algorithm>
#include <stdexcept>

#include "quadrants/rhi/arch.h"

namespace quadrants::lang {

// Per-construct FRONTEND compilation cache stats for one kernel compile. Transient (never serialized): recorded by
// the per-construct frontend split and read back by the codegen driver, carried through the compilation manager into
// `CompileResult`, and surfaced to Python as `PerOffloadCacheObservations`. `construct_*` are `-1` when the split did
// not run for this compile (the whole-kernel path took it, e.g. autodiff or mesh), so Python can tell "split absent"
// from "0 constructs". This PR ships the split with no reuse tier, so when it runs `construct_recompiled ==
// construct_total` and `construct_cache_hit == 0`. The per-task reuse counts are added to this struct by the
// cross-process cache PR (kept named `PerTaskCacheStats` for that continuation).
struct PerTaskCacheStats {
  int construct_total{-1};
  int construct_cache_hit{-1};
  int construct_recompiled{-1};
};

class KernelLaunchHandle {
 public:
  void set_launch_id(int id) {
    launch_id_ = id;
  }

  int get_launch_id() const {
    return launch_id_;
  }

 private:
  int launch_id_{-1};
};

class CompiledKernelDataFile {
 public:
  static constexpr char kHeadStr[] = "QDC";
  static constexpr std::size_t kHeadSize = std::size(kHeadStr);
  static constexpr std::size_t kHashSize = 64;
  enum class Err {
    kNoError,
    kNotQdcFile,
    kCorruptedFile,
    kOutOfMemory,
    kIOStreamError,
  };

  Err dump(std::ostream &os);
  Err load(std::istream &is);

  CompiledKernelDataFile() {
    std::copy(kHeadStr, kHeadStr + kHeadSize, head_);
  }

  void set_arch(Arch arch) {
    arch_ = arch;
  }

  void set_metadata(std::string metadata) {
    metadata_ = std::move(metadata);
  }

  void set_src_code(std::string src) {
    src_code_ = std::move(src);
  }

  const Arch &arch() const {
    return arch_;
  }

  const std::string &metadata() const {
    return metadata_;
  }

  const std::string &src_code() const {
    return src_code_;
  }

 private:
  bool update_hash();

  char head_[kHeadSize];
  Arch arch_;
  std::string metadata_;
  std::string src_code_;
  std::string hash_;
};

class CompiledKernelData {
 public:
  enum class Err {
    kNoError = 0,
    kNotQdcFile,
    kCorruptedFile,
    kParseMetadataFailed,
    kParseSrcCodeFailed,
    kArchNotMatched,
    kSerMetadataFailed,
    kSerSrcCodeFailed,
    kIOStreamError,
    kOutOfMemory,
    kQdWithoutLLVM,
    kQdWithoutSpirv,
    kCompiledKernelDataBroken,
    kUnknown,
  };

  CompiledKernelData() = default;
  CompiledKernelData(const CompiledKernelData &) = delete;
  CompiledKernelData &operator=(const CompiledKernelData &) = delete;
  virtual ~CompiledKernelData() = default;

  virtual Arch arch() const = 0;
  virtual size_t num_tasks() const = 0;

  Err load(std::istream &is);
  Err dump(std::ostream &os) const;

  virtual std::unique_ptr<CompiledKernelData> clone() const = 0;

  virtual Err debug_print(std::ostream &os) const {
    return dump(os);
  }

  virtual Err check() const {
    return Err::kNoError;
  }

  void set_handle(const KernelLaunchHandle &handle) const {
    kernel_launch_handle_ = handle;
  }

  const std::optional<KernelLaunchHandle> &get_handle() const {
    return kernel_launch_handle_;
  }

  // Per-construct frontend-split cache stats for the compile that produced this data (transient; default `-1` for
  // backends / compiles where the split did not run and for data restored from the offline/fast cache).
  virtual PerTaskCacheStats get_per_task_cache_stats() const {
    return {};
  }

  static std::unique_ptr<CompiledKernelData> load(std::istream &is, Err *p_err);

  virtual std::string debug_dump_to_string() const {
    throw std::runtime_error("debug_dump_to_string not implemented");
  }

  static std::string get_err_msg(Err err);

 protected:
  virtual Err load_impl(const CompiledKernelDataFile &file) = 0;
  virtual Err dump_impl(CompiledKernelDataFile &file) const = 0;

 private:
  using Creator = std::unique_ptr<CompiledKernelData>();
  static Creator *const llvm_creator;
  static Creator *const spriv_creator;

  static std::unique_ptr<CompiledKernelData> create(Arch arch, Err &err);

  mutable std::optional<KernelLaunchHandle> kernel_launch_handle_;
};

}  // namespace quadrants::lang
