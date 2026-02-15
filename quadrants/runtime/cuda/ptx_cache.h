// This was copy and hacked from kernel_compilation_manager.h
// It is used to manage the compilation and caching of PTX kernels
#pragma once

#include <ctime>
#include <string>
#include <memory>
#include <unordered_map>

#include "quadrants/util/offline_cache.h"

namespace quadrants::lang {

enum CacheMode { MemCache, MemAndDiskCache };

struct PtxMetadata {
  std::string cache_key;
  std::size_t size{0};          // byte
  std::time_t created_at{0};    // sec
  std::time_t last_used_at{0};  // sec

  CacheMode cache_mode{MemCache};

  // NOTE: The "version" must be the first field to be serialized
  QD_IO_DEF(cache_key, size, created_at, last_used_at);
};

struct WrappedPtx {
  struct PtxMetadata metadata;
  std::optional<std::string> ptx;

  QD_IO_DEF(metadata);
};

struct PtxCacheAllData {
  using Version = std::uint16_t[3];
  Version version{};
  std::size_t size{0};
  std::unordered_map<std::string, WrappedPtx> dataWrapperByCacheKey;

  using DataWrapper = WrappedPtx;

  // NOTE: The "version" must be the first field to be serialized
  QD_IO_DEF(version, size, dataWrapperByCacheKey);
};

class PtxCache final {
 public:
  static constexpr char kMetadataFilename[] = "ptxcache.tcb";
  static constexpr char kCacheFilenameFormat[] = "{}.ptx";
  static constexpr char kMetadataLockName[] = "ptxcache.lock";

  // using WrappedByKey = std::unordered_map<std::string, WrappedPtx>;

  struct Config {
    std::string offline_cache_path;
  };

  explicit PtxCache(const Config init_params,
                    const CompileConfig &compile_config,
                    int compute_capability);

  void dump();
  void clean_offline_cache(offline_cache::CleanCachePolicy policy,
                           int max_bytes,
                           double cleaning_factor) const;
  void store_ptx(const std::string &cache_key, const std::string &ptx);
  std::optional<std::string> load_ptx(const std::string &cache_key);
  std::string make_cache_key(const std::string &llvm_ir,
                             bool use_fast_math) const;

 private:
  std::string make_filename(const std::string &kernel_key) const;
  static CacheMode get_cache_mode(const CompileConfig &compile_config);
  std::optional<std::string> try_load_cached(const std::string &cache_key,
                                             CacheMode cache_mode);
  std::optional<std::string> load_data_from_disk(const std::string &cache_key);

  const Config config_;
  const CompileConfig &compile_config_;
  const int compute_capability_;

  // caching_kernels_ in kernel_compilation_manager
  std::unordered_map<std::string, WrappedPtx> wrapped_by_key_;

  // cached_data_ in kernel_compilation_manager
  PtxCacheAllData cached_all_data_;

  std::vector<WrappedPtx *> updated_data_;
  const std::string cache_dir_;
};

}  // namespace quadrants::lang
