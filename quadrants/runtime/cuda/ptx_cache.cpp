// This was copy and hacked from kernel_compilation_manager.cpp
// It is used to manage the compilation and caching of PTX kernels

#include "picosha2.h"

#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/util/offline_cache.h"
#include "quadrants/util/environ_config.h"

#include "quadrants/runtime/cuda/ptx_cache.h"

namespace quadrants::lang {

namespace offline_cache {

constexpr char kPtxCacheFilenameExt[] = "ptx";

template <>
struct CacheCleanerUtils<PtxCacheAllData> {
  using MetadataType = PtxCacheAllData;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
    write_to_binary_file(
        data, quadrants::join_path(config.path, config.metadata_filename));
    return true;
  }

  static bool save_debugging_metadata(const CacheCleanerConfig &config,
                                      const MetadataType &data) {
    return true;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const CacheCleanerConfig &config,
      const WrappedPtx &wrappedPtx) {
    auto fn = fmt::format(PtxCache::kCacheFilenameFormat,
                          wrappedPtx.metadata.cache_key);
    return {fn};
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
    // Do nothing
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    std::string ext = filename_extension(name);
    return ext == kPtxCacheFilenameExt;
  }
};

}  // namespace offline_cache

PtxCache::PtxCache(const Config config, const CompileConfig &compile_config)
    : config_(std::move(config)),
      compile_config_(compile_config),
      cache_dir_(join_path(config_.offline_cache_path, "ptx_cache")) {
  TI_DEBUG("Create ptxcache with offline_cache_file_path = {}",
           this->cache_dir_);
  auto filepath = join_path(this->cache_dir_, kMetadataFilename);
  auto lock_path = join_path(this->cache_dir_, kMetadataLockName);
  if (path_exists(filepath)) {
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_all_data_, filepath);
    } else {
      TI_WARN(
          "Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
          lock_path, this->cache_dir_);
    }
  }
}

void PtxCache::dump() {
  if (wrapped_by_key_.empty()) {
    return;
  }

  TI_DEBUG("Dumping {} cached kernels to disk", wrapped_by_key_.size());

  quadrants::create_directories(cache_dir_);
  auto filepath = join_path(cache_dir_, kMetadataFilename);
  auto lock_path = join_path(cache_dir_, kMetadataLockName);

  if (!lock_with_file(lock_path)) {
    TI_WARN("Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
            lock_path, cache_dir_);
    wrapped_by_key_.clear();  // Ignore the caching kernels
    return;
  }

  auto _ = make_unlocker(lock_path);
  PtxCacheAllData data;
  data.version[0] = TI_VERSION_MAJOR;
  data.version[1] = TI_VERSION_MINOR;
  data.version[2] = TI_VERSION_PATCH;
  auto &dataWrapperByCacheKey = data.dataWrapperByCacheKey;
  // Load old cached data
  offline_cache::load_metadata_with_checking(data, filepath);
  // Update the cached data
  for (const auto *e : updated_data_) {
    auto iter = dataWrapperByCacheKey.find(e->metadata.cache_key);
    if (iter != dataWrapperByCacheKey.end()) {
      iter->second.metadata.last_used_at = e->metadata.last_used_at;
    }
  }
  // Add new data
  for (auto &[kernel_key, wrapped] : wrapped_by_key_) {
    if (wrapped.metadata.cache_mode == CacheMode::MemAndDiskCache) {
      auto [iter, ok] =
          dataWrapperByCacheKey.insert({kernel_key, std::move(wrapped)});
      TI_ASSERT(!ok || iter->second.metadata.size == 0);
    }
  }
  wrapped_by_key_.clear();
  // Dump cached CompiledKernelData to disk
  for (auto &[_, k] : dataWrapperByCacheKey) {
    if (!k.ptx.has_value()) {
      TI_TRACE("PTX for cache_key {} is not set, skipping dump",
               k.metadata.cache_key);
      continue;
    }
    TI_TRACE("Dumping PTX for cache_key {}", k.metadata.cache_key);
    auto cache_filename = make_filename(k.metadata.cache_key);
    std::ofstream fs{cache_filename, std::ios::out | std::ios::binary};
    TI_ASSERT(fs.is_open());
    fs << k.ptx.value();
    TI_ASSERT(!!fs);
    k.metadata.size = fs.tellp();
    data.size += k.metadata.size;
  }
  // Dump offline cache metadata
  if (!dataWrapperByCacheKey.empty()) {
    write_to_binary_file(data, filepath);
  }
}

void PtxCache::clean_offline_cache(offline_cache::CleanCachePolicy policy,
                                   int max_bytes,
                                   double cleaning_factor) const {
  using CacheCleaner = offline_cache::CacheCleaner<PtxCacheAllData>;
  offline_cache::CacheCleanerConfig config;
  config.path = cache_dir_;
  config.policy = policy;
  config.cleaning_factor = cleaning_factor;
  config.max_size = max_bytes;
  config.metadata_filename = kMetadataFilename;
  config.debugging_metadata_filename = "";
  config.metadata_lock_name = kMetadataLockName;
  CacheCleaner::run(config);
}

std::string PtxCache::make_filename(const std::string &kernel_key) const {
  return join_path(cache_dir_, fmt::format(kCacheFilenameFormat, kernel_key));
}

std::string PtxCache::make_cache_key(const std::string &llvm_ir,
                                     bool use_fast_math) const {
  picosha2::hash256_one_by_one hasher;
  std::string fast_math_str = use_fast_math ? "1" : "0";
  hasher.process(fast_math_str.begin(), fast_math_str.end());
  hasher.process(llvm_ir.begin(), llvm_ir.end());
  hasher.finish();

  auto res = picosha2::get_hash_hex_string(hasher);
  res.insert(res.begin(), 'T');  // The key must start with a letter
  return res;
}

std::optional<std::string> PtxCache::try_load_cached(
    const std::string &cache_key,
    CacheMode cache_mode) {
  {
    // Find in memory-cache
    const auto &kernels = wrapped_by_key_;
    auto iter = kernels.find(cache_key);
    if (iter != kernels.end()) {
      return iter->second.ptx;
    }
  }
  // Find in disk-cache
  if (cache_mode == CacheMode::MemAndDiskCache) {
    auto &dataWrapperByCacheKey = cached_all_data_.dataWrapperByCacheKey;
    auto iter = dataWrapperByCacheKey.find(cache_key);
    if (iter != dataWrapperByCacheKey.end()) {
      auto &k = iter->second;
      TI_DEBUG("Found in cache (key='{}')", cache_key);
      if (k.ptx.has_value()) {
        return k.ptx;
      }
      // If the PTX is not in memory, try to load it from disk
      std::optional<std::string> ptx = load_data_from_disk(cache_key);
      k.ptx = ptx;
      return ptx;
    }
  }
  return std::nullopt;
}

std::optional<std::string> PtxCache::load_data_from_disk(
    const std::string &cache_key) {
  const auto filename = make_filename(cache_key);
  if (std::ifstream ifs(filename, std::ios::in | std::ios::binary);
      ifs.is_open()) {
    std::string ptx = std::string(std::istreambuf_iterator<char>(ifs),
                                  std::istreambuf_iterator<char>());
    if (!ifs) {
      TI_WARN(fmt::format("Failed to read PTX from file {}: {}", filename,
                          std::strerror(errno)));
      return std::nullopt;
    }
    TI_DEBUG("Loaded PTX from file {} (size: {} bytes)", filename, ptx.size());
    if (ptx.empty()) {
      TI_WARN(fmt::format("PTX file {} is empty", filename));
      return std::nullopt;
    }
    return ptx;
  }
  TI_WARN(fmt::format("Failed to load ptx file {}: {}", filename,
                      std::strerror(errno)));
  return std::nullopt;
}

void PtxCache::store_ptx(const std::string &cache_key, const std::string &ptx) {
  TI_DEBUG("Store PTX for cache_key {}", cache_key);
  WrappedPtx k;
  k.ptx = ptx;
  k.metadata.cache_key = cache_key;
  k.metadata.created_at = k.metadata.last_used_at = std::time(nullptr);
  k.metadata.size = 0;  // Populate `size` within the PtxCache::dump()
  k.metadata.cache_mode = CacheMode::MemAndDiskCache;
  wrapped_by_key_[cache_key] = std::move(k);
}

std::optional<std::string> PtxCache::load_ptx(const std::string &cache_key) {
  auto cache_mode = get_cache_mode(compile_config_);
  return try_load_cached(cache_key, cache_mode);
}

CacheMode PtxCache::get_cache_mode(const CompileConfig &compile_config) {
  return CacheMode::MemAndDiskCache;
}

}  // namespace quadrants::lang
