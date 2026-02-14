#include "quadrants/compilation_manager/kernel_compilation_manager.h"

#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/util/offline_cache.h"
#include "quadrants/util/environ_config.h"

namespace quadrants::lang {

namespace offline_cache {

constexpr char kTiCacheFilenameExt[] = "tic";

template <>
struct CacheCleanerUtils<CacheData> {
  using MetadataType = CacheData;
  using DataWrapper = typename MetadataType::DataWrapper;

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
      const DataWrapper &wrapped_data) {
    auto fn = fmt::format(KernelCompilationManager::kCacheFilenameFormat,
                          wrapped_data.metadata.kernel_key);
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
    return ext == kTiCacheFilenameExt;
  }
};

}  // namespace offline_cache

KernelCompilationManager::KernelCompilationManager(Config config)
    : config_(std::move(config)),
      cache_dir_(
          join_path(config_.offline_cache_path, "kernel_compilation_manager")) {
  QD_DEBUG("Create KernelCompilationManager with offline_cache_file_path = {}",
           this->cache_dir_);
  auto filepath = join_path(this->cache_dir_, kMetadataFilename);
  auto lock_path = join_path(this->cache_dir_, kMetadataLockName);
  if (path_exists(filepath)) {
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
    } else {
      QD_WARN(
          "Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
          lock_path, this->cache_dir_);
    }
  }
}

CompileResult KernelCompilationManager::load_or_compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto cache_mode = get_cache_mode(compile_config, kernel_def.ir_is_ast());
  const auto kernel_key = make_kernel_key(compile_config, caps, kernel_def);
  auto cached_kernel = try_load_cached_kernel(kernel_def.get_name(), kernel_key,
                                              compile_config.arch, cache_mode);
  bool cache_hit = (cached_kernel != nullptr);
  return CompileResult{cached_kernel
                           ? *cached_kernel
                           : compile_and_cache_kernel(
                                 kernel_key, compile_config, caps, kernel_def),
                       cache_hit, kernel_key};
}

void KernelCompilationManager::dump() {
  if (caching_kernels_.empty()) {
    return;
  }

  QD_DEBUG("Dumping {} cached kernels to disk", caching_kernels_.size());

  quadrants::create_directories(cache_dir_);
  auto filepath = join_path(cache_dir_, kMetadataFilename);
  auto lock_path = join_path(cache_dir_, kMetadataLockName);

  if (!lock_with_file(lock_path)) {
    QD_WARN("Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
            lock_path, cache_dir_);
    caching_kernels_.clear();  // Ignore the caching kernels
    return;
  }

  auto _ = make_unlocker(lock_path);
  CacheData data;
  data.version[0] = QD_VERSION_MAJOR;
  data.version[1] = QD_VERSION_MINOR;
  data.version[2] = QD_VERSION_PATCH;
  auto &dataWrapperByCacheKey = data.dataWrapperByCacheKey;
  // Load old cached data
  offline_cache::load_metadata_with_checking(data, filepath);
  // Update the cached data
  for (const auto *e : updated_data_) {
    auto iter = dataWrapperByCacheKey.find(e->metadata.kernel_key);
    if (iter != dataWrapperByCacheKey.end()) {
      iter->second.metadata.last_used_at = e->metadata.last_used_at;
    }
  }

  // Add new data
  for (auto &[kernel_key, kernel] : caching_kernels_) {
    if (kernel.metadata.cache_mode == CacheData::MemAndDiskCache) {
      dataWrapperByCacheKey.insert({kernel_key, std::move(kernel)});
    }
  }
  // Clear caching_kernels_
  caching_kernels_.clear();
  // Dump cached CompiledKernelData to disk
  for (auto &[_, k] : dataWrapperByCacheKey) {
    if (k.compiled_kernel_data) {
      auto cache_filename = make_filename(k.metadata.kernel_key);
      std::ofstream fs{cache_filename, std::ios::out | std::ios::binary};
      QD_ASSERT(fs.is_open());
      auto err = k.compiled_kernel_data->dump(fs);
      if (err == CompiledKernelData::Err::kNoError) {
        QD_ASSERT(!!fs);
        k.metadata.size = fs.tellp();
        data.size += k.metadata.size;
      } else {
        QD_DEBUG("Dump cached CompiledKernelData(kernel_key={}) failed: {}",
                 k.metadata.kernel_key, CompiledKernelData::get_err_msg(err));
      }
    }
  }
  // Dump offline cache metadata
  if (!dataWrapperByCacheKey.empty()) {
    write_to_binary_file(data, filepath);
  }
}

void KernelCompilationManager::clean_offline_cache(
    offline_cache::CleanCachePolicy policy,
    int max_bytes,
    double cleaning_factor) const {
  using CacheCleaner = offline_cache::CacheCleaner<CacheData>;
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

std::string KernelCompilationManager::make_filename(
    const std::string &kernel_key) const {
  return join_path(cache_dir_, fmt::format(kCacheFilenameFormat, kernel_key));
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto &compiler = *config_.kernel_compiler;
  auto ir = compiler.compile(compile_config, kernel_def);
  auto ckd = compiler.compile(compile_config, caps, kernel_def, *ir);
  QD_ASSERT(ckd->check() == CompiledKernelData::Err::kNoError);
  return ckd;
}

std::string KernelCompilationManager::make_kernel_key(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto kernel_key = kernel_def.get_cached_kernel_key();
  if (kernel_key.empty()) {
    if (!kernel_def.ir_is_ast()) {
      kernel_key = kernel_def.get_name();
    } else {  // The kernel key is generated from AST
      kernel_key = get_hashed_offline_cache_key(compile_config, caps,
                                                (Kernel *)&kernel_def);
    }

    kernel_def.set_kernel_key_for_cache(kernel_key);
  }
  return kernel_key;
}

const CompiledKernelData *KernelCompilationManager::try_load_cached_kernel(
    const std::string &kernel_name,
    const std::string &kernel_key,
    Arch arch,
    CacheData::CacheMode cache_mode) {
  {  // Find in memory-cache (caching_kernels_)
    const auto &kernels = caching_kernels_;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      QD_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
               kernel_name, kernel_key);
      return iter->second.compiled_kernel_data.get();
    }
  }
  // Find in disk-cache (cached_data_)
  if (cache_mode == CacheData::MemAndDiskCache) {
    auto &dataWrapperByCacheKey = cached_data_.dataWrapperByCacheKey;
    auto iter = dataWrapperByCacheKey.find(kernel_key);
    if (iter != dataWrapperByCacheKey.end()) {
      auto &k = iter->second;
      if (k.compiled_kernel_data) {
        QD_DEBUG("Create kernel '{}' from cache (key='{}')", kernel_name,
                 kernel_key);
        return k.compiled_kernel_data.get();
      } else if (auto loaded = load_ckd(kernel_key, arch)) {
        QD_DEBUG("Create kernel '{}' from cache (key='{}')", kernel_name,
                 kernel_key);
        QD_ASSERT(loaded->arch() == arch);
        k.metadata.last_used_at = std::time(nullptr);
        k.compiled_kernel_data = std::move(loaded);
        updated_data_.push_back(&k);
        return k.compiled_kernel_data.get();
      }
    }
  }
  return nullptr;
}

const CompiledKernelData &KernelCompilationManager::compile_and_cache_kernel(
    const std::string &kernel_key,
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  if (get_environ_config("QD_SHOW_COMPILING")) {
    QD_INFO("Compiling kernel '{}'", kernel_def.get_name());
  }
  std::unique_ptr<CompiledKernelData> compiled_kernel_data =
      compile_kernel(compile_config, caps, kernel_def);
  CompiledKernelData &res = cache_kernel(
      kernel_key, compile_config, std::move(compiled_kernel_data), kernel_def);
  return res;
}

CompiledKernelData &KernelCompilationManager::cache_kernel(
    const std::string &kernel_key,
    const CompileConfig &compile_config,
    std::unique_ptr<CompiledKernelData> compiled_kernel_data,
    const Kernel &kernel_def) {
  auto cache_mode = get_cache_mode(compile_config, kernel_def.ir_is_ast());
  QD_DEBUG_IF(cache_mode == CacheData::MemAndDiskCache,
              "Cache kernel '{}' (key='{}')", kernel_def.get_name(),
              kernel_key);
  QD_ASSERT(caching_kernels_.find(kernel_key) == caching_kernels_.end());
  KernelCacheData k;
  k.metadata.kernel_key = kernel_key;
  k.metadata.created_at = k.metadata.last_used_at = std::time(nullptr);
  // Populate `size` within the KernelCompilationManager::dump()
  k.metadata.size = 0;
  k.metadata.cache_mode = cache_mode;
  k.compiled_kernel_data = std::move(compiled_kernel_data);
  const auto &kernel_data = (caching_kernels_[kernel_key] = std::move(k));
  return *kernel_data.compiled_kernel_data;
}

const CompiledKernelData *KernelCompilationManager::load_fast_cache(
    const std::string &checksum,
    const std::string &kernel_name,
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps) {
  auto cache_mode = get_cache_mode(compile_config, true);
  auto res = try_load_cached_kernel(kernel_name, checksum, compile_config.arch,
                                    cache_mode);
  return res;
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::load_ckd(
    const std::string &kernel_key,
    Arch arch) {
  const auto filename = make_filename(kernel_key);
  if (std::ifstream ifs(filename, std::ios::in | std::ios::binary);
      ifs.is_open()) {
    CompiledKernelData::Err err;
    auto ckd = CompiledKernelData::load(ifs, &err);
    if (err != CompiledKernelData::Err::kNoError) {
      QD_DEBUG("Load cache file {} failed: {}", filename,
               CompiledKernelData::get_err_msg(err));
      return nullptr;
    }
    if (auto err = ckd->check(); err != CompiledKernelData::Err::kNoError) {
      QD_DEBUG("Check CompiledKernelData loaded from {} failed: {}", filename,
               CompiledKernelData::get_err_msg(err));
      return nullptr;
    }
    return ckd;
  }
  return nullptr;
}

CacheData::CacheMode KernelCompilationManager::get_cache_mode(
    const CompileConfig &compile_config,
    bool kernel_ir_is_ast) {
  return compile_config.offline_cache && kernel_ir_is_ast
             ? CacheData::MemAndDiskCache
             : CacheData::MemCache;
}

}  // namespace quadrants::lang
