#pragma once

#include <ctime>
#include <cstdint>
#include <queue>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "quadrants/common/core.h"
#include "quadrants/common/cleanup.h"
#include "quadrants/common/version.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/util/io.h"
#include "quadrants/util/lock.h"
#include "quadrants/program/compile_config.h"

namespace quadrants::lang {
namespace offline_cache {

using Version = std::uint16_t[3];  // {MAJOR, MINOR, PATCH}

enum CleanCacheFlags {
  NotClean = 0b000,
  CleanOldVersion = 0b001,
  CleanOldUsed = 0b010,
  CleanOldCreated = 0b100
};

enum CleanCachePolicy {
  Never = NotClean,
  OnlyOldVersion = CleanOldVersion,
  LRU = CleanOldVersion | CleanOldUsed,
  FIFO = CleanOldVersion | CleanOldCreated
};

inline CleanCachePolicy string_to_clean_cache_policy(const std::string &str) {
  if (str == "never") {
    return Never;
  } else if (str == "version") {
    return OnlyOldVersion;
  } else if (str == "lru") {
    return LRU;
  } else if (str == "fifo") {
    return FIFO;
  }
  return Never;
}

template <typename KernelMetadataType>
struct Metadata {
  using KernelMetadata = KernelMetadataType;

  Version version{};
  std::size_t size{0};  // byte
  std::unordered_map<std::string, KernelMetadata> kernels;

  // NOTE: The "version" must be the first field to be serialized
  QD_IO_DEF(version, size, kernels);
};

enum class LoadMetadataError {
  kNoError,
  kCorrupted,
  kFileNotFound,
  kVersionNotMatched,
};

template <typename MetadataType>
inline LoadMetadataError load_metadata_with_checking(
    MetadataType &result,
    const std::string &filepath) {
  if (!quadrants::path_exists(filepath)) {
    QD_DEBUG("Offline cache metadata file {} not found", filepath);
    return LoadMetadataError::kFileNotFound;
  }

  using VerType = std::remove_reference_t<decltype(result.version)>;
  static_assert(std::is_same_v<VerType, Version>);
  const std::vector<uint8> bytes = read_data_from_file(filepath);

  VerType ver{};
  if (!read_from_binary(ver, bytes.data(), bytes.size(), false)) {
    return LoadMetadataError::kCorrupted;
  }
  if (ver[0] != QD_VERSION_MAJOR || ver[1] != QD_VERSION_MINOR ||
      ver[2] != QD_VERSION_PATCH) {
    QD_DEBUG("The offline cache metadata file {} is old (version={}.{}.{})",
             filepath, ver[0], ver[1], ver[2]);
    return LoadMetadataError::kVersionNotMatched;
  }

  return !read_from_binary(result, bytes.data(), bytes.size())
             ? LoadMetadataError::kCorrupted
             : LoadMetadataError::kNoError;
}

struct CacheCleanerConfig {
  std::string path;
  CleanCachePolicy policy;
  int max_size{0};
  double cleaning_factor{0.f};
  std::string metadata_filename;
  std::string debugging_metadata_filename;
  std::string metadata_lock_name;
};

template <typename MetadataType>
struct CacheCleanerUtils {
  using MetaData = typename MetadataType::Metadata;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
    write_to_binary_file(
        data, quadrants::join_path(config.path, config.metadata_filename));
    return true;
  }

  static bool save_debugging_metadata(const CacheCleanerConfig &config,
                                      const MetadataType &data) {
    TextSerializer ts;
    ts.serialize_to_json("cache", data);
    ts.write_to_file(
        quadrants::join_path(config.path, config.debugging_metadata_filename));
    return true;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const CacheCleanerConfig &config,
      const MetaData &kernel_meta) {
    std::vector<std::string> result;
    for (const auto &f :
         get_possible_llvm_cache_filename_by_key(kernel_meta.kernel_key)) {
      result.push_back(f);
    }
    return result;
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    return true;
  }
};

template <typename MetadataType>
class CacheCleaner {
  using Utils = CacheCleanerUtils<MetadataType>;
  using DataWrapper = typename MetadataType::DataWrapper;

 public:
  static void run(const CacheCleanerConfig &config) {
    QD_ASSERT(!config.path.empty());
    QD_ASSERT(config.max_size > 0);
    QD_ASSERT(!config.metadata_filename.empty());
    QD_ASSERT(!config.metadata_lock_name.empty());
    const auto policy = config.policy;
    const auto &path = config.path;
    const auto metadata_file =
        quadrants::join_path(path, config.metadata_filename);
    const auto debugging_metadata_file =
        quadrants::join_path(path, config.debugging_metadata_filename);

    if (policy == (std::size_t)NotClean) {
      return;
    }
    if (!quadrants::path_exists(path)) {
      return;
    }

    MetadataType cache_data;
    std::vector<std::string> files_to_rm;
    bool ok_rm_meta = false;

    // 1. Remove/Update metadata files
    {
      std::string lock_path =
          quadrants::join_path(path, config.metadata_lock_name);
      if (!lock_with_file(lock_path)) {
        QD_WARN(
            "Lock {} failed. You can run 'ti cache clean -p {}' and try again.",
            lock_path, path);
        return;
      }
      auto _ = make_cleanup([&lock_path]() {
        QD_DEBUG("Stop cleaning cache");
        if (!unlock_with_file(lock_path)) {
          QD_WARN(
              "Unlock {} failed. You can remove this .lock file manually and "
              "try again.",
              lock_path);
        }
      });
      QD_DEBUG("Start cleaning cache");

      using Error = LoadMetadataError;
      Error error = load_metadata_with_checking(cache_data, metadata_file);
      if (error == Error::kFileNotFound) {
        return;
      } else if (error == Error::kCorrupted ||
                 error == Error::kVersionNotMatched) {
        if (policy &
            CleanOldVersion) {  // Remove cache files and metadata files
          QD_DEBUG("Removing all cache files");
          if (quadrants::remove(metadata_file)) {
            quadrants::remove(debugging_metadata_file);
            Utils::remove_other_files(config);
            bool success = quadrants::traverse_directory(
                config.path, [&config](const std::string &name, bool is_dir) {
                  if (!is_dir && Utils::is_valid_cache_file(config, name)) {
                    quadrants::remove(quadrants::join_path(config.path, name));
                  }
                });
            QD_ASSERT(success);
          }
        }
        return;
      }

      if (cache_data.size < config.max_size ||
          static_cast<std::size_t>(config.cleaning_factor *
                                   cache_data.dataWrapperByCacheKey.size()) ==
              0) {
        return;
      }

      // LRU or FIFO
      using KerData = std::pair<const std::string, DataWrapper>;
      using Comparator = std::function<bool(const KerData *, const KerData *)>;
      using PriQueue =
          std::priority_queue<const KerData *, std::vector<const KerData *>,
                              Comparator>;

      Comparator cmp{nullptr};
      if (policy & CleanOldUsed) {  // LRU
        cmp = [](const KerData *a, const KerData *b) -> bool {
          return a->second.metadata.last_used_at <
                 b->second.metadata.last_used_at;
        };
      } else if (policy & CleanOldCreated) {  // FIFO
        cmp = [](const KerData *a, const KerData *b) -> bool {
          return a->second.metadata.created_at < b->second.metadata.created_at;
        };
      }

      if (cmp) {
        PriQueue q(cmp);
        std::size_t cnt =
            config.cleaning_factor * cache_data.dataWrapperByCacheKey.size();
        QD_ASSERT(cnt != 0);
        for (const auto &e : cache_data.dataWrapperByCacheKey) {
          if (q.size() == cnt && cmp(&e, q.top())) {
            q.pop();
          }
          if (q.size() < cnt) {
            q.push(&e);
          }
        }
        QD_ASSERT(q.size() <= cnt);
        while (!q.empty()) {
          const auto *e = q.top();
          for (const auto &f : Utils::get_cache_files(config, e->second)) {
            files_to_rm.push_back(f);
          }
          cache_data.size -= e->second.metadata.size;
          cache_data.dataWrapperByCacheKey.erase(e->first);
          q.pop();
        }

        if (cache_data.dataWrapperByCacheKey.empty()) {  // Remove
          ok_rm_meta = quadrants::remove(metadata_file);
          quadrants::remove(debugging_metadata_file);
          Utils::remove_other_files(config);
        } else {  // Update
          Utils::save_metadata(config, cache_data);
          ok_rm_meta = true;
        }
      }
    }

    // 2. Remove cache files
    if (ok_rm_meta) {
      if (!cache_data.dataWrapperByCacheKey.empty()) {
        // For debugging (Not safe: without locking)
        Utils::save_debugging_metadata(config, cache_data);
      }
      for (const auto &f : files_to_rm) {
        auto file_path = quadrants::join_path(path, f);
        quadrants::remove(file_path);
      }
    }
  }
};

std::string mangle_name(const std::string &primal_name, const std::string &key);
bool try_demangle_name(const std::string &mangled_name,
                       std::string &primal_name,
                       std::string &key);

}  // namespace offline_cache
}  // namespace quadrants::lang
