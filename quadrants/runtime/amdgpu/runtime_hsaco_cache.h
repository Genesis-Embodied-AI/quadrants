#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace quadrants::lang {

struct RuntimeHsacoKey {
  std::string ir_digest;
  std::string mcpu;
  std::string features;
  bool fast_math;

  bool operator==(const RuntimeHsacoKey &other) const {
    return ir_digest == other.ir_digest && mcpu == other.mcpu && features == other.features &&
           fast_math == other.fast_math;
  }
};

// Process-local host bytes only: each JIT session must load and own its own HIP module. Keep compilation under the
// lock so concurrent initializations cannot compile the same runtime twice. Failed compilations leave no entry.
class RuntimeHsacoCache {
 public:
  explicit RuntimeHsacoCache(std::size_t max_entries = 8, std::size_t max_bytes = 8 * 1024 * 1024)
      : max_entries_(max_entries), max_bytes_(max_bytes) {
  }

  template <typename Compile>
  std::string get_or_compile(const RuntimeHsacoKey &key, Compile compile, bool &hit) {
    std::lock_guard<std::mutex> lock(mutex_);
    hit = false;
    for (const auto &entry : entries_) {
      if (entry.first == key) {
        hit = true;
        return entry.second;
      }
    }
    auto bytes = compile();
    if (bytes.empty() || bytes.size() > max_bytes_ || max_entries_ == 0) {
      return bytes;
    }
    while (entries_.size() >= max_entries_ || bytes.size() > max_bytes_ - bytes_) {
      bytes_ -= entries_.front().second.size();
      entries_.pop_front();
    }
    entries_.emplace_back(key, bytes);
    bytes_ += bytes.size();
    return bytes;
  }

 private:
  const std::size_t max_entries_;
  const std::size_t max_bytes_;
  std::size_t bytes_{0};
  std::mutex mutex_;
  std::deque<std::pair<RuntimeHsacoKey, std::string>> entries_;
};

}  // namespace quadrants::lang
