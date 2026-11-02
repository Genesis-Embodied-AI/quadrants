#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace quadrants::lang {

// Retain byte strings for this cache's lifetime. Serialize lookup and construction so a missing key is built once.
template <typename Key>
class ByteCache {
 public:
  template <typename Build>
  std::string get_or_build(const Key &key, Build build, bool &hit) {
    std::lock_guard<std::mutex> lock(mutex_);
    hit = false;
    // Intended for few keys; a linear scan keeps lookup simple without a custom key hash.
    for (const std::pair<Key, std::string> &entry : entries_) {
      if (entry.first == key) {
        hit = true;
        return entry.second;
      }
    }
    std::string bytes = build();
    if (!bytes.empty()) {
      entries_.emplace_back(key, bytes);
    }
    return bytes;
  }

 private:
  std::mutex mutex_;
  std::deque<std::pair<Key, std::string>> entries_;
};

}  // namespace quadrants::lang
