#pragma once

#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "llvm/Support/SHA256.h"

namespace quadrants::lang {

// Own the exact bytes used for both the key and preparation. In particular, an editable build cannot make a cache
// miss hash one file version and then parse a second read of that file. No LLVM object is retained by this snapshot.
struct RuntimeBitcodeInputs {
  std::string target;
  std::map<std::string, std::string> files;

  static RuntimeBitcodeInputs read(const std::string &target, const std::vector<std::string> &paths) {
    RuntimeBitcodeInputs inputs{target, {}};
    for (const auto &path : paths) {
      std::ifstream file(path, std::ios::binary);
      if (!file) {
        throw std::runtime_error("Cannot read runtime bitcode: " + path);
      }
      std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
      if (file.bad()) {
        throw std::runtime_error("Failed reading runtime bitcode: " + path);
      }
      inputs.files.emplace(path, std::move(bytes));
    }
    return inputs;
  }

  std::string key() const {
    llvm::SHA256 hash;
    auto add = [&](const std::string &bytes) {
      // Length-prefix each component so paths, targets, and binary contents cannot have ambiguous boundaries.
      hash.update(std::to_string(bytes.size()));
      hash.update(":");
      hash.update(bytes);
    };
    add(target);
    for (const auto &[path, bytes] : files) {
      add(path);
      add(bytes);
    }
    auto digest = hash.final();
    return {reinterpret_cast<const char *>(digest.data()), digest.size()};
  }
};

// The cache owns serialized modules only. Consumers parse owned copies into their own mutable LLVM contexts.
class RuntimeBitcodeCache {
 public:
  explicit RuntimeBitcodeCache(std::size_t max_entries = 4, std::size_t max_bytes = 32 * 1024 * 1024)
      : max_entries_(max_entries), max_bytes_(max_bytes) {
  }

  template <typename Build>
  std::string get_or_build(const std::string &key, Build build, bool &hit) {
    std::lock_guard<std::mutex> lock(mutex_);
    hit = false;
    for (const auto &[stored_key, bytes] : entries_) {
      if (stored_key == key) {
        hit = true;
        return bytes;
      }
    }
    auto bytes = build();
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
  std::deque<std::pair<std::string, std::string>> entries_;
};

}  // namespace quadrants::lang
