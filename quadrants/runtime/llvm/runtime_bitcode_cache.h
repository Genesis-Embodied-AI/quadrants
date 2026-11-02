#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringRef.h"

namespace quadrants::lang {

// Own the exact bytes used for both the key and preparation. In particular, an editable build cannot make a cache
// miss hash one file version and then parse a second read of that file. No LLVM object is retained by this snapshot.
struct RuntimeBitcodeInputs {
  std::string target;
  std::map<std::string, std::string> files;

  static RuntimeBitcodeInputs read(const std::string &target, const std::vector<std::string> &paths) {
    RuntimeBitcodeInputs inputs{target, {}};
    for (const std::string &path : paths) {
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
    std::array<std::uint8_t, 32> digest = hash.final();
    return {reinterpret_cast<const char *>(digest.data()), digest.size()};
  }
};

// Retain serialized modules until process exit. Consumers parse copies into their own mutable LLVM contexts.
class RuntimeBitcodeCache {
 public:
  template <typename Build>
  std::string get_or_build(const std::string &key, Build build, bool &hit) {
    std::lock_guard<std::mutex> lock(mutex_);
    hit = false;
    // We expect few runtime variants per process; a linear scan avoids a separate index.
    for (const auto &[stored_key, bytes] : entries_) {
      if (stored_key == key) {
        hit = true;
        return bytes;
      }
    }
    std::string bytes = build();
    if (bytes.empty()) {
      return bytes;
    }
    entries_.emplace_back(key, bytes);
    return bytes;
  }

 private:
  std::mutex mutex_;
  std::deque<std::pair<std::string, std::string>> entries_;
};

}  // namespace quadrants::lang
