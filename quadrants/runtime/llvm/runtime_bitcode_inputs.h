#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "quadrants/common/byte_cache.h"
#include "quadrants/common/file_io.h"
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
      inputs.files.emplace(path, read_binary_file(path));
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

// Consumers parse cached bytes into their own mutable LLVM contexts.
using RuntimeBitcodeCache = ByteCache<std::string>;

}  // namespace quadrants::lang
