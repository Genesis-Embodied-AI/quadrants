#pragma once

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "quadrants/runtime/amdgpu/runtime_hsaco_cache.h"

namespace quadrants::lang {

inline RuntimeHsacoKey make_runtime_hsaco_key(const llvm::Module &module,
                                              const std::string &mcpu,
                                              const std::string &features,
                                              bool fast_math) {
  // Hash the complete prepared runtime before codegen mutates it, including linked device libraries and attributes.
  llvm::SmallVector<char, 0> bitcode;
  llvm::raw_svector_ostream stream(bitcode);
  llvm::WriteBitcodeToFile(module, stream);
  llvm::SHA256 hash;
  hash.update(llvm::StringRef(bitcode.data(), bitcode.size()));
  auto digest = hash.final();
  return {std::string(reinterpret_cast<const char *>(digest.data()), digest.size()), mcpu, features, fast_math};
}

}  // namespace quadrants::lang
