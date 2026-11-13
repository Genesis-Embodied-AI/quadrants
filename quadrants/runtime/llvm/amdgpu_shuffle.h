#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

namespace quadrants::lang {

enum class AMDGPUShuffleMode { Wave64, TwoHalves };

// Bind the existing shared helper to one complete C++ implementation. Keep the ordinary and relative shuffle
// wrappers unchanged. Remove the unused algorithm before any unsupported intrinsic is patched.
inline llvm::Function *select_amdgpu_shuffle(llvm::Module &module, AMDGPUShuffleMode mode) {
  auto *entry = module.getFunction("amdgpu_cross_half_shuffle_i32");
  auto *wave64 = module.getFunction("amdgpu_shuffle_wave64_i32");
  auto *two_halves = module.getFunction("amdgpu_shuffle_two_halves_i32");
  if (!entry || !wave64 || !two_halves) {
    llvm::report_fatal_error("Missing AMDGPU shuffle runtime implementation");
  }
  auto *selected = mode == AMDGPUShuffleMode::Wave64 ? wave64 : two_halves;
  auto *unused = mode == AMDGPUShuffleMode::Wave64 ? two_halves : wave64;
  entry->replaceAllUsesWith(selected);
  entry->eraseFromParent();
  selected->setName("amdgpu_cross_half_shuffle_i32");
  unused->eraseFromParent();

  if (auto *swap = module.getFunction("amdgpu_permlane64"); swap && swap->use_empty()) {
    swap->eraseFromParent();
  }
  return selected;
}

}  // namespace quadrants::lang
