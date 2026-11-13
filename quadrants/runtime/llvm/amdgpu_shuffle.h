#pragma once

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
class Module;
}  // namespace llvm

namespace quadrants::lang {

enum class AMDGPUShuffleMode { Wave64, TwoHalves };

// Replace the runtime shuffle entry point with the selected implementation and remove the unused implementation.
llvm::Function *select_amdgpu_shuffle(llvm::Module &module, AMDGPUShuffleMode mode);

// Select the target's shuffle implementation and supply its lane-read and half-swap operations.
void prepare_amdgpu_shuffle(llvm::Module &module, llvm::StringRef mcpu);

}  // namespace quadrants::lang
