#pragma once

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
}  // namespace llvm

namespace quadrants::lang {

// Supply the target-specific lane mask, lane-read operation, and half-swap implementation.
void prepare_amdgpu_shuffle(llvm::Module &module, llvm::StringRef mcpu);

}  // namespace quadrants::lang
