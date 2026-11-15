#pragma once

#include <string>
#include <vector>

#include "llvm/IR/Intrinsics.h"
#include "quadrants/runtime/llvm/llvm_fwd.h"

namespace quadrants::lang {

// Replace a runtime stub with an inline intrinsic wrapper, forwarding its arguments followed by extra_args.
void patch_runtime_intrinsic(llvm::Module &module,
                             std::string name,
                             llvm::Intrinsic::ID intrin,
                             bool ret = true,
                             std::vector<llvm::Type *> types = {},
                             std::vector<llvm::Value *> extra_args = {});

// Preserve the runtime's explicit no-inline marker; otherwise request expansion into the caller.
void mark_runtime_function_inline(llvm::Function *function);

}  // namespace quadrants::lang
