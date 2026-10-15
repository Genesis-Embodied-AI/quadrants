#pragma once

#include "llvm_runtime.h"

// Included in the single runtime bitcode translation unit by runtime.cpp.
extern "C" void adstack_runtime_zero_init(LLVMRuntime *runtime);
