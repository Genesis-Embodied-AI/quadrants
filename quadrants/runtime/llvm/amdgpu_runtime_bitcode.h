#pragma once

#include "quadrants/runtime/llvm/llvm_fwd.h"
#include <memory>
#include <string>
#include <vector>

namespace quadrants::lang {

class QuadrantsLLVMContext;
struct CompileConfig;

// List the device-library files used to prepare the runtime for this GPU target.
std::vector<std::string> amdgpu_libdevice_files(const std::string &mcpu);
// Load the prepared runtime into this context, reusing cached serialized bytes when allowed.
std::unique_ptr<llvm::Module> load_amdgpu_runtime_bitcode(QuadrantsLLVMContext &context,
                                                          const CompileConfig &config,
                                                          const std::string &file);

}  // namespace quadrants::lang
