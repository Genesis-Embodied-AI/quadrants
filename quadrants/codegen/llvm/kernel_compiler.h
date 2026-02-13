#pragma once

#include "quadrants/codegen/kernel_compiler.h"
#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/llvm_context.h"

namespace quadrants::lang {
namespace LLVM {

class KernelCompiler : public lang::KernelCompiler {
 public:
  struct Config {
    QuadrantsLLVMContext *tlctx{nullptr};
  };

  explicit KernelCompiler(Config config);

  IRNodePtr compile(const CompileConfig &compile_config,
                    const Kernel &kernel_def) const override;

  CKDPtr compile(const CompileConfig &compile_config,
                 const DeviceCapabilityConfig &device_caps,
                 const Kernel &kernel_def,
                 IRNode &chi_ir) const override;

 private:
  Config config_;
};

}  // namespace LLVM
}  // namespace quadrants::lang
