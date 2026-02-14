// The AMDGPU backend
#pragma once

#include "quadrants/codegen/codegen.h"
#include "quadrants/codegen/llvm/codegen_llvm.h"

namespace quadrants {
namespace lang {

class KernelCodeGenAMDGPU : public KernelCodeGen {
 public:
  KernelCodeGenAMDGPU(const CompileConfig &config,
                      const Kernel *kernel,
                      IRNode *ir,
                      QuadrantsLLVMContext &tlctx)
      : KernelCodeGen(config, kernel, ir, tlctx) {
  }

// TODO: Stop defining this macro guards in the headers
#ifdef QD_WITH_LLVM
  LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) override;
#endif  // QD_WITH_LLVM
};

}  // namespace lang
}  // namespace quadrants
