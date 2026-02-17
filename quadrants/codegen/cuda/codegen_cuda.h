// The CUDA backend

#pragma once

#include "quadrants/codegen/codegen.h"
#include "quadrants/codegen/llvm/codegen_llvm.h"

namespace quadrants::lang {

class KernelCodeGenCUDA : public KernelCodeGen {
 public:
  explicit KernelCodeGenCUDA(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             QuadrantsLLVMContext &tlctx)
      : KernelCodeGen(compile_config, kernel, ir, tlctx) {
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

}  // namespace quadrants::lang
