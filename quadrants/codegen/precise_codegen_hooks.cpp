#include "quadrants/codegen/precise_codegen_hooks.h"

#include "quadrants/ir/ir.h"

#ifdef QD_WITH_LLVM
#include "quadrants/codegen/llvm/codegen_llvm.h"

namespace quadrants::lang {

void install_precise_hooks_llvm(TaskCodeGenLLVM &cg) {
  // Shared state: the saved FMF from before a precise stmt.
  // Using shared_ptr so the captured state has a stable address across both lambdas.
  auto saved_fmf = std::make_shared<llvm::FastMathFlags>();
  auto active = std::make_shared<bool>(false);

  cg.add_pre_stmt_hook([=](TaskCodeGenLLVM &cg, Stmt *stmt) {
    if (stmt->codegen_hints & (uint32_t)CodegenHint::kDisableFastMath) {
      *saved_fmf = cg.builder->getFastMathFlags();
      cg.builder->setFastMathFlags(llvm::FastMathFlags{});
      *active = true;
    }
  });

  cg.add_post_stmt_hook([=](TaskCodeGenLLVM &cg, Stmt *stmt) {
    if (*active) {
      cg.builder->setFastMathFlags(*saved_fmf);
      *active = false;
    }
  });
}

}  // namespace quadrants::lang
#endif

#if defined(QD_WITH_VULKAN) || defined(QD_WITH_METAL)
#include "quadrants/codegen/spirv/detail/spirv_codegen.h"

namespace quadrants::lang::spirv::detail {

void install_precise_hooks_spirv(TaskCodegen &cg) {
  cg.add_pre_stmt_hook([](TaskCodegen &cg, Stmt *stmt) {
    if (stmt->codegen_hints & (uint32_t)CodegenHint::kDisableFastMath) {
      cg.spirv_builder().set_emit_no_contraction(true);
    }
  });

  cg.add_post_stmt_hook([](TaskCodegen &cg, Stmt *stmt) {
    if (stmt->codegen_hints & (uint32_t)CodegenHint::kDisableFastMath) {
      cg.spirv_builder().set_emit_no_contraction(false);
    }
  });
}

}  // namespace quadrants::lang::spirv::detail
#endif
