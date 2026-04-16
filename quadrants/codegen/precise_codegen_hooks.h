#pragma once

// Precise-feature codegen hook installers. These register generic pre/post stmt hooks on
// the codegen instances. The hooks check the stmt's codegen_hints for kDisableFastMath and
// save/clear/restore the builder's fast-math state accordingly.

#ifdef QD_WITH_LLVM
namespace quadrants::lang {
class TaskCodeGenLLVM;
void install_precise_hooks_llvm(TaskCodeGenLLVM &cg);
}  // namespace quadrants::lang
#endif

#if defined(QD_WITH_VULKAN) || defined(QD_WITH_METAL)
namespace quadrants::lang::spirv::detail {
class TaskCodegen;
void install_precise_hooks_spirv(TaskCodegen &cg);
}  // namespace quadrants::lang::spirv::detail
#endif
