#pragma once

#include "llvm/IR/LLVMContext.h"

#include "quadrants/rhi/arch.h"

namespace quadrants::lang {

// Pick the LLVM syncscope for a user-facing kernel atomic op (`qd.atomic_add`, `qd.atomic_xor`, ..., and the
// corresponding optimized-reduction runtime helpers). On AMDGPU, the default System scope makes the backend refuse the
// native single-instruction atomics (`global_atomic_xor`, `flat_atomic_xor`, `flat_atomic_add_f32`, ...) and fall back
// to a `flat_atomic_cmpswap` retry loop bracketed by cache-flush instructions, because System scope demands mid-kernel
// visibility to the host CPU. That CAS loop livelocks under high contention for non-converging ops like xor (causing
// `test_reduction_single_i32[arch=amdgpu-5]` to hang) and is pathologically slow even for converging ops like fadd
// (causing `test_reduction_single_f32[arch=amdgpu-0]` to hang). `agent` scope is the largest scope a Quadrants kernel
// ever needs (one kernel = one device; the host only reads results after kernel completion, which forces a full cache
// flush regardless), and it lets the AMDGPU backend emit the native atomics. CUDA/NVPTX and CPU already lower
// System-scope seq_cst atomics to single-instruction hardware atomics, so they keep the System default. SPIR-V backends
// spell the same "Device" scope explicitly in `spirv_codegen.cpp`. See `docs/source/user_guide/atomics.md` for the
// per-backend scope summary.
//
// This helper is shared by two emit sites that must agree:
//   - `codegen_llvm.cpp` (JIT-time codegen for `qd.atomic_*` user ops)
//   - `llvm_context.cpp` (runtime-bitcode patcher that rewires `atomic_add_*` stubs in the runtime module, which the
//     AMDGPU/CUDA optimized-reduction path calls into via `reduce_add_f32` etc.)
//
// Internal runtime atomics that touch pinned host memory (e.g. the adstack overflow flag the host polls during kernel
// execution) deliberately do NOT use this helper; they need System scope for correctness.
inline llvm::SyncScope::ID kernel_atomic_syncscope(llvm::LLVMContext *ctx, Arch arch) {
  if (arch == Arch::amdgpu) {
    return ctx->getOrInsertSyncScopeID("agent");
  }
  return llvm::SyncScope::System;
}

}  // namespace quadrants::lang
