#ifdef QD_WITH_LLVM
// GPU-side `qd.checkpoint` gating prologue for the LLVM-IR-generating backends (CUDA / AMDGPU).
//
// Lives in its own translation unit so `codegen_llvm.cpp` doesn't have to grow per-feature.
// Shared by `codegen_cuda.cpp` (pre-Hopper flat-graph path and the in-flight-yield safety net on
// SM 9.0+) and `codegen_amdgpu.cpp` (flat HIP graph path; no conditional-graph-node API).
//
// See `docs/source/user_guide/graph.md` for the user-facing surface and
// `perso_hugh/doc/qipc/reentrant.md` for the design.

#include "quadrants/codegen/llvm/codegen_llvm.h"

namespace quadrants::lang {

// Emits the LLVM-IR equivalent of:
//
//   int32_t *rp = runtime->checkpoint_resume_point_ptr;
//   if (rp != nullptr && *rp > cp_id) goto final_block;
//   int32_t *ys = runtime->checkpoint_yield_signal_ptr;
//   if (ys != nullptr && *ys != -1) goto final_block;
//
// at the top of `func_body_bb` (after `init_offloaded_task_function` opens it). The two
// null-checks let the prologue stay inert for callers that don't populate the
// RuntimeContext slots (e.g. the non-graph CUDA / AMDGPU launch paths that land on the same
// compiled task functions). When the slots are populated (the cached-graph path -- both pre-
// Hopper and SM 9.0+ on CUDA, and the flat HIP graph path on AMDGPU), the check skips the
// body whenever its cp_id is below the current resume_point or some earlier checkpoint in
// this launch already set yield_signal. The `final_block` jump matches the existing early-
// return pattern used by `cpu_assert_failed` gating so all the per-task cleanup (epilogue +
// return) runs unchanged.
//
// On CUDA SM 9.0+ the conditional-graph-node gate prevents the body kernel from launching
// when the checkpoint should be skipped, so the prologue is dead code in steady state. It
// stays in for correctness on the overlapping case where a yield-check kernel earlier in
// the same launched graph set yield_signal between the conditional gate's evaluation and
// the body's execution (the conditional gate captured the value of resume_point at the
// moment the gate kernel ran; if a yield-check kernel later in the graph fires, the body's
// dispatch is still in-flight and the conditional-node skip semantics cover the rest of
// the launch but not already-in-flight bodies). On pre-Hopper CUDA and on AMDGPU (which
// also lacks conditional graph nodes) the prologue *is* the gating mechanism.
//
// The CPU x64 codegen does not call this -- the host launcher does branch gating before
// launch. The GFX (Vulkan / Metal) codegen path also does not call this -- those use indirect
// dispatch with a SPIR-V gate shader that writes per-kernel `dim3` buffers, not a per-thread
// early-return inside the body kernel.
void TaskCodeGenLLVM::emit_checkpoint_gate_prologue(int cp_id) {
  auto *runtime_context_type = get_runtime_type("RuntimeContext");
  // Field indices into RuntimeContext: 0=arg_buffer, 1=runtime, 2=cpu_thread_id,
  // 3=result_buffer, 4=cpu_assert_failed, 5=checkpoint_resume_point_ptr,
  // 6=checkpoint_yield_signal_ptr. The runtime LLVM bitcode is rebuilt whenever
  // `context.h` changes; the indices here must mirror the field order in that struct.
  constexpr unsigned kFieldCheckpointResumePoint = 5;
  constexpr unsigned kFieldCheckpointYieldSignal = 6;
  auto *i32_ty = llvm::Type::getInt32Ty(*llvm_context);
  auto *i32_ptr_ty = llvm::PointerType::get(i32_ty, 0);
  auto *zero_i32 = tlctx->get_constant(0);

  auto load_rt_ctx_ptr = [&](unsigned field_idx) -> llvm::Value * {
    auto *field_ptr =
        builder->CreateGEP(runtime_context_type, get_context(), {zero_i32, tlctx->get_constant((int)field_idx)});
    // The field is `int32_t*`; the GEP'd pointer is `int32_t**`. Cast and load.
    auto *field_ptr_ptr = builder->CreatePointerCast(field_ptr, llvm::PointerType::get(i32_ptr_ty, 0));
    return builder->CreateLoad(i32_ptr_ty, field_ptr_ptr);
  };

  auto *cp_id_const = tlctx->get_constant(cp_id);
  auto *neg_one = tlctx->get_constant(-1);

  auto *check_yield_bb = llvm::BasicBlock::Create(*llvm_context, "qd_ckpt_check_yield", func);
  auto *body_bb = llvm::BasicBlock::Create(*llvm_context, "qd_ckpt_body", func);
  auto *skip_bb = llvm::BasicBlock::Create(*llvm_context, "qd_ckpt_skip", func);

  // Stage 1: resume_point check. Null pointer -> non-gating launcher; treat as "no skip".
  {
    auto *rp_ptr = load_rt_ctx_ptr(kFieldCheckpointResumePoint);
    auto *rp_is_null = builder->CreateICmpEQ(rp_ptr, llvm::ConstantPointerNull::get(i32_ptr_ty));
    auto *rp_load_bb = llvm::BasicBlock::Create(*llvm_context, "qd_ckpt_rp_load", func);
    builder->CreateCondBr(rp_is_null, check_yield_bb, rp_load_bb);
    builder->SetInsertPoint(rp_load_bb);
    auto *rp = builder->CreateLoad(i32_ty, rp_ptr);
    auto *should_skip_rp = builder->CreateICmpSGT(rp, cp_id_const);
    builder->CreateCondBr(should_skip_rp, skip_bb, check_yield_bb);
  }

  // Stage 2: yield_signal check. Null pointer -> treat as "no skip" (kernel has cp >= 0 but
  // no yield_on= on any checkpoint, so the launcher only populated resume_point).
  builder->SetInsertPoint(check_yield_bb);
  {
    auto *ys_ptr = load_rt_ctx_ptr(kFieldCheckpointYieldSignal);
    auto *ys_is_null = builder->CreateICmpEQ(ys_ptr, llvm::ConstantPointerNull::get(i32_ptr_ty));
    auto *ys_load_bb = llvm::BasicBlock::Create(*llvm_context, "qd_ckpt_ys_load", func);
    builder->CreateCondBr(ys_is_null, body_bb, ys_load_bb);
    builder->SetInsertPoint(ys_load_bb);
    auto *ys = builder->CreateLoad(i32_ty, ys_ptr);
    auto *should_skip_ys = builder->CreateICmpNE(ys, neg_one);
    builder->CreateCondBr(should_skip_ys, skip_bb, body_bb);
  }

  builder->SetInsertPoint(skip_bb);
  builder->CreateBr(final_block);

  // Pick up codegen in the body block. Existing task-body codegen will insert into the
  // current builder insertion point; pointing it at `body_bb` continues that flow as if
  // the prologue never inserted itself.
  builder->SetInsertPoint(body_bb);
}

}  // namespace quadrants::lang

#endif  // QD_WITH_LLVM
