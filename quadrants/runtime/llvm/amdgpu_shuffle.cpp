#include "quadrants/runtime/llvm/amdgpu_shuffle.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

namespace quadrants::lang {
namespace {

// These shuffle implementations must be expanded into the GPU kernel, not called out of line.
void inline_shuffle(llvm::Function *function) {
  function->removeFnAttr(llvm::Attribute::OptimizeNone);
  function->removeFnAttr(llvm::Attribute::NoInline);
  function->addFnAttr(llvm::Attribute::AlwaysInline);
}

void patch_shuffle_intrinsic(llvm::Module &module,
                             const char *name,
                             llvm::Intrinsic::ID intrinsic,
                             bool overloaded = false) {
  auto *function = module.getFunction(name);
  if (!function) {
    return;
  }
  auto &context = module.getContext();
  function->deleteBody();
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
  llvm::SmallVector<llvm::Value *> args;
  for (auto &arg : function->args()) {
    args.push_back(&arg);
  }
  auto *callee = overloaded ? llvm::Intrinsic::getOrInsertDeclaration(&module, intrinsic, {builder.getInt32Ty()})
                            : llvm::Intrinsic::getOrInsertDeclaration(&module, intrinsic);
  builder.CreateRet(builder.CreateCall(callee, args));
  inline_shuffle(function);
}

void emulate_half_swap(llvm::Module &module, llvm::Function *permlane64_func) {
  auto *ctx = &module.getContext();
  // gfx10.x RDNA1/2: LDS-based software emulation. Layout: ``[1024 x i32] addrspace(3)`` indexed by ``wave_base +
  // lane``.
  auto i32_ty = llvm::Type::getInt32Ty(*ctx);
  auto buf_ty = llvm::ArrayType::get(i32_ty, 1024);
  auto lds_global = llvm::cast_or_null<llvm::GlobalVariable>(module.getNamedValue("__amdgpu_permlane64_lds"));
  if (!lds_global) {
    lds_global =
        new llvm::GlobalVariable(module, buf_ty, /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
                                 llvm::UndefValue::get(buf_ty), "__amdgpu_permlane64_lds",
                                 /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal, /*AddressSpace=*/3);
    lds_global->setAlignment(llvm::Align(4));
  }

  permlane64_func->deleteBody();
  auto bb = llvm::BasicBlock::Create(*ctx, "entry", permlane64_func);
  llvm::IRBuilder<> builder(*ctx);
  builder.SetInsertPoint(bb);

  // lane = mbcnt_hi(-1, mbcnt_lo(-1, 0))
  auto neg_one = llvm::ConstantInt::get(i32_ty, -1, /*IsSigned=*/true);
  auto zero32 = llvm::ConstantInt::get(i32_ty, 0);
  auto mbcnt_lo_fn = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::amdgcn_mbcnt_lo);
  auto mbcnt_hi_fn = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::amdgcn_mbcnt_hi);
  llvm::Value *mbcnt_lo_args[] = {neg_one, zero32};
  auto lane_lo = builder.CreateCall(mbcnt_lo_fn, llvm::ArrayRef<llvm::Value *>(mbcnt_lo_args));
  llvm::Value *mbcnt_hi_args[] = {neg_one, lane_lo};
  auto lane = builder.CreateCall(mbcnt_hi_fn, llvm::ArrayRef<llvm::Value *>(mbcnt_hi_args));

  // wave_base = (workitem.id.x >> 6) << 6 (wave64-scoped slot offset within the workgroup LDS buffer).
  auto tid_fn = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::amdgcn_workitem_id_x);
  auto tid = builder.CreateCall(tid_fn);
  auto wave_id = builder.CreateLShr(tid, llvm::ConstantInt::get(i32_ty, 6));
  auto wave_base = builder.CreateShl(wave_id, llvm::ConstantInt::get(i32_ty, 6));
  auto slot = builder.CreateAdd(wave_base, lane);

  // lds[slot] = value
  auto value_arg = &*permlane64_func->arg_begin();
  llvm::Value *store_idxs[] = {zero32, slot};
  auto store_ptr = builder.CreateInBoundsGEP(buf_ty, lds_global, llvm::ArrayRef<llvm::Value *>(store_idxs));
  builder.CreateStore(value_arg, store_ptr);

  // Wavefront-scope acquire-release fence -> ``s_waitcnt lgkmcnt(0)`` on AMDGPU; orders LDS writes against
  // subsequent LDS reads within the wave without touching cross-wave state (avoids the ``s_barrier`` that a
  // workgroup-scope fence would emit, which deadlocks if only some waves in the workgroup reach this point).
  llvm::SyncScope::ID wave_scope = ctx->getOrInsertSyncScopeID("wavefront");
  builder.CreateFence(llvm::AtomicOrdering::AcquireRelease, wave_scope);

  // partner_lane = lane ^ 32; result = lds[wave_base + partner_lane]
  auto partner_lane = builder.CreateXor(lane, llvm::ConstantInt::get(i32_ty, 32));
  auto partner_slot = builder.CreateAdd(wave_base, partner_lane);
  llvm::Value *load_idxs[] = {zero32, partner_slot};
  auto load_ptr = builder.CreateInBoundsGEP(buf_ty, lds_global, llvm::ArrayRef<llvm::Value *>(load_idxs));
  auto result = builder.CreateAlignedLoad(i32_ty, load_ptr, llvm::Align(4));
  builder.CreateRet(result);
  inline_shuffle(permlane64_func);
}

}  // namespace

void prepare_amdgpu_shuffle(llvm::Module &module, llvm::StringRef mcpu) {
  auto *ctx = &module.getContext();
  patch_shuffle_intrinsic(module, "amdgpu_ds_bpermute", llvm::Intrinsic::amdgcn_ds_bpermute);
  // The wave64 cross-half subgroup shuffle (``amdgpu_shuffle_i32`` in runtime.cpp) is built from ``ds_bpermute``
  // plus, on some targets, ``permlane64``. How those behave -- and therefore how we patch them -- depends on the
  // architecture family:
  //
  //   * GCN / CDNA (gfx9xx: gfx900/906 Vega, gfx908/gfx90a CDNA1/2, gfx940/gfx941/gfx942 CDNA3, gfx950 CDNA4):
  //     ``ds_bpermute`` addresses the full wave64 directly, so the cross-half shuffle is a single wide
  //     ``ds_bpermute`` (lane mask 63) and ``permlane64`` is unnecessary. Critically, ``v_permlane64_b32`` does
  //     not exist on CDNA -- emitting ``llvm.amdgcn.permlane64`` makes the backend select the ``V_PERMLANE64_B32``
  //     pseudo, which has no valid MC opcode for CDNA, and then crash with a bare SIGSEGV inside
  //     ``SIInstrInfo::getInstSizeInBytes`` during branch relaxation (genesis-world #2962). So on CDNA we patch
  //     ``amdgpu_permlane64`` to the identity, which neutralises the helper's (RDNA-shaped) cross-SIMD branch
  //     without ever emitting the intrinsic.
  //   * RDNA3/4 (gfx11 / gfx12): ``ds_bpermute`` is SIMD32-scoped (lane mask 31), so the top half is reached via
  //     the native single-instruction ``v_permlane64_b32``.
  //   * RDNA1/2 (gfx10.x): ``ds_bpermute`` is SIMD32-scoped (lane mask 31), but ``v_permlane64_b32`` does not
  //     exist yet, so we emulate the lane ``i`` <-> ``i ^ 32`` swap with an LDS roundtrip (below).
  //
  // The LDS emulation writes each lane's ``value`` to ``lds[wave_base + lane]``, issues a wavefront-scope
  // acquire-release fence (lowers to ``s_waitcnt lgkmcnt(0)``: drains outstanding LDS writes without the
  // cross-wave ``s_barrier`` a workgroup-scope fence would emit, which would deadlock if only some waves reach
  // this point), then reads back ``lds[wave_base + (lane ^ 32)]``. ``wave_base`` is ``(workitem.id.x >> 6) << 6``,
  // scoping the slot to a single wave so multi-wave workgroups don't collide. The buffer is a 1024-entry
  // per-workgroup global (4 KiB, the AMDGPU 1024-thread wave64 max), materialised only on this path, so kernels
  // on the other two paths pay zero LDS for cross-half shuffles.
  //
  // ``patch_intrinsic`` for permlane64 passes the explicit ``i32`` type alongside the ID because the intrinsic is
  // overloaded on its element type (signature ``T -> T`` for any 32-bit-or-smaller ``T``); otherwise
  // ``CreateIntrinsic`` segfaults inside ``getDeclaration()`` while resolving the mangled name.
  auto mcpu_str = mcpu;
  bool is_gcn_cdna = (mcpu_str.substr(0, 4) == "gfx9");
  bool has_native_permlane64 = (mcpu_str.substr(0, 5) == "gfx11" || mcpu_str.substr(0, 5) == "gfx12");

  // Patch the ds_bpermute lane mask used by ``amdgpu_shuffle_i32``: 63 where ``ds_bpermute`` is wave64-wide
  // (GCN/CDNA), 31 where it is SIMD32-scoped (RDNA, paired with the permlane64 swap above).
  if (auto mask_func = module.getFunction("amdgpu_ds_bpermute_lane_mask")) {
    mask_func->deleteBody();
    auto bb = llvm::BasicBlock::Create(*ctx, "entry", mask_func);
    llvm::IRBuilder<> builder(*ctx);
    builder.SetInsertPoint(bb);
    builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), is_gcn_cdna ? 63 : 31));
    inline_shuffle(mask_func);
  }

  if (has_native_permlane64) {
    patch_shuffle_intrinsic(module, "amdgpu_permlane64", llvm::Intrinsic::amdgcn_permlane64, true);
  } else if (is_gcn_cdna) {
    // CDNA: the wide ds_bpermute already reaches all 64 lanes, so permlane64 is unnecessary -- and emitting it
    // crashes the backend. Patch it to the identity so the helper's cross-SIMD branch returns the same value as
    // its same-SIMD branch, making the per-lane select a true no-op.
    if (auto permlane64_func = module.getFunction("amdgpu_permlane64")) {
      permlane64_func->deleteBody();
      auto bb = llvm::BasicBlock::Create(*ctx, "entry", permlane64_func);
      llvm::IRBuilder<> builder(*ctx);
      builder.SetInsertPoint(bb);
      builder.CreateRet(&*permlane64_func->arg_begin());
      inline_shuffle(permlane64_func);
    }
  } else if (auto permlane64_func = module.getFunction("amdgpu_permlane64")) {
    emulate_half_swap(module, permlane64_func);
  }
}

}  // namespace quadrants::lang
