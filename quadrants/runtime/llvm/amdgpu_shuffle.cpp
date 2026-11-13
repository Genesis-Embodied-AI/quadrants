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

// Bind the runtime entry point to one complete C++ implementation. Replacing all uses also updates the relative
// shuffle and type-conversion helpers. Remove the unused algorithm before any unsupported intrinsic is patched.
llvm::Function *select_amdgpu_shuffle(llvm::Module &module, AMDGPUShuffleMode mode) {
  auto *entry = module.getFunction("amdgpu_shuffle_i32");
  auto *wave64 = module.getFunction("amdgpu_shuffle_wave64_i32");
  auto *two_halves = module.getFunction("amdgpu_shuffle_two_halves_i32");
  if (!entry || !wave64 || !two_halves) {
    llvm::report_fatal_error("Missing AMDGPU shuffle runtime implementation");
  }
  auto *selected = mode == AMDGPUShuffleMode::Wave64 ? wave64 : two_halves;
  auto *unused = mode == AMDGPUShuffleMode::Wave64 ? two_halves : wave64;
  entry->replaceAllUsesWith(selected);
  entry->eraseFromParent();
  selected->setName("amdgpu_shuffle_i32");
  unused->eraseFromParent();

  if (auto *swap = module.getFunction("amdgpu_permlane64"); swap && swap->use_empty()) {
    swap->eraseFromParent();
  }
  return selected;
}

void prepare_amdgpu_shuffle(llvm::Module &module, llvm::StringRef mcpu) {
  patch_shuffle_intrinsic(module, "amdgpu_ds_bpermute", llvm::Intrinsic::amdgcn_ds_bpermute);
  // Preserve the current routing. The gfx12 full-wave correction is separate (PR #769).
  bool has_wave64_bpermute = mcpu.substr(0, 4) == "gfx9";
  bool has_native_permlane64 = mcpu.substr(0, 5) == "gfx11" || mcpu.substr(0, 5) == "gfx12";
  auto mode = has_wave64_bpermute ? AMDGPUShuffleMode::Wave64 : AMDGPUShuffleMode::TwoHalves;
  inline_shuffle(select_amdgpu_shuffle(module, mode));
  if (has_native_permlane64) {
    patch_shuffle_intrinsic(module, "amdgpu_permlane64", llvm::Intrinsic::amdgcn_permlane64, true);
  } else if (auto *swap = module.getFunction("amdgpu_permlane64")) {
    emulate_half_swap(module, swap);
  }
}

}  // namespace quadrants::lang
