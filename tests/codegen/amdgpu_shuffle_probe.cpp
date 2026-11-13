// CPU-only code-generation probe for the AMDGPU shuffle refactor. This reads a real runtime_amdgpu.bc compiled from
// runtime.cpp, applies the shuffle selection/intrinsic patches, and emits two small kernels for inspection with
// clang --target=amdgcn-amd-amdhsa -mcpu=<target> -mwavefrontsize64 -nogpulib -O2 -S <output.bc>.
// Usage: amdgpu_shuffle_probe <runtime.bc> <output.bc> <gfx942|gfx1100> <before|after>
// The before mode reproduces main's mask/identity patches; after uses the production selection helper. This probe
// intentionally covers one full-wave and one native-swap target, not gfx10's LDS emulation or GPU execution.
#include "quadrants/runtime/llvm/amdgpu_shuffle.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"

int main(int argc, char **argv) {
  if (argc != 5) {
    llvm::errs() << "Usage: probe runtime.bc output.bc gfx942|gfx1100 before|after\n";
    return 1;
  }
  llvm::StringRef target(argv[3]);
  llvm::StringRef revision(argv[4]);
  if ((target != "gfx942" && target != "gfx1100") || (revision != "before" && revision != "after")) {
    llvm::errs() << "Unsupported target or revision\n";
    return 1;
  }
  llvm::LLVMContext context;
  auto input = llvm::MemoryBuffer::getFile(argv[1]);
  if (!input) {
    llvm::errs() << input.getError().message() << '\n';
    return 1;
  }
  auto parsed = llvm::parseBitcodeFile(input.get()->getMemBufferRef(), context);
  if (!parsed) {
    llvm::errs() << llvm::toString(parsed.takeError()) << '\n';
    return 1;
  }
  auto module = std::move(*parsed);
  auto *i32 = llvm::Type::getInt32Ty(context);
  bool full_wave = target == "gfx942";
  auto patch = [&](const char *name, auto emit) {
    auto *function = module->getFunction(name);
    if (!function) {
      llvm::report_fatal_error(llvm::Twine("Missing probe function: ") + name);
    }
    function->deleteBody();
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
    builder.CreateRet(emit(builder, function));
  };
  auto intrinsic = [&](const char *name, llvm::Intrinsic::ID id, bool overloaded = false) {
    patch(name, [&](llvm::IRBuilder<> &builder, llvm::Function *function) {
      auto *callee = overloaded ? llvm::Intrinsic::getOrInsertDeclaration(module.get(), id, {i32})
                                : llvm::Intrinsic::getOrInsertDeclaration(module.get(), id);
      llvm::SmallVector<llvm::Value *> args;
      for (auto &arg : function->args()) {
        args.push_back(&arg);
      }
      return builder.CreateCall(callee, args);
    });
  };
  if (revision == "after") {
    quadrants::lang::select_amdgpu_shuffle(*module, full_wave ? quadrants::lang::AMDGPUShuffleMode::Wave64
                                                              : quadrants::lang::AMDGPUShuffleMode::TwoHalves);
  } else {
    patch("amdgpu_ds_bpermute_lane_mask",
          [&](llvm::IRBuilder<> &builder, llvm::Function *) { return builder.getInt32(full_wave ? 63 : 31); });
  }
  intrinsic("amdgpu_ds_bpermute", llvm::Intrinsic::amdgcn_ds_bpermute);
  intrinsic("amdgpu_mbcnt_lo", llvm::Intrinsic::amdgcn_mbcnt_lo);
  intrinsic("amdgpu_mbcnt_hi", llvm::Intrinsic::amdgcn_mbcnt_hi);
  if (!full_wave) {
    intrinsic("amdgpu_permlane64", llvm::Intrinsic::amdgcn_permlane64, true);
  } else if (revision == "before") {
    patch("amdgpu_permlane64", [&](llvm::IRBuilder<> &, llvm::Function *function) { return function->getArg(0); });
  }
  for (auto &function : *module) {
    if (!function.isDeclaration()) {
      function.setLinkage(llvm::GlobalValue::InternalLinkage);
      function.removeFnAttr(llvm::Attribute::OptimizeNone);
      function.removeFnAttr(llvm::Attribute::NoInline);
      function.addFnAttr(llvm::Attribute::AlwaysInline);
    }
    function.removeFnAttr("target-cpu");
    function.removeFnAttr("target-features");
    function.removeFnAttr("tune-cpu");
  }
  for (auto &global : module->globals()) {
    if (!global.isDeclaration()) {
      global.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }
  for (bool constant_target : {false, true}) {
    auto *kernel_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), {llvm::PointerType::get(context, 1)}, false);
    auto *kernel = llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                                          constant_target ? "shuffle_constant" : "shuffle_dynamic", *module);
    kernel->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", kernel));
    auto *lane = builder.CreateCall(
        llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::amdgcn_workitem_id_x));
    auto *value = builder.CreateAdd(builder.CreateMul(lane, builder.getInt32(3)), builder.getInt32(1));
    llvm::Value *source = constant_target ? builder.getInt32(40) : builder.CreateXor(lane, builder.getInt32(32));
    auto *result = builder.CreateCall(module->getFunction("amdgpu_shuffle_i32"), {source, value});
    builder.CreateStore(result, builder.CreateGEP(i32, kernel->getArg(0), lane));
    builder.CreateRetVoid();
  }
  module->setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));
  module->setDataLayout("");
  if (llvm::verifyModule(*module, &llvm::errs())) {
    return 1;
  }
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager calls;
  llvm::ModuleAnalysisManager modules;
  llvm::PassBuilder passes;
  passes.registerLoopAnalyses(loops);
  passes.registerFunctionAnalyses(functions);
  passes.registerCGSCCAnalyses(calls);
  passes.registerModuleAnalyses(modules);
  passes.crossRegisterProxies(loops, functions, calls, modules);
  llvm::ModulePassManager pipeline;
  pipeline.addPass(llvm::GlobalDCEPass());
  pipeline.addPass(passes.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2));
  pipeline.addPass(llvm::GlobalDCEPass());
  pipeline.run(*module, modules);
  if (llvm::verifyModule(*module, &llvm::errs())) {
    return 1;
  }
  std::error_code error;
  llvm::raw_fd_ostream output(argv[2], error);
  if (error) {
    llvm::errs() << error.message() << '\n';
    return 1;
  }
  llvm::WriteBitcodeToFile(*module, output);
}
