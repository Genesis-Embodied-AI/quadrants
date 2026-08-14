#include "quadrants/runtime/amdgpu/jit_amdgpu.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/runtime/llvm/llvm_context_pass.h"

#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <fstream>
#include <cstdlib>

namespace quadrants {
namespace lang {
#if defined(QD_WITH_AMDGPU)
JITModule *JITSessionAMDGPU ::add_module(std::unique_ptr<llvm::Module> M, int max_reg) {
  auto hsaco = compile_module_to_hsaco(M);
  QD_TRACE("hsaco size: {:.2f}KB", hsaco.size() / 1024.0);

  void *amdgpu_module;
  auto t = Time::get_time();
  AMDGPUDriver::get_instance().module_load_data(&amdgpu_module, hsaco.c_str());
  QD_TRACE("AMDGPU load data from module time : {}ms", (Time::get_time() - t) * 1000);
  modules.push_back(std::make_unique<JITModuleAMDGPU>(amdgpu_module));
  return modules.back().get();
}

std::string JITSessionAMDGPU::compile_module_to_hsaco(std::unique_ptr<llvm::Module> &llvm_module) {
  llvm::legacy::FunctionPassManager function_pass_manager_addrcast(llvm_module.get());
  function_pass_manager_addrcast.add(new AMDGPUConvertAllocaInstAddressSpacePass());
  function_pass_manager_addrcast.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    function_pass_manager_addrcast.run(*func);
  function_pass_manager_addrcast.doFinalization();

  for (auto &F : *llvm_module) {
    // Match CUDA parity: jit_cuda.cpp:332-335 applies unsafe-fp-math when
    // fast_math is enabled. Applied to all functions (not just kernels) because
    // device body functions contain the actual FP compute. Gated on
    // config_.fast_math so that qd.init(fast_math=False) retains IEEE semantics
    // on AMDGPU the same way it does on CUDA and CPU.
    if (this->config_.fast_math) {
      F.addFnAttr("unsafe-fp-math", "true");
      F.addFnAttr("no-signed-zeros-fp-math", "true");
    }

    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      const std::string kernel_name = F.getName().str();
      const bool is_lightweight_cg_subkernel =
          kernel_name.find("_kernel_cg_only_save_prev_grad") !=
              std::string::npos ||
          kernel_name.find("_kernel_update_constraint_forces") !=
              std::string::npos ||
          kernel_name.find("_kernel_update_constraint_qfrc") !=
              std::string::npos ||
          kernel_name.find("_kernel_update_constraint_cost") !=
              std::string::npos ||
          kernel_name.find("_kernel_update_search_direction") !=
              std::string::npos;

      // Each default below is skipped if the kernel already carries that
      // attribute (set upstream in codegen_llvm.cpp from user-supplied
      // @qd.kernel(fn_attrs={...})). User values win.
      if (!is_lightweight_cg_subkernel &&
          !F.hasFnAttribute("amdgpu-waves-per-eu")) {
        F.addFnAttr("amdgpu-waves-per-eu", "1,2");
      }
      if (!F.hasFnAttribute("uniform-work-group-size")) {
        F.addFnAttr("uniform-work-group-size", "true");
      }
      if (!F.hasFnAttribute("amdgpu-ieee")) {
        F.addFnAttr("amdgpu-ieee", "false");
      }
      if (!F.hasFnAttribute("amdgpu-dx10-clamp")) {
        F.addFnAttr("amdgpu-dx10-clamp", "false");
      }
    }
  }

  for (auto &F : *llvm_module) {
    if (F.isDeclaration() || F.empty())
      continue;
    if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    if (F.hasFnAttribute("amdgpu-flat-work-group-size"))
      continue;  // already set (e.g., on runtime kernels via
                 // mark_function_as_amdgpu_kernel-equivalent paths)
    llvm::StringRef inherited;
    for (auto *U : F.users()) {
      auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
      if (!CB)
        continue;
      // Direct call only — function-pointer args (e.g., body fn passed
      // as `RangeForTaskFunc *func` to gpu_parallel_range_for) are
      // skipped because the use is the function pointer itself, not a
      // call to it. `alwaysinline` on gpu_parallel_range_for
      // collapses the function-pointer indirection so the body fn ends
      // up with direct callers in the kernel entry.
      if (CB->getCalledOperand() != &F)
        continue;
      auto *Caller = CB->getFunction();
      if (Caller && Caller->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL &&
          Caller->hasFnAttribute("amdgpu-flat-work-group-size")) {
        inherited =
            Caller->getFnAttribute("amdgpu-flat-work-group-size").getValueAsString();
        break;
      }
    }
    if (inherited.empty())
      inherited = "1,128";  // conservative fallback
    F.addFnAttr("amdgpu-flat-work-group-size", inherited);
  }

  auto *daz_type = llvm::Type::getInt8Ty(llvm_module->getContext());
  auto *daz_init = llvm::ConstantInt::get(daz_type, 1);
  auto *daz_var = new llvm::GlobalVariable(
      *llvm_module, daz_type, true, llvm::GlobalValue::LinkOnceODRLinkage,
      daz_init, "__oclc_daz_opt");
  daz_var->setVisibility(llvm::GlobalValue::HiddenVisibility);


  if (llvm::verifyModule(*llvm_module, &llvm::errs())) {
    llvm_module->print(llvm::errs(), nullptr);
    QD_WARN("Module broken");
  }
  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("quadrants_kernel_amdgpu_llvm_ir_{:04d}.ll", "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }
  auto triple_str = llvm_module->getTargetTriple();
  std::string error_str;
  auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);

  llvm::TargetOptions options;
  options.MCOptions.AsmVerbose = false;
  if (this->config_.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // UnsafeFPMath was removed in LLVM 22; set the individual flags it implied
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
    options.NoSignedZerosFPMath = 1;
    options.NoTrappingFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
    options.NoSignedZerosFPMath = 0;
    options.NoTrappingFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;

  // Force wave64 codegen at the TargetMachine level. Belt-and-suspenders with the per-function target-features
  // attribute set in llvm_context_pass.h: TargetMachine features supply the default when the IR doesn't pin one, and
  // per-function attrs override per call. Both are needed because alloca-pass-cleared functions and freshly created
  // kernel wrappers each take a different code path. Required so the same wave64 runtime works on RDNA (gfx10+) hosts
  // in addition to CDNA.
  const char *kAmdgpuFeatures = "+wavefrontsize64,-wavefrontsize32";
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple_str, AMDGPUContext::get_instance().get_mcpu(), kAmdgpuFeatures, options,
                                  llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOptLevel::Aggressive));

  llvm_module->setDataLayout(machine->createDataLayout());

  if (this->config_.print_kernel_amdgcn) {
    // Amdgcn will not generated during generating hsaco file
    // It's an interim impl
    // while add machine info to pass_manager, the module(LLVM-IR) will add more
    // target-specific info e.g.
    //   call { i1, i32 } @llvm.amdgcn.if.i32(i1 %15)
    // then then `addPassesToEmitFile` will occur an error
    //   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.if
    // related https://github.com/llvm/llvm-project/issues/60727
    //    we can't though the `addPassesToEmitFile` to generate GCN file
    //    directly
    // another way
    //    llvm-objdump -d xxxx.hsaco(can ensure that hsaco and gcn correspond to
    //    each other)

    auto module_clone = llvm::CloneModule(*llvm_module);
    llvm::legacy::PassManager module_gen_gcn_pass_manager;
    llvm::SmallString<0> gcnstr;
    llvm::raw_svector_ostream llvm_stream_gcn(gcnstr);
    std::unique_ptr<llvm::TargetMachine> machine_gen_gcn(
        target->createTargetMachine(triple_str, AMDGPUContext::get_instance().get_mcpu(), kAmdgpuFeatures, options,
                                    llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOptLevel::Aggressive));

    // Replace PassManagerBuilder with PassBuilder API
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PassBuilder pb(machine_gen_gcn.get());
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(*module_clone, mam);

    module_gen_gcn_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine_gen_gcn->getTargetIRAnalysis()));
    machine_gen_gcn->addPassesToEmitFile(module_gen_gcn_pass_manager, llvm_stream_gcn, nullptr,
                                         llvm::CodeGenFileType::AssemblyFile, true);
    module_gen_gcn_pass_manager.run(*module_clone);
    std::string gcn(gcnstr.begin(), gcnstr.end());
    static FileSequenceWriter writer("quadrants_kernel_amdgcn_{:04d}.gcn", "module AMDGCN");
    writer.write(gcn);
  }

  // Replace the main optimization pipeline (lines 114-127)
  llvm::legacy::FunctionPassManager function_pass_manager(llvm_module.get());
  llvm::legacy::PassManager module_pass_manager;

  // Use new PassBuilder API for optimizations
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassBuilder pb(machine.get());
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // Run the new optimization pipeline
  mpm.run(*llvm_module, mam);

  // Keep legacy PassManager for backend code generation
  module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

  machine->Options.MCOptions.AsmVerbose = true;

  auto tmp_dir = get_tmp_dir();
  uint64 random_num = get_random_num();

  auto obj_filename = "quadrants_amdgcn_" + std::to_string(random_num) + ".o";
  auto hsaco_filename = "quadrants_amdgcn_" + std::to_string(random_num) + ".hsaco";
  auto obj_path = tmp_dir + obj_filename;
  auto hsaco_path = tmp_dir + hsaco_filename;
  std::error_code ec;

  llvm::SmallString<0> outstr;
  llvm::raw_svector_ostream llvm_stream(outstr);

  machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr, llvm::CodeGenFileType::ObjectFile, true);

  function_pass_manager.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    function_pass_manager.run(*func);
  function_pass_manager.doFinalization();
  module_pass_manager.run(*llvm_module);

  std::string obj_str(outstr.begin(), outstr.end());
  std::ofstream(obj_path) << obj_str;

  QD_TRACE("Loading module...");
  [[maybe_unused]] auto _ = AMDGPUContext::get_instance().get_lock_guard();

  // Try to find ld.lld from ROCm installation, fallback to system PATH
  std::string lld_executable = "ld.lld";
  const char *rocm_path = std::getenv("ROCM_PATH");
  if (rocm_path) {
    std::string rocm_lld = std::string(rocm_path) + "/llvm/bin/ld.lld";
    std::ifstream test_lld(rocm_lld);
    if (test_lld.good()) {
      lld_executable = rocm_lld;
    }
  }
  // Also try common ROCm installation paths
  if (lld_executable == "ld.lld") {
    std::vector<std::string> common_paths = {
        "/opt/rocm/llvm/bin/ld.lld",
        "/opt/rocm-7.0.0/llvm/bin/ld.lld",
        "/opt/rocm-6.0.0/llvm/bin/ld.lld",
    };
    for (const auto &path : common_paths) {
      std::ifstream test_lld(path);
      if (test_lld.good()) {
        lld_executable = path;
        break;
      }
    }
  }

  std::string lld_cmd = lld_executable + " -shared " + obj_path + " -o " + hsaco_path;
  QD_TRACE("Linking with command: {}", lld_cmd);
  if (std::system(lld_cmd.c_str()))
    QD_ERROR(
        fmt::format("Generate {} Error. Make sure ld.lld is available in your "
                    "ROCm installation, and add path to ROCm path to ROCM_PATH "
                    "if necessary.",
                    hsaco_filename));

  std::string hsaco_str = load_hsaco(hsaco_path);

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer("quadrants_kernel_amdgpu_llvm_ir_optimized_{:04d}.ll",
                                     "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }

  return hsaco_str;
}

std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(QuadrantsLLVMContext *tlctx,
                                                           const CompileConfig &config,
                                                           Arch arch) {
  QD_ASSERT(arch == Arch::amdgpu);
  auto data_layout = QuadrantsLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionAMDGPU>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(QuadrantsLLVMContext *tlctx,
                                                           const CompileConfig &config,
                                                           Arch arch) {
  QD_NOT_IMPLEMENTED
}
#endif

}  // namespace lang
}  // namespace quadrants
