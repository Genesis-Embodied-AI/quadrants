#include "quadrants/runtime/amdgpu/jit_amdgpu.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/runtime/llvm/llvm_context_pass.h"

#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <fstream>
#include <cstdlib>

namespace quadrants {
namespace lang {
#if defined(TI_WITH_AMDGPU)
JITModule *JITSessionAMDGPU ::add_module(std::unique_ptr<llvm::Module> M,
                                         int max_reg) {
  auto hsaco = compile_module_to_hsaco(M);
  TI_TRACE("hsaco size: {:.2f}KB", hsaco.size() / 1024.0);

  void *amdgpu_module;
  auto t = Time::get_time();
  AMDGPUDriver::get_instance().module_load_data(&amdgpu_module, hsaco.c_str());
  TI_TRACE("AMDGPU load data from module time : {}ms",
           (Time::get_time() - t) * 1000);
  modules.push_back(std::make_unique<JITModuleAMDGPU>(amdgpu_module));
  return modules.back().get();
}

std::string JITSessionAMDGPU::compile_module_to_hsaco(
    std::unique_ptr<llvm::Module> &llvm_module) {
  llvm::legacy::FunctionPassManager function_pass_manager_addrcast(
      llvm_module.get());
  function_pass_manager_addrcast.add(
      new AMDGPUConvertFunctionBodyAllocsAddressSpacePass());
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    if (func->getName() == "function_body")
      function_pass_manager_addrcast.run(*func);

  if (llvm::verifyModule(*llvm_module, &llvm::errs())) {
    llvm_module->print(llvm::errs(), nullptr);
    TI_WARN("Module broken");
  }
  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer(
        "quadrants_kernel_amdgpu_llvm_ir_{:04d}.ll",
        "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }
  auto triple_str = llvm_module->getTargetTriple();
  std::string error_str;
  auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);

  llvm::TargetOptions options;
  options.MCOptions.AsmVerbose = false;
  if (this->config_.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      triple_str, AMDGPUContext::get_instance().get_mcpu(), "", options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small,
      llvm::CodeGenOptLevel::Aggressive));

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
        target->createTargetMachine(
            triple_str, AMDGPUContext::get_instance().get_mcpu(), "", options,
            llvm::Reloc::PIC_, llvm::CodeModel::Small,
            llvm::CodeGenOptLevel::Aggressive));

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

    llvm::ModulePassManager mpm =
        pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(*module_clone, mam);

    module_gen_gcn_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
        machine_gen_gcn->getTargetIRAnalysis()));
    machine_gen_gcn->addPassesToEmitFile(
        module_gen_gcn_pass_manager, llvm_stream_gcn, nullptr,
        llvm::CodeGenFileType::AssemblyFile, true);
    module_gen_gcn_pass_manager.run(*module_clone);
    std::string gcn(gcnstr.begin(), gcnstr.end());
    static FileSequenceWriter writer("quadrants_kernel_amdgcn_{:04d}.gcn",
                                     "module AMDGCN");
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

  llvm::ModulePassManager mpm =
      pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // Run the new optimization pipeline
  mpm.run(*llvm_module, mam);

  // Keep legacy PassManager for backend code generation
  module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
      machine->getTargetIRAnalysis()));

  machine->Options.MCOptions.AsmVerbose = true;

  auto tmp_dir = get_tmp_dir();
  uint64 random_num = get_random_num();

  auto obj_filename = "quadrants_amdgcn_" + std::to_string(random_num) + ".o";
  auto hsaco_filename =
      "quadrants_amdgcn_" + std::to_string(random_num) + ".hsaco";
  auto obj_path = tmp_dir + obj_filename;
  auto hsaco_path = tmp_dir + hsaco_filename;
  std::error_code ec;

  llvm::SmallString<0> outstr;
  llvm::raw_svector_ostream llvm_stream(outstr);

  machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr,
                               llvm::CodeGenFileType::ObjectFile, true);

  function_pass_manager.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    function_pass_manager.run(*func);
  function_pass_manager.doFinalization();
  module_pass_manager.run(*llvm_module);

  std::string obj_str(outstr.begin(), outstr.end());
  std::ofstream(obj_path) << obj_str;

  TI_TRACE("Loading module...");
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

  std::string lld_cmd =
      lld_executable + " -shared " + obj_path + " -o " + hsaco_path;
  TI_TRACE("Linking with command: {}", lld_cmd);
  if (std::system(lld_cmd.c_str()))
    TI_ERROR(
        fmt::format("Generate {} Error. Make sure ld.lld is available in your "
                    "ROCm installation, and add path to ROCm path to ROCM_PATH "
                    "if necessary.",
                    hsaco_filename));

  std::string hsaco_str = load_hsaco(hsaco_path);

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "quadrants_kernel_amdgpu_llvm_ir_optimized_{:04d}.ll",
        "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }

  return hsaco_str;
}

std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    QuadrantsLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_ASSERT(arch == Arch::amdgpu);
  auto data_layout = QuadrantsLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionAMDGPU>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    QuadrantsLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

}  // namespace lang
}  // namespace quadrants
