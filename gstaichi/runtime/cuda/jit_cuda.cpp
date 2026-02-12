#include "gstaichi/runtime/cuda/jit_cuda.h"
#include "gstaichi/runtime/llvm/llvm_context.h"
#include "gstaichi/codegen/ir_dump.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Utils.h"

namespace gstaichi::lang {

#if defined(TI_WITH_CUDA)

bool module_has_runtime_initialize(
    const llvm::Module::FunctionListType &function_list) {
  for (auto &func : function_list) {
    if (func.getName() == "runtime_initialize") {
      return true;
    }
  }
  return false;
}

std::string moduleToDumpName(llvm::Module *const M) {
  const auto &function_list = M->getFunctionList();
  if (!function_list.empty() && !module_has_runtime_initialize(function_list)) {
    return function_list.front().getName().str();
  }
  return M->getName().str();
}

JITModuleCUDA::JITModuleCUDA(void *module) : module_(module) {
}

void *JITModuleCUDA::lookup_function(const std::string &name) {
  // TODO: figure out why using the guard leads to wrong tests results
  // auto context_guard = CUDAContext::get_instance().get_guard();
  CUDAContext::get_instance().make_current();
  void *func = nullptr;
  auto t = Time::get_time();
  auto err = CUDADriver::get_instance().module_get_function.call_with_warning(
      &func, module_, name.c_str());
  if (err) {
    TI_ERROR("Cannot look up function {}", name);
  }
  t = Time::get_time() - t;
  TI_TRACE("CUDA module_get_function {} costs {} ms", name, t * 1000);
  TI_ASSERT(func != nullptr);
  return func;
}

void JITModuleCUDA::call(const std::string &name,
                         const std::vector<void *> &arg_pointers,
                         const std::vector<int> &arg_sizes) {
  launch(name, 1, 1, 0, arg_pointers, arg_sizes);
}

void JITModuleCUDA::launch(const std::string &name,
                           std::size_t grid_dim,
                           std::size_t block_dim,
                           std::size_t dynamic_shared_mem_bytes,
                           const std::vector<void *> &arg_pointers,
                           const std::vector<int> &arg_sizes) {
  auto func = lookup_function(name);
  CUDAContext::get_instance().launch(func, name, arg_pointers, arg_sizes,
                                     grid_dim, block_dim,
                                     dynamic_shared_mem_bytes);
}

bool JITModuleCUDA::direct_dispatch() const {
  return false;
}

JITSessionCUDA::JITSessionCUDA(GsTaichiLLVMContext *tlctx,
                               const CompileConfig &config,
                               llvm::DataLayout data_layout,
                               ProgramImpl *program_impl)
    : JITSession(tlctx, config),
      data_layout(data_layout),
      program_impl_(program_impl),
      config_(config) {
  PtxCache::Config ptx_cache_config;
  ptx_cache_config.offline_cache_path = config.offline_cache_file_path;
  int compute_capability = CUDAContext::get_instance().get_compute_capability();
  ptx_cache_ =
      std::make_unique<PtxCache>(ptx_cache_config, config, compute_capability);

  finalizer_ = std::make_unique<Finalizer>(ptx_cache_.get());
  program_impl_->register_needs_finalizing(finalizer_.get());
}

JITModule *JITSessionCUDA::add_module(std::unique_ptr<llvm::Module> M,
                                      int max_reg) {
  const char *dump_ir_env = std::getenv(DUMP_IR_ENV.data());
  if (dump_ir_env != nullptr && std::string(dump_ir_env) == "1") {
    std::filesystem::path ir_dump_dir = config_.debug_dump_path;
    std::filesystem::create_directories(ir_dump_dir);
    std::string dumpName = moduleToDumpName(M.get());
    std::filesystem::path filename =
        ir_dump_dir / (dumpName + "_before_ptx.ll");
    std::error_code EC;
    llvm::raw_fd_ostream dest_file(filename.string(), EC);
    if (!EC) {
      M->print(dest_file, nullptr);
    } else {
      std::cout << "problem dumping file " << filename.string() << ": "
                << EC.message() << std::endl;
      TI_ERROR("Failed to dump LLVM IR to file: {}", filename.string());
    }
  }

  auto ptx = compile_module_to_ptx(M);
  if (this->config_.print_kernel_asm) {
    static FileSequenceWriter writer("gstaichi_kernel_nvptx_{:04d}.ptx",
                                     "module NVPTX");
    writer.write(ptx);
  }

  if (dump_ir_env != nullptr) {
    const std::string dumpOutDir = "/tmp/ptx/";
    std::filesystem::create_directories(dumpOutDir);
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + ".ptx";
    std::ofstream out_file(filename);
    if (out_file.is_open()) {
      out_file << ptx << std::endl;
      out_file.close();
    }
    std::cout << "PTX dumped to: " << filename << std::endl;
  }

  const char *load_ptx_env = std::getenv("GSTAICHI_LOAD_PTX");
  if (load_ptx_env != nullptr) {
    const std::string dumpOutDir = "/tmp/ptx/";
    std::string dumpName = moduleToDumpName(M.get());
    std::string filename = dumpOutDir + "/" + dumpName + ".ptx";
    std::ifstream in_file(filename);
    if (in_file.is_open()) {
      TI_INFO("Loading PTX from file: {}", filename);
      std::ostringstream ptx_stream;
      std::string line;
      while (std::getline(in_file, line)) {
        ptx_stream.write(line.c_str(), line.size());
        ptx_stream.write("\n", 1);
      }
      ptx_stream.write("\0", 1);  // Null-terminate the stream
      ptx = ptx_stream.str();
      in_file.close();
    } else {
      TI_WARN("Failed to open PTX file for loading: {}", filename);
    }
  }

  // TODO: figure out why using the guard leads to wrong tests results
  // auto context_guard = CUDAContext::get_instance().get_guard();
  CUDAContext::get_instance().make_current();
  // Create module for object
  void *cuda_module;
  TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  auto t = Time::get_time();
  TI_TRACE("Loading module...");
  [[maybe_unused]] auto _ = CUDAContext::get_instance().get_lock_guard();

  constexpr int max_num_options = 8;
  int num_options = 0;
  uint32 options[max_num_options];
  void *option_values[max_num_options];

  // Insert options
  if (max_reg != 0) {
    options[num_options] = CU_JIT_MAX_REGISTERS;
    option_values[num_options] = &max_reg;
    num_options++;
  }

  TI_ASSERT(num_options <= max_num_options);

  CUDADriver::get_instance().module_load_data_ex(
      &cuda_module, ptx.c_str(), num_options, options, option_values);
  TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
  // cudaModules.push_back(cudaModule);
  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

llvm::DataLayout JITSessionCUDA::get_data_layout() {
  return data_layout;
}

std::string cuda_mattrs() {
  return "+ptx63";
}

std::string convert(std::string new_name) {
  // Evil C++ mangling on Windows will lead to "unsupported characters in
  // symbol" error in LLVM PTX printer. Convert here.
  for (int i = 0; i < (int)new_name.size(); i++) {
    if (new_name[i] == '@') {
      new_name.replace(i, 1, "_at_");
    } else if (new_name[i] == '?') {
      new_name.replace(i, 1, "_qm_");
    } else if (new_name[i] == '$') {
      new_name.replace(i, 1, "_dl_");
    } else if (new_name[i] == '<') {
      new_name.replace(i, 1, "_lb_");
    } else if (new_name[i] == '>') {
      new_name.replace(i, 1, "_rb_");
    } else if (!std::isalpha(new_name[i]) && !std::isdigit(new_name[i]) &&
               new_name[i] != '_' && new_name[i] != '.') {
      new_name.replace(i, 1, "_xx_");
    }
  }
  if (!new_name.empty())
    TI_ASSERT(isalpha(new_name[0]) || new_name[0] == '_' || new_name[0] == '.');
  return new_name;
}

std::string JITSessionCUDA::compile_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("LLVM Module broken");
  }

  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("gstaichi_kernel_cuda_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::string llvm_ir_str;
  llvm::raw_string_ostream llvm_ir_stream(llvm_ir_str);
  module->print(llvm_ir_stream, nullptr);
  llvm_ir_stream.flush();
  std::string ptx_cache_key =
      ptx_cache_->make_cache_key(llvm_ir_str, this->config_.fast_math);
  std::optional<std::string> maybe_ptx = ptx_cache_->load_ptx(ptx_cache_key);
  if (maybe_ptx.has_value()) {
    TI_TRACE("Loaded PTX from cache for module {}", module->getName().str());
    return maybe_ptx.value();
  }

  for (auto &f : module->globals())
    f.setName(convert(f.getName().str()));
  for (auto &f : *module)
    f.setName(convert(f.getName().str()));

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  if (this->config_.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // See NVPTXISelLowering.cpp
    // Setting UnsafeFPMath true will result in approximations such as
    // sqrt.approx in PTX for both f32 and f64
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

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), CUDAContext::get_instance().get_mcpu(), cuda_mattrs(),
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      CodeGenOptLevel::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassBuilder pb(target_machine.get());
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm =
      pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // NVidia's libdevice library uses a __nvvm_reflect to choose
  // how to handle denormalized numbers. (The pass replaces calls
  // to __nvvm_reflect with a constant via a map lookup. The inliner
  // pass then resolves these situations to fast code, often a single
  // instruction per decision point.)
  //
  // The default is (more) IEEE like handling. FTZ mode flushes them
  // to zero. (This may only apply to single-precision.)
  //
  // The libdevice documentation covers other options for math accuracy
  // such as replacing division with multiply by the reciprocal and
  // use of fused-multiply-add, but they do not seem to be controlled
  // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
  // passes.
  const auto kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);

  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      /* nvptx-f32ftz was deprecated.
       *
       * https://github.com/llvm/llvm-project/commit/a4451d88ee456304c26d552749aea6a7f5154bde#diff-6fda74ef428299644e9f49a2b0994c0d850a760b89828f655030a114060d075a
       */
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");

      // Use unsafe fp math for sqrt.approx instead of sqrt.rn
      fn.addFnAttr("unsafe-fp-math", "true");
    }
  }

  mpm.run(*module, mam);

  llvm::legacy::PassManager legacy_pm;
  legacy_pm.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  legacy_pm.add(llvm::createLoopStrengthReducePass());
  legacy_pm.add(llvm::createSeparateConstOffsetFromGEPPass(false));
  legacy_pm.add(llvm::createEarlyCSEPass(true));

  // Ask the target to add backend passes as necessary.
  bool fail = target_machine->addPassesToEmitFile(
      legacy_pm, ostream, nullptr, llvm::CodeGenFileType::AssemblyFile, true);

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  {
    TI_PROFILER("llvm_module_pass");
    legacy_pm.run(*module);
  }

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "gstaichi_kernel_cuda_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  ptx_cache_->store_ptx(ptx_cache_key, buffer);
  return buffer;
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    GsTaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch,
    ProgramImpl *program_impl) {
  TI_ASSERT(arch == Arch::cuda);
  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
  auto data_layout = GsTaichiLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionCUDA>(tlctx, config, data_layout,
                                          program_impl);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    GsTaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch const ProgramImpl *program_impl) {
  TI_NOT_IMPLEMENTED
}
#endif

}  // namespace gstaichi::lang
