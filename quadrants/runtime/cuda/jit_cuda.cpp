#include <atomic>
#include <chrono>
#include <iterator>
#include <mutex>
#include <random>
#include <vector>

#include "quadrants/runtime/cuda/jit_cuda.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/codegen/ir_dump.h"
#include "quadrants/util/environ_config.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Utils.h"

namespace quadrants::lang {

#if defined(QD_WITH_CUDA)

bool module_has_runtime_initialize(const llvm::Module::FunctionListType &function_list) {
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
  auto err = CUDADriver::get_instance().module_get_function.call_with_warning(&func, module_, name.c_str());
  if (err) {
    QD_ERROR("Cannot look up function {}", name);
  }
  t = Time::get_time() - t;
  QD_TRACE("CUDA module_get_function {} costs {} ms", name, t * 1000);
  QD_ASSERT(func != nullptr);
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
  CUDAContext::get_instance().launch(func, name, arg_pointers, arg_sizes, grid_dim, block_dim,
                                     dynamic_shared_mem_bytes);
}

bool JITModuleCUDA::direct_dispatch() const {
  return false;
}

JITSessionCUDA::JITSessionCUDA(QuadrantsLLVMContext *tlctx,
                               const CompileConfig &config,
                               llvm::DataLayout data_layout,
                               ProgramImpl *program_impl)
    : JITSession(tlctx, config), data_layout(data_layout), program_impl_(program_impl), config_(config) {
  PtxCache::Config ptx_cache_config;
  ptx_cache_config.offline_cache_path = config.offline_cache_file_path;
  int compute_capability = CUDAContext::get_instance().get_compute_capability();
  ptx_cache_ = std::make_unique<PtxCache>(ptx_cache_config, config, compute_capability);

  finalizer_ = std::make_unique<Finalizer>(ptx_cache_.get());
  program_impl_->register_needs_finalizing(finalizer_.get());
}

JITModule *JITSessionCUDA::add_module(std::unique_ptr<llvm::Module> M, int max_reg) {
  // Read through get_environ_config, as codegen_llvm.cpp does for QD_DUMP_IR and QD_LOAD_IR, so that a value of 0
  // turns the variable off here too rather than only its absence.
  const bool dump_ir = get_environ_config(DUMP_IR_ENV.data()) != 0;
  const bool load_ptx_from_dump = get_environ_config(LOAD_PTX_ENV.data()) != 0;

  // Capture the dump name before compile_module_to_ptx renames functions via convert().
  std::string dump_name;
  if (dump_ir || load_ptx_from_dump) {
    dump_name = moduleToDumpName(M.get());
  }

  if (dump_ir && !dump_name.empty()) {
    std::filesystem::path ir_dump_dir = config_.debug_dump_path;
    std::filesystem::create_directories(ir_dump_dir);
    std::filesystem::path filename = ir_dump_dir / (dump_name + "_before_ptx.ll");
    std::error_code EC;
    llvm::raw_fd_ostream dest_file(filename.string(), EC);
    if (!EC) {
      M->print(dest_file, nullptr);
    } else {
      std::cout << "problem dumping file " << filename.string() << ": " << EC.message() << std::endl;
      QD_ERROR("Failed to dump LLVM IR to file: {}", filename.string());
    }
  }

  auto ptx = compile_module_to_ptx(M);
  if (this->config_.print_kernel_asm) {
    static FileSequenceWriter writer("quadrants_kernel_nvptx_{:04d}.ptx", "module NVPTX");
    writer.write(ptx);
  }

  if (dump_ir && !dump_name.empty()) {
    std::filesystem::path ir_dump_dir = config_.debug_dump_path;
    std::filesystem::create_directories(ir_dump_dir);
    std::filesystem::path ptx_path = ir_dump_dir / (dump_name + ".ptx");
    if (std::ofstream out_file(ptx_path); out_file.is_open()) {
      out_file << ptx << std::endl;
      std::cout << "PTX dumped to: " << ptx_path.string() << std::endl;
    }
  }

  if (load_ptx_from_dump && !dump_name.empty()) {
    std::filesystem::path ir_dump_dir = config_.debug_dump_path;
    std::filesystem::path ptx_path = ir_dump_dir / (dump_name + ".ptx");
    std::ifstream in_file(ptx_path);
    if (in_file.is_open()) {
      QD_INFO("Loading PTX from file: {}", ptx_path.string());
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
      QD_WARN("Failed to open PTX file for loading: {}", ptx_path.string());
    }
  }

  // TODO: figure out why using the guard leads to wrong tests results
  // auto context_guard = CUDAContext::get_instance().get_guard();
  CUDAContext::get_instance().make_current();
  // Create module for object
  void *cuda_module;
  QD_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  auto t = Time::get_time();
  QD_TRACE("Loading module...");
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

  QD_ASSERT(num_options <= max_num_options);

  CUDADriver::get_instance().module_load_data_ex(&cuda_module, ptx.c_str(), num_options, options, option_values);
  QD_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
  // cudaModules.push_back(cudaModule);
  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

// PTX -> *relocatable* cubin via `ptxas -c`. Relocatable is mandatory: cuLink rejects an executable cubin (err 209).
static std::vector<char> ptx_to_relocatable_cubin(const std::string &ptx, const std::string &arch) {
  namespace fs = std::filesystem;
  static std::atomic<uint64_t> ctr{0};
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  auto stem = fmt::format("qd_culink_{}_{}", (unsigned long long)now, ctr.fetch_add(1));
  auto ptx_path = fs::temp_directory_path() / (stem + ".ptx");
  auto cubin_path = fs::temp_directory_path() / (stem + ".cubin");
  {
    // ptxas (CUDA 13+) reads the trailing NUL compile_module_to_ptx adds (for the driver) as premature EOF; trim it.
    std::streamsize ptx_len = (std::streamsize)ptx.size();
    while (ptx_len > 0 && ptx[ptx_len - 1] == '\0')
      --ptx_len;
    std::ofstream o(ptx_path, std::ios::binary);
    o.write(ptx.data(), ptx_len);
  }
  auto cmd = fmt::format("ptxas -c -arch={} {} -o {} 2>/dev/null", arch, ptx_path.string(), cubin_path.string());
  int rc = std::system(cmd.c_str());
  QD_ERROR_IF(rc != 0, "ptxas -c failed (rc={}) arch={}", rc, arch);
  std::ifstream in(cubin_path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::error_code ec;
  fs::remove(ptx_path, ec);
  fs::remove(cubin_path, ec);
  return bytes;
}

// LLVM->PTX stays serialised: per-task modules may share an LLVMContext, unsafe for concurrent codegen.
static std::mutex g_ptxgen_mu;

// Per-task relocatable-cubin disk cache under <offline_cache>/culink_cubins_<mcpu> (namespaced by SM, since ptxas
// output isn't portable). A warm unchanged task skips PTX + ptxas. Keyed on the module's LLVM-IR text, which carries
// the entry-point symbol names, so distinct kernels can't collide. offline_cache=False -> mem-only, no disk touched.
std::vector<char> JITSessionCUDA::get_or_build_construct_cubin(std::unique_ptr<llvm::Module> &module) {
  namespace fs = std::filesystem;
  const std::string mcpu = CUDAContext::get_instance().get_mcpu();
  std::error_code ec;

  // Captured before compile_module_to_ptx, which renames functions via convert().
  const bool dump_ir = get_environ_config(DUMP_IR_ENV.data()) != 0;

  const bool on_disk = config_.offline_cache;
  std::string cubin_path;
  if (on_disk) {
    const std::string leaf = "culink_cubins_" + mcpu;
    const std::string dir =
        config_.offline_cache_file_path.empty() ? ("/tmp/qd_" + leaf) : (config_.offline_cache_file_path + "/" + leaf);
    fs::create_directories(dir, ec);

    std::string llvm_ir_str;
    llvm::raw_string_ostream os(llvm_ir_str);
    module->print(os, nullptr);
    os.flush();
    const std::string key = ptx_cache_->make_cache_key(llvm_ir_str, config_.fast_math);
    cubin_path = (fs::path(dir) / (key + ".cubin")).string();

    // A cached cubin skips PTX, so bypass the read under QD_DUMP_IR (this path owns the per-kernel PTX dump).
    if (!dump_ir && fs::exists(cubin_path)) {
      std::ifstream in(cubin_path, std::ios::binary);
      return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
  }

  const std::string dump_name = dump_ir ? moduleToDumpName(module.get()) : std::string();
  std::string ptx;
  {
    std::lock_guard<std::mutex> g(g_ptxgen_mu);
    ptx = compile_module_to_ptx(module);
  }
  if (dump_ir && !dump_name.empty()) {
    fs::path ir_dump_dir = config_.debug_dump_path;
    fs::create_directories(ir_dump_dir, ec);
    if (std::ofstream out(ir_dump_dir / (dump_name + ".ptx")); out.is_open()) {
      out << ptx << std::endl;
    }
  }
  if (on_disk)
    return assemble_and_store_cubin(ptx, cubin_path);
  return ptx_to_relocatable_cubin(ptx, mcpu);  // mem-only: assemble without persisting
}

std::vector<char> JITSessionCUDA::assemble_and_store_cubin(const std::string &ptx, const std::string &cubin_path) {
  namespace fs = std::filesystem;
  auto cubin = ptx_to_relocatable_cubin(ptx, CUDAContext::get_instance().get_mcpu());
  // temp + rename so a concurrent reader never sees a partial cubin
  std::error_code ec;
  auto tmp = cubin_path +
             fmt::format(".tmp{}", (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count());
  {
    std::ofstream o(tmp, std::ios::binary);
    o.write(cubin.data(), (std::streamsize)cubin.size());
  }
  fs::rename(tmp, cubin_path, ec);
  if (ec)
    fs::remove(tmp, ec);
  return cubin;
}

JITModule *JITSessionCUDA::add_module_culink(std::vector<PerConstructArtifact> artifacts, int max_reg) {
  // A cached relocatable cubin per task, device-linked into one CUmodule.
  constexpr uint32 kCuJitInputCubin = 0;
  constexpr uint32 kCuJitErrorLogBuffer = 5;
  constexpr uint32 kCuJitErrorLogBufferSizeBytes = 6;
  auto &drv = CUDADriver::get_instance();

  std::vector<std::vector<char>> cubins;
  cubins.reserve(artifacts.size());
  for (auto &art : artifacts) {
    cubins.push_back(get_or_build_construct_cubin(art.module));
  }

  CUDAContext::get_instance().make_current();
  [[maybe_unused]] auto _ = CUDAContext::get_instance().get_lock_guard();

  std::vector<char> err_log(8192, 0);
  std::size_t err_sz = err_log.size();
  uint32 link_opts[3];
  void *link_optvals[3];
  int n_lopt = 0;
  if (max_reg != 0) {
    link_opts[n_lopt] = CU_JIT_MAX_REGISTERS;
    link_optvals[n_lopt++] = &max_reg;
  }
  link_opts[n_lopt] = kCuJitErrorLogBuffer;
  link_optvals[n_lopt++] = err_log.data();
  link_opts[n_lopt] = kCuJitErrorLogBufferSizeBytes;
  link_optvals[n_lopt++] = (void *)err_sz;

  void *link = nullptr;
  QD_ERROR_IF(drv.link_create.call(n_lopt, link_opts, link_optvals, &link), "cuLinkCreate (culink-pertask) failed");
  for (int i = 0; i < (int)cubins.size(); i++) {
    auto name = fmt::format("construct_{}", i);
    auto e = drv.link_add_data.call(link, kCuJitInputCubin, (void *)cubins[i].data(), cubins[i].size(), name.c_str(), 0,
                                    nullptr, nullptr);
    if (e != 0)
      QD_ERROR("cuLinkAddData construct {} failed err={} log=[{}]", i, e, err_log.data());
  }
  void *cubin = nullptr;
  std::size_t cubin_sz = 0;
  auto e_cmp = drv.link_complete.call(link, &cubin, &cubin_sz);
  if (e_cmp != 0)
    QD_ERROR("cuLinkComplete (culink-pertask) failed err={} log=[{}]", e_cmp, err_log.data());
  void *cuda_module = nullptr;
  QD_ERROR_IF(drv.module_load_data.call(&cuda_module, cubin), "module_load_data (culink-pertask) failed");
  drv.link_destroy.call(link);

  this->modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return this->modules.back().get();
}

llvm::DataLayout JITSessionCUDA::get_data_layout() {
  return data_layout;
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
    } else if (!std::isalpha(new_name[i]) && !std::isdigit(new_name[i]) && new_name[i] != '_' && new_name[i] != '.') {
      new_name.replace(i, 1, "_xx_");
    }
  }
  if (!new_name.empty())
    QD_ASSERT(isalpha(new_name[0]) || new_name[0] == '_' || new_name[0] == '.');
  return new_name;
}

std::string JITSessionCUDA::compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  QD_AUTO_PROF
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    QD_ERROR("LLVM Module broken");
  }

  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("quadrants_kernel_cuda_llvm_ir_{:04d}.ll", "unoptimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::string llvm_ir_str;
  llvm::raw_string_ostream llvm_ir_stream(llvm_ir_str);
  module->print(llvm_ir_stream, nullptr);
  llvm_ir_stream.flush();
  std::string ptx_cache_key = ptx_cache_->make_cache_key(llvm_ir_str, this->config_.fast_math);
  std::optional<std::string> maybe_ptx = ptx_cache_->load_ptx(ptx_cache_key);
  if (maybe_ptx.has_value()) {
    QD_TRACE("Loaded PTX from cache for module {}", module->getName().str());
    return maybe_ptx.value();
  }

  for (auto &f : module->globals())
    f.setName(convert(f.getName().str()));
  for (auto &f : *module)
    f.setName(convert(f.getName().str()));

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target = TargetRegistry::lookupTarget(triple.str(), err_str);
  QD_ERROR_UNLESS(target, err_str);

  TargetOptions options;
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

  std::unique_ptr<TargetMachine> target_machine(
      target->createTargetMachine(triple, CUDAContext::get_instance().get_mcpu(), "", options, llvm::Reloc::PIC_,
                                  llvm::CodeModel::Small, CodeGenOptLevel::Aggressive));

  QD_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  QuadrantsLLVMContext::strip_nvvmir_version(module.get());

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

  llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

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
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", kFTZDenorms);

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
  legacy_pm.add(createTargetTransformInfoWrapperPass(target_machine->getTargetIRAnalysis()));

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  legacy_pm.add(llvm::createLoopStrengthReducePass());
  legacy_pm.add(llvm::createSeparateConstOffsetFromGEPPass(false));
  legacy_pm.add(llvm::createEarlyCSEPass(true));

  // Ask the target to add backend passes as necessary.
  bool fail =
      target_machine->addPassesToEmitFile(legacy_pm, ostream, nullptr, llvm::CodeGenFileType::AssemblyFile, true);

  QD_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  {
    QD_PROFILER("llvm_module_pass");
    legacy_pm.run(*module);
  }

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer("quadrants_kernel_cuda_llvm_ir_optimized_{:04d}.ll", "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::string buffer(outstr.begin(), outstr.end());
  append_compute_cache_bypass_nonce_if_disabled(buffer, this->config_);

  // Null-terminate the ptx source
  buffer.push_back(0);
  ptx_cache_->store_ptx(ptx_cache_key, buffer);
  return buffer;
}

void append_compute_cache_bypass_nonce_if_disabled(std::string &ptx, const CompileConfig &compile_config) {
  // CUDA_CACHE_DISABLE is captured at libcuda init time so it cannot be toggled mid-process; appending a per-session
  // comment shifts the PTX hash that the driver compute cache uses, forcing ptxas to re-run cold. The nonce is
  // `static` so that within one process every call sees the same value: two kernels in the same run that produce
  // identical PTX still hit the driver cache and skip a second ptxas invocation, while only cross-process reuse is
  // broken - the property needed for clean cold-cache measurements when offline_cache=false. We mix a high-resolution
  // timestamp with a random_device draw so back-to-back processes (CI matrix runs, `for i in {1..N}; do ./bench`)
  // cannot collide on the same wall-clock second the way std::time(nullptr) would.
  if (compile_config.offline_cache) {
    return;
  }
  static const std::string session_nonce = []() {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    auto entropy = std::random_device{}();
    return fmt::format("\n// quadrants-session-nonce: {}-{:08x}\n", static_cast<int64_t>(now), entropy);
  }();
  ptx.append(session_nonce);
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(QuadrantsLLVMContext *tlctx,
                                                         const CompileConfig &config,
                                                         Arch arch,
                                                         ProgramImpl *program_impl) {
  QD_ASSERT(arch == Arch::cuda);
  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
  auto data_layout = QuadrantsLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionCUDA>(tlctx, config, data_layout, program_impl);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(QuadrantsLLVMContext *tlctx,
                                                         const CompileConfig &config,
                                                         Arch arch const ProgramImpl *program_impl) {
  QD_NOT_IMPLEMENTED
}
#endif

}  // namespace quadrants::lang
