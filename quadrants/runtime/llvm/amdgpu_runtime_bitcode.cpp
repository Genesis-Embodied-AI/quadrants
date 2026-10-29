#include "quadrants/runtime/llvm/amdgpu_runtime_bitcode.h"
#include "quadrants/runtime/llvm/runtime_bitcode_cache.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/util/environ_config.h"
#include "quadrants/program/compile_config.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/Error.h"
#include <atomic>
#include <cstdlib>
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#ifdef QD_WITH_AMDGPU
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#endif

namespace quadrants::lang {
static void dump_template(llvm::Module &module) {
  if (const char *path = std::getenv("QD_CONTEXT_COMPARE")) {
    static std::atomic<int> sequence{0};
    std::error_code ec;
    llvm::raw_fd_ostream out(std::string(path) + std::to_string(sequence++) + ".ll", ec);
    module.print(out, nullptr);
  }
}

std::vector<std::string> amdgpu_libdevice_files(const std::string &mcpu) {
  return {"ocml.bc",
          "oclc_wavefrontsize64_on.bc",
          "ockl.bc",
          "oclc_abi_version_400.bc",
          "oclc_correctly_rounded_sqrt_off.bc",
          "oclc_daz_opt_off.bc",
          "oclc_finite_only_off.bc",
          "oclc_isa_version_" + mcpu.substr(3, 4) + ".bc",
          "oclc_unsafe_math_off.bc",
          "opencl.bc"};
}

std::unique_ptr<llvm::Module> load_amdgpu_runtime_bitcode(QuadrantsLLVMContext &context,
                                                          const CompileConfig &config,
                                                          const std::string &file) {
#ifdef QD_WITH_AMDGPU
  if (!config.offline_cache || config.print_kernel_llvm_ir || config.print_kernel_llvm_ir_optimized ||
      config.print_kernel_amdgcn || get_environ_config("QD_DUMP_IR") != 0) {
    QD_TRACE("AMDGPU runtime bitcode cache bypass");
    return context.module_from_file(file);
  }

  const auto target = AMDGPUContext::get_instance().get_mcpu();
  std::vector<std::string> paths{runtime_lib_dir() + "/" + file};
  for (const auto &library : amdgpu_libdevice_files(target)) {
    paths.push_back(runtime_lib_dir() + "_rocm70/" + library);
  }
  const auto inputs = RuntimeBitcodeInputs::read(target, paths);
  static RuntimeBitcodeCache cache;
  std::unique_ptr<llvm::Module> prepared;
  bool hit;
  const auto bitcode = cache.get_or_build(
      inputs.key(),
      [&] {
        prepared = context.module_from_file(file, &inputs);
        dump_template(*prepared);
        std::string bytes;
        llvm::raw_string_ostream stream(bytes);
        llvm::WriteBitcodeToFile(*prepared, stream);
        return bytes;
      },
      hit);
  QD_TRACE("AMDGPU runtime bitcode cache {}", hit ? "hit" : "miss");
  if (prepared) {
    return prepared;
  }
  // Parse into this program/thread's fresh context; mutations cannot reach any other program or cached template.
  // Like clone_module_to_context, parse the prepared intermediate without running the verifier here: device-library
  // functions still have CommonLinkage, which the later runtime/kernel linking passes resolve before code generation.
  auto module = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bitcode, "prepared_amdgpu_runtime"),
                                       *context.get_this_thread_context());
  if (!module) {
    QD_ERROR("Failed to parse prepared AMD runtime bitcode: {}", llvm::toString(module.takeError()));
  }
  dump_template(*module.get());
  return std::move(module.get());
#else
  QD_NOT_IMPLEMENTED;
#endif
}

}  // namespace quadrants::lang
