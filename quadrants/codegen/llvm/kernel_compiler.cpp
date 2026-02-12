#include "quadrants/codegen/codegen.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/transforms.h"

#include "quadrants/codegen/llvm/kernel_compiler.h"
#include "quadrants/codegen/llvm/compiled_kernel_data.h"

namespace quadrants::lang {
namespace LLVM {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) const {
  auto ir = irpass::analysis::clone(kernel_def.ir.get());
  bool verbose = compile_config.print_ir;
  if (kernel_def.is_accessor && !compile_config.print_accessor_ir) {
    verbose = false;
  }
  bool use_ad_stack = compile_config.ad_stack_experimental_enabled;
  irpass::compile_to_offloads(ir.get(), compile_config, &kernel_def,
                              /*verbose=*/verbose,
                              /*autodiff_mode=*/kernel_def.autodiff_mode,
                              /*ad_use_stack=*/use_ad_stack,
                              /*start_from_ast=*/kernel_def.ir_is_ast());
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  LLVM::CompiledKernelData::InternalData data;
  auto codegen = KernelCodeGen::create(compile_config, &kernel_def, &chi_ir,
                                       *config_.tlctx);
  data.compiled_data = codegen->compile_kernel_to_module();
  data.args.reserve(kernel_def.nested_parameters.size());
  for (const auto &p : kernel_def.nested_parameters)
    data.args.push_back(p);
  data.rets = kernel_def.rets;
  data.args_type = kernel_def.args_type;
  data.args_size = kernel_def.args_size;
  data.ret_type = kernel_def.ret_type;
  data.ret_size = kernel_def.ret_size;
  return std::make_unique<LLVM::CompiledKernelData>(compile_config.arch, data);
}

}  // namespace LLVM
}  // namespace quadrants::lang
