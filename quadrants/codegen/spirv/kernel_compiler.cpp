#include "quadrants/codegen/spirv/kernel_compiler.h"

#include <mutex>

#include "quadrants/ir/analysis.h"
#include "quadrants/codegen/spirv/spirv_codegen.h"
#include "quadrants/codegen/spirv/compiled_kernel_data.h"
#include "quadrants/program/program.h"
#include "quadrants/program/per_construct_cache.h"

namespace quadrants::lang {
namespace spirv {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(const CompileConfig &compile_config, const Kernel &kernel_def) const {
  auto ir = irpass::analysis::clone(kernel_def.ir.get());
  irpass::compile_to_executable(ir.get(), compile_config, &kernel_def, kernel_def.autodiff_mode,
                                /*ad_use_stack=*/compile_config.ad_stack_experimental_enabled, compile_config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(const CompileConfig &compile_config,
                                               const DeviceCapabilityConfig &device_caps,
                                               const Kernel &kernel_def,
                                               IRNode &chi_ir) const {
  QD_TRACE("VK codegen for Quadrants kernel={}", kernel_def.name);
  KernelCodegen::Params params;
  params.qd_kernel_name = kernel_def.name;
  params.kernel = &kernel_def;
  params.ir_root = &chi_ir;
  params.compiled_structs = *config_.compiled_struct_data;
  params.arch = compile_config.arch;
  params.caps = device_caps;
  params.enable_spv_opt = compile_config.external_optimization_level > 0;
  params.compile_config = &compile_config;
  spirv::KernelCodegen codegen(params);
  spirv::CompiledKernelData::InternalData internal_data;
  codegen.run(internal_data.metadata.kernel_attribs, internal_data.src.spirv_src);
  internal_data.metadata.num_snode_trees = config_.compiled_struct_data->size();
  // Carry the frontend split's no-alias assumption onto the compiled kernel so a cache hit still arms the launch guard
  // (mirrors the LLVM path in codegen.cpp). Absent entry (whole-kernel compile) leaves the default empty.
  if (kernel_def.program != nullptr) {
    auto &cc = kernel_def.program->per_construct_cache();
    std::lock_guard<std::mutex> g(cc.mu);
    auto it = cc.last_stats.find(kernel_def.get_name());
    if (it != cc.last_stats.end())
      internal_data.metadata.split_assumed_disjoint_pairs = it->second.assumed_disjoint_pairs;
  }
  return std::make_unique<spirv::CompiledKernelData>(compile_config.arch, internal_data);
}

}  // namespace spirv
}  // namespace quadrants::lang
