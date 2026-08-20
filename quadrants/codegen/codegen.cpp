// Driver class for kernel codegen

#include "codegen.h"

#if defined(QD_WITH_LLVM)
#include "quadrants/codegen/cpu/codegen_cpu.h"
#include "quadrants/runtime/program_impls/llvm/llvm_program.h"
#endif
#if defined(QD_WITH_CUDA)
#include "quadrants/codegen/cuda/codegen_cuda.h"
#endif
#if defined(QD_WITH_AMDGPU)
#include "quadrants/codegen/amdgpu/codegen_amdgpu.h"
#endif
#include "quadrants/system/timer.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/program/per_construct_cache.h"

#include <mutex>

namespace quadrants::lang {

KernelCodeGen::KernelCodeGen(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             QuadrantsLLVMContext &tlctx)
    : prog(kernel->program), kernel(kernel), ir(ir), compile_config_(compile_config), tlctx_(tlctx) {
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(const CompileConfig &compile_config,
                                                     const Kernel *kernel,
                                                     IRNode *ir,
                                                     QuadrantsLLVMContext &tlctx) {
#ifdef QD_WITH_LLVM
  const auto arch = compile_config.arch;
  if (arch_is_cpu(arch)) {
    return std::make_unique<KernelCodeGenCPU>(compile_config, kernel, ir, tlctx);
  } else if (arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    return std::make_unique<KernelCodeGenCUDA>(compile_config, kernel, ir, tlctx);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    return std::make_unique<KernelCodeGenAMDGPU>(compile_config, kernel, ir, tlctx);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else {
    QD_NOT_IMPLEMENTED
  }
#else
  QD_ERROR("Llvm disabled");
#endif
}
#ifdef QD_WITH_LLVM

LLVMCompiledKernel KernelCodeGen::compile_kernel_to_module() {
  auto block = dynamic_cast<Block *>(ir);
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  QD_ASSERT(block);

  auto &offloads = block->statements;
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(offloads.size());
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      Block blk;
      blk.insert(std::move(offload));
      auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
    };
    worker.enqueue(compile_func);
  }
  worker.flush();

  // Per-task path (CUDA-only): one self-contained module per task, built BEFORE the whole-module link consumes
  // `data`. Only the CUDA launcher consumes per_construct_artifacts, so gate on CUDA to avoid paying an extra
  // per-task link + optimize on CPU / AMDGPU (7 and 8 extend consumption to those backends).
  std::vector<PerConstructArtifact> per_construct_artifacts;
  if (compile_config_.arch == Arch::cuda) {
    for (int i = 0; i < (int)data.size(); i++) {
      if (!data[i] || !data[i]->module)
        continue;
      PerConstructArtifact art;
      std::vector<std::unique_ptr<LLVMCompiledTask>> one;
      one.push_back(std::make_unique<LLVMCompiledTask>(data[i]->clone()));
      auto linked_one = tlctx_.link_compiled_tasks(std::move(one));
      optimize_module(linked_one.module.get());
      art.module = std::move(linked_one.module);
      per_construct_artifacts.push_back(std::move(art));
    }
  }

  auto llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
  optimize_module(llvm_compiled_kernel.module.get());
  llvm_compiled_kernel.per_construct_artifacts = std::move(per_construct_artifacts);
  // If the per-construct frontend split ran for this kernel, surface its cache stats. Recorded by
  // `split_frontend_per_construct` on the program-scoped construct-stats record, keyed by kernel name. Left at the
  // default `-1` (split absent) for kernels that took the whole-kernel path (autodiff, mesh).
  {
    auto &cc = kernel->program->per_construct_cache();
    std::lock_guard<std::mutex> g(cc.mu);
    auto it = cc.last_stats.find(kernel->get_name());
    if (it != cc.last_stats.end()) {
      llvm_compiled_kernel.per_task_cache_stats.construct_total = it->second.total;
      llvm_compiled_kernel.per_task_cache_stats.construct_cache_hit = it->second.hit;
      llvm_compiled_kernel.per_task_cache_stats.construct_recompiled = it->second.recompiled;
    }
  }
  return llvm_compiled_kernel;
}

#endif
}  // namespace quadrants::lang
