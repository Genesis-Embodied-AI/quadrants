// Driver class for kernel codegen

#include "codegen.h"

#if defined(QD_WITH_LLVM)
#include "quadrants/codegen/cpu/codegen_cpu.h"
#include "quadrants/codegen/llvm/per_task_module_cache.h"
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
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/rhi/device_capability.h"

#include <atomic>
#include <mutex>
#include <string>

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
  // The per-task cache key needs the device caps; fetch once (cheap) so the per-task codegen cache is always active.
  const DeviceCapabilityConfig pertask_caps = prog->get_device_caps();
  auto &task_cache = get_llvm_program(kernel->program)->per_task_module_cache();
  std::atomic<int> n_cache_hit{0}, n_recompiled{0};
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      // The task key is name/index-independent (re-id'd body + config + caps + touched-SNode layout + autodiff
      // mode). But the compiled module bakes in `task_codegen_id` -- the offload's index within THIS kernel -- into the
      // entry function name (`{kernel}_{id}_{taskname}`), `shared_array_t{id}_...` globals, and the
      // `adstack_row_counters[id]` indices. So the in-memory cache key must include the task index: otherwise two
      // byte-identical tasks at different indices (e.g. repeated `deactivate` loops emit identical serial/gc tasks)
      // alias to one cached module and produce duplicate symbols that collide at link time. Reuse across *different*
      // kernels at the same index is still sound -- the reused module keeps the original kernel's (unique) symbol
      // names, which stay unique within the new kernel's linked module.
      const std::string key =
          get_hashed_per_task_cache_key(compile_config_, pertask_caps, offload->as<OffloadedStmt>(), kernel);
      const std::string cache_key = key + "#" + std::to_string(i);

      // Autodiff tasks register per-task AdStack sizing into the program-scoped adstack cache as a compile-time side
      // effect keyed on {kernel_name, task_codegen_id}; a cross-kernel cache hit would skip that registration, so keep
      // adstack-bearing tasks off the reuse path and always lower them.
      bool has_adstack = false;
      irpass::analysis::gather_statements(offload.get(), [&has_adstack](Stmt *s) {
        if (s->is<AdStackAllocaStmt>()) {
          has_adstack = true;
        }
        return false;
      });
      const bool cacheable = !has_adstack;

      if (cacheable) {  // Cache hit: reuse the cached task module by cloning it into this worker's LLVM context.
        std::lock_guard<std::mutex> g(task_cache.mu);
        auto it = task_cache.entries.find(cache_key);
        if (it != task_cache.entries.end()) {
          auto mod = tlctx_.clone_module_to_this_thread_context(it->second.module.get());
          data[i] = std::make_unique<LLVMCompiledTask>(it->second.tasks, std::move(mod), it->second.used_tree_ids,
                                                       it->second.struct_for_tls_sizes);
          n_cache_hit.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }

      // Cache miss: lower the task (the expensive step, done outside the cache lock) and store it.
      Block blk;
      blk.insert(std::move(offload));
      auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
      if (cacheable) {
        std::lock_guard<std::mutex> g(task_cache.mu);
        if (task_cache.entries.find(cache_key) == task_cache.entries.end()) {
          PerTaskModuleCache::Entry e;
          e.module = tlctx_.clone_module_to_context(new_data.module.get(), &task_cache.ctx);
          e.tasks = new_data.tasks;
          e.used_tree_ids = new_data.used_tree_ids;
          e.struct_for_tls_sizes = new_data.struct_for_tls_sizes;
          task_cache.entries.emplace(cache_key, std::move(e));
        }
      }
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
      n_recompiled.fetch_add(1, std::memory_order_relaxed);
    };
    worker.enqueue(compile_func);
  }
  worker.flush();

  auto llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
  optimize_module(llvm_compiled_kernel.module.get());
  llvm_compiled_kernel.per_task_cache_stats = {(int)offloads.size(), n_cache_hit.load(), n_recompiled.load()};
  return llvm_compiled_kernel;
}

#endif
}  // namespace quadrants::lang
