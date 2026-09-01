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
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/codegen/llvm/per_task_artifact_cache.h"
#include "quadrants/program/per_construct_cache.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_set>

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
  const int n = (int)offloads.size();
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(n);

  // Cross-process per-task artifact cache: a hit skips a task's entire compilation and carries cached backend code +
  // launch metadata to the launcher. CUDA + AMDGPU + CPU; an empty dir means offline cache is off (tier disabled).
  const std::string art_dir = pertask_artifact_dir_ref();
  const bool artifact_tier = (compile_config_.arch == Arch::cuda || compile_config_.arch == Arch::amdgpu ||
                              arch_is_cpu(compile_config_.arch)) &&
                             !art_dir.empty();
  const PerTaskArtifactCache artifact_cache(art_dir);
  const DeviceCapabilityConfig pertask_caps = prog->get_device_caps();
  std::vector<std::string> pertask_keys(n);
  std::vector<std::vector<char>> artifact_codes(n);
  std::atomic<int> n_hit{0}, n_recompiled{0};

  for (int i = 0; i < n; i++) {
    // Only used to give per-task QD_DUMP_CFG dumps a collision-free name; inert in codegen.
    offloads[i]->as<OffloadedStmt>()->task_index = i;
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      // Skip the artifact cache for tasks whose printed body doesn't faithfully/deterministically capture their
      // compiled code (empty key => the JIT fill also skips the store):
      //  - adstack: lowering registers per-task AdStack sizing as a side effect a hit would skip;
      //  - external-func: the body has the so/bc path + name, not its contents, so a stale update keeps the key;
      //  - real-func (FuncCallStmt): the printer emits only the callee name, but codegen inlines its body;
      //  - mem_access_opt (BLS/read-only hints): serialized from an unordered_map, so its order varies by process.
      bool artifact_eligible = artifact_tier && offload->as<OffloadedStmt>()->mem_access_opt.get_all().empty();
      if (artifact_eligible) {
        irpass::analysis::gather_statements(offload.get(), [&artifact_eligible](Stmt *s) {
          if (s->is<AdStackAllocaStmt>() || s->is<ExternalFuncCallStmt>() || s->is<FuncCallStmt>()) {
            artifact_eligible = false;
          }
          return false;
        });
      }

      // Key on `#index` too: the module bakes the task's kernel-wide index into its symbol names, so two
      // byte-identical tasks at different indices must not alias (they would collide at link).
      std::string cache_key;
      if (artifact_eligible) {
        cache_key = get_hashed_per_task_cache_key(compile_config_, pertask_caps, offload->as<OffloadedStmt>(), kernel) +
                    "#" + std::to_string(i);
        // Under kernel profiling, drop the name-free cross-kernel aliasing: the artifact carries OffloadedTask::name,
        // which the profiler bills against, so a cross-kernel hit would misattribute time. Costs sharing only here.
        if (compile_config_.kernel_profiler) {
          cache_key += "@" + kernel->get_name();
        }
        pertask_keys[i] = cache_key;
        PerTaskArtifact rec;
        if (artifact_cache.try_load(cache_key, &rec)) {
          // Hit: reconstruct a metadata-only task (no LLVM module exists here) and keep the PTX for the JIT.
          data[i] = std::make_unique<LLVMCompiledTask>(
              rec.tasks, nullptr, std::unordered_set<int>(rec.used_tree_ids.begin(), rec.used_tree_ids.end()),
              std::unordered_set<int>(rec.struct_for_tls_sizes.begin(), rec.struct_for_tls_sizes.end()));
          artifact_codes[i] = std::move(rec.code);
          n_hit.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }

      Block blk;
      blk.insert(std::move(offload));
      auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
      n_recompiled.fetch_add(1, std::memory_order_relaxed);
    };
    worker.enqueue(compile_func);
  }
  worker.flush();

  // Build one self-contained artifact per task before the whole-module link consumes `data`: a hit carries cached
  // backend code (null module), a miss builds+optimizes a module for the JIT to compile and store.
  //
  // AMDGPU links each task with a separate `ld.lld`, and CPU serializes each task to a host object, so their per-task
  // path only pays off when the tier is on; with it off they stay on the single whole-module link. CUDA's per-task
  // load is cheap, so it always takes the per-task path.
  std::vector<PerConstructArtifact> per_construct_artifacts;
  const bool build_per_construct_artifacts =
      compile_config_.arch == Arch::cuda ||
      ((compile_config_.arch == Arch::amdgpu || arch_is_cpu(compile_config_.arch)) && artifact_tier);
  if (build_per_construct_artifacts) {
    for (int i = 0; i < n; i++) {
      if (!data[i])
        continue;
      PerConstructArtifact art;
      art.key = pertask_keys[i];
      if (!artifact_codes[i].empty()) {
        art.code = std::move(artifact_codes[i]);
        art.tasks = data[i]->tasks;
      } else {
        if (!data[i]->module)
          continue;
        std::vector<std::unique_ptr<LLVMCompiledTask>> one;
        one.push_back(std::make_unique<LLVMCompiledTask>(data[i]->clone()));
        auto linked_one = tlctx_.link_compiled_tasks(std::move(one));
        optimize_module(linked_one.module.get());
        art.module = std::move(linked_one.module);
        art.tasks = linked_one.tasks;
      }
      // Sorted for deterministic on-disk bytes (unordered_set upstream).
      art.used_tree_ids.assign(data[i]->used_tree_ids.begin(), data[i]->used_tree_ids.end());
      art.struct_for_tls_sizes.assign(data[i]->struct_for_tls_sizes.begin(), data[i]->struct_for_tls_sizes.end());
      std::sort(art.used_tree_ids.begin(), art.used_tree_ids.end());
      std::sort(art.struct_for_tls_sizes.begin(), art.struct_for_tls_sizes.end());
      per_construct_artifacts.push_back(std::move(art));
    }
  }

  // A cross-process hit leaves a task with no module, so skip the whole-kernel link; the launcher assembles the
  // backend module from the per-task artifacts. Still concatenate every task's metadata into `tasks`, which it runs
  // off.
  const bool code_only_tasks = std::any_of(data.begin(), data.end(), [](const auto &d) { return d && !d->module; });
  LLVMCompiledKernel llvm_compiled_kernel;
  if (code_only_tasks) {
    for (auto &d : data) {
      if (!d)
        continue;
      for (auto &t : d->tasks) {
        llvm_compiled_kernel.tasks.push_back(t);
      }
    }
  } else {
    llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
    optimize_module(llvm_compiled_kernel.module.get());
  }
  llvm_compiled_kernel.per_construct_artifacts = std::move(per_construct_artifacts);
  // Persist the kernel as an ordered key list (rebuilt on load) only when every task is keyed: an unkeyed (ineligible)
  // task would write a `.qdc` that never loads, so leave the list empty and let dump_impl skip persistence.
  if (code_only_tasks) {
    const auto &arts = llvm_compiled_kernel.per_construct_artifacts;
    const bool all_keyed = std::all_of(arts.begin(), arts.end(), [](const auto &a) { return !a.key.empty(); });
    if (all_keyed) {
      llvm_compiled_kernel.per_task_artifact_keys.reserve(arts.size());
      for (const auto &a : arts) {
        llvm_compiled_kernel.per_task_artifact_keys.push_back(a.key);
      }
    }
  }
  // Record per-task reuse counts for PerOffloadCacheObservations.tasks_* (read back by the compilation manager). Only
  // when the tier ran, so non-CUDA / cache-off compiles keep the -1 sentinel instead of a misleading 0.
  if (artifact_tier && prog != nullptr) {
    auto &cc = prog->per_construct_cache();
    std::lock_guard<std::mutex> g(cc.mu);
    cc.last_task_stats[kernel->get_name()] = {n, n_hit.load(), n_recompiled.load()};
  }
  return llvm_compiled_kernel;
}

#endif
}  // namespace quadrants::lang
