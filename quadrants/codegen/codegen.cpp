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

  // Cross-process per-task artifact cache. On a hit this process skips a task's ENTIRE compilation (CHI->LLVM, link,
  // optimize, PTX, ptxas) and carries the cached PTX + launch metadata straight to the launcher. Only the CUDA JIT
  // fills it today, and reuse needs the per-task composite module path, so the tier is CUDA-only; the empty dir
  // (offline cache off) disables it. Each task's IR key is kept so the artifacts below can be stored under it.
  const std::string art_dir = pertask_artifact_dir_ref();
  const bool artifact_tier = compile_config_.arch == Arch::cuda && !art_dir.empty();
  const PerTaskArtifactCache artifact_cache(art_dir);
  const DeviceCapabilityConfig pertask_caps = prog->get_device_caps();
  std::vector<std::string> pertask_keys(n);
  std::vector<std::vector<char>> artifact_codes(n);
  std::atomic<int> n_hit{0}, n_recompiled{0};

  for (int i = 0; i < n; i++) {
    // Record each task's kernel-wide index (clone() carries it onto the per-task copy below). Each task is then
    // lowered in isolation, where its one-task block's local index is always 0, so QD_DUMP_CFG reads this to give
    // per-task CFG dumps a collision-free `_task<i>` name. Unconditional and inert -- never affects codegen.
    offloads[i]->as<OffloadedStmt>()->task_index = i;
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      // Tasks kept entirely off the artifact cache -- neither probed nor stored (an empty key makes the JIT fill skip
      // the store), because the printed body the key hashes does not faithfully capture their compiled code:
      //  - adstack tasks: lowering registers per-task AdStack sizing as a compile-time side effect a hit would skip;
      //  - external-func tasks: the body carries the so/bc path + name but not its contents, so an in-place update keeps
      //    the key and a hit would run stale external code;
      //  - real-func calls (FuncCallStmt): the printer emits only the callee name, but codegen inlines its body, so a
      //    callee-body edit keeps the key and a hit would run stale code. (The whole-kernel key folds callee bodies in
      //    via emit_dependencies; matching that for the per-task key is deferred with the production binary key.)
      bool artifact_eligible = artifact_tier;
      if (artifact_eligible) {
        irpass::analysis::gather_statements(offload.get(), [&artifact_eligible](Stmt *s) {
          if (s->is<AdStackAllocaStmt>() || s->is<ExternalFuncCallStmt>() || s->is<FuncCallStmt>()) {
            artifact_eligible = false;
          }
          return false;
        });
      }

      // The key is name/index-independent, but the compiled module bakes the task's kernel-wide index into its
      // entry-fn / shared-array / adstack symbol names, so key on `#index` too: two byte-identical tasks at different
      // indices (e.g. repeated deactivate loops) must not alias to one artifact and collide at link.
      std::string cache_key;
      if (artifact_eligible) {
        cache_key = get_hashed_per_task_cache_key(compile_config_, pertask_caps, offload->as<OffloadedStmt>(), kernel) +
                    "#" + std::to_string(i);
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

  // Per-task path (CUDA-only): one self-contained artifact per task, built BEFORE the whole-module link consumes
  // `data`. A hit carries the cached PTX (`code`) with a null module; a miss builds the module for the JIT to
  // compile and store. The launch/graph metadata (`tasks`, tree ids) travels with each so the JIT can persist a
  // complete record -- the code alone cannot be launched. 7 and 8 extend consumption to CPU / AMDGPU.
  std::vector<PerConstructArtifact> per_construct_artifacts;
  if (compile_config_.arch == Arch::cuda) {
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

  // A cross-process hit leaves a task with no LLVM module here, so the whole-kernel link is impossible (and
  // unnecessary -- the launcher assembles the CUmodule from the per-task artifacts). The kernel-level `tasks` list
  // must still be the in-order concatenation of every task's metadata: the launcher and CUDA graph builder run off
  // it. Otherwise take the normal whole-module path.
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
  // Persistable form of the artifacts: only with no whole-kernel module (the `.qdc` describes the kernel purely as an
  // ordered list of per-task artifact keys, rebuilt from the cache on load) AND only when every task is keyed. A kernel
  // mixing a cache hit with an ineligible task (external-func / adstack carry no key) can't be described this way:
  // persisting the empty key would write a `.qdc` that never loads (`try_load("")` always misses), forcing a recompile
  // every process. Leaving the keys empty makes `dump_impl` skip the `.qdc` (unpersistable) instead of writing a dud.
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
  // Record per-task reuse counts on the program-scoped surface; the compilation manager reads them back and Python
  // surfaces them as `PerOffloadCacheObservations.tasks_*`. Only when the artifact tier ran, so non-CUDA / cache-off
  // compiles keep the -1 sentinel rather than reporting a misleading 0. n_hit + n_recompiled == n by construction
  // (every task takes exactly one branch), so `n` is the task total.
  if (artifact_tier && prog != nullptr) {
    auto &cc = prog->per_construct_cache();
    std::lock_guard<std::mutex> g(cc.mu);
    cc.last_task_stats[kernel->get_name()] = {n, n_hit.load(), n_recompiled.load()};
  }
  return llvm_compiled_kernel;
}

#endif
}  // namespace quadrants::lang
