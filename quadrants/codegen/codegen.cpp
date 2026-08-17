// Driver class for kernel codegen

#include "codegen.h"

#if defined(QD_WITH_LLVM)
#include "quadrants/codegen/cpu/codegen_cpu.h"
#include "quadrants/codegen/llvm/per_task_module_cache.h"
#include "quadrants/codegen/llvm/per_task_artifact_cache.h"
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
#include "quadrants/program/per_construct_cache.h"
#include "quadrants/rhi/device_capability.h"

#include <algorithm>
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
  // Each task's IR key, kept so the artifacts below can be stored under it.
  std::vector<std::string> pertask_keys(offloads.size());
  // The per-task cache key needs the device caps; fetch once (cheap) so the per-task codegen cache is always active.
  const DeviceCapabilityConfig pertask_caps = prog->get_device_caps();
  auto &task_cache = get_llvm_program(kernel->program)->per_task_module_cache();
  std::atomic<int> n_cache_hit{0}, n_recompiled{0}, n_artifact_hit{0};

  // EXPERIMENT (no-inmem variation): bypass the in-memory per-task LLVM-module cache to measure the cost of relying on
  // the on-disk artifact/cubin tiers alone. The cross-process artifact probe above and the relocatable-cubin disk
  // cache below are deliberately left untouched -- only the in-process LLVM-module reuse is disabled. Flip to `true`
  // to restore the baseline behaviour.
  constexpr bool kUseInMemoryTaskCache = false;

  // Cross-process per-task artifact cache. On a hit we skip this task's ENTIRE compilation -- CHI->LLVM codegen,
  // link, optimize, PTX and ptxas -- and carry the cached cubin straight to the cuLink assembly, using the record's
  // `OffloadedTask` metadata for launch and graph construction.
  const PerTaskArtifactCache artifact_cache(pertask_artifact_dir_for(compile_config_.offline_cache_file_path));
  std::vector<std::vector<char>> artifact_cubins(offloads.size());

  // The frontend split may have replaced whole constructs with placeholder tasks, having found a manifest naming
  // their already-compiled artifacts. Those tasks must not be keyed or lowered at all -- their IR is an empty stub.
  // The plan also carries, per task, which construct produced it, so manifests can be recorded below.
  ConstructDiskPlan disk_plan;
  if (kernel->program != nullptr) {
    auto &cc_plan = kernel->program->per_construct_cache();
    std::lock_guard<std::mutex> g(cc_plan.mu);
    auto it = cc_plan.disk_plans.find(kernel->get_name());
    if (it != cc_plan.disk_plans.end()) {
      disk_plan = it->second;
    }
  }
  auto placeholder_key = [&disk_plan](int i) -> std::string {
    return (i < (int)disk_plan.artifact_key_by_task.size()) ? disk_plan.artifact_key_by_task[i] : std::string();
  };
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();

      // Placeholder: the split already resolved this task to a compiled artifact via a construct manifest. Load it
      // and skip everything -- no clone, no key, no lowering. This is what removes the whole-kernel frontend cost for
      // unchanged constructs, since their IR was never produced in this process.
      if (const std::string pkey = placeholder_key(i); !pkey.empty()) {
        PerTaskArtifact rec;
        if (artifact_cache.try_load(pkey, &rec)) {
          data[i] = std::make_unique<LLVMCompiledTask>(
              rec.tasks, nullptr, std::unordered_set<int>(rec.used_tree_ids.begin(), rec.used_tree_ids.end()),
              std::unordered_set<int>(rec.struct_for_tls_sizes.begin(), rec.struct_for_tls_sizes.end()));
          artifact_cubins[i] = std::move(rec.cubin);
          pertask_keys[i] = pkey;
          n_artifact_hit.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        // The split verified the artifact existed before committing to a placeholder, so losing it in between means
        // the store was pruned mid-compile. There is no real IR to fall back on, so fail loudly rather than emit a
        // kernel with a missing task.
        QD_ERROR("per-construct manifest referenced missing artifact {} (kernel {} task {})", pkey, kernel->get_name(),
                 i);
      }

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
      pertask_keys[i] = cache_key;

      // Cross-process artifact hit: this exact task was fully compiled by an earlier process. Reconstruct the
      // metadata-only `LLVMCompiledTask` (module stays null -- there is no LLVM for it in this process) and keep the
      // cubin for the cuLink assembly. Nothing below this point runs for the task.
      {
        PerTaskArtifact rec;
        if (artifact_cache.try_load(cache_key, &rec)) {
          data[i] = std::make_unique<LLVMCompiledTask>(
              rec.tasks, nullptr, std::unordered_set<int>(rec.used_tree_ids.begin(), rec.used_tree_ids.end()),
              std::unordered_set<int>(rec.struct_for_tls_sizes.begin(), rec.struct_for_tls_sizes.end()));
          artifact_cubins[i] = std::move(rec.cubin);
          n_artifact_hit.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }

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

      if (kUseInMemoryTaskCache && cacheable) {  // Cache hit: reuse the cached task module by cloning it into this
                                                 // worker's LLVM context.
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
      if (kUseInMemoryTaskCache && cacheable) {
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

  // Record, per construct, the ordered per-task artifact keys it produced, so the next process can skip that
  // construct's frontend entirely. Only now are the keys known -- they embed the task's index in the reassembled
  // kernel. Constructs that were themselves manifest hits are skipped: their manifest already exists and is correct.
  if (!disk_plan.construct_key_by_task.empty()) {
    const ConstructManifestCache manifests(construct_manifest_dir_for(compile_config_.offline_cache_file_path));
    std::vector<std::string> order;
    std::unordered_map<std::string, std::vector<std::string>> by_construct;
    for (int i = 0; i < (int)pertask_keys.size() && i < (int)disk_plan.construct_key_by_task.size(); i++) {
      const auto &ck = disk_plan.construct_key_by_task[i];
      if (ck.empty() || !placeholder_key(i).empty())
        continue;
      if (!by_construct.count(ck))
        order.push_back(ck);
      by_construct[ck].push_back(pertask_keys[i]);
    }
    for (const auto &ck : order) {
      ConstructManifest m;
      m.task_keys = by_construct[ck];
      manifests.store(ck, m);
    }
  }

  // Per-task cuLink path: produce one self-contained artifact per offloaded task -- link its runtime deps and
  // optimize it -- BEFORE the whole-module link consumes `data`. These flow to the CUDA JIT, which compiles each to a
  // relocatable cubin (hitting the on-disk cubin cache when the task is unchanged) and `cuLink`s them into one
  // CUmodule.
  std::vector<PerConstructArtifact> per_construct_artifacts;
  for (int i = 0; i < (int)data.size(); i++) {
    if (!data[i])
      continue;
    PerConstructArtifact art;
    art.key = pertask_keys[i];
    if (!artifact_cubins[i].empty()) {
      // Artifact-cache hit: no module exists in this process. The metadata came out of the cached record, so the
      // task is fully described without any codegen having run.
      art.cubin = std::move(artifact_cubins[i]);
      art.tasks = data[i]->tasks;
    } else {
      if (!data[i]->module)
        continue;
      std::vector<std::unique_ptr<LLVMCompiledTask>> one;
      one.push_back(std::make_unique<LLVMCompiledTask>(data[i]->clone()));
      auto linked_one = tlctx_.link_compiled_tasks(std::move(one));
      optimize_module(linked_one.module.get());
      art.module = std::move(linked_one.module);
      // Carry the launch/graph metadata down to the JIT so it can persist a complete `PerTaskArtifact`: the cubin
      // alone cannot be launched.
      art.tasks = linked_one.tasks;
    }
    // Sorted for deterministic on-disk bytes (these are unordered_sets upstream).
    art.used_tree_ids.assign(data[i]->used_tree_ids.begin(), data[i]->used_tree_ids.end());
    art.struct_for_tls_sizes.assign(data[i]->struct_for_tls_sizes.begin(), data[i]->struct_for_tls_sizes.end());
    std::sort(art.used_tree_ids.begin(), art.used_tree_ids.end());
    std::sort(art.struct_for_tls_sizes.begin(), art.struct_for_tls_sizes.end());
    per_construct_artifacts.push_back(std::move(art));
  }

  // If any task came from the cross-process artifact cache it has no LLVM module in this process, so the
  // whole-module link is both impossible and unnecessary -- the cuLink path assembles the CUmodule from per-task
  // cubins instead. Skipping it also removes a double build (link+optimize once per task AND once for the whole
  // kernel). The kernel-level `tasks` list still has to be the in-order concatenation of every task's metadata,
  // because the launcher and the CUDA graph builder are driven by it.
  const bool code_only_tasks = std::any_of(data.begin(), data.end(), [](const auto &d) { return d && !d->module; });
  LLVMCompiledKernel llvm_compiled_kernel;
  if (code_only_tasks) {
    for (auto &d : data) {
      if (!d)
        continue;
      for (auto &t : d->tasks)
        llvm_compiled_kernel.tasks.push_back(t);
    }
  } else {
    llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
    optimize_module(llvm_compiled_kernel.module.get());
  }
  llvm_compiled_kernel.per_construct_artifacts = std::move(per_construct_artifacts);
  // Persistable form of the artifacts: only meaningful when there is no whole-kernel module, i.e. when the `.qdc`
  // entry must describe this kernel purely as an ordered list of per-task artifacts.
  if (code_only_tasks) {
    llvm_compiled_kernel.per_task_artifact_keys.reserve(llvm_compiled_kernel.per_construct_artifacts.size());
    for (const auto &a : llvm_compiled_kernel.per_construct_artifacts) {
      llvm_compiled_kernel.per_task_artifact_keys.push_back(a.key);
    }
  }
  // Artifact-cache hits count as cache hits for observation purposes: from the caller's point of view the task was
  // reused rather than recompiled, it just came from disk rather than from this process's memory.
  llvm_compiled_kernel.per_task_cache_stats = {(int)offloads.size(), n_cache_hit.load() + n_artifact_hit.load(),
                                               n_recompiled.load()};
  // If the per-construct frontend split ran for this kernel, surface its cache stats alongside the per-task ones.
  // Recorded by `split_frontend_per_construct` on the program-scoped construct cache, keyed by kernel name.
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
