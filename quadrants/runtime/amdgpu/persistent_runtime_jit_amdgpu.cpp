#include "quadrants/runtime/amdgpu/persistent_runtime_jit_amdgpu.h"

#ifdef QD_WITH_AMDGPU

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "quadrants/jit/jit_module.h"
#include "quadrants/jit/jit_session.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/runtime/amdgpu/jit_amdgpu.h"
#include "quadrants/runtime/llvm/llvm_context.h"

namespace quadrants::lang {
namespace {

// Each persistent entry owns the small backing state the JITSession holds non-owning references to.
// `unique_ptr` for each so the entry's address is stable across cache rehashes.
struct PersistentRuntimeJitEntry {
  // Stable backing storage for `JITSession::config_`. Default-constructed and patched with only the fields the
  // AMDGPU codegen path reads (`arch`, `fast_math`, the three `print_kernel_*` toggles).
  std::unique_ptr<CompileConfig> stable_config;
  // Stable backing storage for `JITSession::tlctx_`. The persistent JITSession outlives every per-init
  // `LlvmRuntimeExecutor`, so we cannot share a per-init LLVM context (it would be destroyed with the executor
  // and leave `tlctx_` dangling). The runtime IR loaded here is independent of any per-init state.
  std::unique_ptr<QuadrantsLLVMContext> stable_llvm_ctx;
  // Owns the `JITModuleAMDGPU` we hand out. Destruction unloads the `hipModule_t` (via `~JITModuleAMDGPU`),
  // which only runs at process exit because we never erase entries.
  std::unique_ptr<JITSession> jit_session;
  // Non-owning. Points into `jit_session->modules`. Stable for the entry's lifetime.
  JITModule *jit_module{nullptr};
};

std::string cache_key_for(const CompileConfig &c) {
  // The runtime HSACO bytes are determined by:
  //   - `mcpu` (e.g. "gfx942"), since codegen targets it.
  //   - `fast_math`, since it flips the IEEE-conformance options the AMDGPU `TargetMachine` is configured with.
  // Other `CompileConfig` fields (default_gpu_block_dim, saturating_grid_dim, device_memory_GB, debug, etc.)
  // either don't reach the runtime IR at all or only affect runtime PARAMETERS that are passed into
  // `runtime_initialize` at materialization time, never baked into the HSACO.
  return AMDGPUContext::get_instance().get_mcpu() + "|" + (c.fast_math ? "fast" : "strict");
}

std::mutex &cache_mu() {
  // Heap-allocated, never freed. Mirrors `AMDGPUContext::get_instance()`'s `new` pattern: at process exit the
  // entries below would otherwise be destroyed in undefined order with respect to AMDGPU's libamdhip64 teardown,
  // and `~JITModuleAMDGPU()` -> `hipModuleUnload` would race the HIP runtime shutdown and crash on
  // `hipCtxSetCurrent`. Leaking the cache is the same shape `AMDGPUContext` itself uses to dodge that race.
  static auto *p = new std::mutex();
  return *p;
}

std::unordered_map<std::string, std::unique_ptr<PersistentRuntimeJitEntry>> &cache() {
  static auto *p = new std::unordered_map<std::string, std::unique_ptr<PersistentRuntimeJitEntry>>();
  return *p;
}

}  // namespace

JITModule *get_or_build_persistent_runtime_jit_amdgpu(const CompileConfig &per_init_config) {
  std::lock_guard<std::mutex> _(cache_mu());
  const std::string key = cache_key_for(per_init_config);
  auto &g_cache = cache();
  auto it = g_cache.find(key);
  if (it != g_cache.end()) {
    return it->second->jit_module;
  }

  auto entry = std::make_unique<PersistentRuntimeJitEntry>();

  // Snapshot only the fields the AMDGPU codegen path actually reads. Default-constructing then patching is
  // intentional: a wholesale `*entry->stable_config = per_init_config` would tie the persistent entry to
  // potentially mutable per-init state (e.g. logging knobs the user toggled mid-session), which could subtly
  // diverge the cached HSACO from a fresh build the user might expect.
  entry->stable_config = std::make_unique<CompileConfig>();
  entry->stable_config->arch = Arch::amdgpu;
  entry->stable_config->fast_math = per_init_config.fast_math;
  // The persistent path is for production runs, so disable the diagnostic dump knobs unconditionally; the
  // per-init `LlvmRuntimeExecutor` already handles user-facing diagnostic dumps for kernel modules separately.
  entry->stable_config->print_kernel_llvm_ir = false;
  entry->stable_config->print_kernel_llvm_ir_optimized = false;
  entry->stable_config->print_kernel_amdgcn = false;

  entry->stable_llvm_ctx = std::make_unique<QuadrantsLLVMContext>(*entry->stable_config, Arch::amdgpu);

  auto runtime_module = entry->stable_llvm_ctx->clone_runtime_module();
  // Apply the per-arch IR transforms (kernel-marking, address-space conversion, unused-function elimination)
  // before handing the module to the session — this matches what `LlvmRuntimeExecutor::init_runtime_jit_module`
  // does on the per-init path.
  entry->stable_llvm_ctx->init_runtime_module(runtime_module.get());

  // ProgramImpl is unused by the AMDGPU JITSession factory; pass nullptr explicitly.
  entry->jit_session = JITSession::create(entry->stable_llvm_ctx.get(), *entry->stable_config, Arch::amdgpu,
                                          /*program_impl=*/nullptr);
  entry->jit_module = entry->jit_session->add_module(std::move(runtime_module), /*max_reg=*/0);

  JITModule *result = entry->jit_module;
  g_cache.emplace(key, std::move(entry));
  return result;
}

}  // namespace quadrants::lang

#endif  // QD_WITH_AMDGPU
