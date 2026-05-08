#pragma once

#ifdef QD_WITH_AMDGPU

namespace quadrants::lang {

class JITModule;
struct CompileConfig;

// Returns a process-resident `JITModule *` whose underlying `hipModule_t` is the loaded runtime HSACO for the
// effective key `(mcpu, fast_math)`. Builds the entry on first call and reuses it on every subsequent call across
// `qd.init` / `qd.reset` cycles. Thread-safe.
//
// Why this exists:
//   The runtime LLVM bitcode (`runtime/llvm/runtime.cpp` -> `runtime_amdgpu.bc`) is content-deterministic for a
//   given key. With the default per-init flow each `qd.init` re-runs the whole AMDGPU codegen pipeline (LLVM `O3`
//   passes + `ld.lld` fork to link the object into a `.hsaco`) and then `hipModuleLoadData`'s the resulting bytes
//   into the GPU. The codegen is several tens of ms of CPU work; the load consumes per-VF kernarg / signal pool
//   memory. Under 8-worker `pytest-xdist` contention on a single MI300X VF, the simultaneous-init burst was the
//   trigger for `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (surfaced as `hipErrorUnknown` from `hipStreamSynchronize`,
//   or `hipErrorOutOfMemory` from a fresh worker's first `hipMalloc`). Caching the HSACO process-wide eliminates
//   the fork, the file I/O, and the per-init `hipModuleLoadData`, which removes the contention spike.
//
// Cache key: only the fields that flow into the runtime HSACO. The runtime IR has no dependence on
// `default_gpu_block_dim`, `saturating_grid_dim`, `device_memory_GB`, etc. — those are runtime-time parameters
// passed through `runtime_initialize`.
//
// Lifetime: cache entries live until process exit. They share lifetime with the `AMDGPUContext` singleton (which
// retains the HIP primary context), so the `hipModule_t` is valid for the whole process. Destroying entries
// earlier would re-introduce the per-init pool churn this cache is meant to avoid.
//
// Why amdgpu-specific:
//   - On AMDGPU the per-init compile cost dominates (`ld.lld` shells out a child process per init) and the
//     `hipModuleLoadData` allocates per-VF resources we have no other way to share across processes-on-the-same-VF.
//   - On CUDA `cuModuleLoadData` consumes PTX through a fast in-driver cache; the per-init cost is negligible
//     and persisting it across `qd.init` does not help the failure modes we have evidence for.
//   - On CPU there is no module-load step.
JITModule *get_or_build_persistent_runtime_jit_amdgpu(const CompileConfig &per_init_config);

}  // namespace quadrants::lang

#endif  // QD_WITH_AMDGPU
