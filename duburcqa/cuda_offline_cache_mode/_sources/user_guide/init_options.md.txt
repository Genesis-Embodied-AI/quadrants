# qd.init options

`qd.init(...)` accepts every field of the underlying `CompileConfig` struct as a keyword argument; the same fields are also reachable as environment variables of the form `QD_<UPPERCASE_NAME>` (e.g. `QD_OFFLINE_CACHE=0`). This page covers some of the knobs that are commonly tuned in practice. The underlying source of truth is `quadrants/program/compile_config.h`.

## Caching

### `offline_cache`

Whether the compilation caches **persist on disk across Python invocations**. Default `True`. The "offline" in the name refers to the fact that this cache outlives the process: it is what makes the *second* time you start a Python interpreter and run a kernel cheap, by reusing artifacts from the first run.

Setting `offline_cache=False` is intended to emulate cold-start, i.e. a fresh Python process with no prior on-disk artifacts available. In-process caches operate independently of this flag: within a single Python session, identical kernels are never recompiled. The flag therefore controls only whether the next Python invocation observes a warm or a cold disk.

When `offline_cache=True`, three persistent layers cooperate. The first two share the cache directory configured by `offline_cache_file_path` (default `~/.cache/quadrants`); the third is owned by libcuda and lives outside that path.

1. The cross-backend kernel-IR / compiled-kernel cache (driven by `KernelCompilationManager`). When the IR-and-config hash hits, the previously compiled kernel data is loaded from disk and the entire compile pipeline is skipped. Active for every backend (CPU, CUDA, AMDGPU, Metal, Vulkan).
2. The CUDA per-arch PTX cache, written under `<offline_cache_file_path>/ptx_cache_sm_*` (driven by `PtxCache`). When the LLVM-IR hash hits, the previously emitted PTX is loaded from disk and `ptxas` is skipped.
3. The NVIDIA driver compute cache at `~/.nv/ComputeCache`, keyed by PTX content hash. When this hits, even `ptxas` work that the per-arch PTX cache would have triggered is skipped because the SASS itself is reused. This cache is owned by libcuda and not by Quadrants.

Setting `offline_cache=False` (or `QD_OFFLINE_CACHE=0`) disables every disk-persistent layer so a fresh Python session sees a true cold start:

- Layer 1 falls back to memory-only. The kernel-IR cache is not consulted on construction and not written on shutdown, so kernels are compiled from source on every Python invocation.
- Layer 2 falls back to memory-only. PTX is still cached within one process so kernels with identical LLVM IR share `ptxas` output, but nothing is read from or written to disk.
- Layer 3 cannot be controlled by the libcuda environment variable `CUDA_CACHE_DISABLE` from inside Python because the variable is captured by libcuda at process start. Quadrants instead appends a per-process nonce comment to the PTX it submits to `cuModuleLoadDataEx`. The nonce is constant within one process - kernels with identical PTX still share a cubin in the same run - and changes between processes so cross-run hits cannot quietly serve stale SASS.

When to set it to `False`:
- Taking compile-time profiles where any cached SASS would mask the real cost.
- Investigating a stale-cache bug or suspected cache corruption.
- Reproducing first-run behavior in CI matrix runs that would otherwise warm the caches across iterations.

For normal use, leave it at `True`; the cache layers are the dominant source of fast warm-up.
