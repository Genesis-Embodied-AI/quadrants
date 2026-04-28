# qd.init options

`qd.init(...)` accepts every field of the underlying `CompileConfig` struct as a keyword argument; the same fields are also reachable as environment variables of the form `QD_<UPPERCASE_NAME>` (e.g. `QD_OFFLINE_CACHE=0`). This page covers some of the knobs that are commonly tuned in practice. The underlying source of truth is [`quadrants/program/compile_config.h`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/program/compile_config.h).

## Caching

### `offline_cache`

Whether the compilation caches **persist on disk across Python invocations**. Default `True`. The "offline" in the name refers to the fact that this cache outlives the process: it is what makes the *second* time you start a Python interpreter and run a kernel cheap, by reusing artifacts from the first run.

Setting `offline_cache=False` is intended to emulate cold-start, i.e. a fresh Python process with no prior on-disk artifacts available. In-process caches operate independently of this flag: within a single Python session, identical kernels are never recompiled. The flag therefore controls only whether the next Python invocation observes a warm or a cold disk.

When `offline_cache=True`, three persistent layers cooperate. The first two share the cache directory configured by `offline_cache_file_path` (default `~/.cache/quadrants/qdcache`); the third is owned by libcuda and lives outside that path.

1. The cross-backend kernel-IR / compiled-kernel cache (driven by `KernelCompilationManager`). When the IR-and-config hash hits, the previously compiled kernel data is loaded from disk and the entire compile pipeline is skipped. Active for every backend (CPU, CUDA, AMDGPU, Metal, Vulkan).
2. The CUDA per-arch PTX cache, written under `<offline_cache_file_path>/ptx_cache_sm_*` (driven by `PtxCache`). When the LLVM-IR hash hits, the previously emitted PTX is loaded from disk and the LLVM-to-PTX compilation pipeline (LLVM optimization passes plus the NVPTX backend's PTX emission) is skipped. `ptxas` itself runs later inside `cuModuleLoadDataEx` and is governed by Layer 3.
3. The NVIDIA driver compute cache at `~/.nv/ComputeCache`, keyed by PTX content hash. When this hits, `ptxas` work is skipped because the SASS itself is reused. This cache is owned by libcuda and not by Quadrants.

Setting `offline_cache=False` (or `QD_OFFLINE_CACHE=0`) disables every disk-persistent layer so a fresh Python session sees a true cold start:

- Layer 1 falls back to memory-only. The disk cache is not consulted for kernel data and new kernels are not persisted, so kernels are compiled from source on every Python invocation.
- Layer 2 falls back to memory-only. PTX is still cached within one process so kernels with identical LLVM IR share PTX output, but nothing is read from or written to disk.
- Layer 3 cannot be controlled by the libcuda environment variable `CUDA_CACHE_DISABLE` from inside Python because the variable is captured by libcuda at process start. Quadrants instead appends a per-process nonce comment to the PTX it submits to `cuModuleLoadDataEx`. The nonce is constant within one process - kernels with identical PTX still share a cubin in the same run - and changes between processes so cross-run hits cannot quietly serve stale SASS.

When to set it to `False`:
- Taking compile-time profiles where any cached SASS would mask the real cost.
- Investigating a stale-cache bug or suspected cache corruption.
- Reproducing first-run behavior in CI matrix runs that would otherwise warm the caches across iterations.

For normal use, leave it at `True`; the cache layers are the dominant source of fast warm-up.

## Compile-time tuning

### `cfg_optimization`

Whether to run the control-flow-graph optimization pass. Default `True`. Setting it to `False` makes compilation up to 6x faster while costing 1-5% of runtime speed; consider disabling it if compile time is the bottleneck and the runtime delta is acceptable.

### `fast_math`

Whether to enable IEEE-relaxed floating-point optimizations (FMA fusion, no NaN / infinity / signed-zero guarantees). Default `True`. Disable when investigating numerical anomalies or running deterministic-tolerance tests.

### `num_compile_threads`

Number of host threads used when compiling kernels. Default `4`. Raise on machines with many idle cores compiling many kernels back-to-back; lower (or set to `1`) on memory-pressure-bound systems where concurrent LLVM compilations thrash.

## Debugging

### `debug`

Enables IR verification between every compiler pass plus additional runtime checks (integer-overflow guards on arithmetic, linear-index overflow guards on tensor indexing, adstack push-bounds at the runtime helper level). Default `False`. Compile time slows substantially because the verifier walks the IR after every transform and the extra runtime checks expand the emitted code; ~21s additional has been observed on adstack-heavy kernels. Turn this on while iterating on a kernel that is producing incorrect numerics or while developing a new compiler pass; turn it back off once the bug is found.

### `check_out_of_bound`

Enables runtime bounds-checking for tensor indexing. Default `False`. Costs runtime performance proportional to indexing density; leave off for benchmarks. Backends that do not expose the `assertion` extension (currently Metal and Vulkan) cannot honor this flag.
