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

## Reverse-mode autodiff

See [Autodiff](./autodiff.md) for the reverse-mode pipeline overview.

### `ad_stack_experimental_enabled`

Enables the dynamic-loop reverse-mode pipeline (the *adstack*). Default `False`. Required when a reverse-mode kernel has a runtime-bounded loop carrying a non-linear primal; without it, such kernels either compile-error or produce silently-wrong gradients depending on the loop shape. See [Autodiff with dynamic loops](./autodiff.md#autodiff-with-dynamic-loops) for the rules. Adstack-on is safe even when not strictly needed, but it does come with a few drawbacks:

- **Memory.** The reverse pass replays each iteration of the dynamic loop, so the adstack stores per-iteration intermediate values for every thread. See [Memory footprint](./autodiff.md#memory-footprint) for the exact formula and the knobs that shrink it (`ad_stack_size`, `ad_stack_sparse_threshold_bytes`).
- **Per-launch overhead.** Every backward kernel launch incurs a small fixed CPU-to-GPU data transfer. Kernels whose dynamic loop is gated by a sparse predicate (e.g. `for i in range(n): if active[i] > 0: ...`) additionally run a fast GPU pre-step that counts how many threads pass the gate so that the adstack can be tightly sized instead of upper-bounded by worst case.

*Note.* These drawbacks affect only reverse-mode kernels that actually use the adstack; forward-only kernels and reverse-mode kernels without a dynamic non-linear inner loop pay nothing extra. In other words, enabling adstack globally is effectively free except for kernels that need it anyway!

### `ad_stack_size`

Forces every adstack in the program to exactly `N` slots and bypasses the launch-time sizer. Default `0`, meaning "let the sizer decide" (the recommended setting for day-to-day use). Setting a positive `N` is meant for stress tests or working around a suspected sizer bug; it defeats the per-launch-exact sizing so every dispatch allocates the full `N` slots whether or not the kernel actually needs them. Has no effect when `ad_stack_experimental_enabled=False`.

### `ad_stack_sparse_threshold_bytes`

Cutoff (in bytes) below which the gate-passing-count sizing path described in [Memory footprint](./autodiff.md#memory-footprint) is skipped in favour of the eager `dispatched_threads * stride` heap. Default `100 MiB`. The sparse path saves memory on kernels of the shape `for i in range(...): if field[i] cmp literal: <adstack work>` but pays a per-launch reducer dispatch; below the threshold that overhead outweighs the savings. Set to `0` to always use the sparse path; lower it if the default still skips kernels you want shrunk. No effect when `ad_stack_experimental_enabled=False` or when the kernel has no such gate.

## Apple Metal

### `external_metal_command_queue`

An `MTLCommandQueue*` pointer (as an integer) to use instead of creating a new Metal command queue. Default `0` (create a new queue). When non-zero, Quadrants dispatches all GPU work on the provided queue, which enables GPU-side ordering with other frameworks that share the same queue (most notably PyTorch MPS).

### `external_metal_command_queue_is_torch_queue`

Default `False`. Set to `True` when the `external_metal_command_queue` is PyTorch MPS's command queue. This tells Quadrants that both frameworks share the same Metal queue, so the explicit `qd.sync()` / `torch.mps.synchronize()` calls at `to_torch` / `from_torch` interop points can be skipped. When `False` (or when no external queue is set), the interop syncs are preserved.

See [Shared Metal command queue](./metal_shared_queue.md) for the full setup guide, including how to extract the queue pointer from PyTorch and the synchronisation implications.

## Debugging

See [Debug mode](./debug.md) for runnable examples and a typical develop / benchmark workflow.

### `debug`

Default `False`. Turns on every available correctness check. Use while iterating on a kernel that produces wrong numerics or while developing a new compiler pass; turn off for benchmarks and production.

Enables:
- field-bounds check on tensor indexing (out-of-range index raises `RuntimeError`);
- kernel `assert` statements;
- integer-overflow guards on arithmetic;
- IR verification after every compiler pass.

The adstack-overflow check on reverse-mode autodiff runs unconditionally on every backend regardless of `debug`; see [Autodiff -> What can go wrong](autodiff.md) for the contract.

**Cost.** Significant on both compile time (verifier walks the IR after every transform; extra runtime checks expand the emitted code; ~21s extra observed on adstack-heavy kernels) and runtime. For just the field-bounds check in a release build without the rest, use [`check_out_of_bound`](#check_out_of_bound) below.

### `check_out_of_bound`

Default `False`. Enables the field-bounds check on tensor indexing - an out-of-range index raises `RuntimeError`.

**Cost.** Scales with how often kernels index into tensors. Cheaper than `debug=True`. Still leave off for benchmarks.

Interaction with `debug`:

| Flags | Field bounds | Other `debug` checks |
|-------|--------------|----------------------|
| neither | off | off |
| `check_out_of_bound=True` only | on | off |
| `debug=True` | on | on |

- `debug=True` always implies `check_out_of_bound=True` (the field-bounds check fires whenever debug mode is on).

Per-backend support:

| Backend | Field bounds check |
|---------|--------------------|
| CPU | with `check_out_of_bound=True` or `debug=True` |
| CUDA | with `check_out_of_bound=True` or `debug=True` |
| AMDGPU | with `check_out_of_bound=True` or `debug=True` |
| Metal | never (no in-kernel assertion mechanism) |
| Vulkan | never (no in-kernel assertion mechanism) |

Metal and Vulkan lack the assertion extension that the field-bounds check relies on; `check_out_of_bound=True` is silently reset to `False` on those backends at `qd.init` time and a warning is logged.
