# qd.init options

`qd.init(...)` accepts a range of keyword options that tune how Quadrants compiles and runs your kernels; most are also settable as an environment variable of the form `QD_<UPPERCASE_NAME>` (e.g. `QD_OFFLINE_CACHE=0`). This page walks through the options most commonly tuned in practice; the [full list of options](#all-options) is at the bottom.

## Caching

### `offline_cache`

Whether the compilation caches **persist on disk across Python invocations**. Default `True`. The "offline" in the name refers to the fact that this cache outlives the process: it is what makes the *second* time you start a Python interpreter and run a kernel cheap, by reusing artifacts from the first run.

Setting `offline_cache=False` is intended to emulate cold-start, i.e. a fresh Python process with no prior on-disk artifacts available. In-process caches operate independently of this flag: while a runtime is alive, identical kernels are not recompiled regardless of its value (though `qd.reset()` clears that in-process cache, so kernels recompile after a reset). The flag therefore controls only whether the next Python invocation observes a warm or a cold disk.

When `offline_cache=True`, compilation artifacts persist on disk under `offline_cache_file_path` (default `~/.cache/quadrants/qdcache`), so a later Python process reuses them instead of recompiling. Setting `offline_cache=False` (or `QD_OFFLINE_CACHE=0`) forces a cold start: Quadrants recompiles kernels and neither reads nor writes its own on-disk cache. (On CUDA the driver keeps its own separate cache of compiled GPU code at `~/.nv/ComputeCache` that this flag does not disable; `offline_cache=False` only stops that cache from serving results across runs. Set `CUDA_CACHE_DISABLE=1` to turn it off entirely.)

The separate source-level cache used by [fastcache](./fastcache.md) kernels is controlled by `src_ll_cache` (on by default), not by `offline_cache`; with `offline_cache=False` it still writes its own bookkeeping files to disk, so set `src_ll_cache=False` as well to stop that too.

When to set it to `False`:
- Taking compile-time profiles where a cached kernel would mask the real cost.
- Investigating a stale-cache bug or suspected cache corruption.
- Reproducing first-run behavior in CI matrix runs that would otherwise warm the caches across iterations.

For normal use, leave it at `True`; the caches are the main reason a repeated run starts up quickly.

## Compile-time tuning

### `cfg_optimization`

Whether to run the control-flow-graph optimization (an internal compile-time optimization of your kernel's branches and loops). Default `True`. Setting it to `False` makes compilation up to 6x faster while costing 1-5% of runtime speed; consider disabling it if compile time is the bottleneck and the runtime delta is acceptable.

### `fast_math`

Whether to enable relaxed floating-point optimizations (fusing multiply-add operations, and dropping NaN / infinity / signed-zero guarantees). Default `True`. Disable when investigating numerical anomalies or running deterministic-tolerance tests.

### `num_compile_threads`

Number of host threads used to compile a single kernel's internal tasks in parallel. Default `4`. When Quadrants compiles a kernel it first splits it into several tasks (roughly one per parallel loop) and hands them to a pool of this many threads, so a kernel that splits into many tasks compiles faster on a machine with idle cores. (Distinct kernels are still each compiled lazily the first time they run; this option speeds up the compilation of one such kernel, not scheduling across kernels.) Lower it, or set `1`, on memory-constrained systems where many concurrent compilations would thrash memory. Only the LLVM backends (CPU, CUDA, AMDGPU) use it.

## Reverse-mode autodiff

See [Autodiff](./autodiff.md) for the reverse-mode pipeline overview.

### `ad_stack_experimental_enabled`

Enables the dynamic-loop reverse-mode pipeline (the *adstack*). Default `False`. Required when a reverse-mode kernel has a runtime-bounded loop carrying a non-linear primal (a value computed on the forward pass that then feeds a non-linear operation); without it, such kernels either compile-error or produce silently-wrong gradients depending on the loop shape. See [Autodiff with dynamic loops](./autodiff.md#autodiff-with-dynamic-loops) for the rules. Adstack-on is safe even when not strictly needed, but it does come with a few drawbacks:

- **Memory.** The reverse pass replays each iteration of the dynamic loop, so the adstack stores per-iteration intermediate values for every thread. See [Memory footprint](./autodiff.md#memory-footprint) for the exact formula and the knobs that shrink it (`ad_stack_size`, `ad_stack_sparse_threshold_bytes`).
- **Per-launch overhead.** Every backward kernel launch incurs a small fixed CPU-to-GPU data transfer. Kernels whose dynamic loop is gated by a sparse predicate (e.g. `for i in range(n): if active[i] > 0: ...`) additionally run a fast GPU pre-step that counts how many threads pass the gate so that the adstack can be tightly sized instead of upper-bounded by worst case.

*Note.* These drawbacks affect only reverse-mode kernels that actually use the adstack; forward-only kernels and reverse-mode kernels without a dynamic non-linear inner loop pay nothing extra. In other words, enabling adstack globally is effectively free except for kernels that need it anyway!

### `ad_stack_size`

Forces every adstack in the program to exactly `N` slots and bypasses the launch-time sizer. Default `0`, meaning "let the sizer decide" (the recommended setting for day-to-day use). Setting a positive `N` is meant for stress tests or working around a suspected sizer bug; it defeats the per-launch-exact sizing so every dispatch allocates the full `N` slots whether or not the kernel actually needs them. Has no effect when `ad_stack_experimental_enabled=False`.

### `ad_stack_sparse_threshold_bytes`

Cutoff (in bytes) below which the gate-passing-count sizing path described in [Memory footprint](./autodiff.md#memory-footprint) is skipped in favor of the eager worst-case heap (sized for the full thread count instead of the gate-passing count). Default `100 MiB`. The sparse path saves memory on kernels of the shape `for i in range(...): if field[i] cmp literal: <adstack work>` but pays a per-launch reducer dispatch; below the threshold that overhead outweighs the savings. Set to `0` to always use the sparse path; lower it if the default still skips kernels you want shrunk. No effect when `ad_stack_experimental_enabled=False` or when the kernel has no such gate.

## Apple Metal

### `external_metal_command_queue`

An `MTLCommandQueue*` pointer (as an integer) to use instead of creating a new Metal command queue. Default `0` (create a new queue). When non-zero, Quadrants dispatches all GPU work on the provided queue, which enables GPU-side ordering with other frameworks that share the same queue (most notably PyTorch MPS).

### `external_metal_command_queue_is_torch_queue`

Default `False`. Set to `True` when the `external_metal_command_queue` is PyTorch MPS's command queue. This tells Quadrants that both frameworks share the same Metal queue, so the explicit `qd.sync()` / `torch.mps.synchronize()` calls at `to_torch` / `from_torch` interop points can be skipped. When `False` (or when no external queue is set), the interop syncs are preserved.

See [Shared Metal command queue](./metal_shared_queue.md) for the full setup guide, including how to extract the queue pointer from PyTorch and the synchronization implications.

## Debugging

See [Debug mode](./debug.md) for runnable examples and a typical develop / benchmark workflow.

### `debug`

Default `False`. Turns on every available correctness check. Use while iterating on a kernel that produces wrong numerics; turn off for benchmarks and production.

Enables:
- field-bounds check on tensor indexing (out-of-range index raises `RuntimeError`);
- kernel `assert` statements;
- integer-overflow guards on arithmetic;
- extra internal consistency checks throughout compilation.

The adstack-overflow check on reverse-mode autodiff runs unconditionally on every backend regardless of `debug`; see [Autodiff -> What can go wrong](./autodiff.md#what-can-go-wrong) for the contract.

**Cost.** Significant on both compile time (extra checks are inserted and validated throughout compilation; e.g. ~21s of added compile time observed on adstack-heavy kernels) and runtime. For just the field-bounds check in a release build without the rest, use [`check_out_of_bound`](#check_out_of_bound) below.

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

## All options

Most `qd.init` keywords set a compiler-configuration option. Each option below can be passed as a keyword argument to `qd.init(...)`, and after initializing a compiled backend it is also readable and writable as an attribute on the configuration object `qd.cfg` (e.g. `qd.cfg.opt_level`). Because each option is both a `qd.init` argument and a `qd.cfg` attribute, the list below documents each as a *property* of `qd.cfg`, with its type, default value, and a short description.

These are compiler settings, so most do not apply to the pure-Python `qd.python` backend, for which `qd.cfg` is `None`. A few instead set language defaults (such as the default numeric types `default_fp` and `default_ip`) and still take effect on `qd.python`.

```{eval-rst}
.. autoclass:: quadrants._lib.core.quadrants_python.CompileConfig
   :members:
   :exclude-members: default_up
```

`qd.init` also accepts a number of options that are handled on the Python side and so do not appear in the generated list above. Some can only be set in the `qd.init` call; others are also reachable through a `QD_` environment variable, as noted per option:

- `enable_fallback` (`bool`, default `True`): fall back to the CPU backend when the requested `arch` is unavailable, instead of raising an error. No environment variable equivalent.
- `src_ll_cache` (`bool`, default `True`): use an additional source-level on-disk cache that speeds up loading previously compiled kernels. It only applies to kernels declared `@qd.kernel(fastcache=True)` (or the deprecated `@qd.kernel(pure=True)`; see [fastcache](./fastcache.md)). Reusing a kernel's compiled code across processes also needs the offline cache, so with `offline_cache=False` it no longer speeds up loading, but it still does source-cache bookkeeping on disk; set `src_ll_cache=False` to turn it off entirely. No environment variable equivalent.
- `require_version` (`str`): raise an error unless the installed Quadrants version is compatible with the given `major.minor.patch` string (same major version, and at least the given minor and patch). No environment variable equivalent.
- `print_non_pure` (`bool`, default `False`): print the name of each executed kernel that is not *declared* pure, i.e. not marked `@qd.kernel(fastcache=True)` (or the deprecated `@qd.kernel(pure=True)`). This is a declaration check, not an analysis of what the kernel actually touches: a plain `@qd.kernel` is reported even if it only uses its explicit parameters. Only kernels declared pure can use [fastcache](./fastcache.md) to speed up load, so use this to find kernels that could opt in. No environment variable equivalent.
- `log_level` (`str`, default `"info"`): logging verbosity; one of `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`, `"critical"`, or `"off"` to disable logging (also settable via `QD_LOG_LEVEL`).
- `gdb_trigger` (`bool`, default `False`): drop into gdb when Quadrants' compiled C++ runtime crashes (a native crash rather than a Python exception) (also settable via `QD_GDB_TRIGGER`).
- `short_circuit_operators` (`bool`, default `True`): use short-circuit evaluation for `and`/`or` inside kernels (also settable via `QD_SHORT_CIRCUIT_OPERATORS`).
- `print_full_traceback` (`bool`, default `False`): print the full Python traceback when an exception propagates out of Quadrants (also settable via `QD_PRINT_FULL_TRACEBACK`).
- `unrolling_limit` (`int`, default `32`): maximum number of iterations a static loop may be unrolled before a warning is emitted; `0` disables the warning (also settable via `QD_UNROLLING_LIMIT`).
