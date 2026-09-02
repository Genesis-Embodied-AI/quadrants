# Kernel caching: whole-kernel vs per-offload

Quadrants caches compiled kernels so that re-running an unchanged kernel does not recompile it. There are two granularities at which this caching happens. Both produce identical results; they differ only in how much has to be recompiled after you edit a kernel.

- **Whole-kernel caching.** The whole `@qd.kernel` is compiled and cached as a single unit. Editing any part of the kernel invalidates that entry, so the entire kernel recompiles.
- **Per-offload caching** *(default on CUDA/AMD GPUs)*. The kernel is split into its top-level constructs (independent top-level loops or serial runs) and each is compiled in isolation, so the compiled result is cached per offloaded task. Editing one construct recompiles only that construct; the others are reused.

## Which is better

| | Advantage |
|---|---|
| Per-offload | Fast incremental recompiles. Editing one task in a large kernel (for example a big [`qd.graph.do_while`](graph.md) loop) reuses the other tasks' compiled code instead of recompiling the whole kernel. This is the win for large kernels you iterate on. |
| Whole-kernel | Simplest and universal. It is always correct and is the only option for kernels the split cannot compile construct-by-construct equivalently. |

Per-offload caching is **on by default on CUDA and AMD GPUs** with the [offline cache](init_options.md#offline_cache) enabled (itself the default). On other backends (CPU, Metal, Vulkan), or with the offline cache disabled, Quadrants uses whole-kernel caching. Beyond that, Quadrants falls back to the whole-kernel path automatically and transparently whenever the split would not be equivalent, or cannot be verified as safe for a given launch. You never choose between them and the results are identical either way; the only observable difference is compile time.

## When it falls back to whole-kernel

Some fallbacks are decided once, at compile time (the kernel never uses the split). Others are decided per launch (the kernel is split, but a particular call runs the whole-kernel variant instead).

**Compile-time fallback** (this kernel stays whole-kernel):

| Condition | Reason |
|---|---|
| No per-task reuse tier: CPU/Metal/Vulkan, or the offline cache disabled | There is nowhere to reuse per-construct results, so splitting would only add compile time. |
| [Autodiff](autodiff.md) (gradient) kernels | Gradient computation is not construct-isolatable. |
| A value carried between constructs | A local one construct builds up and another reads cannot be recomputed in isolation. |
| A snapshot read a later construct would re-read after an intervening write | Isolating it would read the overwritten value. Includes `boundary="clamp"` accesses, where an out-of-range index can collapse onto a written element. |
| Cross-argument gradient disjointness | Whether two gradient buffers alias cannot be checked at launch, so the split is refused. |
| Certain specialized kernels | For example task barriers or side-effecting constructs. |

**Runtime fallback** (a per-launch decision for a split kernel):

| Condition | Reason |
|---|---|
| Two parameters the split assumed disjoint are bound to the same buffer | Recomputing a read across the aliased write could change the result, so the call uses the whole-kernel variant. |
| A parameter's backing buffer cannot be read to confirm disjointness | For example a raw NumPy array or PyTorch tensor passed directly. The call conservatively uses the whole-kernel variant. Passing `qd.ndarray` / `qd.Tensor` objects avoids this. |

A runtime fallback emits a one-time `[PER_OFFLOAD][FALLBACK]` warning naming the kernel. It is suppressible through the logging level (it uses the standard `warn` level).

## Inspecting the split

See [Inspecting the split](optimization_passes.md#inspecting-the-split) for the `per_offload_cache_observations` attribute. `frontend_constructs_total == -1` means the split did not run for that compile: either it took a fallback above, or the kernel was served from the [offline cache](init_options.md#offline_cache) or [fastcache](fastcache.md), so no frontend ran at all.
