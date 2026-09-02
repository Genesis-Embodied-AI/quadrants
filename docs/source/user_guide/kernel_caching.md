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

**Compile-time fallback** (this kernel is never split; it always uses whole-kernel caching):

| Condition | What it means |
|---|---|
| [Autodiff](autodiff.md) (gradient) kernels | The backward pass links the constructs together, so they cannot be compiled one at a time. |
| [One construct produces a value a later construct consumes](#a-value-carried-between-constructs) | The later construct cannot be rebuilt on its own without redoing the earlier one. |
| [A construct re-reads array data an earlier construct overwrote](#re-reading-overwritten-data) | Compiling it in isolation would observe the new value instead of the original. This includes clamped indexing (`boundary="clamp"`), where an out-of-range index can land on an element that was written. |
| [Two array arguments where safety would depend on them not sharing memory, and one is a gradient](#possibly-aliasing-gradient-arguments) | Whether two gradient buffers overlap cannot be verified even at launch, so Quadrants refuses to split rather than risk a wrong result. |
| A side effect, random draw, or volatile read would be repeated | Recomputing it inside another construct would change what the kernel does. |
| Kernels that iterate over an unstructured mesh | Mesh iteration is not expressed as independent top-level constructs. |
| A region explicitly scheduled to run concurrently | Concurrent constructs share one scratch buffer, so isolating them would corrupt it. |
| A diagnostic mode is enabled ([line coverage](kernel_coverage.md), or a compiler debug dump, see [optimization passes](optimization_passes.md)) | These need the whole-kernel form to keep their diagnostics meaningful. |

**Runtime fallback** (a per-launch decision for a split kernel):

| Condition | Reason |
|---|---|
| Two parameters the split assumed disjoint are bound to the same buffer | Recomputing a read across the aliased write could change the result, so the call uses the whole-kernel variant. |
| A parameter's backing buffer cannot be read to confirm disjointness | For example a raw NumPy array or PyTorch tensor passed directly. The call conservatively uses the whole-kernel variant. Passing `qd.ndarray` / `qd.Tensor` objects avoids this. |

Each compile-time fallback logs a one-line `debug`-level message naming the kernel and the condition, so raising Quadrants' log level to `debug` tells you why a given kernel stayed whole-kernel. A runtime fallback instead emits a one-time `[PER_OFFLOAD][FALLBACK]` warning naming the kernel (at the standard `warn` level, so it is on by default and suppressible through the logging level).

## Compile-time fallback examples

Each kernel below trips one of the compile-time fallbacks above and stays whole-kernel. You can confirm it with [`per_offload_cache_observations`](optimization_passes.md#inspecting-the-split) (`frontend_constructs_total` is `-1`) or by reading the `debug`-level log line.

### A value carried between constructs

The first loop accumulates `total`; the second loop reads it. Because `total` exists only once the first loop has run, the split cannot compile the second loop on its own.

```python
@qd.kernel
def running_total(x: qd.types.NDArray[qd.f32, 1], out: qd.types.NDArray[qd.f32, 1]):
    total = 0.0
    for i in range(x.shape[0]):    # construct 1: build up a running total
        total += x[i]
    for i in range(out.shape[0]):  # construct 2: reuse it
        out[i] = total
```

Supporting a split for cases like this is possible but involved; see [Advanced: making these cases splittable](#advanced-making-these-cases-splittable).

### Re-reading overwritten data

`first` snapshots `a[0]`, then the first loop overwrites `a`. Recomputing that snapshot inside the second loop would read the overwritten value instead of the original.

```python
@qd.kernel
def stale_snapshot(a: qd.types.NDArray[qd.f32, 1], out: qd.types.NDArray[qd.f32, 1]):
    first = a[0]                   # snapshot a[0]
    for i in range(a.shape[0]):    # construct 1: overwrite a
        a[i] = 0.0
    for i in range(out.shape[0]):  # construct 2: reuse the snapshot
        out[i] = first
```

This shares the same fix as the previous case; see [Advanced: making these cases splittable](#advanced-making-these-cases-splittable).

### Possibly-aliasing gradient arguments

`a` and `b` are different parameters, so the split would treat their buffers as disjoint and reuse that assumption while recomputing. But one access is a gradient buffer, and a caller can make `a` and `b` (or a primal and a gradient) share memory, which cannot be verified even at launch, so Quadrants refuses to split.

```python
@qd.kernel
def grad_snapshot(a: qd.types.ndarray(dtype=qd.f32, ndim=1, needs_grad=True),
                  b: qd.types.ndarray(dtype=qd.f32, ndim=1, needs_grad=True),
                  out: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    g = a.grad[0]                  # snapshot a's gradient
    for i in range(b.shape[0]):    # construct 1: write b's gradient (a different argument)
        b.grad[i] = 0.0
    for i in range(out.shape[0]):  # construct 2: reuse the snapshot
        out[i] = g
```

This one is fixable too, though for a narrower case; see [Advanced: making these cases splittable](#advanced-making-these-cases-splittable).

## Advanced: making these cases splittable

This section is internal background for the curious; you never need it to use Quadrants.

To support [a value carried between constructs](#a-value-carried-between-constructs), the split would need to let one construct hand the value to the next through a small buffer that persists between them, instead of rebuilding each construct from the kernel's arguments alone. That buffer becomes a shared interface the cache has to track (its type and layout) so that editing one construct can't silently break the other. It's a real amount of machinery for an uncommon case, so today Quadrants takes the simpler, always-correct route and caches the whole kernel.

The same persisted-buffer idea covers [re-reading overwritten data](#re-reading-overwritten-data): capture the snapshot before the overwrite and store it for the second loop, instead of rebuilding it from the arguments (where it would re-read the overwritten `a`).

[Possibly-aliasing gradient arguments](#possibly-aliasing-gradient-arguments) is fixable too, but for a fairly narrow case (a non-autodiff kernel that manually reads two `.grad` buffers as plain arrays; true autodiff kernels are already excluded from the split). Today it refuses to split rather than assuming disjointness it can't verify.

## Inspecting the split

See [Inspecting the split](optimization_passes.md#inspecting-the-split) for the `per_offload_cache_observations` attribute. `frontend_constructs_total == -1` means the split did not run for that compile: either it took a fallback above, or the kernel was served from the [offline cache](init_options.md#offline_cache) or [fastcache](fastcache.md), so no compilation happened at all.
