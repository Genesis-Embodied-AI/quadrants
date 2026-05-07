# Grid primitives

Grid-level primitives operate across **all blocks of a single kernel launch** — i.e. the entire device for the duration of one kernel. They sit one tier above block-scope primitives and one tier below "finish the kernel and launch a new one" (the only fully cross-block thread synchronization Quadrants offers).

Grid ops live under `qd.simt.grid`. The namespace currently contains a single op, the device-scope memory fence:

## What's available

| Op                       | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|--------------------------|------|--------|-------------------------|
| `qd.simt.grid.memfence()`| yes  | no     | no                      |

Calling a backend marked "no" raises a runtime / link-time error. AMDGPU and SPIR-V lowerings are tracked as future work; until they land, kernels that need a grid-scope fence are CUDA-only.

Naming note: `qd.simt.grid.memfence()` will be renamed to `qd.simt.grid.mem_fence()` (note the underscore) in the near future, for consistency with the `mem_fence` spelling used at other scopes. The new name is not yet available; this page uses the current name throughout.

## Barrier vs fence at grid scope

There is no `grid.sync()` — Quadrants does not expose a thread-converging barrier across blocks within a single kernel launch. The reasons are practical: CUDA cooperative groups need launch-time opt-in, AMDGPU and SPIR-V either lack a comparable primitive or expose it only under non-portable extensions, and the latency of an in-kernel grid barrier is comparable to a kernel relaunch on most hardware.

What Quadrants does provide at grid scope is a pure **memory fence**:

- **`qd.simt.grid.memfence()`** orders memory operations issued by the calling thread so that prior writes are visible to other threads **anywhere in the grid** (across blocks) before any subsequent read in the calling thread can be reordered ahead of the fence. It does **not** synchronize threads.
- For full thread synchronization across the grid, finish the current kernel and launch a new one — the implicit kernel-end barrier is the canonical cross-block synchronization in Quadrants.

## Semantics

### `qd.simt.grid.memfence()`

**Planned rename: `qd.simt.grid.mem_fence()`** (with underscore). The op will be renamed in a future release for consistency with the `mem_fence` spelling at other scopes; the current `memfence` name remains the only spelling available today, and the rest of this section uses it.

A device-scope memory fence. Lowers to `__threadfence()` (`nvvm_membar_gl`) on CUDA. No convergence requirement — safe to call from divergent control flow (e.g. inside `if tid == 0`).

Use this when one block (or one thread per block) needs to publish data to global memory and have the publication be visible to other blocks **without** waiting at a kernel boundary. The canonical use case is the decoupled-look-back pattern in Onesweep-style device scans:

```python
@qd.kernel
def lookback_scan(...) -> None:
    bid = qd.simt.block.global_thread_idx() // BLOCK_SIZE
    tid = qd.simt.block.global_thread_idx() %  BLOCK_SIZE

    block_sum = ...

    if tid == 0:
        partials[bid] = block_sum
        qd.simt.grid.memfence()
        flags[bid] = STATE_AGGREGATE

    if tid == 0:
        prev = bid - 1
        while prev >= 0:
            while flags[prev] == STATE_INVALID:
                qd.simt.grid.memfence()
            qd.simt.grid.memfence()
            block_sum += partials[prev]
            ...
```

The three `grid.memfence()` calls are doing different jobs:

1. The first orders the publication: any block reading `flags[bid] == STATE_AGGREGATE` is guaranteed to also see the published `partials[bid]`.
2. The second is **inside** the spin-wait, and is what forces each iteration to actually re-fetch `flags[prev]` from global memory (see "Spin-wait gotcha" below).
3. The third is the symmetric reader-side fence: after observing the predecessor's flag, the reader needs to refresh its view of `partials[prev]` (the scope of which is already global, but the fence pins down the ordering).

The fence does not require thread convergence, which is why it appears inside `if tid == 0` without deadlocking — `qd.simt.block.sync()` would deadlock there; `grid.memfence()` is safe.

#### Spin-wait gotcha

Quadrants ndarray reads compile to plain LLVM loads with no ordering or volatility. A naive spin-wait

```python
while flags[prev] == STATE_INVALID:
    pass
```

is therefore unsafe: LLVM's loop-invariant-code-motion will hoist the load out of the loop (the loop body has no aliasing writes), so once the first read sees `STATE_INVALID` the loop never observes the producer's update — it spins forever. Two correct spellings:

- **Fence inside the loop body** (used in the example above):

  ```python
  while flags[prev] == STATE_INVALID:
      qd.simt.grid.memfence()
  ```

  `__threadfence()` has compiler-visible global side effects, which prevents LICM from hoisting the `flags[prev]` load across it; each iteration is forced to re-fetch from global memory. Cost: one fence per spin iteration.

- **Atomic load via `atomic_or` with zero**:

  ```python
  while qd.atomic_or(flags[prev], 0) == STATE_INVALID:
      pass
  ```

  `qd.atomic_or(x, 0)` is an atomic op that returns the current value without modifying it, which forces a real memory access on every iteration and additionally pins down ordering. Slightly cheaper than a per-iteration full grid fence on most hardware, and arguably the cleaner expression of "atomically load this slot".

## Performance and portability notes

- **CUDA-only today.** AMDGPU and SPIR-V lowerings are not implemented; calling `grid.memfence()` on those backends raises at trace time. Cross-platform code that needs a grid-scope fence must currently CUDA-bound the kernel, or restructure to use the kernel-launch boundary.
- **Cost scales with the global-cache invalidation domain.** A grid fence drains the L2 (and on some GPUs the L1) caches of all SMs / CUs touching the address. On A100 / H100 the cost is on the order of tens to low hundreds of nanoseconds per call; in tight loops, prefer batching multiple cross-block updates per fence.
- **Pair with the right ordering of memory ops.** The fence orders the *calling thread*'s memory ops; readers in other blocks need their own fence (or an atomic load) to refresh their view. The producer-fence + consumer-fence pattern is the canonical idiom.
- **Not a substitute for atomics on contended locations.** A fence orders writes but does not serialize them. If multiple blocks write to the same location, you need an atomic regardless of how the fence is placed.

## Related

- `qd.simt.block.*` — the block-scope counterpart, including `qd.simt.block.mem_sync()` (block-scope fence) and `qd.simt.block.sync()` (block-scope barrier).
- `qd.simt.subgroup.*` — subgroup-scope barriers, fences, shuffles, and reductions.
- [parallelization](parallelization.md) — the broader synchronization story; explains how grid-scope fences fit relative to atomics, block barriers, and the kernel boundary.
