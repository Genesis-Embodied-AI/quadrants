# Grid primitives

Grid-level primitives operate across **all blocks of a single kernel launch** — i.e. the entire device for the duration of one kernel. They sit one tier above block-scope primitives and one tier below "finish the kernel and launch a new one" (the only fully cross-block thread synchronization Quadrants offers).

Grid ops live under `qd.simt.grid`. The namespace currently contains a single op, the device-scope memory fence:

## What's available

| Op                       | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|--------------------------|------|--------|-------------------------|
| `qd.simt.grid.memfence()`| yes  | no     | no                      |

Calling a backend marked "no" raises a runtime / link-time error. AMDGPU and SPIR-V lowerings are tracked as future work; until they land, kernels that need a grid-scope fence are CUDA-only.

## Barrier vs fence at grid scope

There is no `grid.sync()` — Quadrants does not expose a thread-converging barrier across blocks within a single kernel launch. The reasons are practical: CUDA cooperative groups need launch-time opt-in, AMDGPU and SPIR-V either lack a comparable primitive or expose it only under non-portable extensions, and the latency of an in-kernel grid barrier is comparable to a kernel relaunch on most hardware.

What Quadrants does provide at grid scope is a pure **memory fence**:

- **`qd.simt.grid.memfence()`** orders memory operations issued by the calling thread so that prior writes are visible to other threads **anywhere in the grid** (across blocks) before any subsequent read in the calling thread can be reordered ahead of the fence. It does **not** synchronize threads.
- For full thread synchronization across the grid, finish the current kernel and launch a new one — the implicit kernel-end barrier is the canonical cross-block synchronization in Quadrants.

The corresponding distinctions at narrower scopes:

- Block scope: `qd.simt.block.sync()` (barrier) vs `qd.simt.block.mem_sync()` (fence).
- Subgroup scope: `qd.simt.subgroup.barrier()` (barrier) vs `qd.simt.subgroup.memory_barrier()` (fence).

A useful mental model: barriers converge threads, fences order memory; grid scope only offers the fence.

## Semantics

### `qd.simt.grid.memfence()`

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
                pass
            qd.simt.grid.memfence()
            block_sum += partials[prev]
            ...
```

The two `grid.memfence()` calls are doing different jobs:

1. The first orders the publication: any block reading `flags[bid] == STATE_AGGREGATE` is guaranteed to also see the published `partials[bid]`.
2. The second is the symmetric reader-side fence: after observing the predecessor's flag, the reader needs to refresh its view of `partials[prev]` (the scope of which is already global, but the fence pins down the ordering).

The fence does not require thread convergence, which is why it appears inside `if tid == 0` without deadlocking — `qd.simt.block.sync()` would deadlock there; `grid.memfence()` is safe.

## When to use which fence

| Scope you need to publish to        | Use                                  |
|-------------------------------------|--------------------------------------|
| Other lanes in the same subgroup    | `qd.simt.subgroup.memory_barrier()`  |
| Other threads in the same block     | `qd.simt.block.mem_sync()`           |
| Other blocks in the same grid       | `qd.simt.grid.memfence()`            |
| Threads of a *future* kernel launch | (implicit at the kernel boundary; no explicit fence required from Python) |

A wider scope is always sound — `grid.memfence()` is a strict superset of `block.mem_sync()`'s ordering — but slower, because more caches need to be drained. Pick the narrowest scope that matches your sharing pattern.

## Examples

### Cross-block reduction with a single kernel

```python
NUM_BLOCKS = 64
BLOCK_SIZE = 256

partials = qd.field(qd.f32, shape=(NUM_BLOCKS,))
flags    = qd.field(qd.i32, shape=(NUM_BLOCKS,))

@qd.kernel
def reduce_one_pass(input: qd.types.NDArray[qd.f32, 1], result: qd.types.NDArray[qd.f32, 1]) -> None:
    qd.loop_config(block_dim=BLOCK_SIZE)

    bid = qd.simt.block.global_thread_idx() // BLOCK_SIZE
    tid = qd.simt.block.global_thread_idx()  % BLOCK_SIZE

    local = qd.f32(0.0)
    i = bid * BLOCK_SIZE + tid
    if i < input.shape[0]:
        local = input[i]

    block_sum = qd.simt.subgroup.reduce_all_add(local, 5)

    if tid == 0:
        partials[bid] = block_sum
        qd.simt.grid.memfence()
        flags[bid] = 1

    if bid == NUM_BLOCKS - 1 and tid == 0:
        for b in range(NUM_BLOCKS - 1):
            while flags[b] == 0:
                pass
            qd.simt.grid.memfence()
        total = qd.f32(0.0)
        for b in range(NUM_BLOCKS):
            total += partials[b]
        result[0] = total
```

The two-stage publish-then-flag pattern is exactly what Onesweep, decoupled-look-back scan, and persistent-thread reductions are built on. Without the `grid.memfence()`, a reader could observe `flags[b] == 1` while still seeing the old value of `partials[b]`.

### When to *not* use it

If you only need to publish across threads of the same block, `qd.simt.block.mem_sync()` is several times cheaper. Use `grid.memfence()` only for true cross-block coordination.

If you need every thread in the grid to *converge* (not just see each other's writes), there is no in-kernel primitive — finish the kernel and launch a new one. Quadrants kernels are inherently asynchronous from each other only through the launch boundary, which doubles as a cross-grid barrier.

## Performance and portability notes

- **CUDA-only today.** AMDGPU and SPIR-V lowerings are not implemented; calling `grid.memfence()` on those backends raises at trace time. Cross-platform code that needs a grid-scope fence must currently CUDA-bound the kernel, or restructure to use the kernel-launch boundary.
- **Cost scales with the global-cache invalidation domain.** A grid fence drains the L2 (and on some GPUs the L1) caches of all SMs / CUs touching the address. On A100 / H100 the cost is on the order of tens to low hundreds of nanoseconds per call; in tight loops, prefer batching multiple cross-block updates per fence.
- **Pair with the right ordering of memory ops.** The fence orders the *calling thread*'s memory ops; readers in other blocks need their own fence (or an atomic load) to refresh their view. The producer-fence + consumer-fence pattern in the example above is the canonical idiom.
- **Not a substitute for atomics on contended locations.** A fence orders writes but does not serialize them. If multiple blocks write to the same location, you need an atomic regardless of how the fence is placed.

## Related

- `qd.simt.block.*` — the block-scope counterpart, including `qd.simt.block.mem_sync()` (block-scope fence) and `qd.simt.block.sync()` (block-scope barrier).
- `qd.simt.subgroup.*` — subgroup-scope barriers, fences, shuffles, and reductions.
- [parallelization](parallelization.md) — the broader synchronization story; explains how grid-scope fences fit relative to atomics, block barriers, and the kernel boundary.
