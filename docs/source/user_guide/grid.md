# Grid primitives

Grid-level primitives operate across **all blocks of a single kernel launch** — i.e. the entire device for the duration of one kernel. They sit one tier above block-scope primitives and one tier below "finish the kernel and launch a new one" (the only fully cross-block thread synchronization Quadrants offers).

Grid ops live under `qd.simt.grid`. The namespace currently contains a single op, the device-scope memory fence:

## What's available

| Op                        | CUDA | AMDGPU | Vulkan | Metal |
|---------------------------|------|--------|--------|-------|
| `qd.simt.grid.mem_fence()`| yes  | yes    | yes    | yes\* |

\* On Metal, `grid.mem_fence()` lowers (via MoltenVK / SPIRV-Cross → MSL) to `atomic_thread_fence(metal::memory_scope_device)`. Per the Metal Shading Language specification, that builtin only synchronizes *atomic* and *relaxed-atomic* memory accesses across the device, not plain (non-atomic) loads / stores. So a Metal-portable cross-workgroup producer-consumer pattern needs the produced value itself to be published through an atomic op (e.g. `qd.atomic_or(a[i], 1)`); CUDA, AMDGPU, and native Vulkan (Linux / Windows) order non-atomic stores around `grid.mem_fence()` strictly. The same caveat applies to **Vulkan on macOS**, since on macOS Vulkan is really MoltenVK lowering SPIR-V to MSL — treat `qd.vulkan` on Darwin the same as `qd.metal` for `grid.mem_fence()` purposes. Empirically validate any other pattern, or split into two kernel launches.

Per-backend lowering:

- **CUDA**: `__threadfence()` (NVVM intrinsic `nvvm_membar_gl`).
- **AMDGPU**: an LLVM IR `fence syncscope("agent") acq_rel` synthesized in `llvm_context.cpp::patch_fence` after the bitcode is retargeted to AMDGCN. (We can't use `__builtin_amdgcn_fence` directly because the runtime stubs are compiled host-side.)
- **Vulkan**: SPIR-V `OpMemoryBarrier(ScopeDevice, AcquireRelease | UniformMemory)` emitted by `spirv_codegen.cpp` for the `gridMemoryBarrier` internal op.
- **Metal**: same SPIR-V op as Vulkan; MoltenVK / SPIRV-Cross translates it to MSL `atomic_thread_fence(memory_scope_device)`. Available since MSL 2.0 (macOS 10.13+ / iOS 11+).

### Naming and deprecated alias

The op was previously named `qd.simt.grid.memfence()` (no underscore). It is still callable under that name as a deprecated alias that emits `DeprecationWarning` on first use; new code should use `qd.simt.grid.mem_fence()`. The alias will be removed in a future release.

## Barrier vs fence at grid scope

There is no `grid.sync()` — Quadrants does not expose a thread-converging barrier across blocks within a single kernel launch. The reasons are practical: CUDA cooperative groups need launch-time opt-in, AMDGPU and SPIR-V either lack a comparable primitive or expose it only under non-portable extensions, and the latency of an in-kernel grid barrier is comparable to a kernel relaunch on most hardware.

What Quadrants does provide at grid scope is a pure **memory fence**:

- **`qd.simt.grid.mem_fence()`** orders memory operations issued by the calling thread so that prior writes are visible to other threads **anywhere in the grid** (across blocks) before any subsequent read in the calling thread can be reordered ahead of the fence. It does **not** synchronize threads.
- For full thread synchronization across the grid, finish the current kernel and launch a new one — the implicit kernel-end barrier is the canonical cross-block synchronization in Quadrants.

## Semantics

### `qd.simt.grid.mem_fence()`

A device-scope memory fence. Lowering on each backend is described in "Per-backend lowering" above. No convergence requirement on any backend — safe to call from divergent control flow (e.g. inside `if tid == 0`).

Use this when one block (or one thread per block) needs to publish data to global memory and have the publication be visible to other blocks **without** waiting at a kernel boundary. The canonical use case is the decoupled-look-back pattern in Onesweep-style device scans:

```python
@qd.kernel
def lookback_scan(...) -> None:
    bid = qd.simt.block.global_thread_idx() // BLOCK_SIZE
    tid = qd.simt.block.global_thread_idx() %  BLOCK_SIZE

    block_sum = ...

    if tid == 0:
        partials[bid] = block_sum
        qd.simt.grid.mem_fence()
        flags[bid] = STATE_AGGREGATE

    if tid == 0:
        prev = bid - 1
        while prev >= 0:
            while flags[prev] == STATE_INVALID:
                qd.simt.grid.mem_fence()
            qd.simt.grid.mem_fence()
            block_sum += partials[prev]
            ...
```

The three `grid.mem_fence()` calls are doing different jobs:

1. The first orders the publication: any block reading `flags[bid] == STATE_AGGREGATE` is guaranteed to also see the published `partials[bid]`.
2. The second is **inside** the spin-wait, and is what forces each iteration to actually re-fetch `flags[prev]` from global memory (see "Spin-wait gotcha" below).
3. The third is the symmetric reader-side fence: after observing the predecessor's flag, the reader needs to refresh its view of `partials[prev]` (the scope of which is already global, but the fence pins down the ordering).

The fence does not require thread convergence, which is why it appears inside `if tid == 0` without deadlocking — `qd.simt.block.sync()` would deadlock there; `grid.mem_fence()` is safe.

> **Metal / Vulkan-on-macOS portability:** the example as written above relies on the producer's plain (non-atomic) store `flags[bid] = STATE_AGGREGATE` becoming visible to other workgroups once `grid.mem_fence()` retires. CUDA, AMDGPU, and native Vulkan honor this strictly; Metal (and therefore Vulkan-on-macOS) does **not** — `atomic_thread_fence(memory_scope_device)` only orders atomic accesses across the device. To make this idiom Metal-portable, publish through an atomic store (`qd.atomic_or(flags[bid], STATE_AGGREGATE)`) or split the producer and consumer phases into separate kernel launches.

#### Spin-wait gotcha

Quadrants ndarray reads compile to plain LLVM loads with no ordering or volatility. A naive spin-wait

```python
while flags[prev] == STATE_INVALID:
    pass
```

is therefore unsafe: LLVM's loop-invariant-code-motion will hoist the load out of the loop (the loop body has no aliasing writes), so once the first read sees `STATE_INVALID` the loop never observes the producer's update — it spins forever. Three correct spellings, in order of preference:

- **Volatile load** (recommended; cross-backend, cheapest):

  ```python
  while qd.volatile_load(flags[prev]) == STATE_INVALID:
      pass
  ```

  `qd.volatile_load` lowers to LLVM `load volatile` on CUDA / AMDGPU and to `OpLoad` with the SPIR-V `Volatile` `MemoryAccess` mask on Vulkan / Metal — the optimiser is forbidden from hoisting / merging the load on every backend, with no per-iteration cache-flush or atomic-RMW overhead. See [atomics](atomics.md) for the full primitive description, including the producer-side pairing requirements (atomic store, or plain store + fence on non-Metal backends).

- **Fence inside the loop body** (used in the example above; legacy spelling):

  ```python
  while flags[prev] == STATE_INVALID:
      qd.simt.grid.mem_fence()
  ```

  The fence has compiler-visible global side effects, which prevents LICM from hoisting the `flags[prev]` load across it; each iteration is forced to re-fetch from global memory. Cost: one fence per spin iteration (an order of magnitude more than a volatile load). Works on CUDA, AMDGPU, and native Vulkan; **does not work on Metal / Vulkan-on-macOS** (the producer's plain store may never become visible — see the portability note above). Prefer `qd.volatile_load` for new code.

- **Atomic load via `atomic_or` with zero** (legacy spelling):

  ```python
  while qd.atomic_or(flags[prev], 0) == STATE_INVALID:
      pass
  ```

  `qd.atomic_or(x, 0)` is an atomic op that returns the current value without modifying it, which forces a real memory access on every iteration and additionally pins down ordering. Cheaper than a per-iteration full grid fence but still pays the atomic-RMW hardware cost (and contention with concurrent stores on the same cell). Was previously the only Metal-portable spelling; `qd.volatile_load` now covers Metal too and is preferred for new code. Useful as a paired idiom with an atomic *store* on the producer side (`qd.atomic_or(flags[bid], STATE_AGGREGATE)`) when the producer also wants atomic-store semantics.

## Performance and portability notes

- **GPU-portable.** All four GPU backends implement `grid.mem_fence()` natively (see "Per-backend lowering"). The one residual portability gap is Metal / Vulkan-on-macOS only ordering *atomic* memory accesses across the device — patterns relying on plain stores becoming visible across workgroups must publish through atomics on those targets.
- **Cost scales with the global-cache invalidation domain.** A grid fence drains the L2 (and on some GPUs the L1) caches of all SMs / CUs touching the address. On A100 / H100 the cost is on the order of tens to low hundreds of nanoseconds per call; AMDGPU `acq_rel` agent fences and Vulkan `OpMemoryBarrier(ScopeDevice)` are in the same ballpark on contemporary hardware. In tight loops, prefer batching multiple cross-block updates per fence.
- **Pair with the right ordering of memory ops.** The fence orders the *calling thread*'s memory ops; readers in other blocks need their own fence (or an atomic load) to refresh their view. The producer-fence + consumer-fence pattern is the canonical idiom.
- **Not a substitute for atomics on contended locations.** A fence orders writes but does not serialize them. If multiple blocks write to the same location, you need an atomic regardless of how the fence is placed.

## Related

- `qd.simt.block.*` — the block-scope counterpart, including `qd.simt.block.mem_fence()` (block-scope fence) and `qd.simt.block.sync()` (block-scope barrier).
- `qd.simt.subgroup.*` — subgroup-scope barriers, fences, shuffles, and reductions.
- [parallelization](parallelization.md) — the broader synchronization story; explains how grid-scope fences fit relative to atomics, block barriers, and the kernel boundary.
