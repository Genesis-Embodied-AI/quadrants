# Block primitives

Block-level primitives operate on the threads of a single CUDA thread block (CTA) / AMDGPU workgroup / Vulkan or Metal workgroup. They include thread barriers, memory fences, shared memory, and per-thread indexing helpers — the building blocks for cooperation among threads of the same block.

Block ops live under `qd.simt.block`. They are written so the same Python source compiles to the right vendor primitive on each backend; calling a backend that has not implemented an op raises `ValueError` from the Python layer at trace time.

The closely-related grid-level fence (`qd.simt.grid.memfence()`) is documented at the end of this page, since users picking between a block-scope and a device-scope fence need to see both side by side.

## What's available

| Op                                              | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|-------------------------------------------------|------|--------|-------------------------|
| `block.sync()`                                  | yes  | yes    | yes                     |
| `block.sync_all_nonzero(predicate)`             | yes  | no     | no                      |
| `block.sync_any_nonzero(predicate)`             | yes  | no     | no                      |
| `block.sync_count_nonzero(predicate)`           | yes  | no     | no                      |
| `block.mem_sync()`                              | yes\*| no     | yes                     |
| `block.SharedArray(shape, dtype)`               | yes  | yes    | yes                     |
| `block.global_thread_idx()`                     | yes  | yes    | —                       |
| `block.thread_idx()`                            | no   | no     | yes                     |
| `block.reduce_{add,min,max}(v, tid, ...)`       | yes  | yes    | yes                     |
| `block.reduce_all_{add,min,max}(v, tid, ...)`   | yes  | yes    | yes                     |
| `grid.memfence()` (device-scope, see below)     | yes  | no     | no                      |

Calling a backend marked "no" raises `ValueError` from the Python layer at trace time. `global_thread_idx()` is verified on CUDA and AMDGPU; the SPIR-V codepath in the wrapper exists but is currently unreachable due to a control-flow quirk in the dispatch and is therefore left undocumented here.

\* On CUDA, `block.mem_sync()` currently lowers via `block_barrier` (i.e. `__syncthreads()`), which doubles as a memory fence but additionally requires thread convergence — meaning calling it from divergent control flow today deadlocks. A fix to lower `mem_sync()` to a pure `__threadfence_block()` is in flight as [quadrants#637](https://github.com/Genesis-Embodied-AI/quadrants/pull/637); once merged, the divergent-branch pattern shown in the `block.mem_sync()` semantics section below works as written. Until then, prefer calling `mem_sync()` from uniform control flow on CUDA.

Naming note: two of the names on this page are planned to be renamed in a future release, to align with the project's "fence vs barrier" terminology and to use a consistent `mem_fence` spelling:

- `block.mem_sync()` will be renamed to `block.mem_fence()`.
- `grid.memfence()` will be renamed to `grid.mem_fence()` (note the underscore).

The new names are not yet available; this page uses the current names throughout.

## Barrier vs fence: the distinction that matters

Two of these ops sound similar but have very different semantics, and mixing them up deadlocks the GPU. The summary:

- `block.sync()` is a **thread-converging barrier**. Every thread in the block must reach the call site before any thread proceeds. It also implies a memory fence at block scope.
- `block.mem_sync()` (to be renamed `block.mem_fence()` in a future release) is a **memory fence only**, at block scope. It orders memory operations but does not require thread convergence — it is safe to call from divergent control flow (e.g. inside `if tid == 0`).

Concretely, on the SPIR-V backend `sync()` lowers to `workgroupBarrier` and `mem_sync()` lowers to `workgroupMemoryBarrier`. On CUDA, `sync()` lowers to `__syncthreads()`; `mem_sync()` is intended to lower to `__threadfence_block()` (a pure fence with no convergence requirement) — see the support-table caveat above. Calling `sync()` from a path that not all threads reach (a divergent `if`, an early `return`, etc.) is a classic GPU deadlock and applies to both backends.

The corresponding distinction at device scope is `grid.memfence()` (to be renamed `grid.mem_fence()` in a future release; memory fence across the entire grid, no thread synchronization). There is no block-style "device barrier" — to synchronize threads across blocks, finish the kernel and launch a new one.

## Semantics

### `block.sync()`

A block-wide thread-converging barrier. All threads in the block stop at the call until every thread has reached it; once all have arrived, all proceed. Reads and writes issued before the barrier are visible to other threads in the block after the barrier.

- Lowers to `__syncthreads()` (`nvvm_barrier_cta_sync_aligned_all`) on CUDA, `s_barrier` on AMDGPU, `workgroupBarrier` on SPIR-V.
- Must be called from uniform control flow within the block. Calling from a divergent branch deadlocks.

### `block.sync_all_nonzero(predicate)` / `sync_any_nonzero` / `sync_count_nonzero`

Block-wide barriers that also reduce a per-thread `i32` predicate across the block:

- `sync_all_nonzero(p)` returns non-zero if `p` is non-zero on **every** thread (logical AND).
- `sync_any_nonzero(p)` returns non-zero if `p` is non-zero on **any** thread (logical OR).
- `sync_count_nonzero(p)` returns the number of threads for which `p` is non-zero (popcount).

Each call performs both the synchronization (same convergence requirement as `sync()`) and the reduction in a single instruction. Only available on CUDA today; they lower to the NVPTX `barrier.cta.red` family of intrinsics (`block_barrier_and_i32`, `block_barrier_or_i32`, `block_barrier_count_i32`).

### `block.mem_sync()`

**Planned rename: `block.mem_fence()`.** This op will be renamed to `block.mem_fence()` in a future release. The current name (`mem_sync()`) remains the only spelling available today; the rest of this section uses it.

A block-scope memory fence. Orders memory operations issued by the calling thread so that prior writes are visible to other threads in the block before any subsequent read by the calling thread can be reordered ahead of the fence. It does **not** synchronize threads — no convergence requirement (subject to the CUDA caveat in the support table).

- Lowers to `__threadfence_block()` (`nvvm_membar_cta`) — the intended target — on CUDA, and to `workgroupMemoryBarrier` on SPIR-V.
- AMDGPU support is currently unimplemented and raises `ValueError` at trace time.
- Use this when one thread in the block (e.g. lane 0) needs to publish data to shared memory and have the publication be visible to the rest of the block without forcing the publishing thread to wait at a barrier. The pattern is typically:

  ```python
  if tid == 0:
      shared[...] = computed_value
      qd.simt.block.mem_sync()
      shared_flag[0] = 1
  qd.simt.block.sync()
  ```

  The `mem_sync()` here orders the data write before the flag write; the `sync()` is what converges the other threads so they observe the published flag.

### `block.SharedArray(shape, dtype)`

Allocates a shared-memory array, scoped to the calling block.

- `shape`: an `int` (1-D) or a `tuple` / `list` of `int`s (multi-dim). Must be compile-time constants — shared memory is statically allocated per block.
- `dtype`: a scalar Quadrants dtype (`qd.f32`, `qd.i32`, ...) or a `qd.types.matrix(...)` / `qd.types.vector(...)` type. Matrix types are flattened to their element tensor type.

Element access uses the standard `arr[i]` / `arr[i, j]` subscript syntax inside a kernel.

A worked example with `Tile16x16` interaction is in [tile16](tile16.md).

### `block.global_thread_idx()`

Returns the global thread index of the calling thread within the kernel launch. Verified on CUDA and AMDGPU.

On CUDA / AMDGPU this is the natural way to identify which work-item a thread should process when the kernel uses `qd.loop_config(block_dim=...)` — together with `block_dim`, you can recover the in-block thread index via `global_thread_idx() % block_dim`.

### `block.thread_idx()`

Returns the local thread index within the block. Currently only implemented on SPIR-V backends (Vulkan / Metal); on CUDA / AMDGPU it raises. On CUDA / AMDGPU, use `global_thread_idx()` together with `qd.loop_config(block_dim=...)` and recover the in-block index via `global_thread_idx() % block_dim`.

### `block.reduce_{add,min,max}(value, tid, block_dim, log2_warp, dtype)`

Block-scope reductions following CUB's `BLOCK_REDUCE_WARP_REDUCTIONS` strategy: each warp reduces its lanes via a `shuffle_down` tree, lane 0 of each warp publishes the warp aggregate to shared memory, then thread 0 sequentially folds the warp aggregates with the same operator. The result is valid in **thread 0 only**; other threads retain partial values. For the broadcast-to-every-thread variants see `block.reduce_all_{add,min,max}` below.

Arguments:

- `value`: per-thread input.
- `tid`: calling thread's block-local index. Pass `i % block_dim` from a `qd.loop_config(block_dim=...)` kernel, or `qd.simt.block.thread_idx()` on backends that expose it.
- `block_dim`: threads per block (compile-time `template()`; must be a multiple of `2**log2_warp`).
- `log2_warp`: `log2(warp_size)`, compile-time `template()`. Pass 5 on CUDA / Metal / RDNA AMDGPU (wave32) and 6 on CDNA AMDGPU (wave64).
- `dtype`: scalar dtype for the inter-warp shared-memory staging slot; must match `value`'s type.

Cost: `log2_warp` shuffles + 1 shared-memory write/read per warp + 1 `block.sync()` + `(block_dim / 2**log2_warp) - 1` ops on thread 0. When the block is exactly one warp the shared-memory path is short-circuited at trace time.

```python
@qd.kernel
def kern(src: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1)):
    qd.loop_config(block_dim=128)
    for i in range(N):
        tid = i % 128
        agg = qd.simt.block.reduce_add(src[i], tid, 128, 5, qd.f32)
        if tid == 0:
            out[i // 128] = agg
```

A generic `block.reduce(value, tid, block_dim, log2_warp, op, dtype)` is also available for custom associative operators (e.g. bitwise ops, custom monoids). It accepts an `op: template()` `@qd.func` taking `(a, b)` and returning the same type as `value`.

### `block.reduce_all_{add,min,max}(value, tid, block_dim, log2_warp, dtype)`

The broadcast variants of the above. Identical semantics, but the result is published to a one-slot `SharedArray` and read back by every thread after a second `block.sync()`. Use this when downstream code on every thread needs the block-wide aggregate (e.g. normalising each thread's value by the block sum). Cost: one extra `block.sync()` plus one shared-memory hop vs. the lane-0-only variants. The corresponding generic form is `block.reduce_all(value, tid, block_dim, log2_warp, op, dtype)`.

## Grid-scope fence: `qd.simt.grid.memfence()`

**Planned rename: `qd.simt.grid.mem_fence()`** (note the underscore). This op will be renamed in a future release for consistency with the planned `block.mem_fence()`. The current name (`grid.memfence()`) remains the only spelling available today; the rest of this section uses it.

`grid.memfence()` is the device-scope counterpart of `block.mem_sync()` (to be renamed `block.mem_fence()` in a future release). It orders memory operations across the entire grid, so writes made by one block become visible to other blocks after the fence. CUDA only today; lowers to `__threadfence()` (`nvvm_membar_gl`).

Use it when you need cross-block coordination via global memory (decoupled look-back scan, inter-block flag publishing, single-pass reductions, etc.). For coordination within a single block, prefer `block.mem_sync()` — it is cheaper.

There is no built-in grid-wide barrier (cooperative-groups-style); the only way to converge threads across blocks is to finish the kernel and launch a new one.

## Related

- [parallelization](parallelization.md) — kernel-launch and grid-stride patterns.
- [subgroup](subgroup.md) — primitives that operate within a single subgroup (warp / wavefront), one tier below block scope.
- [tile16](tile16.md) — `Tile16x16` register-resident tiles, built on `subgroup.shuffle`.
- `qd.atomic_add` / `qd.atomic_min` / ... — global-memory atomics, the other common cross-block coordination mechanism.
