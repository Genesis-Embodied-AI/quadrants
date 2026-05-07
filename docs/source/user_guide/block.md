# Block primitives

Block-level primitives operate on the threads of a single CUDA thread block (CTA) / AMDGPU workgroup / Vulkan or Metal workgroup. They include thread barriers, memory fences, shared memory, and per-thread indexing helpers — the building blocks for cooperation among threads of the same block.

Block ops live under `qd.simt.block`. They are written so the same Python source compiles to the right vendor primitive on each backend; calling a backend that has not implemented an op raises `ValueError` from the Python layer at trace time.

The closely-related grid-level fence (`qd.simt.grid.mem_fence()`) is documented at the end of this page, since users picking between a block-scope and a device-scope fence need to see both side by side.

## What's available

| Op                                              | CUDA | AMDGPU | Vulkan | Metal |
|-------------------------------------------------|------|--------|--------|-------|
| `block.sync()`                                  | yes  | yes    | yes    | yes   |
| `block.sync_all_nonzero(predicate)`             | yes  | no     | no     | no    |
| `block.sync_any_nonzero(predicate)`             | yes  | no     | no     | no    |
| `block.sync_count_nonzero(predicate)`           | yes  | no     | no     | no    |
| `block.mem_fence()`                             | yes\*| yes    | yes    | yes   |
| `block.SharedArray(shape, dtype)`               | yes  | yes    | yes    | yes   |
| `block.global_thread_idx()`                     | yes  | yes    | yes    | yes   |
| `block.thread_idx()`                            | no   | no     | yes    | yes   |
| `grid.mem_fence()` (device-scope, see below)    | yes  | yes    | yes    | yes\*\* |

Calling a backend marked "no" raises `ValueError` from the Python layer at trace time. Vulkan and Metal share a SPIR-V codegen path (Metal goes through MoltenVK → MSL); they are listed as separate columns because a few ops have Metal-specific limitations that are called out below.

\* On CUDA, `block.mem_fence()` currently lowers via `block_barrier` (i.e. `__syncthreads()`), which doubles as a memory fence but additionally requires thread convergence — meaning calling it from divergent control flow today deadlocks. A fix to lower `mem_fence()` to a pure `__threadfence_block()` is in flight as [quadrants#637](https://github.com/Genesis-Embodied-AI/quadrants/pull/637); once merged, the divergent-branch pattern shown in the `block.mem_fence()` semantics section below works as written. Until then, prefer calling `mem_fence()` from uniform control flow on CUDA.

\*\* On Metal, `grid.mem_fence()` lowers (via MoltenVK / SPIRV-Cross → MSL) to `atomic_thread_fence(metal::memory_scope_device)`, available since MSL 2.0 (macOS 10.13+ / iOS 11+). Cross-workgroup memory-ordering guarantees are stronger on Apple Silicon (A11+) than on older Apple hardware or very old macOS Intel GPUs; for those targets, validate empirically that producer-consumer patterns across blocks behave as expected, or fall back to splitting the kernel.

Naming note: two of the names on this page were recently renamed for consistency with the project's "fence vs barrier" terminology and to use a consistent `mem_fence` spelling. The old names are still available as deprecated aliases that emit a `DeprecationWarning` (shown once per process, courtesy of `warnings.filterwarnings("once", ..., module="quadrants")` configured by `quadrants/lang/misc.py`):

- `block.mem_sync()` → `block.mem_fence()`.
- `grid.memfence()` → `grid.mem_fence()` (note the underscore).

New code should use the `mem_fence` spellings. The aliases are retained for backward compatibility and may be removed in a future release.

## Barrier vs fence: the distinction that matters

Two of these ops sound similar but have very different semantics, and mixing them up deadlocks the GPU. The summary:

- `block.sync()` is a **thread-converging barrier**. Every thread in the block must reach the call site before any thread proceeds. It also implies a memory fence at block scope.
- `block.mem_fence()` is a **memory fence only**, at block scope. It orders memory operations but does not require thread convergence — it is safe to call from divergent control flow (e.g. inside `if tid == 0`).

Concretely, on the SPIR-V backend `sync()` lowers to `workgroupBarrier` and `mem_fence()` lowers to `workgroupMemoryBarrier`. On CUDA, `sync()` lowers to `__syncthreads()`; `mem_fence()` is intended to lower to `__threadfence_block()` (a pure fence with no convergence requirement) — see the support-table caveat above. Calling `sync()` from a path that not all threads reach (a divergent `if`, an early `return`, etc.) is a classic GPU deadlock and applies to both backends.

The corresponding distinction at device scope is `grid.mem_fence()` (memory fence across the entire grid, no thread synchronization). There is no block-style "device barrier" — to synchronize threads across blocks, finish the kernel and launch a new one.

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

### `block.mem_fence()`

A block-scope memory fence. Orders memory operations issued by the calling thread so that prior writes are visible to other threads in the block before any subsequent read by the calling thread can be reordered ahead of the fence. It does **not** synchronize threads — no convergence requirement (subject to the CUDA caveat in the support table).

- Lowers to `__threadfence_block()` (`nvvm_membar_cta`) — the intended target — on CUDA, to `__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup")` on AMDGPU (which the AMDGCN backend lowers to `s_waitcnt lgkmcnt(0)` and friends), and to `workgroupMemoryBarrier` on SPIR-V (Vulkan / Metal).
- Use this when one thread in the block (e.g. lane 0) needs to publish data to shared memory and have the publication be visible to the rest of the block without forcing the publishing thread to wait at a barrier. The pattern is typically:

  ```python
  if tid == 0:
      shared[...] = computed_value
      qd.simt.block.mem_fence()
      shared_flag[0] = 1
  qd.simt.block.sync()
  ```

  The `mem_fence()` here orders the data write before the flag write; the `sync()` is what converges the other threads so they observe the published flag.

The deprecated alias `block.mem_sync()` calls `block.mem_fence()` and emits a `DeprecationWarning` on first use.

### `block.SharedArray(shape, dtype)`

Allocates a shared-memory array, scoped to the calling block.

- `shape`: an `int` (1-D) or a `tuple` / `list` of `int`s (multi-dim). Must be compile-time constants — shared memory is statically allocated per block.
- `dtype`: a scalar Quadrants dtype (`qd.f32`, `qd.i32`, ...) or a `qd.types.matrix(...)` / `qd.types.vector(...)` type. Matrix types are flattened to their element tensor type.

Element access uses the standard `arr[i]` / `arr[i, j]` subscript syntax inside a kernel.

A worked example with `Tile16x16` interaction is in [tile16](tile16.md).

### `block.global_thread_idx()`

Returns the global thread index of the calling thread within the kernel launch.

On CUDA / AMDGPU this lowers to the in-block thread index (`nvvm_read_ptx_sreg_tid_x` / `amdgcn_workitem_id_x`) plus the grid offset that the offload framework adds; on Vulkan / Metal it lowers to `globalInvocationId` (MoltenVK maps this to MSL `thread_position_in_grid`).

On CUDA / AMDGPU this is the natural way to identify which work-item a thread should process when the kernel uses `qd.loop_config(block_dim=...)` — together with `block_dim`, you can recover the in-block thread index via `global_thread_idx() % block_dim`.

### `block.thread_idx()`

Returns the local thread index within the block. Currently only implemented on SPIR-V backends (Vulkan / Metal); on CUDA / AMDGPU it raises. On CUDA / AMDGPU, use `global_thread_idx()` together with `qd.loop_config(block_dim=...)` and recover the in-block index via `global_thread_idx() % block_dim`.

## Grid-scope fence: `qd.simt.grid.mem_fence()`

`grid.mem_fence()` is the device-scope counterpart of `block.mem_fence()`. It orders memory operations across the entire grid, so writes made by one block become visible to other blocks after the fence. Per-backend lowering:

- CUDA: `__threadfence()` (`nvvm_membar_gl`).
- AMDGPU: `__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent")`, lowered by the AMDGCN backend to the appropriate `s_waitcnt` / cache-flush sequence.
- Vulkan: `OpMemoryBarrier(ScopeDevice, AcquireRelease | UniformMemory | WorkgroupMemory)`.
- Metal: same SPIR-V op as Vulkan, translated by MoltenVK to MSL `atomic_thread_fence(metal::memory_scope_device)` (see Metal caveat in the support table).

Use it when you need cross-block coordination via global memory (decoupled look-back scan, inter-block flag publishing, single-pass reductions, etc.). For coordination within a single block, prefer `block.mem_fence()` — it is cheaper.

There is no built-in grid-wide barrier (cooperative-groups-style); the only way to converge threads across blocks is to finish the kernel and launch a new one.

The deprecated alias `grid.memfence()` calls `grid.mem_fence()` and emits a `DeprecationWarning` on first use.

## Related

- [parallelization](parallelization.md) — kernel-launch and grid-stride patterns.
- [subgroup](subgroup.md) — primitives that operate within a single subgroup (warp / wavefront), one tier below block scope.
- [tile16](tile16.md) — `Tile16x16` register-resident tiles, built on `subgroup.shuffle`.
- `qd.atomic_add` / `qd.atomic_min` / ... — global-memory atomics, the other common cross-block coordination mechanism.
