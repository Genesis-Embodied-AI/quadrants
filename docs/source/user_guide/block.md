# Block primitives

Block-level primitives operate on the threads of a single CUDA thread block (CTA) / Vulkan
workgroup. They include thread barriers, memory fences, shared memory, and per-thread
indexing helpers.

Block ops live under `qd.simt.block`. They are written so the same Python source compiles
to the right vendor primitive on each backend.

The closely-related grid-level fence (`qd.simt.grid.memfence()`) is documented at the end
of this page, since users picking between block-scope and device-scope fences need to see
both.

## What's available

| Op                                              | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|-------------------------------------------------|------|--------|-------------------------|
| `block.sync()`                                  | yes  | yes    | yes                     |
| `block.sync_all_nonzero(predicate)`             | yes  | no     | no                      |
| `block.sync_any_nonzero(predicate)`             | yes  | no     | no                      |
| `block.sync_count_nonzero(predicate)`           | yes  | no     | no                      |
| `block.mem_sync()`                              | yes  | no     | yes                     |
| `block.global_thread_idx()`                     | yes  | yes    | —                       |
| `block.thread_idx()`                            | no   | no     | yes                     |
| `block.SharedArray(shape, dtype)`               | yes  | yes    | yes                     |
| `grid.memfence()` (device-scope, see below)     | yes  | no     | no                      |

Calling a backend marked "no" raises `ValueError` from the Python layer at trace time.
`global_thread_idx()` is verified on CUDA and AMDGPU; the SPIR-V codepath in the wrapper
exists but is currently unreachable due to a control-flow quirk in the dispatch and is
therefore left undocumented here.

## Barrier vs fence: the distinction that matters

Two of these ops sound similar but have very different semantics, and mixing them up
deadlocks the GPU. The summary:

- `block.sync()` is a **thread-converging barrier**. Every thread in the block must reach
  the call site before any thread proceeds. It also implies a memory fence at block scope.
- `block.mem_sync()` is a **memory fence only**, at block scope. It orders memory operations
  but does not require thread convergence — it is safe to call from divergent control flow
  (e.g. inside `if tid == 0`).

Concretely, on CUDA `sync()` lowers to `__syncthreads()` and `mem_sync()` lowers to
`__threadfence_block()`. Calling `sync()` from a path that not all threads reach (a
divergent `if`, an early `return`, etc.) is a classic GPU deadlock.

The corresponding distinction at device scope is `grid.memfence()` (memory fence across the
entire grid, no thread sync). There is no block-style "device barrier" — to synchronize
threads across blocks, finish the kernel and launch a new one.

## Semantics

### `block.sync()`

A block-wide thread-converging barrier. All threads in the block stop at the call until
every thread has reached it; once all have arrived, all proceed. Reads and writes issued
before the barrier are visible to other threads in the block after the barrier.

- Lowers to `__syncthreads()` on CUDA, `s_barrier` on AMDGPU, `workgroupBarrier` on SPIR-V.
- Must be called from uniform control flow within the block. Calling from a divergent
  branch deadlocks.

### `block.sync_all_nonzero(predicate)` / `sync_any_nonzero` / `sync_count_nonzero`

Block-wide barriers that also reduce a per-thread `i32` predicate across the block:

- `sync_all_nonzero(p)` returns non-zero if `p` is non-zero on **every** thread (logical AND).
- `sync_any_nonzero(p)` returns non-zero if `p` is non-zero on **any** thread (logical OR).
- `sync_count_nonzero(p)` returns the number of threads for which `p` is non-zero (popcount).

Each call performs both the synchronization (same convergence requirement as `sync()`) and
the reduction in a single instruction. Only available on CUDA today; they lower to the
NVPTX `barrier.cta.red` family of intrinsics.

### `block.mem_sync()`

A block-scope memory fence. Orders memory operations issued by the calling thread so that
prior writes are visible to other threads in the block before any subsequent read by the
calling thread can be reordered ahead of the fence. It does **not** synchronize threads —
no convergence requirement.

- Lowers to `__threadfence_block()` (`nvvm_membar_cta`) on CUDA, `workgroupMemoryBarrier`
  on SPIR-V.
- Safe to call from divergent control flow.
- Use this when one thread in the block (e.g. lane 0) needs to publish data to shared
  memory and have the publication be visible to the rest of the block, without forcing the
  publishing thread to wait at a barrier. The pattern is typically:

  ```python
  if tid == 0:
      shared[...] = computed_value
      qd.simt.block.mem_sync()
      shared_flag[0] = 1
  qd.simt.block.sync()
  # all threads can now safely read shared[...]
  ```

  The `mem_sync()` here orders the data write before the flag write; the `sync()` is what
  converges the other threads so they observe the published flag.

### `block.global_thread_idx()`

Returns the global thread index of the calling thread within the kernel launch. Verified on
CUDA and AMDGPU.

### `block.thread_idx()`

Returns the local thread index within the block. Currently only implemented on SPIR-V
backends (Vulkan / Metal); on CUDA / AMDGPU it raises. On CUDA / AMDGPU, use
`global_thread_idx()` and the loop-driven indexing pattern instead — see Examples.

### `block.SharedArray(shape, dtype)`

Allocates a shared-memory array, scoped to the calling block.

- `shape`: an `int` (1-D) or a `tuple` / `list` of `int`s (multi-dim). Must be compile-time
  constants — shared memory is statically allocated per block.
- `dtype`: a scalar Quadrants dtype (`qd.f32`, `qd.i32`, ...) or a `qd.types.matrix(...)` /
  `qd.types.vector(...)` type. Matrix types are flattened to their element tensor type.

Element access uses the standard `arr[i]` / `arr[i, j]` subscript syntax inside a kernel.

A worked example with `Tile16x16` interaction is in [tile16](tile16.md).

## Grid-scope fence: `qd.simt.grid.memfence()`

`grid.memfence()` is the device-scope counterpart of `block.mem_sync()`. It orders memory
operations across the entire grid, so writes made by one block become visible to other
blocks after the fence. CUDA only today; lowers to `__threadfence()` (`nvvm_membar_gl`).

Use it when you need cross-block coordination via global memory (decoupled lookback,
inter-block flag publishing, etc.). For coordination within a single block, prefer
`block.mem_sync()` — it is cheaper.

## Examples

### Block barrier with shared memory

```python
import quadrants as qd

BLOCK = 256

@qd.kernel
def reduce_per_block(a: qd.types.ndarray(dtype=qd.f32, ndim=1),
                     out: qd.types.ndarray(dtype=qd.f32, ndim=1)) -> None:
    sh = qd.simt.block.SharedArray(BLOCK, qd.f32)
    qd.loop_config(block_dim=BLOCK)
    for i in range(a.shape[0]):
        tid = i % BLOCK
        sh[tid] = a[i]
        qd.simt.block.sync()
        # ... cooperative reduction over `sh` ...
```

Every thread in the block writes its slot in `sh`, then `sync()` ensures every write is
visible to every other thread before the reduction begins.

### Memory fence in a divergent branch

```python
@qd.kernel
def publish_then_sync() -> None:
    sh = qd.simt.block.SharedArray(1, qd.i32)
    flag = qd.simt.block.SharedArray(1, qd.i32)
    qd.loop_config(block_dim=32)
    for i in range(32):
        tid = i % 32
        if tid == 0:
            sh[0] = 42
            qd.simt.block.mem_sync()  # order data write before flag write
            flag[0] = 1
        qd.simt.block.sync()           # converge all threads
        # all threads now see sh[0] == 42 once flag[0] == 1
```

`mem_sync()` runs on a single thread inside the divergent branch. Substituting `sync()`
here would deadlock, since the other 31 threads never enter the `if`.

### Block-wide predicate reductions

```python
@qd.kernel
def vote(b: qd.types.ndarray(dtype=qd.i32, ndim=1),
         out_all: qd.types.ndarray(dtype=qd.i32, ndim=1),
         out_any: qd.types.ndarray(dtype=qd.i32, ndim=1),
         out_cnt: qd.types.ndarray(dtype=qd.i32, ndim=1)) -> None:
    qd.loop_config(block_dim=32)
    for i in range(32):
        out_all[i] = qd.simt.block.sync_all_nonzero(b[i])
        out_any[i] = qd.simt.block.sync_any_nonzero(b[i])
        out_cnt[i] = qd.simt.block.sync_count_nonzero(b[i])
```

Each op both synchronizes the block and returns the reduction result, in a single
instruction. CUDA only.
