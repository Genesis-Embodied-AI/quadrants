# Block primitives

Block-level primitives operate on the threads of a single CUDA thread block (CTA) / AMDGPU workgroup / Vulkan or Metal workgroup. They include thread barriers, memory fences, shared memory, and per-thread indexing helpers — the building blocks for cooperation among threads of the same block.

Block ops live under `qd.simt.block`. They are written so the same Python source compiles to the right vendor primitive on each backend. As of this writing every op on this page is portable across CUDA, AMDGPU, Vulkan, and Metal; the only remaining caveat (called out in the support-table footnote below) is a perf trade-off for the emulated `block.sync_*_nonzero` ops on non-CUDA backends, not a correctness gap. If a future op is added that is not yet portable, the Python layer will raise `ValueError` at trace time on the unsupported backend.

The closely-related device-scope memory fence is documented separately in [grid](grid.md). Users picking between a block-scope and a device-scope fence should read that page for the device-scope side.

## What's available

| Op                                              | CUDA | AMDGPU  | Vulkan | Metal |
|-------------------------------------------------|------|---------|--------|-------|
| `block.sync()`                                  | yes  | yes     | yes    | yes   |
| `block.sync_all_nonzero(predicate)`             | yes  | yes\*   | yes\*  | yes\* |
| `block.sync_any_nonzero(predicate)`             | yes  | yes\*   | yes\*  | yes\* |
| `block.sync_count_nonzero(predicate)`           | yes  | yes\*   | yes\*  | yes\* |
| `block.mem_fence()`                             | yes  | yes     | yes    | yes   |
| `block.SharedArray(shape, dtype)`               | yes  | yes     | yes    | yes   |
| `block.global_thread_idx()`                     | yes  | yes     | yes    | yes   |
| `block.thread_idx()`                            | yes  | yes     | yes    | yes   |
| `block.reduce_{add,min,max}(v, tid, ...)`       | yes  | yes\*\* | yes    | yes   |
| `block.reduce_all_{add,min,max}(v, tid, ...)`   | yes  | yes\*\* | yes    | yes   |
| `block.inclusive_{add,min,max}(v, tid, ...)`    | yes  | yes\*\* | yes    | yes   |
| `block.exclusive_{add,min,max}(v, tid, ...)`    | yes  | yes\*\* | yes    | yes   |
| `block.radix_rank_match_atomic_or(...)`         | yes  | wave32  | yes    | yes   |

Vulkan and Metal share a SPIR-V codegen path (Metal goes through MoltenVK → MSL); they are listed as separate columns because a couple of ops have Metal-specific caveats called out below. Footnoted entries are still functional, just with the limitations the footnote describes.

\* On AMDGPU, Vulkan, and Metal the `block.sync_{all,any,count}_nonzero(p)` ops are *emulated* via shared memory (one shared `i32` slot + 2 block barriers + a single `atomic_add` per contributing thread) rather than a single hardware-fused barrier-with-reduction. CUDA has the fused NVPTX `barrier.cta.red.{and,or,popc}.aligned.all.sync` family of intrinsics so it stays on the fast path; the other backends do not have a direct analog (in particular, SPIR-V `OpGroupNonUniform*` only operates at subgroup scope reliably across Vulkan + Metal). All three reductions are routed through `atomic_add` rather than `atomic_or` / `atomic_and`: the latter trip a Metal-specific bug where `OpAtomicOr` on threadgroup memory silently no-ops via MoltenVK / SPIRV-Cross. The emulation is correct and portable but costs two `block.sync()`s plus one shared-memory atomic per call instead of a single barrier instruction; if you have an inner loop calling these ops millions of times, consider whether you can batch the predicate before reducing it.

\*\* On AMDGPU the block reduce / scan ops require `log2_warp` to match the active hardware wave size: pass `5` on RDNA (wave32) and `6` on CDNA (wave64). Passing wave32 on a wave64 device gives wrong results because per-warp `shuffle_down` falls back to `ds_bpermute` whose OOB index wraps modulo the wave size (e.g. lane 48 with offset 16 reads lane 0 instead of returning the lane's own value as `__shfl_down_sync(width=32)` would on CUDA), contaminating the per-warp aggregate of the second logical 32-lane warp inside each wave. CUDA / Metal / RDNA Vulkan are wave32 in practice, so `log2_warp=5` is the natural choice there.

`block.radix_rank_match_atomic_or` is **wave32-only** by construction: the atomic-OR match phase is keyed on a 32-lane `i32` ballot mask plus `clz` / `popcnt` of that mask. Calling it on CDNA AMDGPU (wave64) requires the kernel to launch with a 32-thread subgroup or to wait for a wave64 path (which would need a `u64` bin mask, `u64` `clz` / `popcnt`, and 64-lane `subgroup.shuffle`).

Naming note: `block.mem_sync()` was recently renamed to `block.mem_fence()` for consistency with the project's "fence vs barrier" terminology. The old name is still available as a deprecated alias that emits `DeprecationWarning` on first use; new code should use `block.mem_fence()`.

## Barrier vs fence: the distinction that matters

Two of these ops sound similar but have very different semantics, and mixing them up deadlocks the GPU. The summary:

- `block.sync()` is a **thread-converging barrier**. Every thread in the block must reach the call site before any thread proceeds. It also implies a memory fence at block scope.
- `block.mem_fence()` is a **memory fence only**, at block scope. It orders memory operations but does not require thread convergence — it is safe to call from divergent control flow (e.g. inside `if tid == 0`).

Concretely:

- CUDA: `sync()` lowers to `__syncthreads()`; `mem_fence()` lowers to `__threadfence_block()` (a pure fence with no convergence requirement).
- AMDGPU: `sync()` lowers to `s_barrier`; `mem_fence()` lowers to `fence acquire_release syncscope("workgroup")`.
- Vulkan / Metal (SPIR-V): `sync()` lowers to `workgroupBarrier`; `mem_fence()` lowers to `workgroupMemoryBarrier`.

Calling `sync()` from a path that not all threads reach (a divergent `if`, an early `return`, etc.) is a classic GPU deadlock and applies to all backends.

The corresponding distinction at device scope is the grid-scope memory fence (memory fence across the entire grid, no thread synchronization), documented in [grid](grid.md).

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

Each call performs both the synchronization (same convergence requirement as `sync()`) and the reduction.

- On CUDA, this lowers to a single hardware-fused instruction from the NVPTX `barrier.cta.red` family — `block_barrier_and_i32`, `block_barrier_or_i32`, `block_barrier_count_i32`.
- On AMDGPU, Vulkan, and Metal, there is no direct hardware-fused barrier-with-reduction, so the op is emulated in Quadrants Python (`_block_reduce_*_emulated` in `python/quadrants/lang/simt/block.py`) as: lane 0 zeroes a 1-element `SharedArray(i32)` → `block.sync()` → every thread folds its predicate via `qd.atomic_or` / `qd.atomic_add` → `block.sync()` → every thread reads the broadcasted result. Two block barriers plus one shared-memory atomic per call. See the support-table footnote for the perf trade-off.

### `block.mem_fence()`

A block-scope memory fence. Orders memory operations issued by the calling thread so that prior writes are visible to other threads in the block before any subsequent read by the calling thread can be reordered ahead of the fence. It does **not** synchronize threads — no convergence requirement, so it is safe to call from divergent control flow (e.g. inside `if tid == 0`) on every backend.

- Lowers to `__threadfence_block()` (`nvvm_membar_cta`) — the intended target — on CUDA, to an LLVM IR `fence acquire_release syncscope("workgroup")` on AMDGPU (which the AMDGCN backend lowers to the appropriate `s_waitcnt` / cache-flush sequence; emitted via a body-replacement in `llvm_context.cpp` rather than `__builtin_amdgcn_fence`, since the `runtime.cpp` is built with a host-targeted clang that doesn't know AMDGCN builtins), and to `workgroupMemoryBarrier` on SPIR-V (Vulkan / Metal).
- Use this when one thread in the block needs to publish data to shared memory and have other threads observe it via polling, without going through a thread-converging barrier. The canonical pattern is a flag-published producer + spin-waiting consumers:

  ```python
  if tid == 0:
      shared[...] = computed_value
      qd.simt.block.mem_fence()  # order the data write before the flag store
      shared_flag[0] = 1
  else:
      while shared_flag[0] == 0:
          pass
      use(shared[...])  # without the fence above, may observe stale shared[...]
  ```

  `block.sync()` does not work here — it deadlocks, because `tid == 0` and the other threads take divergent paths and never converge at a single call site. `block.sync()` would also be sufficient by itself (it implies a block-scope fence) when the producer and consumers *can* converge; reach for `block.mem_fence()` specifically when they cannot.

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

Returns the in-block (workgroup-local) thread index of the calling thread. Available on every supported GPU backend.

- CUDA: `nvvm_read_ptx_sreg_tid_x` (i.e. `threadIdx.x`).
- AMDGPU: `amdgcn_workitem_id_x`.
- Vulkan: `localInvocationId` (`gl_LocalInvocationID.x`).
- Metal: same SPIR-V op as Vulkan; MoltenVK / SPIRV-Cross translates to MSL `thread_position_in_threadgroup`.

This is the thread's index *within its own block / workgroup*. To get the across-grid index, use `block.global_thread_idx()`. The historical workaround on CUDA / AMDGPU of recovering the in-block index via `global_thread_idx() % block_dim` is still valid but no longer necessary; prefer the direct `block.thread_idx()` call for clarity.

Today only the X dimension is exposed (1-D blocks). For 2-D / 3-D blocks the calling code should compute the linear index from `block.thread_idx()` and the block-Y / Z dimensions itself, or stick to 1-D blocks (the dominant Quadrants idiom — `qd.loop_config(block_dim=N)` always sets the X extent).

### `block.reduce_{add,min,max}(value, tid, block_dim, log2_warp, dtype)`

Block-scope reductions following the standard two-stage warp-reduction strategy: each warp reduces its lanes via a `shuffle_down` tree, lane 0 of each warp publishes the warp aggregate to shared memory, then thread 0 sequentially folds the warp aggregates with the same operator. The result is valid in **thread 0 only**; other threads retain partial values. For the broadcast-to-every-thread variants see `block.reduce_all_{add,min,max}` below.

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

### `block.inclusive_{add,min,max}(value, tid, block_dim, log2_warp, dtype)`

Block-scope inclusive prefix scans via the standard two-stage warp-scan strategy: each warp does a Hillis-Steele scan via `subgroup` shuffles, the last lane of each warp publishes the warp aggregate to shared memory, then every thread sequentially folds the cross-warp prefix and applies its own warp's prefix to its scan value. **All threads receive a valid result.** After the call, thread `i` holds `op(v[0], v[1], ..., v[i])`.

Args match `block.reduce_add` (`value, tid, block_dim, log2_warp, dtype`). Cost: per-warp Hillis-Steele tree (`log2_warp` shuffles) + 1 shared-memory write/read per warp + 1 `block.sync()` + `(block_dim / 2**log2_warp) - 1` ops on every thread (the cross-warp prefix is computed redundantly to avoid a second barrier). When the block is exactly one warp the shared-memory path is short-circuited at trace time.

```python
@qd.kernel
def kern(src: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1)):
    qd.loop_config(block_dim=128)
    for i in range(N):
        tid = i % 128
        out[i] = qd.simt.block.inclusive_add(src[i], tid, 128, 5, qd.i32)
```

The corresponding generic form is `block.inclusive_scan(value, tid, block_dim, log2_warp, op, dtype)` for custom monoids.

### `block.exclusive_{add,min,max}(value, tid, block_dim, log2_warp[, identity], dtype)`

Block-scope exclusive prefix scans. Same strategy and cost profile as `inclusive_*`, but each thread receives the prefix `op(v[0], ..., v[i-1])` instead — and thread 0 receives the operator's identity.

- `exclusive_add`: identity is the additive zero; derived from `value - value` so callers do not need to pass it. After the call, thread 0 holds 0.
- `exclusive_min(..., identity, dtype)`: pass `identity` greater than or equal to every legal element of the input — typically `+∞` for floats or the dtype's maximum for integers. Thread 0 holds `identity`. There is no portable type-extreme derivable from `value` alone, so this op takes an explicit `identity` argument (mirrors `subgroup.exclusive_min`).
- `exclusive_max(..., identity, dtype)`: pass `identity` less than or equal to every legal element of the input — typically `-∞` for floats or the dtype's minimum for integers. Thread 0 holds `identity`.

The corresponding generic form is `block.exclusive_scan(value, tid, block_dim, log2_warp, op, identity, dtype)`.

### `block.radix_rank_match_atomic_or(key, tid, block_dim, log2_warp, radix_bits, bit_start, num_bits, bins, excl_prefix)`

Block-level radix ranking via the atomic-OR match-and-count strategy (the workhorse of an SM90-style onesweep radix sort). Each thread holds one `u32` key; the function returns the key's stable rank within the block under the digit `(key >> bit_start) & ((1 << num_bits) - 1)`, and writes the per-digit count and exclusive-prefix arrays to two caller-supplied `block.SharedArray` outparams.

Constraints (currently):

- `block_dim` must equal `1 << radix_bits` (each digit gets exactly one thread for the per-thread bin / exclusive-prefix output). Typical configuration is `radix_bits=8, block_dim=256`.
- `log2_warp` must be 5 (warp size 32). The match path is built around 32-lane `i32` ballot masks; wave64 callers should pass 5 and arrange to launch with a 32-thread subgroup, or wait for a wave64 path.
- One key per thread (`items_per_thread = 1`). Multi-item per thread is a future extension.
- `num_bits <= radix_bits`; `bit_start` is the offset of the digit's low bit.

Args:

- `key`: per-thread `u32` input.
- `tid`: calling thread's block-local index.
- `block_dim`, `log2_warp`, `radix_bits`, `bit_start`, `num_bits`: all compile-time `template()`.
- `bins`: `block.SharedArray((1 << radix_bits,), qd.i32)`. After the call, `bins[d]` holds the count of keys whose digit equals `d`.
- `excl_prefix`: `block.SharedArray((1 << radix_bits,), qd.i32)`. After the call, `excl_prefix[d]` holds the exclusive prefix sum of `bins` up to digit `d`.

Cost: 2 `block.sync()` + a handful of `subgroup.sync()` calls + 1 block exclusive scan + per-key `atomic_or` + leader-only `atomic_add` on shared memory. Shared-memory footprint is `2 * BLOCK_WARPS * RADIX_DIGITS` ints = 16 KiB at the default `log2_warp=5, radix_bits=8` configuration (8 warps × 256 digits × 2 banks × 4 B).

```python
@qd.kernel
def kern(keys_in: qd.types.ndarray(ndim=1), ranks_out: qd.types.ndarray(ndim=1)):
    qd.loop_config(block_dim=256)
    for i in range(256):
        tid = i % 256
        bins = qd.simt.block.SharedArray((256,), qd.i32)
        excl = qd.simt.block.SharedArray((256,), qd.i32)
        ranks_out[i] = qd.simt.block.radix_rank_match_atomic_or(
            keys_in[i], tid, 256, 5, 8, 0, 8, bins, excl
        )
```

The function inserts the necessary `block.sync()` retires before returning, so callers can read `bins` / `excl_prefix` immediately after the call without an extra barrier.

## Related

- [grid](grid.md) — the device-scope counterpart of `block.mem_fence()`. For coordination within a single block, prefer `block.mem_fence()` — it is cheaper.
- [parallelization](parallelization.md) — kernel-launch and grid-stride patterns.
- [subgroup](subgroup.md) — primitives that operate within a single subgroup (warp / wavefront), one tier below block scope.
- [tile16](tile16.md) — `Tile16x16` register-resident tiles, built on `subgroup.shuffle`.
- `qd.atomic_add` / `qd.atomic_min` / ... — global-memory atomics, the other common cross-block coordination mechanism.
