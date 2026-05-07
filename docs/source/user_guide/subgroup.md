# Subgroup primitives

Subgroup operations let threads within the same subgroup (warp on NVIDIA, wave on AMD, subgroup on Vulkan / Metal) cooperate directly — exchanging register values, voting on predicates, scanning, and electing a leader — without going through shared memory or block barriers. They are the building block for fast in-warp data exchange and are used internally by `Tile16x16` (see [tile16](tile16.md)).

Subgroup ops live under `qd.simt.subgroup` and are written so the same Python source compiles to the right vendor primitive on each backend. Calling a backend that has not implemented an op fails at trace or codegen time.

## What's available

The full Python API is grouped here by category. The first column lists each op, the next three columns indicate which backend currently lowers it. Cells marked "no" either raise from the Python wrapper, fail to link in the runtime, or return `None` from a `# TODO` stub — in every case the op is unusable on that backend today.

### Data movement

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                       |
|---------------------------------------------|------|--------|-------------------------|------------------------------|
| `subgroup.shuffle(value, index)`            | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_down(value, offset)`      | yes  | yes\*  | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_up(value, offset)`        | no   | no     | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_xor(value, mask)`         | no   | no     | no                      | — (TODO stub on every backend) |
| `subgroup.broadcast(value, index)`          | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.broadcast_first(value)`           | no   | no     | no                      | — (TODO stub on every backend) |

\* AMDGPU `shuffle_down` (and therefore `reduce_add`, which is built on it) is currently emulated via `ds_bpermute` (~50 cycle latency).

The remaining shuffle flavours (`shuffle_up`, `shuffle_xor`) are exposed in the Python module but are not yet implemented across backends. Calling them will fail at codegen. Use `shuffle` with an explicit lane index in the meantime — every shuffle pattern can be expressed that way.

### Identification and control

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|------|--------|-------------------------|
| `subgroup.invocation_id()`                  | yes  | yes    | yes                     |
| `subgroup.group_size()`                     | no   | no     | yes                     |
| `subgroup.elect()`                          | no   | no     | yes                     |
| `subgroup.barrier()`                        | no   | no     | yes                     |
| `subgroup.memory_barrier()`                 | no   | no     | yes                     |

Naming note: two of the names above are planned to be renamed in a future release, to align with the project's naming conventions across scopes:

- `subgroup.barrier()` will be renamed to `subgroup.sync()` (matching `block.sync()`).
- `subgroup.memory_barrier()` will be renamed to `subgroup.mem_fence()` (matching the planned `block.mem_fence()` and `grid.mem_fence()`).

The new names are not yet available; this page uses the current names throughout.

### Voting and predicate ops

Although these functions exist, we will migrate them to have an additional `log2_size` parameter, similar to `reduce_add` — so that the vote is over the first `2**log2_size` lanes rather than the full subgroup.

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|------|--------|-------------------------|
| `subgroup.all_true(predicate)`              | no   | no     | no                      |
| `subgroup.any_true(predicate)`              | no   | no     | no                      |
| `subgroup.all_equal(value)`                 | no   | no     | no                      |

All three are TODO stubs in `python/quadrants/lang/simt/subgroup.py` and currently return `None` on every backend. The CUDA-only counterparts on `qd.simt.warp` (`warp.all_nonzero`, `warp.any_nonzero`, `warp.unique`) are usable today if you can afford to be CUDA-bound.

### Reductions and scans

`reduce_add` and `reduce_all_add` already take a `log2_size` parameter. The `inclusive_*` and `exclusive_*` rows below do not. Although these functions exist, we will migrate the `inclusive_*` and `exclusive_*` ops to have an additional `log2_size` parameter, similar to `reduce_add` — so that the scan is over the first `2**log2_size` lanes rather than the full subgroup.

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                       |
|---------------------------------------------|------|--------|-------------------------|------------------------------|
| `subgroup.reduce_add(v, log2_size)`         | yes  | yes\*  | yes                     | any type supporting `+`      |
| `subgroup.reduce_all_add(v, log2_size)`     | yes  | yes    | yes                     | any type supporting `+`      |
| `subgroup.inclusive_add(value)`             | no   | no     | yes                     | integer + float             |
| `subgroup.inclusive_mul(value)`             | no   | no     | yes                     | integer + float             |
| `subgroup.inclusive_min(value)`             | no   | no     | yes                     | integer + float             |
| `subgroup.inclusive_max(value)`             | no   | no     | yes                     | integer + float             |
| `subgroup.inclusive_and(value)`             | no   | no     | yes                     | integer                      |
| `subgroup.inclusive_or(value)`              | no   | no     | yes                     | integer                      |
| `subgroup.inclusive_xor(value)`             | no   | no     | yes                     | integer                      |
| `subgroup.exclusive_add` / `_mul` / `_min` / `_max` / `_and` / `_or` / `_xor` | no | no | no | — (TODO stubs) |

The SPV-only no-arg reductions (`subgroup.reduce_mul` / `reduce_min` / `reduce_max` / `reduce_and` / `reduce_or` / `reduce_xor`, plus the original `reduce_add(value)` with no `log2_size`) have been removed in favour of the portable sized API (`reduce_add(v, log2_size)` / `reduce_all_add(v, log2_size)`). For reductions other than sum, build a sized helper on top of `shuffle_down` / `shuffle` following the same pattern.

## Semantics

All of these ops operate within a single subgroup: they do not move data through memory and do not synchronise across subgroups.

### `shuffle(value, index)`

Each lane returns the `value` held by the lane whose subgroup-local id equals `index`.

- `value` is a scalar in a register. Supported dtypes are 32-bit and 64-bit signed/unsigned ints and `f32`/`f64`. (64-bit types are split into two 32-bit shuffles on AMDGPU; CUDA dispatches to its native 64-bit helpers.)
- `index` is a `u32`. If `index` is out of range for the active subgroup the result is implementation-defined, so pass `subgroup.invocation_id()`-derived values or known-good lane ids.

### `shuffle_down(value, offset)`

Lane `i` returns the `value` held by lane `i + offset`. Lanes near the top of the subgroup — where `i + offset >= subgroup_size` — receive an implementation-defined value (typically their own `value`), so reduction patterns must only trust lane 0's final result, or mask out the out-of-range lanes.

- `value` and `offset` dtypes: same as `shuffle` above; `offset` is a `u32`.
- Maps to `__shfl_down_sync` on CUDA and `OpGroupNonUniformShuffleDown` on SPIR-V. On AMDGPU it is currently emulated with `ds_bpermute` (see the support matrix above).

### `shuffle_up(value, offset)`

Lane `i` returns the `value` held by lane `i - offset`. Lanes near the bottom of the subgroup — where `i - offset < 0` — receive an implementation-defined value (typically their own `value`).

- Same dtype rules as `shuffle` / `shuffle_down`.
- Currently lowered only on SPIR-V (`OpGroupNonUniformShuffleUp`). CUDA and AMDGPU fall through to the generic LLVM runtime-call path and fail to link; until they are added, emulate with an explicit `shuffle(value, lane - offset)`.

### `broadcast(value, index)`

Every lane in the subgroup returns the `value` held by the lane whose subgroup-local id equals `index`. Equivalent to `shuffle(value, index)` but expresses intent ("read lane `index`") more directly and may map to a cheaper instruction where the backend has a dedicated broadcast.

- Same dtype rules as `shuffle`.
- Maps to `__shfl_sync` on CUDA, `ds_bpermute` on AMDGPU, and `OpGroupNonUniformBroadcast` on SPIR-V.

### Common to the data-movement ops

- All shuffles / broadcasts are issued under a full active mask on CUDA (`0xFFFFFFFF`). Call them from uniform control flow; calling from divergent control flow is undefined on most backends. (This means: every thread has to execute the shuffle.)
- Subgroup size varies by backend (32 on NVIDIA, 32 or 64 on AMD depending on wavefront mode, 32 in Vulkan compute on most GPUs). Use `subgroup.group_size()` to query at runtime on SPIR-V; on CUDA / AMDGPU use a compile-time constant.

### `invocation_id()`

Returns this lane's subgroup-local index — `0..subgroup_size - 1`. Used both as a lane id when computing a target lane for `shuffle` / `broadcast`, and as a per-lane identifier in cooperative algorithms.

- Returns `i32`.
- Available on every backend.

### `group_size()`

Returns the subgroup size in effect for the current launch. Currently SPIR-V only; on CUDA the active warp size is statically `32`, and on AMDGPU it is `32` or `64` depending on the wavefront mode chosen at compile time, so a runtime query is typically unnecessary on those backends.

### `elect()`

Picks one lane in the subgroup as the "leader". Returns `1` on the elected lane and `0` on every other lane. The choice of which lane is elected is implementation-defined but stable for the duration of the call.

- Useful for "exactly one lane does X" patterns where you don't care which lane it is — e.g. emitting a single global write per subgroup.
- Currently SPIR-V only (`OpGroupNonUniformElect`). On CUDA / AMDGPU, emulate with `subgroup.invocation_id() == 0`.

### `barrier()` / `memory_barrier()`

**Planned renames: `subgroup.sync()` (in place of `barrier()`) and `subgroup.mem_fence()` (in place of `memory_barrier()`).** Both will be renamed in a future release; the current names remain the only spellings available today, and the rest of this section uses them.

`barrier()` is a subgroup-scope thread-converging barrier — every lane in the subgroup must reach the call before any lane proceeds. `memory_barrier()` is a subgroup-scope memory fence: it orders memory operations within the subgroup without requiring thread convergence.

- Both currently SPIR-V only (`OpControlBarrier` / `OpMemoryBarrier`, both scoped to `Subgroup`). On CUDA / AMDGPU, subgroups (warps) execute in lockstep and these are typically unnecessary; the equivalent under divergent control flow on CUDA is `__syncwarp(active_mask)`, which is not currently exposed through `qd.simt.subgroup`.

### `reduce_add(value, log2_size)`

Sums `value` across `2**log2_size` consecutive lanes via a `shuffle_down` tree. The result is valid **in lane 0** of each group; other lanes hold partial sums and should be considered undefined.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into exactly `log2_size` `shuffle_down + add` pairs in the calling kernel's IR, with no runtime loop overhead.
- `2**log2_size` must not exceed the active subgroup size on the target (32 on CUDA / Metal and on RDNA, 64 on CDNA). Passing a larger value produces implementation-defined results; it does not error.
- The reduction works on any type that supports `+` and `shuffle_down`; in practice this means i32, u32, f32, f64, i64, u64.
- Decorated with `@qd.func` and inlined into the calling kernel — there is no kernel-launch overhead and no separate symbol to link.

Lanes 1..`2**log2_size - 1` receive undefined-but-safe partial sums (they never touch out-of-range lanes because the tree shrinks each step), but only lane 0's result is meaningful for the caller.

### `reduce_all_add(value, log2_size)`

Same sum as `reduce_add`, but broadcast to **every lane** in each `2**log2_size` group. Implemented as a butterfly using `shuffle` with `lane ^ mask`, `mask` stepping through `1, 2, 4, ..., 2**(log2_size-1)`.

- Same `log2_size` template + size-cap contract as `reduce_add`.
- Use this when every lane needs the reduction result (e.g. to divide by the sum, or to branch on it uniformly). It costs exactly the same number of shuffles as `reduce_add` but leaves the answer in all lanes, so it replaces a `reduce_add` + `shuffle`/broadcast pair.
- Uses `subgroup.shuffle` under the hood.

### `inclusive_add` / `inclusive_mul` / `inclusive_min` / `inclusive_max` / `inclusive_and` / `inclusive_or` / `inclusive_xor`

Per-lane inclusive scans over the full subgroup. Lane `i` receives `v[0] op v[1] op ... op v[i]`, where `op` is the operation indicated by the suffix and `v[k]` is the `value` held by lane `k`.

- Currently SPIR-V only (`OpGroupNonUniformInclusiveScan` parameterised by the operation). On CUDA / AMDGPU, fall through to the generic LLVM runtime-call path and fail to link; until they are added, build the equivalent on top of `shuffle_down` / `shuffle_up` with a log2-step loop.
- `_add`, `_mul`, `_min`, `_max` accept integer and float dtypes; `_and`, `_or`, `_xor` accept integer dtypes only.
- The operation is over the full subgroup; there is no sized variant. For sums, a sized portable inclusive scan can be derived from `reduce_add` plus a prefix-fixup pattern.

### `shuffle_xor`, `broadcast_first`, `exclusive_*`, `all_true`, `any_true`, `all_equal`

These names are present in `python/quadrants/lang/simt/subgroup.py` but are currently `# TODO` stubs that return `None` on every backend. They are listed in the support matrices above for completeness — calling them produces a tracing failure rather than a useful operation. Do not depend on them today.

## Examples

### Broadcast lane 0 to all lanes

```python
import quadrants as qd
from quadrants.lang.simt import subgroup

@qd.kernel
def broadcast(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=64)
    for i in range(a.shape[0]):
        a[i] = subgroup.shuffle(a[i], qd.u32(0))
```

After the kernel, every lane in a subgroup holds the original value of its lane 0. `subgroup.broadcast(a[i], qd.u32(0))` is interchangeable here.

### Identity shuffle (each lane reads its own id)

Useful as a sanity check:

```python
@qd.kernel
def identity(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
             dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=64)
    for i in range(src.shape[0]):
        lane = subgroup.invocation_id()
        dst[i] = subgroup.shuffle(src[i], qd.cast(lane, qd.u32))
```

`dst[i]` equals `src[i]` on every lane.

### Swap neighbours (xor pattern via explicit lane)

```python
@qd.kernel
def swap_pairs(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
               dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=64)
    for i in range(src.shape[0]):
        lane = subgroup.invocation_id()
        dst[i] = subgroup.shuffle(src[i], qd.cast(lane ^ 1, qd.u32))
```

Pairs `(0,1)`, `(2,3)`, ... swap their values.

### Arbitrary per-lane gather

```python
@qd.kernel
def reverse4(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
             dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=64)
    for i in range(src.shape[0]):
        lane = subgroup.invocation_id()
        group_base = (lane // 4) * 4
        src_lane = group_base + 3 - lane % 4
        dst[i] = subgroup.shuffle(src[i], qd.cast(src_lane, qd.u32))
```

Within each group of 4 contiguous lanes the values are reversed.

### Tree reduction with `shuffle_down`

Classic warp-level sum of 4 values — after the second step, lane 0 of each group of 4 holds the total:

```python
@qd.kernel
def reduce4(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
            dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=64)
    for i in range(src.shape[0]):
        val = src[i]
        val = val + subgroup.shuffle_down(val, qd.u32(2))
        val = val + subgroup.shuffle_down(val, qd.u32(1))
        dst[i] = val
```

Extend the pattern (offsets 16, 8, 4, 2, 1, ...) to reduce a full subgroup; only lane 0's final value is meaningful, because the lanes near the top read past the end of the subgroup.

### Sum 32 lanes with `reduce_add`

The same tree, packaged as a one-liner. Lane 0 of each group of 32 holds the total; other lanes hold partial sums:

```python
@qd.kernel
def sum32(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
          dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(src.shape[0]):
        total = subgroup.reduce_add(src[i], 5)
        if subgroup.invocation_id() == 0:
            dst[i // 32] = total
```

`5` is `log2_size`; `2**5 == 32` matches the block dim. The body of `reduce_add` unrolls at trace time into five `shuffle_down + add` pairs, so the generated IR is identical to a hand-written tree reduction.

### Broadcast the sum to all lanes with `reduce_all_add`

When every lane needs the reduction result — e.g. to normalise by the sum — use the butterfly variant. No follow-up broadcast needed:

```python
@qd.kernel
def normalize32(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        total = subgroup.reduce_all_add(a[i], 5)
        a[i] = a[i] / total
```

Every lane in each group of 32 sees the same `total`.

### Partial-subgroup reductions

`log2_size` does not have to match the full subgroup. Sum groups of 8 with `reduce_add(v, 3)` or groups of 16 with `reduce_all_add(v, 4)`; the caller just ensures `2**log2_size <= subgroup_size` (so up to 5 on CUDA / Metal / RDNA, up to 6 on CDNA).

### Inclusive scan on SPIR-V

```python
@qd.kernel
def cumsum(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        a[i] = subgroup.inclusive_add(a[i])
```

After the call, lane `k` holds `a[0] + a[1] + ... + a[k]`. This compiles only on SPIR-V backends today; for a portable inclusive scan, build one on `shuffle_down` / `shuffle_up` with a log2-step loop.

## Performance notes

- Shuffles are register-to-register on CUDA (`__shfl_sync`, `__shfl_down_sync`) and on SPIR-V where the GPU has hardware support — typically a handful of cycles, no memory traffic.
- AMDGPU `shuffle` and `shuffle_down` both go through `ds_permute` / `ds_bpermute` today (LDS-routed, roughly tens of cycles).
- `reduce_add` and `reduce_all_add` both issue exactly `log2_size` shuffles and `log2_size` adds per call. No barriers, no shared memory, no launch overhead (they inline).
- Pick `reduce_all_add` over `reduce_add + broadcast` when you need the result in every lane — same cost, one fewer shuffle.
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.
- `inclusive_*` (SPIR-V) lowers to a single `OpGroupNonUniform*` instruction — hardware-assisted on most modern GPUs.

## Related

- [tile16](tile16.md) — `Tile16x16` builds on `subgroup.shuffle` to implement register-resident 16×16 matrix tiles.
- `qd.simt.warp.*` — CUDA-only counterparts to the still-stubbed `subgroup.{ballot, all_true, any_true, match_*, active_mask, ...}`. Useful as a fallback when the portable version is not yet implemented; loses cross-backend portability.
