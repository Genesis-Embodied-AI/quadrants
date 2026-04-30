# Subgroup primitives

Subgroup operations let threads within the same subgroup (warp on NVIDIA, wave on AMD) exchange register values directly, without using shared memory or barriers. They are the building block for fast in-warp data exchange — broadcasts, neighbour exchanges, permutations, reductions — and are used internally by `Tile16x16` (see [tile16](tile16.md)).

Subgroup ops live under `qd.simt.subgroup` and are written so the same Python source compiles to the right vendor primitive on each backend.

## What's available

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan) | dtypes                       |
|---------------------------------------------|------|--------|-----------------|------------------------------|
| `subgroup.shuffle(v, idx)`                  | yes  | yes    | yes             | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_down(v, n)`               | yes  | yes\*  | yes             | i32, u32, f32, f64, i64, u64 |
| `subgroup.ballot(predicate)`                | yes  | yes    | yes             | i32 predicate → u32 bitmask  |
| `subgroup.reduce_add(v, log2_size)`         | yes  | yes\*  | yes             | any type supporting `+`      |
| `subgroup.reduce_all_add(v, log2_size)`     | yes  | yes    | yes             | any type supporting `+`      |

\* AMDGPU `shuffle_down` (and therefore `reduce_add`, which is built on it) is currently emulated via `ds_bpermute` (~50 cycle latency).

The remaining shuffle flavours (`shuffle_up`, `shuffle_xor`) are exposed in the Python module but are not yet implemented across backends. Calling them will fail at codegen. Use `shuffle` with an explicit lane index in the meantime — every shuffle pattern can be expressed that way.

The SPIR-V-only no-arg reductions (`subgroup.reduce_mul` / `reduce_min` / `reduce_max` / `reduce_and` / `reduce_or` / `reduce_xor`, plus the original `reduce_add(value)` with no `log2_size`) have been removed in favour of the portable sized API described below. For reductions other than sum, build a sized helper on top of `shuffle_down` / `shuffle` following the same pattern.

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

### Common to both

- Ops are issued under a full active mask on CUDA (`0xFFFFFFFF`). Call them from uniform control flow; calling from divergent control flow is undefined on most backends. (this means: all threads have to execute the shuffle)
- Subgroup size varies by backend (32 on NVIDIA, 32 or 64 on AMD, 32 in Vulkan compute on most GPUs).

### `ballot(predicate)`

Each lane evaluates `predicate` (an `i32`; non-zero is true, zero is false) and the result is a `u32` bitmask where bit `i` is set if lane `i`'s predicate was non-zero.

- Returns a `u32`. Bit 0 corresponds to lane 0, bit 1 to lane 1, etc.
- On CUDA, maps to `__ballot_sync(0xFFFFFFFF, predicate)`. On SPIR-V, maps to `OpGroupNonUniformBallot` (component 0 of the uvec4 result). On AMDGPU, maps to the `ballot.i32` intrinsic.
- The result covers the first 32 lanes. On AMDGPU CDNA with 64-wide wavefronts only the low 32 bits are returned; the upper 32 lanes are not represented. This is consistent with the 32-bit return type.
- Must be called from uniform control flow (all active lanes must execute the ballot).

Ballot is a building block for warp-cooperative algorithms: population counts (`popcount(ballot(cond))` counts how many lanes satisfy `cond`), prefix masks, and lane compaction.

### `reduce_add(value, log2_size)`

Sums `value` across `2**log2_size` consecutive lanes via a `shuffle_down` tree. The result is valid **in lane 0** of each group; other lanes hold partial sums and should be considered undefined.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into exactly `log2_size` `shuffle_down + add` pairs in the calling kernel's IR, with no runtime loop overhead.
- `2**log2_size` must not exceed the active subgroup size on the target (32 on CUDA/Metal and on RDNA, 64 on CDNA). Passing a larger value produces implementation-defined results; it does not error.
- The reduction works on any type that supports `+` and `shuffle_down`; in practice this means i32, u32, f32, f64, i64, u64.
- Decorated with `@qd.func` and inlined into the calling kernel — there is no kernel-launch overhead and no separate symbol to link.

Lanes 1..`2**log2_size - 1` receive undefined-but-safe partial sums (they never touch out-of-range lanes because the tree shrinks each step), but only lane 0's result is meaningful for the caller.

### `reduce_all_add(value, log2_size)`

Same sum as `reduce_add`, but broadcast to **every lane** in each `2**log2_size` group. Implemented as a butterfly using `shuffle` with `lane ^ mask`, `mask` stepping through `1, 2, 4, ..., 2**(log2_size-1)`.

- Same `log2_size` template + size-cap contract as `reduce_add`.
- Use this when every lane needs the reduction result (e.g. to divide by the sum, or to branch on it uniformly). It costs exactly the same number of shuffles as `reduce_add` but leaves the answer in all lanes, so it replaces a `reduce_add` + `shuffle`/broadcast pair.
- Uses `subgroup.shuffle` under the hood.

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

After the kernel, every lane in a subgroup holds the original value of its lane 0.

### Ballot: count how many lanes satisfy a condition

```python
@qd.kernel
def count_positive(a: qd.types.ndarray(dtype=qd.f32, ndim=1),
                   counts: qd.types.ndarray(dtype=qd.u32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        mask = subgroup.ballot(qd.i32(a[i] > 0.0))
        if subgroup.invocation_id() == 0:
            counts[i // 32] = mask
```

After the kernel, `counts[g]` contains a bitmask of which lanes in group `g` had positive values. Use `popcount(mask)` on the host to get the count.

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

`log2_size` does not have to match the full subgroup. Sum groups of 8 with `reduce_add(v, 3)` or groups of 16 with `reduce_all_add(v, 4)`; the caller just ensures `2**log2_size <= subgroup_size` (so 5 on CUDA / Metal / RDNA, up to 6 on CDNA).

## Performance notes

- Shuffles are register-to-register on CUDA (`__shfl_sync`, `__shfl_down_sync`) and on SPIR-V where the GPU has hardware support — typically a handful of cycles, no memory traffic.
- AMDGPU `shuffle` and `shuffle_down` both go through `ds_permute`/`ds_bpermute` today (LDS-routed, roughly tens of cycles).
- `ballot` is a single hardware instruction on all backends — one cycle on CUDA (`__ballot_sync`), one instruction on AMDGPU (`v_ballot_b32`), and `OpGroupNonUniformBallot` on SPIR-V.
- `reduce_add` and `reduce_all_add` both issue exactly `log2_size` shuffles and `log2_size` adds per call. No barriers, no shared memory, no launch overhead (they inline).
- Pick `reduce_all_add` over `reduce_add + broadcast` when you need the result in every lane — same cost, one fewer shuffle.
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.

## Related

- [tile16](tile16.md) — `Tile16x16` builds on `subgroup.shuffle` to implement register-resident 16x16 matrix tiles.
- `subgroup.invocation_id()` — returns this lane's subgroup-local index.
- `subgroup.group_size()` — returns the active subgroup size.
- `subgroup.ballot` — returns a u32 bitmask of lanes where the predicate is non-zero (see above).
- `subgroup.reduce_add` / `subgroup.reduce_all_add` — portable sized sum reductions built on `shuffle_down` / `shuffle` (see above).
