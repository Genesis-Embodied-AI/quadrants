# Atomics and bit operations

Per-thread primitives — operations a single thread executes on its own registers or on a single memory location. They do not synchronize threads; the only ordering they provide is the per-location atomicity of the read-modify-write itself. For cooperative ops across threads see the `qd.simt.block.*`, `qd.simt.subgroup.*`, and `qd.simt.grid.*` namespaces.

This page covers two groups: atomic read-modify-write operations on global / shared memory (`qd.atomic_*`), and bit-counting helpers on integer registers (`qd.math.popcnt`, `qd.math.clz`).

## What's available

### Atomics

All atomic ops follow the same shape: `qd.atomic_op(x, y)` performs `x = op(x, y)` atomically and returns the **old** value of `x`. `x` must be a writable memory target (a field element, ndarray element, or matrix slot); scalars and constant expressions are not allowed.

| Op             | Semantics                              | i32 | u32 | i64 | u64 | f32 | f64 |
|----------------|----------------------------------------|-----|-----|-----|-----|-----|-----|
| `atomic_add`   | `x += y`                               | yes | yes | yes | yes | yes | \*  |
| `atomic_sub`   | `x -= y`                               | yes | yes | yes | yes | yes | \*  |
| `atomic_mul`   | `x *= y`                               | yes | yes | yes | yes | yes | \*  |
| `atomic_min`   | `x = min(x, y)`                        | yes | yes | yes | yes | yes | \*  |
| `atomic_max`   | `x = max(x, y)`                        | yes | yes | yes | yes | yes | \*  |
| `atomic_and`   | `x &= y`                               | yes | yes | yes | yes | —   | —   |
| `atomic_or`    | `x \|= y`                              | yes | yes | yes | yes | —   | —   |
| `atomic_xor`   | `x ^= y`                               | yes | yes | yes | yes | —   | —   |

\* `f64` atomic add / sub / mul / min / max is hardware-dependent: supported on CUDA sm_60+ for `add`, falls back to a CAS loop elsewhere or raises at codegen time on older targets and on backends that do not lower a CAS loop. Prefer `f32` on hot paths if portability matters.

There is no `atomic_cas` (compare-and-swap) exposed in Python today. The C++ runtime uses CmpXchg internally; surfacing it requires extending `AtomicOpType`.

All atomic ops work on both global memory (fields, ndarrays) and block-shared memory (`qd.simt.block.SharedArray`). They are sequentially consistent on the location they touch; they are **not** memory fences for the rest of the address space — to publish other writes alongside an atomic, pair the atomic with `qd.simt.block.mem_sync()` (block scope) or `qd.simt.grid.memfence()` (device scope).

### Bit-counting helpers

| Op                  | What it returns                              | i32 | u32 | i64 | u64 |
|---------------------|----------------------------------------------|-----|-----|-----|-----|
| `qd.math.popcnt(x)` | Number of set bits in `x`                    | yes | yes | yes | yes |
| `qd.math.clz(x)`    | Number of leading zero bits in `x`           | yes | \*  | yes | \*  |

\* `qd.math.clz` on CUDA currently rejects unsigned 32- and 64-bit inputs (`QD_NOT_IMPLEMENTED`); cast through `qd.bit_cast(x, qd.i32)` / `qd.i64` as a workaround. On SPIR-V, `qd.math.clz` is hard-coded to 32-bit (`FindMSB`); 64-bit input is silently truncated.

The classic CUDA bit-tricks `__ffs` (find first set bit) and `__fns` (find n-th set bit in a mask) are not exposed; for a leading-zero count of a u32, the `bit_cast` workaround above is the canonical approach.

## Semantics

### `qd.atomic_add(x, y)` — and the rest of the family

```python
old = qd.atomic_add(x, y)
# Effect:
#   tmp = load(x)
#   store(x, op(tmp, y))
#   old = tmp
# all three steps execute as a single atomic transaction on x.
```

Properties common to every `qd.atomic_*`:

- **Returns the old value**, not the new one. This matches CUDA's `atomicAdd` and is what enables building reservation patterns: `slot = qd.atomic_add(counter, 1)` gives every thread a unique index.
- **Per-location atomicity, no fence on the rest of memory.** Writes you issued before an atomic on `x` are not necessarily visible to other threads after they observe the new `x`. Pair the atomic with `qd.simt.block.mem_sync()` or `qd.simt.grid.memfence()` if you need that ordering.
- **Vector / matrix arguments fan out element-wise.** `qd.atomic_add(field_of_vec3, qd.Vector([1.0, 2.0, 3.0]))` issues three independent scalar atomic-adds, one per component. There is no all-or-nothing guarantee across the components.

### `qd.atomic_min(x, y)` / `qd.atomic_max(x, y)`

Atomically writes back `min(x, y)` (resp. `max(x, y)`). Returns the old value of `x`. Floating-point min/max follow IEEE rules — `NaN` propagates: if either input is `NaN`, the result is `NaN`.

### `qd.atomic_and(x, y)` / `qd.atomic_or(x, y)` / `qd.atomic_xor(x, y)`

Bitwise atomics. Integer dtypes only — passing `f32` / `f64` raises a type error at trace time.

### `qd.atomic_sub(x, y)` / `qd.atomic_mul(x, y)`

Atomic subtract and atomic multiply. `atomic_sub` is supported natively on most backends; `atomic_mul` on integer types lowers to a CAS loop on hardware without a native multiply atomic and is intentionally not heavily optimised — prefer reducing to a different scheme on hot paths.

### `qd.math.popcnt(x)`

Counts set bits in `x` and returns an `i32`. Lowers to `__popc` / `__popcll` on CUDA, `OpBitCount` on SPIR-V, `__builtin_amdgcn_popcnt` on AMDGPU. Defined for all integer dtypes.

### `qd.math.clz(x)`

Counts leading zero bits in `x` and returns an `i32`. For a 32-bit input, `clz(0) = 32`; otherwise the result is in `[0, 31]`. Lowers to `__nv_clz` / `__nv_clzll` on CUDA, `FindMSB` on SPIR-V (with `bitwidth - 1 - FindMSB` to convert MSB index into leading-zero count), `__builtin_amdgcn_sffbh_i32` on AMDGPU. See the cross-backend caveats in the support table.

## Examples

### Reserving a slot in an output array

```python
counter = qd.field(qd.i32, shape=())
output  = qd.field(qd.f32, shape=(MAX_OUTPUTS,))

@qd.kernel
def emit(values: qd.types.NDArray[qd.f32, 1], threshold: qd.f32) -> None:
    for i in range(values.shape[0]):
        if values[i] > threshold:
            slot = qd.atomic_add(counter[None], 1)
            output[slot] = values[i]
```

Every thread that passes the predicate gets a unique `slot` from the counter. The pattern is the workhorse of select / compact and contact-pair generation.

### Histogram

```python
hist = qd.field(qd.i32, shape=(NBINS,))

@qd.kernel
def histogram(samples: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(samples.shape[0]):
        b = qd.i32(samples[i] * NBINS)
        if 0 <= b < NBINS:
            qd.atomic_add(hist[b], 1)
```

### Bitset population count

```python
@qd.kernel
def count_bits(masks: qd.types.NDArray[qd.u32, 1], total: qd.types.NDArray[qd.i32, 1]) -> None:
    n = 0
    for i in range(masks.shape[0]):
        n += qd.math.popcnt(masks[i])
    qd.atomic_add(total[0], n)
```

### Highest set bit (Morton-code depth)

```python
@qd.func
def msb(x: qd.i32) -> qd.i32:
    return 31 - qd.math.clz(x)
```

For `qd.u32` input on CUDA, cast first: `qd.math.clz(qd.bit_cast(x, qd.i32))`.

## Performance and portability notes

- **Atomic contention is the silent killer of throughput.** The cost of `qd.atomic_add(counter, 1)` from every thread is dominated by serialization at the location, not by the per-thread arithmetic. If many threads hit the same slot, prefer a two-stage scheme: per-warp / per-block reduction first (`qd.simt.block.reduce` if available, or `qd.simt.subgroup.reduce_add`), then a single atomic per warp / block.
- **Pair atomics with the right fence scope.** A bare atomic only orders the location it touches. To make other writes visible to readers that observe the new atomic value, follow the atomic with a fence: block-scope (`qd.simt.block.mem_sync()`) for shared-memory publishing, or grid-scope (`qd.simt.grid.memfence()`) for cross-block coordination.
- **`f64` atomics fall off the fast path** on most backends; if you only need monotonic accumulation, consider Kahan summation in registers and a single atomic-add at the end of the block.
- **`atomic_mul` is generally a CAS loop** under the hood; don't put it on the hot path.
- **Bit-trick portability is uneven.** `qd.math.popcnt` is fully cross-backend; `qd.math.clz` has the dtype caveats noted above. Tests that depend on `qd.math.clz` over u32 or u64 should `bit_cast` to the matching signed type for portability.

## Related

- `qd.simt.block.*` — block-scope barriers and memory fences (`qd.simt.block.mem_sync()`).
- `qd.simt.subgroup.*` — warp-scope reductions and shuffles, the recommended pre-aggregation step before an atomic.
- `qd.simt.grid.*` — device-scope memory fence (`qd.simt.grid.memfence()`).
- [parallelization](parallelization.md) — thread-synchronization patterns and how atomics fit into the broader synchronization story.
