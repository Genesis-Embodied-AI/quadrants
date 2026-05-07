# Atomics

Atomic read-modify-write operations on a single memory location. They do not synchronize threads; the only ordering they provide is the per-location atomicity of the read-modify-write itself. For cooperative ops across threads see the `qd.simt.block.*`, `qd.simt.subgroup.*`, and `qd.simt.grid.*` namespaces. Bit-counting helpers on integer registers (`qd.math.popcnt`, `qd.math.clz`) are documented in [math](math.md).

## What's available

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

## Performance and portability notes

- **Atomic contention is the silent killer of throughput.** The cost of `qd.atomic_add(counter, 1)` from every thread is dominated by serialization at the location, not by the per-thread arithmetic. If many threads hit the same slot, prefer a two-stage scheme: per-warp / per-block reduction first (`qd.simt.block.reduce` if available, or `qd.simt.subgroup.reduce_add`), then a single atomic per warp / block.
- **Pair atomics with the right fence scope.** A bare atomic only orders the location it touches. To make other writes visible to readers that observe the new atomic value, follow the atomic with a fence: block-scope (`qd.simt.block.mem_sync()`) for shared-memory publishing, or grid-scope (`qd.simt.grid.memfence()`) for cross-block coordination.
- **`f64` atomics fall off the fast path** on most backends; if you only need monotonic accumulation, consider Kahan summation in registers and a single atomic-add at the end of the block.
- **`atomic_mul` is generally a CAS loop** under the hood; don't put it on the hot path.

## Related

- [math](math.md) — `qd.math.*`, including the bit-counting helpers (`popcnt`, `clz`) commonly paired with atomics in select / compact patterns.
- `qd.simt.block.*` — block-scope barriers and memory fences (`qd.simt.block.mem_sync()`).
- `qd.simt.subgroup.*` — warp-scope reductions and shuffles, the recommended pre-aggregation step before an atomic.
- `qd.simt.grid.*` — device-scope memory fence (`qd.simt.grid.memfence()`).
- [parallelization](parallelization.md) — thread-synchronization patterns and how atomics fit into the broader synchronization story.
