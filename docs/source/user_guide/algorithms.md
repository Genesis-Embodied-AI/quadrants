# Algorithms

The algorithms here operate at device-level, using all available gpu cores, and contrast with:
- [Per-thread linear algorithms](linalg_per_thread.md): operate on per-thread level
- [Subgroup-level operations](subgroup.md): operate on per-subgroup level
- [Block-level operations](block.md): operate on per-block level

Each function is a kernel-side qd.func functions, which must be launched from inside a qd.kernel. They are fully compatible with graph. Every op requires caller-owned **scratch**, covered in a later section.

## What's available

| Op                                                          | What it does                                                       | CUDA | AMDGPU | Vulkan | Metal |
|-------------------------------------------------------------|--------------------------------------------------------------------|------|--------|--------|-------|
| `qd.algorithms.reduce_{add,min,max}(arr, out, scratch, n, DTYPE, LOG256_MAX_N)` | `out[0] = sum/min/max(arr[0:n])` (fixed-depth tree reduction; identity derived from `DTYPE` for min / max). Composed at the top level of your own kernel (device-resident count `n`, compile-time `LOG256_MAX_N`). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.exclusive_scan_{add,min,max}(arr, out, scratch, n, DTYPE, LOG256_MAX_N)` | `out[i] = sum/min/max(arr[0:i])` (three-pass Blelloch-style scan; 32-bit + 64-bit scalars; identity derived from `DTYPE` for min / max). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.select(arr, flags, out, num_out, scratch, n, LOG256_MAX_N)` | Stream compaction: copy `arr[i]` to a dense prefix of `out` for every `flags[i] == 1` (`flags` must be exactly 0/1; no `DTYPE` - the scatter is dtype-agnostic). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.sort(keys, tmp_keys, values, tmp_values, scratch, n, KEY_DTYPE, HAS_VALUES, END_BIT, LOG256_MAX_N)` | LSB radix sort (32-bit / 64-bit scalar keys, optional key-value). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch, n, VALUE_DTYPE, LOG256_MAX_N)` | Collapse each consecutive run of equal keys into `(key, sum_of_values)` (`VALUE_DTYPE` only for the `values_out` zero-init). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.{reduce,exclusive_scan,select,reduce_by_key,sort}_scratch_slots(...)` | Host- and kernel-callable helpers returning the scratch slot count each op needs. | —    | —      | —      | —     |
| `qd.algorithms.parallel_sort`                               | Odd-even merge sort (in-place, key or key-value). **Deprecated**: prefer `sort`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.PrefixSumExecutor`                           | Inclusive in-place prefix sum (i32 only). **Deprecated**: prefer `exclusive_scan_add`. | yes  | no     | yes    | no    |

\* `reduce_{add,min,max}`, `exclusive_scan_{add,min,max}`, `select`, `sort`, `reduce_by_key_add`, and `parallel_sort` run anywhere a Quadrants kernel runs; portability is inherited from the underlying block / subgroup primitives.

## Composable `@qd.func` ops

Every algorithm is a composable `@qd.func` (e.g. `reduce_add`, `sort`): call it at the **top level** of your own `@qd.kernel`, so the op is captured into the same kernel / graph as your surrounding phases. The function is annotated with `requires_top_level=True`, so attempting to use it outside of top level will throw an exception [FIXME: implement this]. Note that within a `qd.checkpoint`, `qd.loop_do_while` both count as being at top-level.

## Key sizing parameters

Two parameters control sizing:

- **live element count**, `n`: , the number of elements that will actually be handled by the algorithm. This is a runtime value, provided in a scalar tensor. It can be modified without recompiling the kernel.
- **maximum capacity**, `LOG256_MAX_N`: a **compile-time** constant. One compiled kernel can by used for any live count up to that capacity — see [Capacity (`LOG256_MAX_N`)](#capacity-log256_max_n).

Because each op runs entirely as device code it does no host-side validation (a size check would force a device→host read of the count that defeats graph capture), so you must size its `scratch` correctly up front — see [Scratch space](#scratch-space).

## Capacity (`LOG256_MAX_N`)

These functions bake their **maximum capacity** in as a compile-time constant, `LOG256_MAX_N`. An op compiled for a given `LOG256_MAX_N` correctly handles any live count `n` with `0 <= n <= 256 ** LOG256_MAX_N`. Passing in an `n` outside of these bounds is unchecked undefined behavior.

The capacity grows fast — each level multiplies it by 256:

| `LOG256_MAX_N` | Max count (`256 ** LOG256_MAX_N`) |
|----------------|-----------------------------------|
| `1`            | `256` |
| `2`            | `65,536` |
| `3`            | `16,777,216` |
| `4`            | `4,294,967,296` (≈ 4.3 billion) |

`LOG256_MAX_N = 4` already exceeds what a 32-bit index can address (`256 ** 4 == 2 ** 32`), so in practice you rarely need more than 3–4.

It must be compile-time because it fixes the number and order of the internal launches: the staircase is statically unrolled to `LOG256_MAX_N` levels, and that fixed launch topology is exactly what lets **one captured graph replay for any count up to the capacity** without re-tracing. The live count `n` flows as a device value while `LOG256_MAX_N` is frozen at trace time.

**Pick it from an upper bound, not the current count.** Use the smallest `LOG256_MAX_N` whose capacity covers the largest count you will ever feed the captured graph — `LOG256_MAX_N = ceil(log256(capacity))`, floored at `1`. Size it against a *provisioned* upper bound (a buffer capacity, qipc's `padded_N`, ...), not today's `n`, so the same graph serves the whole range below it:

```python
def log256_max_n(capacity: int) -> int:
    d = 1
    while 256 ** d < capacity:
        d += 1
    return d
```

- **Over-specifying is safe.** A capacity larger than the live count just adds staircase levels that operate on length-1 buffers — harmless identity no-ops. The only cost is a few extra empty launches and marginally larger scratch, so when in doubt, round up.
- **Under-specifying is a bug.** If `256 ** LOG256_MAX_N < n` the op cannot address the tail of the input, and there is no host-side guard to catch it. Always cover your worst case.

Size `scratch` with the **same** `LOG256_MAX_N` you compile the op for — `reduce_scratch_slots(capacity, log256_max_n)`, `exclusive_scan_scratch_slots(capacity, log256_max_n)`, `sort_scratch_slots(capacity, log256_max_n)` — see [Scratch space](#scratch-space). (The `*_scratch_slots` helpers can also derive the minimal depth from `N` for you when you omit the argument — but that path is host-only, so when composing the op into a kernel always pass the explicit depth.)

## Scratch space

The algorithms need scratch space in order to run. This is temporary space used by the algorithms. The caller owns the scratch space. It does not need to be initialized in any way. It does need to exist, and be correctly sized, and typed. Every algorithm takes a mandatory `scratch` argument, to receive the scratch space.

**Ask first, then allocate.** Each algorithm ships a companion `*_scratch_slots(N)` function - branch-free integer arithmetic, no device round-trip - that returns the minimum number of slots needed for a length-`N` input. Allocate at least that many; allocating more is fine. These functions are both **host- and kernel-callable**: pass a Python `int` to size an allocation up front, or call the same function inside a `@qd.kernel` on a device-read `N` to recompute the requirement on-device and validate it against `scratch.shape[0]` without ever reading `N` back to the host..

| Algorithm | Sizing function | Scratch dtype |
|-----------|-----------------|---------------|
| `reduce_{add,min,max}` | `reduce_scratch_slots(N[, log256_max_n])` | `u32` (4-byte `arr`) / `u64` (8-byte `arr`) |
| `exclusive_scan_{add,min,max}` | `exclusive_scan_scratch_slots(N[, log256_max_n])` | `u32` (4-byte `arr`) / `u64` (8-byte `arr`) |
| `select` | `select_scratch_slots(N)` | `u32` (always) |
| `reduce_by_key_add` | `reduce_by_key_scratch_slots(N)` | `u32` (always) |
| `sort` | `sort_scratch_slots(N[, log256_max_n])` | `u32` (always, regardless of key width) |

The slot count is **dtype-width-independent** (it is a count, not a byte count). For the 4-byte / 8-byte algorithms (`reduce`, `scan`) you allocate the *same number of slots* but in a `u32` buffer for 4-byte element dtypes and a `u64` buffer for 8-byte ones - the partials are `bit_cast` to / from the element dtype. `select`, `reduce_by_key_add`, and the radix sort always use `u32` scratch (they stage counts / indices / tile histograms, which are `u32` regardless of the element / key dtype).

```python
import quadrants as qd

N = 1_000_000   # capacity: an upper bound on the live count you will feed the captured graph
D = 3           # LOG256_MAX_N: 256**3 = 16,777,216 >= N

# 4-byte reduce: u32 scratch, sized for the same depth D the func is compiled for.
scratch = qd.ndarray(qd.u32, shape=qd.algorithms.reduce_scratch_slots(N, D))

# 8-byte reduce: same slot count, u64 buffer.
scratch64 = qd.ndarray(qd.u64, shape=qd.algorithms.reduce_scratch_slots(N, D))
```

**No on-device check.** The composable `@qd.func` forms run directly as device code, so they do **no** scratch-sufficiency check. Size `scratch` correctly up front with the matching helper — `reduce_scratch_slots(N, log256_max_n)`, `exclusive_scan_scratch_slots(N, log256_max_n)`, `select_scratch_slots(N)`, `reduce_by_key_scratch_slots(N)`, or `sort_scratch_slots(N, log256_max_n)` — for the capacity you compile the op against. An undersized buffer corrupts the output (or reads / writes out of bounds) rather than raising.

## Semantics

The active ops below share a calling convention and several rules; these are stated once in **Common conventions**, and only the op-specific behaviour is repeated per op. The internal algorithm for each op is in [Under the hood](#under-the-hood).

Each op section ends with a runnable toy example. They all assume this prelude:

```python
import numpy as np
import quadrants as qd

qd.init(arch=qd.gpu)

N, D = 8, 1   # 8 elements; D = LOG256_MAX_N = 1 → capacity 256**1 = 256 ≥ N
```

### Common conventions

**Call site.** Every op is a composable `@qd.func`; call it at the **top level** of your own `@qd.kernel` (see [Composable `@qd.func` ops](#composable-qdfunc-ops)). **Never** nest the call in ordinary runtime `for` / `if` / `while` control flow: that demotes the op's internal phase loops out of top-level position and drops the per-phase grid-wide barriers it relies on, silently corrupting the result.

**Shared arguments.** Every op takes a device-resident live count `n` and a compile-time capacity `LOG256_MAX_N`; the ops that handle typed values also take the element dtype as an explicit template (`DTYPE` / `KEY_DTYPE` / `VALUE_DTYPE`):

- `n` — the live count, read **on-device** (e.g. `count[0]`, or `n[()]` for a 0-d count). See [Key sizing parameters](#key-sizing-parameters).
- `LOG256_MAX_N` — the compile-time phase count; the op handles any count `<= 256 ** LOG256_MAX_N`, so one captured graph replays across that whole range. See [Capacity (`LOG256_MAX_N`)](#capacity-log256_max_n).
- `DTYPE` / `KEY_DTYPE` / `VALUE_DTYPE` — the element dtype, passed explicitly because an `ndarray` kernel argument exposes no in-kernel `.dtype`.
- `scratch` — caller-owned workspace; size it with the matching `*_scratch_slots(capacity, LOG256_MAX_N)` helper for the capacity you compile against. See [Scratch space](#scratch-space).

**Tensor polymorphism.** Every 1-D tensor argument is an `ndarray` kernel parameter, so it is polymorphic over `qd.field`, `qd.ndarray`, and `qd.Tensor`.

**Scalar dtypes & scratch width (`reduce` / `exclusive_scan`).** The element dtype is one of `{qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64}`; narrower / wider scalar dtypes (`qd.i16`, `qd.f16`, …) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage their partials through a `u32` scratch and 8-byte dtypes through a `u64` scratch (same slot count either way; see [Scratch space](#scratch-space)).

**Identity value (`reduce` / `exclusive_scan`, min / max).** The *identity* is the value that leaves a result unchanged when combined under the op — `add` → `0`, `min` → the largest representable value, `max` → the smallest. It pads out-of-range lanes and seeds `exclusive_scan`'s `out[0]`. It is derived in-kernel from the element dtype, so there is no runtime identity argument (mirroring the `block.reduce_min` / `subgroup.reduce_min` typed wrappers): `0` for `add`; for `min`, `+inf` (floats) / `INT{32,64}_MAX` (signed ints) / `UINT{32,64}_MAX` (unsigned); for `max`, `-inf` (floats) / `INT{32,64}_MIN` (signed ints) / `0` (unsigned).

**Floating-point non-associativity.** For `add` on `f32` / `f64`, the device combine order differs from a left-to-right host pass, so results are **not** bitwise-equal to `numpy.sum` / `numpy.cumsum` (nor reproducible across `N` changes). Tests tolerate a small relative error rather than asserting bitwise equality.

**Composition shape.** Each op is one call placed among your other top-level phases:

```python
@qd.kernel
def my_pipeline(
    arr: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    count: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    # ... other top-level phases ...
    qd.algorithms.reduce_add(arr, out, scratch, count[0], qd.f32, LOG256_MAX_N)
    # ... more top-level phases ...
```

The per-op snippets below show only the call line; drop it into a kernel like the one above.

### `qd.algorithms.reduce_{add,min,max}`

Device-wide tree reduction over a 1-D tensor: `out[0]` holds `sum(arr[0:n])` / `min(arr[0:n])` / `max(arr[0:n])`. Signature `reduce_{add,min,max}(arr, out, scratch, n, DTYPE, LOG256_MAX_N)`.

Arguments (see [Common conventions](#common-conventions) for `n` / `DTYPE` / `LOG256_MAX_N`, `out`, and tensor polymorphism):

- `arr`: 1-D input tensor.
- `out`: 1-element tensor with the same dtype as `arr`.
- `scratch`: `reduce_scratch_slots(N, LOG256_MAX_N)` slots — `u32` for 4-byte `arr`, `u64` for 8-byte.

Constraints (plus the shared scalar-dtype set and `f32` / `f64` non-associativity from [Common conventions](#common-conventions)):

- **Shape:** `arr` must be 1-D and `out.shape` must be `(1,)`; both share the same dtype.

Scratch footprint: `reduce_scratch_slots(N)` ≈ `ceil(N / BLOCK_DIM)` slots, where `BLOCK_DIM = 256` (`N = 1G` is ~4M slots). See [Scratch space](#scratch-space).

Example — sum of an array:

```python
arr     = qd.field(qd.i32, shape=N)
out     = qd.field(qd.i32, shape=1)
scratch = qd.field(qd.u32, shape=qd.algorithms.reduce_scratch_slots(N, D))
count   = qd.field(qd.i32, shape=1)

arr.from_numpy(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))
count.from_numpy(np.array([N], dtype=np.int32))

@qd.kernel
def run():
    qd.algorithms.reduce_add(arr, out, scratch, count[0], qd.i32, D)

run()
print(out.to_numpy()[0])   # 31   (3 + 1 + 4 + 1 + 5 + 9 + 2 + 6)
```

`reduce_min` / `reduce_max` are identical apart from the call name (they give `1` and `9` for this input).

### `qd.algorithms.exclusive_scan_{add,min,max}`

Device-wide exclusive prefix scan over a 1-D tensor: `out[i]` holds the reduction (`sum` / `min` / `max`) of `arr[0:i]`, and `out[0]` is always the op's [identity value](#common-conventions) (`0` for `add`). Signature `exclusive_scan_{add,min,max}(arr, out, scratch, n, DTYPE, LOG256_MAX_N)`.

Arguments (see [Common conventions](#common-conventions) for `n` / `DTYPE` / `LOG256_MAX_N`):

- `arr` / `out`: 1-D input / output tensors, same shape and dtype; `out` must be a **distinct** buffer from `arr` (see constraints).
- `scratch`: `exclusive_scan_scratch_slots(N, LOG256_MAX_N)` slots — `u32` for 4-byte `arr`, `u64` for 8-byte.

Constraints (plus the shared scalar-dtype set and `add` non-associativity from [Common conventions](#common-conventions)):

- **Shape:** `arr` and `out` must both be 1-D with the same shape and dtype.
- **No in-place scan:** `out` must be a distinct buffer from `arr`. Calling with `out is arr` raises `ValueError`. (The kernels do not protect against same-buffer aliasing; allocating one extra buffer once is cheap relative to the scan itself.)

Scratch footprint: `exclusive_scan_scratch_slots(N)` ≈ `ceil(N / BLOCK_DIM)` slots for the per-tile partials plus the recursive staircase above them (`4112` slots at `N = 1M`). See [Scratch space](#scratch-space).

Example — running total:

```python
arr     = qd.field(qd.i32, shape=N)
out     = qd.field(qd.i32, shape=N)   # must be a distinct buffer from arr
scratch = qd.field(qd.u32, shape=qd.algorithms.exclusive_scan_scratch_slots(N, D))
count   = qd.field(qd.i32, shape=1)

arr.from_numpy(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))
count.from_numpy(np.array([N], dtype=np.int32))

@qd.kernel
def run():
    qd.algorithms.exclusive_scan_add(arr, out, scratch, count[0], qd.i32, D)

run()
print(out.to_numpy())   # [ 0  3  4  8  9 14 23 25 ]   (out[i] = sum(arr[:i]))
```

### `qd.algorithms.select`

Stream compaction. Copy every `arr[i]` whose corresponding `flags[i]` is `1` into a dense prefix of `out`, in stable input order, and write the count of selected elements to `num_out[0]`. Signature `select(arr, flags, out, num_out, scratch, n, LOG256_MAX_N)` — there is **no `DTYPE` argument**: the scatter `out[idx] = arr[i]` lowers per-field, so `select` works for scalar *and* struct element dtypes unchanged.

Arguments (see [Common conventions](#common-conventions) for `n` / `LOG256_MAX_N` and the `num_out` host-hop rule):

- **`arr`:** 1-D input of any scalar dtype in `{qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64}` *or* any `qd.types.struct(...)` / `qd.Struct.field({...})` composite (e.g. libuipc `Vector2i` / `Vector3i` / `Vector4i` / `LinearBVHAABB`-style structs).
- **`flags`:** 1-D `qd.i32` tensor with the same shape as `arr`. **Every entry must be exactly `0` or `1`** (`1` selects). The algorithm prefix-sums `flags` directly as counts, so non-0/1 values produce wrong indices and a wrong `num_out` count — the caller is responsible for normalization (no implicit normalization pass). Populate it with a kernel applying whatever predicate you want.
- **`out`:** 1-D tensor, same dtype as `arr`, with `len(out) >= len(arr)` so the worst-case all-selected run is safe. Only `out[0 : num_out[0]]` carries meaningful data on return; the tail is left untouched.
- **`num_out`:** 1-element `qd.i32` tensor receiving the selected count.
- **`scratch`:** `select_scratch_slots(N)` slots — always `u32`, regardless of `arr.dtype`. See [Scratch space](#scratch-space).

Scratch footprint: `select_scratch_slots(N)` ≈ `N` u32 slots (one write index per input element). See [Scratch space](#scratch-space).

Example — keep the flagged elements:

```python
arr     = qd.field(qd.i32, shape=N)
flags   = qd.field(qd.i32, shape=N)
out     = qd.field(qd.i32, shape=N)   # len(out) >= len(arr)
num_out = qd.field(qd.i32, shape=1)
scratch = qd.field(qd.u32, shape=qd.algorithms.select_scratch_slots(N))
count   = qd.field(qd.i32, shape=1)

arr.from_numpy(np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int32))
flags.from_numpy(np.array([1,  0,  1,  1,  0,  0,  1,  0], dtype=np.int32))
count.from_numpy(np.array([N], dtype=np.int32))

@qd.kernel
def run():
    qd.algorithms.select(arr, flags, out, num_out, scratch, count[0], D)

run()
k = int(num_out.to_numpy()[0])
print(k)                    # 4
print(out.to_numpy()[:k])   # [10 12 13 16]   (the flagged elements, in input order)
```

### `qd.algorithms.sort`

Ascending in-place LSB radix sort over a 1-D tensor of 32-bit or 64-bit scalar keys (`u32` / `i32` / `f32` / `u64` / `i64` / `f64`), with optional lock-step permutation of a `values` tensor (key-value sort). Called as a `@qd.func` at the **top level** of your own `@qd.kernel` (the qipc path) so the sort composes with your other phases into one compiled kernel / captured graph:

`sort(keys, tmp_keys, values, tmp_values, scratch, n, KEY_DTYPE, HAS_VALUES, END_BIT, LOG256_MAX_N)`

Here `n` is a 0-d `i32` ndarray and the compile-time flags (`KEY_DTYPE`, `HAS_VALUES`, `END_BIT`, `LOG256_MAX_N`) are passed explicitly (see [Common conventions](#common-conventions) for `n` / `KEY_DTYPE` / `LOG256_MAX_N`). Pass real `values` / `tmp_values` with `HAS_VALUES=True` for a key-value sort; **for a keys-only sort pass `keys` / `tmp_keys` again in the `values` / `tmp_values` slots** as placeholders and set `HAS_VALUES=False` (every value access is `HAS_VALUES`-guarded). The func does **no** host-side validation or scratch-sufficiency check (a DtoH would defeat graph capture), so size `scratch` correctly up front.

Arguments:

- `keys`: 1-D tensor (`qd.field`, `qd.ndarray`, or `qd.Tensor`). Sorted **in place**.
- `tmp_keys`: ping-pong workspace, same shape & dtype as `keys`, distinct buffer. Contents on return are intermediate and should be considered garbage.
- `values`, `tmp_values`: the key-value buffers (any supported scalar dtype, independent of the key dtype, same shape as `keys`, distinct from each other) when `HAS_VALUES=True`. For a keys-only sort pass `keys` / `tmp_keys` here as placeholders and set `HAS_VALUES=False`.
- `scratch`: required caller-owned 1-D `qd.u32` tensor used as the per-pass tile-histogram + scan workspace. Size it with `qd.algorithms.sort_scratch_slots(N, LOG256_MAX_N)` (the footprint is dtype-independent — tile histograms are `u32` regardless of key width). The func does no sufficiency check, so size it correctly up front. See [Scratch space](#scratch-space).
- `n`: 0-d `i32` ndarray (`shape=()`) holding the element count **on-device** (read as `n[()]`).
- `KEY_DTYPE`: the key element dtype, passed explicitly (see [Common conventions](#common-conventions)).
- `HAS_VALUES`: compile-time bool — whether `values` / `tmp_values` are real buffers (`True`) or placeholders (`False`).
- `END_BIT`: number of low key bits to sort. Use the full key width (32 for 4-byte keys, 64 for 8-byte) unless the high bits are known to be zero (e.g. `16` for keys `< 2**16`, to save passes). Must be a positive multiple of `8` that yields an even number of digit passes so the result lands back in `keys`.
- `LOG256_MAX_N`: scan depth `D` (the compile-time capacity; see [Common conventions](#common-conventions)). Size `scratch` with the same `D`.

Constraints:

- **Dtypes:** the key dtype and value dtype are each independently one of `{qd.u32, qd.i32, qd.f32, qd.u64, qd.i64, qd.f64}`. Narrower scalar dtypes (`qd.i16`, `qd.f16`, ...) and struct dtypes raise `NotImplementedError` at compile time. 8-byte keys run 8 digit passes per sort; 4-byte keys run 4. Scratch footprint is the same for both widths (the per-tile histograms are `u32` regardless).
- **Aliasing:** `keys` and `tmp_keys` must be distinct buffers; same for real `values` / `tmp_values`. `sort` does not check this (passing the same buffer corrupts the sort).
- **Stability:** stable sort - equal keys keep their original input order in the output.
- **NaN handling (f32):** matches `numpy.sort` (NaNs land at the end). NaNs are not tested separately and should not be relied on for ordering invariants beyond `numpy.sort`.

Scratch footprint: `num_blocks * 256 + ...` u32 slots (where `num_blocks = ceil(N / 256)`), plus the scan staircase. Call `qd.algorithms.sort_scratch_slots(N, LOG256_MAX_N)` to get the exact slot count (pure host arithmetic, no device round-trip). See [Scratch space](#scratch-space).

Example — key-value sort. Unlike the others, `sort` takes its count as a **0-d** tensor (read on-device as `n[()]`) and its buffers as kernel arguments. Here `values` rides along with the keys (the original indices), `HAS_VALUES=True`, and `END_BIT=32` is the full `i32` key width:

```python
i32_1d = qd.types.ndarray(qd.i32, ndim=1)
u32_1d = qd.types.ndarray(qd.u32, ndim=1)
i32_0d = qd.types.ndarray(qd.i32, ndim=0)

@qd.kernel
def run(keys: i32_1d, tmp_keys: i32_1d, values: i32_1d, tmp_values: i32_1d, scratch: u32_1d, n: i32_0d):
    qd.algorithms.sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.i32, True, 32, D)

keys       = qd.ndarray(qd.i32, shape=(N,))
tmp_keys   = qd.ndarray(qd.i32, shape=(N,))
values     = qd.ndarray(qd.i32, shape=(N,))
tmp_values = qd.ndarray(qd.i32, shape=(N,))
scratch    = qd.ndarray(qd.u32, shape=(qd.algorithms.sort_scratch_slots(N, D),))
n          = qd.ndarray(qd.i32, shape=())

keys.from_numpy(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))
values.from_numpy(np.arange(N, dtype=np.int32))   # payload: each key's original index
n.fill(N)

run(keys, tmp_keys, values, tmp_values, scratch, n)
print(keys.to_numpy())     # [1 1 2 3 4 5 6 9]
print(values.to_numpy())   # [1 3 6 0 2 4 7 5]   (original indices; stable for the tied 1s)
```

### `qd.algorithms.reduce_by_key_add`

Collapse every **consecutive run of equal keys** into a single output entry `(unique_key, sum_of_values_in_run)`. Keys that compare equal but are separated by other keys form separate runs. For a global per-key sum, sort by key first (e.g. with `qd.algorithms.sort`) and then reduce-by-key. Signature `reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch, n, VALUE_DTYPE, LOG256_MAX_N)`.

Arguments (see [Common conventions](#common-conventions) for `n` / `LOG256_MAX_N` and the `num_runs` host-hop rule):

- `keys_in`: 1-D tensor of `u32` / `i32` / `f32`.
- `values_in`: 1-D tensor of `u32` / `i32` / `f32`, same shape as `keys_in`.
- `keys_out`: 1-D tensor of the same dtype as `keys_in`, with `len(keys_out) >= len(keys_in)` so the worst-case-all-unique run is safe. Only `keys_out[0 : num_runs[0]]` carries meaningful data on return; the tail is untouched.
- `values_out`: 1-D tensor of the same dtype as `values_in`, same length requirement. The first `num_runs[0]` slots are overwritten; the tail past that prefix is left untouched.
- `num_runs`: 1-element `qd.i32` tensor receiving the number of runs.
- `scratch`: `reduce_by_key_scratch_slots(N)` slots — always `u32`, regardless of key / value dtype. See [Scratch space](#scratch-space).
- `VALUE_DTYPE`: the values dtype, passed explicitly (used only to write the typed zero into `values_out` before the scatter's `atomic_add`; keys are handled generically).

Constraints:

- **Dtypes (first land):** `keys_in.dtype` and `values_in.dtype` ∈ {`qd.i32`, `qd.u32`, `qd.f32`}. Other dtypes raise `NotImplementedError`.
- **Reduction:** only `add` is exposed for first land. `min` / `max` variants need `atomic_min` / `atomic_max` for `f32`, which has spottier cross-backend support; defer to a follow-up gated on real qipc usage.
- **f32 non-associativity:** the order of additions inside a run is set by hardware atomic ordering, not host order, so `f32` results are *not* bitwise-equal to a serial scan. Tests tolerate a small relative error.
- **NaN handling (f32 keys):** `NaN != NaN` is true, so each NaN-keyed element becomes its own run. Consistent with treating NaN as "different from everything", which matches the run-length-encoding spirit.

Scratch footprint: `reduce_by_key_scratch_slots(N)` ≈ `1.004 * N` u32 slots. See [Scratch space](#scratch-space).

Example — sum consecutive runs of equal keys:

```python
keys_in    = qd.field(qd.i32, shape=N)
values_in  = qd.field(qd.i32, shape=N)
keys_out   = qd.field(qd.i32, shape=N)
values_out = qd.field(qd.i32, shape=N)
num_runs   = qd.field(qd.i32, shape=1)
scratch    = qd.field(qd.u32, shape=qd.algorithms.reduce_by_key_scratch_slots(N))
count      = qd.field(qd.i32, shape=1)

keys_in.from_numpy(np.array([1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32))
values_in.from_numpy(np.array([5, 2, 1, 4, 4, 6, 1, 1], dtype=np.int32))
count.from_numpy(np.array([N], dtype=np.int32))

@qd.kernel
def run():
    qd.algorithms.reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch, count[0], qd.i32, D)

run()
r = int(num_runs.to_numpy()[0])
print(r)                          # 3
print(keys_out.to_numpy()[:r])    # [1 2 3]
print(values_out.to_numpy()[:r])  # [8 8 8]   (5+2+1, 4+4, 6+1+1)
```

### `qd.algorithms.parallel_sort(keys, values=None)`

> **Deprecated.** New code should call the LSB radix sort `qd.algorithms.sort` (a `@qd.func`) instead. The radix sort is asymptotically `O(N log_radix N)` rather than `O(N log^2 N)`, is **stable** (odd-even merge sort is not), supports 32-bit and 64-bit scalar keys across CUDA / AMDGPU / Vulkan / Metal, and accepts `qd.field`, `qd.ndarray`, and `qd.Tensor` (`parallel_sort` is field-only). The only thing `parallel_sort` is competitive on is very small N (~4K and below); even there the radix path is comparable on modern hardware. To migrate, allocate `tmp_keys` of the same shape and dtype as `keys` plus a `u32` `scratch` buffer, then call `sort` at the top level of a kernel (see its section above for the full signature). `parallel_sort` is kept for one release cycle for backward compat and will be removed thereafter.

In-place sort. Reorders `keys` ascending; if `values` is provided, applies the same permutation to `values` (key-value sort). Both arguments must be 1-D `qd.field` - `parallel_sort` reaches into `snode.ptr.offset` internally, so `ndarray` is **not** supported and will fail at compile time with an `AttributeError`.

```python
import quadrants as qd

keys = qd.field(qd.i32, shape=(N,))
qd.algorithms.parallel_sort(keys)
```

```python
keys = qd.field(qd.i32, shape=(N,))
vals = qd.field(qd.f32, shape=(N,))
qd.algorithms.parallel_sort(keys, vals)
```

- **Algorithm.** Batcher's odd-even merge sort. Time complexity `O(N log² N)`, work-efficient for small / mid-sized arrays.
- **Key dtype.** Whatever the key field's dtype is, as long as `<` is meaningful for it (integer and float types).
- **Stability.** Odd-even merge sort is *not* a stable sort - equal keys may be reordered relative to one another. If stability matters, encode tiebreakers into the keys (e.g. pack the original index into the low bits).
- **Memory.** Strictly in-place - no auxiliary buffers from the caller's perspective.
- **Performance characteristic.** Beats radix-style sorts for small N (roughly N ≲ 4K).

### `qd.algorithms.PrefixSumExecutor`

> **Deprecated.** New code should call `qd.algorithms.exclusive_scan_add` instead. `PrefixSumExecutor` is **inclusive**-only, **`i32`**-only, and **CUDA / Vulkan**-only; the new functional API covers `{i32, u32, f32, i64, u64, f64}` on every supported backend and runs the exclusive variant directly. To migrate from inclusive in-place to exclusive out-of-place, drop the `Executor` wrapper, allocate a distinct `out` field, and post-process if you actually need the inclusive form (`inclusive[i] = exclusive[i] + arr[i]`). `PrefixSumExecutor` is kept for one release cycle for backward compat and will be removed in a future release.

Inclusive in-place prefix sum (scan) over a 1-D `i32` field. Construct once with the array length, then call `.run(field)` to scan.

```python
psum = qd.algorithms.PrefixSumExecutor(N)
arr  = qd.field(qd.i32, shape=(N,))
# ... fill arr ...
psum.run(arr)
# arr now holds the inclusive prefix sum: arr[i] = sum(arr_original[0..=i]).
```

Constructor:

- `length: int` - the **fixed** number of elements the executor will scan on every `.run()` call. Internally allocates an auxiliary `qd.field(i32, shape=padded_length)` sized to the Kogge-Stone hierarchy (block size = 64).

`run(input_arr)`:

- `input_arr` must be a 1-D `qd.field(qd.i32, shape=(length,))` - its length must match the constructor's `length` exactly. `run()` always blits `length` elements between `input_arr` and the internal buffer; passing a shorter field results in out-of-bounds reads / writes (no runtime check today).
- Returns nothing; `input_arr` is overwritten with the scan result.

Constraints:

- **Dtype:** `qd.i32` only. Calling with any other dtype raises `RuntimeError("Only qd.i32 type is supported for prefix sum.")`.
- **Inclusive only.** No exclusive variant exposed. To convert to exclusive, post-process: `exclusive[i] = inclusive[i] - input_original[i]`.
- **Backend coverage.** CUDA and Vulkan only. AMDGPU and Metal raise `RuntimeError(f"{arch} is not supported for prefix sum.")` at compile time.

The implementation is a Kogge-Stone hierarchical scan: per-block inclusive scan on shared memory, then a small recursive scan over per-block totals, then a uniform-add pass to propagate back. This means the executor reuses the underlying buffer across calls, which is why it's a class (allocate once, run many times) rather than a free function.

No explicit fence is required between a kernel that writes the input and the subsequent `.run()` call. `.run()` launches its own kernels under the hood, and the kernel boundary serializes against prior writes from host-launched kernels.

## Under the hood

Implementation detail for the curious — not needed to *use* the ops. In every case the func emits a fixed-depth (`LOG256_MAX_N`) staircase of phases inside your kernel; each phase is a separate offloaded launch, so correctness relies on the same launch-boundary serialization as your surrounding top-level loops. `BLOCK_DIM = 256` throughout.

### `reduce_{add,min,max}`

- Fixed-depth tree reduction. Each phase uses `BLOCK_DIM = 256` threads per block and reduces 256 elements per block via `block.reduce_{add,min,max}`. `LOG256_MAX_N = 1` covers `N <= 256`; `2` covers up to `256² = 65536`; and so on. Out-of-range lanes contribute the [identity value](#common-conventions).
- Per-phase partials are written to the caller's `scratch`; the final phase writes `out[0]` directly.

### `exclusive_scan_{add,min,max}`

Blelloch 1990 three-pass exclusive scan:

1. **Pass 1** — per-block tile reduce of `arr` into the caller's `scratch` (one slot per block).
2. **Pass 2** — exclusive-scan the partials buffer in place. For `N ≤ BLOCK_DIM²` (= 65536) a single block does this. For larger `N`, the staircase recurses `D - 2` further levels: another tile-reduce on the partials, a recursive scan, then a downsweep that applies the higher-level prefixes.
3. **Pass 3** — per-block tile scan + add the block prefix from scratch. Each block re-reads its tile from `arr`, runs `block.exclusive_scan` to get per-thread tile prefixes, and adds its `block_prefix` from the scanned partials.

Total scratch at `N = 1M` is `exclusive_scan_scratch_slots(N)` = `4096 + 16 = 4112` slots (~16 KB for 4-byte dtypes, ~32 KB for 8-byte).

### `select`

The textbook scan-based compaction:

1. **Exclusive scan of `flags`** into the caller's `u32` scratch, producing per-element write indices. Same staircase phases as `exclusive_scan_add` (out-of-place: `flags` stays intact for the scatter / count, the indices land in `scratch[0:N]` and the partials above them).
2. **Scatter:** a phase reads each `(arr[i], flags[i], indices[i])` and, if the flag is set, writes `out[indices[i]] = arr[i]`. No races, by construction of the exclusive scan over 0 / 1 flags.
3. **Count tail:** a one-thread phase computes `indices[N-1] + flags[N-1]` and stores it in `num_out[0]`.

### `sort`

- Classical LSB radix sort with 8-bit digits, four passes for `u32` / `i32` / `f32` (eight for the 64-bit dtypes). Each digit pass is three internal kernels:
  1. **Histogram** — every block computes its per-digit count into shared memory, then publishes the 256-bin tile histogram to the shared u32 scratch (digit-major layout: `tile_histograms[d * num_blocks + b]`).
  2. **Scan** — in-place exclusive scan over the flat tile_histograms buffer. The digit-major layout makes a single 1-D scan enough to produce per-(digit, block) global offsets.
  3. **Scatter** — each block ranks its keys via `block.radix_rank_match_atomic_or` (wave32 + wave64 clean), looks up the per-(digit, block) global offset from the scan output, and scatters keys (and values, if provided) to the destination buffer.
- After each pass `keys` ↔ `tmp_keys` are swapped. An even pass count lands the sorted keys back in `keys`.
- Signed-integer (`i32` / `i64`) and floating-point (`f32` / `f64`) keys are mapped to a sortable unsigned representation (`u32` / `u64`) before the first pass and mapped back after the last via in-place "twiddle" kernels (signed: XOR sign bit; float: flip sign bit on positives, flip all bits on negatives — the standard sortable-key transform). `u32` / `u64` keys are sorted directly with no twiddle.

### `reduce_by_key_add`

Scan + scatter + atomic_add — no segmented-scan primitive needed:

1. **Head-flag pass.** `head_flags[i] = 1` if `i == 0` or `keys[i] != keys[i-1]`, else `0`. Written to the caller's `u32` scratch (bit-cast from `i32`).
2. **In-place exclusive scan** of `head_flags` (same staircase phases as `exclusive_scan_add`). After this, `scratch[i] = sum(head_flags[0:i])`.
3. **Zero-init `values_out[0:N]`.** The scatter uses `atomic_add`; slots must start at the additive identity `0`.
4. **Scatter.** For each `i`, recompute `head_flag(i)` from `keys[i]` / `keys[i-1]`, derive the run index `pos = scratch[i] + head_flag(i) - 1` (inclusive scan minus 1), and write `keys_out[pos] = keys[i]` + `atomic_add(values_out[pos], values[i])`.
5. **Count.** `num_runs[0] = scratch[N-1] + head_flag(N-1)`.

## Related

- `qd.simt.block.*` - the block-scope reductions and shared-memory primitives that algorithm kernels build on.
- `qd.simt.subgroup.*` - `inclusive_add` and friends, what the per-block scan stage of `PrefixSumExecutor` actually calls.
- `qd.simt.grid.mem_fence()` - the grid-scope memory fence that decoupled-look-back scans (a more efficient alternative to Kogge-Stone) require.
- [parallelization](parallelization.md) - broader synchronization story, including how `qd.algorithms` operations compose with hand-written kernels.
