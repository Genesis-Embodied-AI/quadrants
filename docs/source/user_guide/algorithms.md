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
scratch = qd.ndarray(qd.u32, shape=max(qd.algorithms.reduce_scratch_slots(N, D), 1))

# 8-byte reduce: same slot count, u64 buffer.
scratch64 = qd.ndarray(qd.u64, shape=max(qd.algorithms.reduce_scratch_slots(N, D), 1))
```

Several sizing functions return `0` for the trivial / single-tile case (e.g. `reduce_scratch_slots(N)` for `N <= 1`, `exclusive_scan_scratch_slots(N)` for `N <= 256`); wrap the count in `max(..., 1)` so the `qd.Tensor` allocation stays legal, since the algorithm won't touch the buffer in those cases anyway.

**No on-device check.** The composable `@qd.func` forms run directly as device code, so they do **no** scratch-sufficiency check (a host-side check would force an `N` device-to-host read that defeats graph capture). Size `scratch` correctly up front with the matching helper — `reduce_scratch_slots(N, log256_max_n)`, `exclusive_scan_scratch_slots(N, log256_max_n)`, `select_scratch_slots(N)`, `reduce_by_key_scratch_slots(N)`, or `sort_scratch_slots(N, log256_max_n)` — for the capacity you compile the op against. An undersized buffer corrupts the output (or reads / writes out of bounds) rather than raising.

The per-algorithm sections below restate the sizing function and footprint for each op.

## Semantics

### `qd.algorithms.reduce_{add,min,max}`

Device-wide tree reduction over a 1-D tensor, called at the **top level** of your own `@qd.kernel`: `out[0]` holds `sum(arr[0:n])` / `min(arr[0:n])` / `max(arr[0:n])`. The monoid identity is derived from the `DTYPE` template automatically — the element dtype, which the op takes explicitly because an `ndarray` kernel argument exposes no in-kernel `.dtype` (`0` for `add`; largest representable value for `min` - `+inf` for floats, `INT{32,64}_MAX` for signed ints, `UINT{32,64}_MAX` for unsigned; smallest representable value for `max` - `-inf` for floats, `INT{32,64}_MIN` for signed ints, `0` for unsigned), mirroring the `block.reduce_min` / `subgroup.reduce_min` typed wrappers which don't take an identity for the same reason.

Arguments:

- `arr`: 1-D input tensor. As an `ndarray` kernel argument it is polymorphic over `qd.field`, `qd.ndarray`, and `qd.Tensor`.
- `out`: 1-element tensor with the same dtype as `arr`. Caller-supplied so the call is fully asynchronous - there is no implicit device→host sync. To get a Python scalar, do `out.to_numpy()[0]` explicitly once the enclosing kernel has run; this makes the host hop visible at the call site rather than hidden inside the algorithm.
- `scratch`: caller-owned 1-D workspace of `reduce_scratch_slots(N, LOG256_MAX_N)` slots, `u32` for 4-byte `arr` dtypes and `u64` for 8-byte ones. See [Scratch space](#scratch-space).
- `n`, `DTYPE`, `LOG256_MAX_N`: the device-resident live count, the element dtype, and the compile-time capacity — see "Composing the func inside your own kernel" below.

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through a `u32` scratch and 8-byte dtypes through a `u64` scratch; see [Scratch space](#scratch-space) for the mechanics.
- **Shape:** `arr` must be 1-D; `out.shape` must be `(1,)`. Both must share the same dtype.
- **f32 / f64 non-associativity:** `reduce_add` on a floating-point dtype is not bitwise-reproducible across `N` changes, nor bitwise-equal to host `numpy.sum`. Tests tolerate a small relative error rather than asserting bitwise.

Implementation:

- Fixed-depth tree reduction. The func emits a fixed-depth (`LOG256_MAX_N`) staircase of phases inside the enclosing kernel; each phase uses `BLOCK_DIM = 256` threads per block and reduces 256 elements per block via `block.reduce_{add,min,max}`. `LOG256_MAX_N = 1` covers `N <= 256`; `2` covers up to `256^2 = 65536`; and so on. Out-of-range lanes contribute the monoid identity, derived in-kernel from the `DTYPE` template (no runtime identity argument).
- Per-phase partials are written to the caller's `scratch` buffer (u32 for 4-byte dtypes, u64 for 8-byte dtypes; see [Scratch space](#scratch-space)); the final phase writes `out[0]` directly. The phases are separate offloaded launches inside the enclosing kernel, so correctness relies on the same launch-boundary serialization as the surrounding phases.

Composing the func inside your own kernel (qipc-style): call `reduce_{add,min,max}(arr, out, scratch, n, DTYPE, LOG256_MAX_N)` at the **top level** of your `@qd.kernel`. `n` is the live count read **on-device** (e.g. `count[0]`); `DTYPE` is the element dtype (an `ndarray` kernel argument exposes no `.dtype` inside the kernel, so pass it explicitly); `LOG256_MAX_N` is the compile-time phase count - the emitted reduce handles any count `<= 256**LOG256_MAX_N`, so a graph captured for a given `LOG256_MAX_N` is reusable across all such counts. Size `scratch` with `reduce_scratch_slots(capacity_n, LOG256_MAX_N)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow (that demotes the phase loops and drops the per-phase grid-wide barriers).

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

Scratch footprint: `reduce_scratch_slots(N)` ≈ `ceil(N / BLOCK_DIM)` slots, where `BLOCK_DIM = 256` (`N = 1G` is ~4M slots). See [Scratch space](#scratch-space).

### `qd.algorithms.exclusive_scan_{add,min,max}`

Device-wide exclusive prefix scan over a 1-D tensor, called at the **top level** of your own `@qd.kernel`: `out[i]` holds the reduction (`sum` / `min` / `max`) of `arr[0:i]`. `out[0]` is always the monoid identity, which is derived from the `DTYPE` template automatically — the element dtype, which the op takes explicitly because an `ndarray` kernel argument exposes no in-kernel `.dtype` (`0` for `add`; largest representable value for `min` - `+inf` for floats, `INT{32,64}_MAX` for signed ints, `UINT{32,64}_MAX` for unsigned; smallest representable value for `max` - `-inf` for floats, `INT{32,64}_MIN` for signed ints, `0` for unsigned), mirroring the `block.exclusive_min` / `subgroup.exclusive_min_tiled` typed wrappers.

Arguments:

- `arr` / `out`: 1-D input / output tensors, same shape and dtype; `out` must be a distinct buffer (see constraints).
- `scratch`: caller-owned 1-D workspace of `exclusive_scan_scratch_slots(N, LOG256_MAX_N)` slots, `u32` for 4-byte `arr` dtypes and `u64` for 8-byte ones. See [Scratch space](#scratch-space).
- `n`, `DTYPE`, `LOG256_MAX_N`: the device-resident live count, the element dtype, and the compile-time capacity — see "Composing the func inside your own kernel" below.

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through a `u32` scratch and 8-byte dtypes through a `u64` scratch; see [Scratch space](#scratch-space) for the mechanics.
- **Shape:** `arr` and `out` must both be 1-D with the same shape and dtype.
- **No in-place scan:** `out` must be a distinct buffer from `arr`. Calling with `out is arr` raises `ValueError`. (The kernels do not protect against same-buffer aliasing; allocating one extra buffer once is cheap relative to the scan itself.)
- **Float non-associativity:** the order of additions inside a scan tree is not the same as a left-to-right host scan, so `f32` / `f64` results are *not* bitwise-equal to `numpy.cumsum`. Tests tolerate a small relative error (scaled by dtype precision).

Implementation:

- Blelloch 1990 three-pass exclusive scan, emitted as a fixed-depth (`LOG256_MAX_N`) staircase inside the enclosing kernel:
  1. **Pass 1** - per-block tile reduce of `arr` into the caller's `scratch` (one slot per block).
  2. **Pass 2** - exclusive-scan the partials buffer in place. For `N ≤ BLOCK_DIM²` (= 65536) a single block does this. For larger `N`, the staircase recurses `D - 2` further levels: another tile-reduce on the partials, a recursive scan, then a downsweep that applies the higher-level prefixes.
  3. **Pass 3** - per-block tile scan + add the block prefix from scratch. Each block re-reads its tile from `arr`, runs `block.exclusive_scan` to get per-thread tile prefixes, and adds its `block_prefix` from the scanned partials.
- `BLOCK_DIM = 256`. Total scratch usage at `N = 1M` is `exclusive_scan_scratch_slots(N)` = `4096 + 16 = 4112` slots (~16 KB for 4-byte dtypes, ~32 KB for 8-byte). See [Scratch space](#scratch-space).

Composing the func inside your own kernel (qipc-style): call `exclusive_scan_{add,min,max}(arr, out, scratch, n, DTYPE, LOG256_MAX_N)` at the **top level** of your `@qd.kernel`. Like `reduce_{add,min,max}`, `n` is the live count read **on-device** (e.g. `count[0]`); `DTYPE` is the element dtype (an `ndarray` kernel argument exposes no `.dtype` inside the kernel, so pass it explicitly); `LOG256_MAX_N` is the compile-time phase count - the emitted scan handles any count `<= 256**LOG256_MAX_N`, so a graph captured for a given `LOG256_MAX_N` is reusable across all such counts. The scan is out-of-place (`out` distinct from `arr`); size `scratch` with `exclusive_scan_scratch_slots(capacity_n, LOG256_MAX_N)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow (that demotes the phase loops and drops the per-phase grid-wide barriers).

```python
@qd.kernel
def my_pipeline(
    arr: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    count: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    # ... other top-level phases ...
    qd.algorithms.exclusive_scan_add(arr, out, scratch, count[0], qd.f32, LOG256_MAX_N)
    # ... more top-level phases ...
```

### `qd.algorithms.select`

Stream compaction, called at the **top level** of your own `@qd.kernel`. Copy every `arr[i]` whose corresponding `flags[i]` is `1` into a dense prefix of `out`, in stable input order, and write the count of selected elements to `num_out[0]`. Flags must be exactly `0` or `1` - see the constraints below.

Constraints:

- **Dtypes:** `arr.dtype` is any scalar dtype in `{qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64}` *or* any `qd.types.struct(...)` / `qd.Struct.field({...})` composite (e.g. libuipc `Vector2i` / `Vector3i` / `Vector4i` / `LinearBVHAABB`-style structs). The scatter is `dst[idx] = src[i]`, which lowers per-field, so the algorithm is dtype-agnostic - no scratch reinterpretation needed for wider or composite element types.
- **`flags`:** 1-D `qd.i32` tensor with the same shape as `arr`. **Every entry must be exactly `0` or `1`** (`1` selects). The algorithm prefix-sums `flags` directly as counts, so non-0/1 values produce wrong indices and a wrong `num_out` count - the caller is responsible for normalization, no implicit normalization pass is performed. `flags` is caller-built - populate it with a kernel applying whatever predicate you want, writing exactly `1` for selected and `0` otherwise.
- **`out`:** 1-D tensor, same dtype as `arr`, with `len(out) >= len(arr)` so the worst-case all-selected run is safe. Only `out[0 : num_out[0]]` carries meaningful data on return; the tail is left untouched (whatever was in `out` before the call remains).
- **`num_out`:** 1-element `qd.i32` tensor. Same explicit-host-hop rule: do `int(num_out.to_numpy()[0])` after the call to get the count as a Python scalar.
- **`scratch`:** caller-owned 1-D `qd.u32` tensor of `select_scratch_slots(N)` slots (always `u32`, regardless of `arr.dtype`). See [Scratch space](#scratch-space).

Algorithm: the textbook scan-based compaction, emitted as a fixed-depth (`LOG256_MAX_N`) staircase inside the enclosing kernel.

1. **Exclusive scan of `flags`** into the caller's `u32` scratch, producing per-element write indices. Same staircase phases as `exclusive_scan_add` (out-of-place: `flags` stays intact for the scatter / count, the indices land in `scratch[0:N]` and the partials above them).
2. **Scatter:** a phase reads each `(arr[i], flags[i], indices[i])` and, if the flag is set, writes `out[indices[i]] = arr[i]`. No races by construction of the exclusive scan over 0 / 1 flags.
3. **Count tail:** a one-thread phase computes `indices[N-1] + flags[N-1]` and stores it in `num_out[0]`.

Composing the func inside your own kernel (qipc-style): call `select(arr, flags, out, num_out, scratch, n, LOG256_MAX_N)` at the **top level** of your `@qd.kernel`. Like the other ops, `n` is the live count read **on-device** (e.g. `count[0]`) and `LOG256_MAX_N` is the compile-time phase count (the compaction handles any count `<= 256**LOG256_MAX_N`). Unlike `reduce_{add,min,max}` / `exclusive_scan_{add,min,max}` there is **no `DTYPE` argument** - the scatter `out[idx] = arr[i]` lowers per-field, so `select` works for scalar *and* struct element dtypes unchanged. Size `scratch` with `select_scratch_slots(capacity_n)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow.

```python
@qd.kernel
def my_pipeline(
    arr: qd.types.ndarray(dtype=qd.f32, ndim=1),
    flags: qd.types.ndarray(dtype=qd.i32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    num_out: qd.types.ndarray(dtype=qd.i32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    count: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    # ... a phase that fills flags via your predicate ...
    qd.algorithms.select(arr, flags, out, num_out, scratch, count[0], LOG256_MAX_N)
    # ... more top-level phases ...
```

Scratch footprint: `select_scratch_slots(N)` ≈ `N` u32 slots (one write index per input element). See [Scratch space](#scratch-space).

### `qd.algorithms.sort`

Ascending in-place LSB radix sort over a 1-D tensor of 32-bit or 64-bit scalar keys (`u32` / `i32` / `f32` / `u64` / `i64` / `f64`), with optional lock-step permutation of a `values` tensor (key-value sort). Called as a `@qd.func` at the **top level** of your own `@qd.kernel` (the qipc path) so the sort composes with your other phases into one compiled kernel / captured graph:

`sort(keys, tmp_keys, values, tmp_values, scratch, n, KEY_DTYPE, HAS_VALUES, END_BIT, LOG256_MAX_N)`

Here `n` is a 0-d `i32` ndarray (the count lives **on-device**, so one captured graph replays for any count `<= 256 ** LOG256_MAX_N`), and the compile-time flags (`KEY_DTYPE`, `HAS_VALUES`, `END_BIT`, `LOG256_MAX_N`) are passed explicitly. Pass real `values` / `tmp_values` with `HAS_VALUES=True` for a key-value sort; **for a keys-only sort pass `keys` / `tmp_keys` again in the `values` / `tmp_values` slots** as placeholders and set `HAS_VALUES=False` (every value access is `HAS_VALUES`-guarded). The func does **no** host-side validation or scratch-sufficiency check (a DtoH would defeat graph capture), so size `scratch` correctly up front.

Composing the func inside your own kernel (qipc-style; `KEY_DTYPE` is the key element dtype you already know):

```python
from quadrants.types import ndarray as ndarray_ann

@qd.kernel
def my_pipeline(
    keys: ndarray_ann(dtype=qd.u32, ndim=1),
    tmp:  ndarray_ann(dtype=qd.u32, ndim=1),
    scratch: ndarray_ann(dtype=qd.u32, ndim=1),
    n: ndarray_ann(dtype=qd.i32, ndim=0),
):
    # ... other top-level phases ...
    qd.algorithms.sort(keys, tmp, keys, tmp, scratch, n, qd.u32, False, 32, D)
    # ... more top-level phases ...
```

Arguments:

- `keys`: 1-D tensor (`qd.field`, `qd.ndarray`, or `qd.Tensor`). Sorted **in place**.
- `tmp_keys`: ping-pong workspace, same shape & dtype as `keys`, distinct buffer. Contents on return are intermediate and should be considered garbage.
- `values`, `tmp_values`: the key-value buffers (any supported scalar dtype, independent of the key dtype, same shape as `keys`, distinct from each other) when `HAS_VALUES=True`. For a keys-only sort pass `keys` / `tmp_keys` here as placeholders and set `HAS_VALUES=False`.
- `scratch`: required caller-owned 1-D `qd.u32` tensor used as the per-pass tile-histogram + scan workspace. Size it with `qd.algorithms.sort_scratch_slots(N, LOG256_MAX_N)` (the footprint is dtype-independent — tile histograms are `u32` regardless of key width). The func does no sufficiency check, so size it correctly up front. See [Scratch space](#scratch-space).
- `n`: 0-d `i32` ndarray (`shape=()`) holding the element count **on-device**.
- `KEY_DTYPE`: the key element dtype (one of the supported set). Passed explicitly because an `ndarray` kernel argument exposes no `.dtype` inside the kernel.
- `HAS_VALUES`: compile-time bool — whether `values` / `tmp_values` are real buffers (`True`) or placeholders (`False`).
- `END_BIT`: number of low key bits to sort. Use the full key width (32 for 4-byte keys, 64 for 8-byte) unless the high bits are known to be zero (e.g. `16` for keys `< 2**16`, to save passes). Must be a positive multiple of `8` that yields an even number of digit passes so the result lands back in `keys`.
- `LOG256_MAX_N`: scan depth `D`; the emitted sort handles any element count `<= 256**D`, so a graph captured for a given `D` is reusable across all such counts. Size `scratch` with the same `D`.

Constraints:

- **Dtypes:** the key dtype and value dtype are each independently one of `{qd.u32, qd.i32, qd.f32, qd.u64, qd.i64, qd.f64}`. Narrower scalar dtypes (`qd.i16`, `qd.f16`, ...) and struct dtypes raise `NotImplementedError` at compile time. 8-byte keys run 8 digit passes per sort; 4-byte keys run 4. Scratch footprint is the same for both widths (the per-tile histograms are `u32` regardless).
- **Aliasing:** `keys` and `tmp_keys` must be distinct buffers; same for real `values` / `tmp_values`. `sort` does not check this (passing the same buffer corrupts the sort).
- **Stability:** stable sort - equal keys keep their original input order in the output.
- **NaN handling (f32):** matches `numpy.sort` (NaNs land at the end). NaNs are not tested separately and should not be relied on for ordering invariants beyond `numpy.sort`.

Implementation:

- Classical LSB radix sort with 8-bit digits, four passes for `u32` / `i32` / `f32`. Each digit pass is three internal kernels:
  1. **Histogram** - every block computes its per-digit count into shared memory, then publishes the 256-bin tile histogram to the shared u32 scratch (digit-major layout: `tile_histograms[d * num_blocks + b]`).
  2. **Scan** - in-place exclusive scan over the flat tile_histograms buffer. The digit-major layout makes a single 1-D scan enough to produce per-(digit, block) global offsets.
  3. **Scatter** - each block ranks its keys via `block.radix_rank_match_atomic_or` (wave32 + wave64 clean), looks up the per-(digit, block) global offset from the scan output, and scatters keys (and values, if provided) to the destination buffer.
- After each pass we swap `keys` ↔ `tmp_keys`. Four passes is even, so the sorted keys end up back in `keys`.
- Signed-integer (`i32` / `i64`) and floating-point (`f32` / `f64`) keys are mapped to a sortable unsigned representation (`u32` / `u64`) before the first pass and mapped back after the last pass via in-place "twiddle" kernels (signed: XOR sign bit; float: flip sign bit on positives, flip all bits on negatives - the standard sortable-key transform). `u32` / `u64` keys are sorted directly with no twiddle.

Scratch footprint: `num_blocks * 256 + ...` u32 slots (where `num_blocks = ceil(N / 256)`), plus the scan staircase. Call `qd.algorithms.sort_scratch_slots(N, LOG256_MAX_N)` to get the exact slot count (pure host arithmetic, no device round-trip). See [Scratch space](#scratch-space).

### `qd.algorithms.reduce_by_key_add`

Collapse every **consecutive run of equal keys** into a single output entry `(unique_key, sum_of_values_in_run)`, called at the **top level** of your own `@qd.kernel`. Keys that compare equal but are separated by other keys form separate runs. For a global per-key sum, sort by key first (e.g. with `qd.algorithms.sort`) and then reduce-by-key.

Arguments:

- `keys_in`: 1-D tensor of `u32` / `i32` / `f32`. Pass a `qd.field`, `qd.ndarray`, or `qd.Tensor`.
- `values_in`: 1-D tensor of `u32` / `i32` / `f32`, same shape as `keys_in`.
- `keys_out`: 1-D tensor of the same dtype as `keys_in`, with `len(keys_out) >= len(keys_in)` so the worst-case-all-unique run is safe. Only `keys_out[0 : num_runs[0]]` carries meaningful data on return; the tail is untouched.
- `values_out`: 1-D tensor of the same dtype as `values_in`, same length requirement. The first `num_runs[0]` slots are overwritten; the tail past that prefix is left untouched.
- `num_runs`: 1-element `qd.i32` tensor receiving the number of runs. Same explicit-host-hop rule: do `int(num_runs.to_numpy()[0])` after the call to get the count as a Python scalar.
- `scratch`: caller-owned 1-D `qd.u32` tensor of `reduce_by_key_scratch_slots(N)` slots (always `u32`, regardless of key / value dtype). See [Scratch space](#scratch-space).
- `n`, `VALUE_DTYPE`, `LOG256_MAX_N`: the device-resident live count, the values dtype (used only to write the typed zero into `values_out` before the scatter's `atomic_add`), and the compile-time capacity — see "Composing the func inside your own kernel" below.

Constraints:

- **Dtypes (first land):** `keys_in.dtype` and `values_in.dtype` ∈ {`qd.i32`, `qd.u32`, `qd.f32`}. Other dtypes raise `NotImplementedError`.
- **Reduction:** only `add` is exposed for first land. `min` / `max` variants need `atomic_min` / `atomic_max` for `f32`, which has spottier cross-backend support; defer to a follow-up gated on real qipc usage.
- **f32 non-associativity:** the order of additions inside a run is set by hardware atomic ordering, not host order, so `f32` results are *not* bitwise-equal to a serial scan. Tests tolerate a small relative error.
- **NaN handling (f32 keys):** `NaN != NaN` is true, so each NaN-keyed element becomes its own run. Consistent with treating NaN as "different from everything", which matches the run-length-encoding spirit.

Algorithm: scan + scatter + atomic_add - no segmented-scan primitive needed. Emitted as a fixed-depth (`LOG256_MAX_N`) staircase inside the enclosing kernel.

1. **Head-flag pass.** `head_flags[i] = 1` if `i == 0` or `keys[i] != keys[i-1]`, else `0`. Written to the caller's `u32` scratch (bit-cast from `i32`).
2. **In-place exclusive scan** of `head_flags` (using the same staircase phases as `exclusive_scan_add`). After this, `scratch[i] = sum(head_flags[0:i])`.
3. **Zero-init `values_out[0:N]`.** The scatter uses `atomic_add`; slots must start at the additive identity `0`.
4. **Scatter.** For each `i`, recompute `head_flag(i)` from `keys[i]` / `keys[i-1]`, derive the run index `pos = scratch[i] + head_flag(i) - 1` (inclusive scan minus 1), and write `keys_out[pos] = keys[i]` + `atomic_add(values_out[pos], values[i])`.
5. **Count.** `num_runs[0] = scratch[N-1] + head_flag(N-1)`.

Composing the func inside your own kernel (qipc-style): call `reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch, n, VALUE_DTYPE, LOG256_MAX_N)` at the **top level** of your `@qd.kernel`. `n` is the live count read **on-device** (e.g. `count[0]`); `LOG256_MAX_N` is the compile-time phase count (any count `<= 256**LOG256_MAX_N`); `VALUE_DTYPE` is the values dtype - needed only to write the typed zero into `values_out` before the scatter's `atomic_add` (keys are handled generically). Size `scratch` with `reduce_by_key_scratch_slots(capacity_n)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow.

```python
@qd.kernel
def my_pipeline(
    keys_in: qd.types.ndarray(dtype=qd.i32, ndim=1),
    values_in: qd.types.ndarray(dtype=qd.f32, ndim=1),
    keys_out: qd.types.ndarray(dtype=qd.i32, ndim=1),
    values_out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    num_runs: qd.types.ndarray(dtype=qd.i32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    count: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    # ... (typically sort keys_in / values_in by key first) ...
    qd.algorithms.reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch, count[0], qd.f32, LOG256_MAX_N)
    # ... more top-level phases ...
```

Scratch footprint: `reduce_by_key_scratch_slots(N)` ≈ `1.004 * N` u32 slots. See [Scratch space](#scratch-space).

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

## Examples

### Sort indices by per-element key

```python
N = 1000
D = 1
while 256 ** D < N:
    D += 1

keys     = qd.ndarray(qd.f32, shape=(N,))
tmp_keys = qd.ndarray(qd.f32, shape=(N,))
indices  = qd.ndarray(qd.i32, shape=(N,))
tmp_idx  = qd.ndarray(qd.i32, shape=(N,))
scratch  = qd.ndarray(qd.u32, shape=max(qd.algorithms.sort_scratch_slots(N, D), 1))
n        = qd.ndarray(qd.i32, shape=())
n.fill(N)

@qd.kernel
def sort_indices(
    keys: qd.types.ndarray(dtype=qd.f32, ndim=1),
    tmp_keys: qd.types.ndarray(dtype=qd.f32, ndim=1),
    indices: qd.types.ndarray(dtype=qd.i32, ndim=1),
    tmp_idx: qd.types.ndarray(dtype=qd.i32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    n: qd.types.ndarray(dtype=qd.i32, ndim=0),
) -> None:
    for i in range(N):
        keys[i] = qd.random()
        indices[i] = i
    # Key-value radix sort at the top level: keys carry the order, indices ride along.
    qd.algorithms.sort(keys, tmp_keys, indices, tmp_idx, scratch, n, qd.f32, True, 32, D)

sort_indices(keys, tmp_keys, indices, tmp_idx, scratch, n)
# keys is now ascending; indices[k] is the original index of the k-th smallest key. (Stable: ties between equal keys preserve their input-order indices.)
```

### Compact-array offsets via prefix sum

```python
N = 100_000
flags  = qd.field(qd.i32, shape=(N,))   # 0 or 1 per element
offsets = qd.field(qd.i32, shape=(N,))

@qd.kernel
def populate(data: qd.types.NDArray[qd.f32, 1], threshold: qd.f32) -> None:
    for i in range(N):
        flags[i] = 1 if data[i] > threshold else 0

@qd.kernel
def copy_flags() -> None:
    for i in range(N):
        offsets[i] = flags[i]

scan = qd.algorithms.PrefixSumExecutor(N)

populate(data, 0.5)
copy_flags()
scan.run(offsets)
# offsets[i] is now the 1-based output position of element i if it was selected.
```

The compact-output kernel reads `offsets[i]` (or `offsets[i] - flags[i]` for 0-based) to decide where to write surviving elements. This is the textbook scan-based select / compact pattern; the only Quadrants-specific note is the `i32`-only restriction.

## Related

- `qd.simt.block.*` - the block-scope reductions and shared-memory primitives that algorithm kernels build on.
- `qd.simt.subgroup.*` - `inclusive_add` and friends, what the per-block scan stage of `PrefixSumExecutor` actually calls.
- `qd.simt.grid.mem_fence()` - the grid-scope memory fence that decoupled-look-back scans (a more efficient alternative to Kogge-Stone) require.
- [parallelization](parallelization.md) - broader synchronization story, including how `qd.algorithms` operations compose with hand-written kernels.
