# Algorithms

Device-wide algorithms are primitives that consume and produce whole arrays, executed as one or more kernel launches under the hood. They sit one tier above block and subgroup primitives: they *use* `block.reduce`, `block.exclusive_scan`, `block.radix_rank_match_atomic_or`, and `subgroup` reductions internally, and rely on the kernel-launch boundary (plus `atomic_add` in a few places) for cross-block synchronization rather than any in-kernel grid-scope barrier. Most are called from host (Python) code, not from inside a kernel. Some also ship a composable `@qd.func` form (the `_func` suffix) you can call at the **top level** of your own kernel so the op fuses with your other phases into one compiled kernel / captured graph - currently the device-wide reduce (`reduce_{add,min,max}_func`), the exclusive scan (`exclusive_scan_{add,min,max}_func`), stream compaction (`select_func`), reduce-by-key (`reduce_by_key_add_func`), and the LSB radix sort (`radix_sort_func`). These `_func` forms take the live element count as a **device-resident** `Expr` and a compile-time recursion `DEPTH` (the radix sort calls it `LOG256_MAX_N`), so one captured graph replays for every count up to the depth's capacity. Every algorithm also has a friendly host (Python) entry that validates inputs, derives the depth, and launches the work standalone; the host entry is the right choice unless you specifically need to fuse the op into your own captured graph.

## What's available

| Op                                                          | What it does                                                       | CUDA | AMDGPU | Vulkan | Metal |
|-------------------------------------------------------------|--------------------------------------------------------------------|------|--------|--------|-------|
| `qd.algorithms.reduce_{add,min,max}(arr, out, scratch)`         | `out[0] = sum/min/max(arr)` (fixed-depth tree reduction, one launch; identity derived from `arr.dtype` for min / max). Friendly host entry. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.reduce_{add,min,max}_func(arr, out, scratch, n, DTYPE, DEPTH)` | Same reduce as a `@qd.func`, to compose at the top level of your own kernel (device-resident count `n`, compile-time `DEPTH`). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.exclusive_scan_{add,min,max}(arr, out, scratch)` | `out[i] = sum/min/max(arr[0:i])` (three-pass Blelloch-style scan, one launch; 32-bit + 64-bit scalars; identity derived from `arr.dtype` for min / max). Friendly host entry. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.exclusive_scan_{add,min,max}_func(arr, out, scratch, n, DTYPE, DEPTH)` | Same scan as a `@qd.func`, to compose at the top level of your own kernel (device-resident count `n`, compile-time `DEPTH`). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.select(arr, flags, out, num_out, scratch)`     | Stream compaction: copy `arr[i]` to a dense prefix of `out` for every `flags[i] == 1` (`flags` must be exactly 0/1). Friendly host entry. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.select_func(arr, flags, out, num_out, scratch, n, DEPTH)` | Same compaction as a `@qd.func`, to compose at the top level of your own kernel (device-resident count `n`, compile-time `DEPTH`; no `DTYPE` - the scatter is dtype-agnostic). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.radix_sort(keys, tmp_keys, scratch, *, values=None, tmp_values=None, end_bit=None, log256_max_n=None)` | LSB radix sort (32-bit / 64-bit scalar keys, optional key-value). Friendly host entry (derives passes / depth / device-`N`, validates scratch). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.radix_sort_func(keys, tmp_keys, values, tmp_values, scratch, N, KEY_DTYPE, HAS_VALUES, END_BIT, LOG256_MAX_N)` | Same sort as a `@qd.func`, to compose at the top level of your own kernel. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch)` | Collapse each consecutive run of equal keys into `(key, sum_of_values)`. Friendly host entry. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.reduce_by_key_add_func(keys_in, values_in, keys_out, values_out, num_runs, scratch, n, VALUE_DTYPE, DEPTH)` | Same reduce-by-key as a `@qd.func`, to compose at the top level of your own kernel (device-resident count `n`, compile-time `DEPTH`; `VALUE_DTYPE` only for the `values_out` zero-init). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.parallel_sort`                               | Odd-even merge sort (in-place, key or key-value). **Deprecated**: prefer `radix_sort`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.PrefixSumExecutor`                           | Inclusive in-place prefix sum (i32 only). **Deprecated**: prefer `exclusive_scan_add`. | yes  | no     | yes    | no    |

\* `reduce_*`, `exclusive_scan_*`, `select`, `radix_sort` / `radix_sort_func`, `reduce_by_key_add`, and `parallel_sort` run anywhere a Quadrants kernel runs; portability is inherited from the underlying block / subgroup primitives.

## Scratch space

Every device-wide algorithm in this module decomposes into "per-block partial → cross-block combine → finalize" passes (tree reduction, three-pass Blelloch scan, four-pass radix sort, scan-then-scatter compaction). The per-block partials need somewhere to live between kernel launches - that buffer is called **scratch**, and **the caller owns it**. Every algorithm takes a mandatory `scratch` argument.

**Ask first, then allocate.** Each algorithm ships a companion `*_scratch_slots(N)` function - branch-free integer arithmetic, no device round-trip - that returns the minimum number of slots needed for a length-`N` input. Allocate at least that many; allocating more is fine. These functions are **host- and kernel-callable**: pass a Python `int` to size an allocation up front, or call the same function inside a `@qd.kernel` on a device-read `N` to recompute the requirement on-device and validate it against `scratch.shape[0]` (e.g. raise an overflow flag for a yield-and-realloc loop) without ever reading `N` back to the host. The fixed-depth ops follow the same convention: `reduce_scratch_slots(N, DEPTH)`, `exclusive_scan_scratch_slots(N, DEPTH)`, and `radix_sort_scratch_slots(N, log256_max_n)` are host- and kernel-callable when you pass the compile-time depth the op is compiled for; called as `reduce_scratch_slots(N)` / `exclusive_scan_scratch_slots(N)` / `radix_sort_scratch_slots(N)` (depth omitted) they auto-pick the minimal depth from `N` and are host-only.

| Algorithm | Sizing function | Scratch dtype |
|-----------|-----------------|---------------|
| `reduce_{add,min,max}` / `reduce_*_func` | `reduce_scratch_slots(N[, DEPTH])` | `u32` (4-byte `arr`) / `u64` (8-byte `arr`) |
| `exclusive_scan_{add,min,max}` / `exclusive_scan_*_func` | `exclusive_scan_scratch_slots(N[, DEPTH])` | `u32` (4-byte `arr`) / `u64` (8-byte `arr`) |
| `select` | `select_scratch_slots(N)` | `u32` (always) |
| `reduce_by_key_add` | `reduce_by_key_scratch_slots(N)` | `u32` (always) |
| `radix_sort` / `radix_sort_func` | `radix_sort_scratch_slots(N[, log256_max_n])` | `u32` (always, regardless of key width) |

The slot count is **dtype-width-independent** (it is a count, not a byte count). For the 4-byte / 8-byte algorithms (`reduce`, `scan`) you allocate the *same number of slots* but in a `u32` buffer for 4-byte element dtypes and a `u64` buffer for 8-byte ones - the partials are `bit_cast` to / from the element dtype. `select`, `reduce_by_key_add`, and the radix sort always use `u32` scratch (they stage counts / indices / tile histograms, which are `u32` regardless of the element / key dtype).

```python
import quadrants as qd

N = 1_000_000

# 4-byte reduce: u32 scratch.
scratch = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.reduce_scratch_slots(N), 1)))
qd.algorithms.reduce_add(inp_i32, out=out_i32, scratch=scratch)

# 8-byte reduce: same slot count, u64 buffer.
scratch64 = qd.Tensor(qd.ndarray(qd.u64, shape=max(qd.algorithms.reduce_scratch_slots(N), 1)))
qd.algorithms.reduce_add(inp_f64, out=out_f64, scratch=scratch64)
```

Several sizing functions return `0` for the trivial / single-tile case (e.g. `reduce_scratch_slots(N)` for `N <= 1`, `exclusive_scan_scratch_slots(N)` for `N <= 256`); wrap the count in `max(..., 1)` so the `qd.Tensor` allocation stays legal, since the algorithm won't touch the buffer in those cases anyway.

**Too-small buffer.** The host-launched algorithms (`reduce_*`, `exclusive_scan_*`, `select`, `reduce_by_key_add`) validate the supplied `scratch` against `*_scratch_slots(N)` and raise `qd.algorithms.InsufficientScratchError` (a `RuntimeError` subclass) **before** launching anything, so a failed call never corrupts the caller's inputs. The exception carries the sizes programmatically:

```python
try:
    qd.algorithms.reduce_add(arr, out=out, scratch=scratch)
except qd.algorithms.InsufficientScratchError as err:
    scratch = qd.Tensor(qd.ndarray(qd.u32, shape=err.required_slots))   # err.required_bytes / err.provided_slots also available
    qd.algorithms.reduce_add(arr, out=out, scratch=scratch)
```

This is the "try and fail with the size" path; `*_scratch_slots` is the "ask first" path. Either is fine; pick whichever fits your control flow. The composable `@qd.func` forms are the exception: `reduce_*_func`, `exclusive_scan_*_func`, `select_func`, `reduce_by_key_add_func`, and `radix_sort_func` run directly as device code (a host-side scratch check would force an `N` device-to-host read that defeats graph capture), so they do **no** such check - size `scratch` correctly up front with `reduce_scratch_slots(N, DEPTH)` / `exclusive_scan_scratch_slots(N, DEPTH)` / `select_scratch_slots(N)` / `reduce_by_key_scratch_slots(N)` / `radix_sort_scratch_slots(N, log256_max_n)`. The host entries (`reduce_*`, `exclusive_scan_*`, `select`, `reduce_by_key_add`) keep the up-front check.

The per-algorithm sections below restate the sizing function and footprint for each op.

## Semantics

### `qd.algorithms.reduce_{add,min,max}(arr, out, scratch)`

Device-wide tree reduction over a 1-D tensor: `out[0]` holds `sum(arr)` / `min(arr)` / `max(arr)`. The monoid identity is derived from `arr.dtype` automatically (`0` for `add`; largest representable value for `min` - `+inf` for floats, `INT{32,64}_MAX` for signed ints, `UINT{32,64}_MAX` for unsigned; smallest representable value for `max` - `-inf` for floats, `INT{32,64}_MIN` for signed ints, `0` for unsigned), mirroring the `block.reduce_min` / `subgroup.reduce_min` typed wrappers which don't take an identity for the same reason.

```python
import quadrants as qd

inp = qd.field(qd.f32, shape=N)
out = qd.field(qd.f32, shape=1)
scratch = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.reduce_scratch_slots(N), 1)))
# ... fill inp ...

qd.algorithms.reduce_add(inp, out=out, scratch=scratch)
total = float(out.to_numpy()[0])   # explicit device->host hop
```

Arguments:

- `arr`: 1-D input tensor. Pass a `qd.field`, `qd.ndarray`, or `qd.Tensor` wrapper around either - the kernels are polymorphic via the `qd.Tensor` annotation.
- `out`: 1-element tensor with the same dtype as `arr`. Caller-supplied so the call is fully asynchronous - there is no implicit device→host sync. To get a Python scalar, do `out.to_numpy()[0]` explicitly after the call. This makes the host hop visible at the call site rather than hidden inside the algorithm.
- `scratch`: caller-owned 1-D workspace of `reduce_scratch_slots(N)` slots, `u32` for 4-byte `arr` dtypes and `u64` for 8-byte ones. See [Scratch space](#scratch-space).

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through a `u32` scratch and 8-byte dtypes through a `u64` scratch; see [Scratch space](#scratch-space) for the mechanics.
- **Shape:** `arr` must be 1-D; `out.shape` must be `(1,)`. Both must share the same dtype.
- **f32 / f64 non-associativity:** `reduce_add` on a floating-point dtype is not bitwise-reproducible across `N` changes, nor bitwise-equal to host `numpy.sum`. Tests tolerate a small relative error rather than asserting bitwise.

Implementation:

- Fixed-depth tree reduction emitted as a single kernel launch. The host entry derives the minimal depth `D` (smallest `D` with `256**D >= N`) from `N` and launches one `@qd.kernel` that emits `D` phases internally; each phase uses `BLOCK_DIM = 256` threads per block and reduces 256 elements per block via `block.reduce_{add,min,max}`. For `N <= 256` one phase suffices; up to `256^2 = 65536`, two; and so on. Out-of-range lanes contribute the monoid identity, derived in-kernel from the element dtype (no runtime identity argument).
- Per-phase partials are written to the caller's `scratch` buffer (u32 for 4-byte dtypes, u64 for 8-byte dtypes; see [Scratch space](#scratch-space)); the final phase writes `out[0]` directly. The phases are separate offloaded launches inside the one kernel, so correctness relies on the same launch-boundary serialization as before.

Composing the func inside your own kernel (qipc-style): call `reduce_{add,min,max}_func(arr, out, scratch, n, DTYPE, DEPTH)` at the **top level** of your `@qd.kernel`. `n` is the live count read **on-device** (e.g. `count[0]`); `DTYPE` is the element dtype (an `ndarray` kernel argument exposes no `.dtype` inside the kernel, so pass it explicitly); `DEPTH` is the compile-time phase count - the emitted reduce handles any count `<= 256**DEPTH`, so a graph captured for a given `DEPTH` is reusable across all such counts. Size `scratch` with `reduce_scratch_slots(capacity_n, DEPTH)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow (that demotes the phase loops and drops the per-phase grid-wide barriers).

```python
@qd.kernel
def my_pipeline(
    arr: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    count: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    # ... other top-level phases ...
    qd.algorithms.reduce_add_func(arr, out, scratch, count[0], qd.f32, DEPTH)
    # ... more top-level phases ...
```

Scratch footprint: `reduce_scratch_slots(N)` ≈ `ceil(N / BLOCK_DIM)` slots, where `BLOCK_DIM = 256` (`N = 1G` is ~4M slots). See [Scratch space](#scratch-space).

### `qd.algorithms.exclusive_scan_{add,min,max}(arr, out, scratch)`

Device-wide exclusive prefix scan over a 1-D tensor: `out[i]` holds the reduction (`sum` / `min` / `max`) of `arr[0:i]`. `out[0]` is always the monoid identity, which is derived from `arr.dtype` automatically (`0` for `add`; largest representable value for `min` - `+inf` for floats, `INT{32,64}_MAX` for signed ints, `UINT{32,64}_MAX` for unsigned; smallest representable value for `max` - `-inf` for floats, `INT{32,64}_MIN` for signed ints, `0` for unsigned), mirroring the `block.exclusive_min` / `subgroup.exclusive_min_tiled` typed wrappers.

```python
import quadrants as qd

N = 1_000_000
inp = qd.field(qd.f32, shape=N)
out = qd.field(qd.f32, shape=N)
scratch = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.exclusive_scan_scratch_slots(N), 1)))
# ... fill inp ...

qd.algorithms.exclusive_scan_add(inp, out=out, scratch=scratch)
# out[0] == 0.0; out[i] == sum(inp[0:i]) for i > 0.
```

Arguments:

- `arr` / `out`: 1-D input / output tensors, same shape and dtype; `out` must be a distinct buffer (see constraints).
- `scratch`: caller-owned 1-D workspace of `exclusive_scan_scratch_slots(N)` slots, `u32` for 4-byte `arr` dtypes and `u64` for 8-byte ones. See [Scratch space](#scratch-space).

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through a `u32` scratch and 8-byte dtypes through a `u64` scratch; see [Scratch space](#scratch-space) for the mechanics.
- **Shape:** `arr` and `out` must both be 1-D with the same shape and dtype.
- **No in-place scan:** `out` must be a distinct buffer from `arr`. Calling with `out is arr` raises `ValueError`. (The kernels do not protect against same-buffer aliasing; allocating one extra buffer once is cheap relative to the scan itself.)
- **Float non-associativity:** the order of additions inside a scan tree is not the same as a left-to-right host scan, so `f32` / `f64` results are *not* bitwise-equal to `numpy.cumsum`. Tests tolerate a small relative error (scaled by dtype precision).

Implementation:

- Blelloch 1990 three-pass exclusive scan, emitted as a fixed-depth staircase inside a single kernel launch (the host entry derives the minimal depth `D` from `N`, the same way `reduce` does, then launches one `@qd.kernel`):
  1. **Pass 1** - per-block tile reduce of `arr` into the caller's `scratch` (one slot per block).
  2. **Pass 2** - exclusive-scan the partials buffer in place. For `N ≤ BLOCK_DIM²` (= 65536) a single block does this. For larger `N`, the staircase recurses `D - 2` further levels: another tile-reduce on the partials, a recursive scan, then a downsweep that applies the higher-level prefixes.
  3. **Pass 3** - per-block tile scan + add the block prefix from scratch. Each block re-reads its tile from `arr`, runs `block.exclusive_scan` to get per-thread tile prefixes, and adds its `block_prefix` from the scanned partials.
- `BLOCK_DIM = 256`. Total scratch usage at `N = 1M` is `exclusive_scan_scratch_slots(N)` = `4096 + 16 = 4112` slots (~16 KB for 4-byte dtypes, ~32 KB for 8-byte). See [Scratch space](#scratch-space).

Composing the func inside your own kernel (qipc-style): call `exclusive_scan_{add,min,max}_func(arr, out, scratch, n, DTYPE, DEPTH)` at the **top level** of your `@qd.kernel`. Like `reduce_*_func`, `n` is the live count read **on-device** (e.g. `count[0]`); `DTYPE` is the element dtype (an `ndarray` kernel argument exposes no `.dtype` inside the kernel, so pass it explicitly); `DEPTH` is the compile-time phase count - the emitted scan handles any count `<= 256**DEPTH`, so a graph captured for a given `DEPTH` is reusable across all such counts. The scan is out-of-place (`out` distinct from `arr`); size `scratch` with `exclusive_scan_scratch_slots(capacity_n, DEPTH)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow (that demotes the phase loops and drops the per-phase grid-wide barriers).

```python
@qd.kernel
def my_pipeline(
    arr: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    count: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    # ... other top-level phases ...
    qd.algorithms.exclusive_scan_add_func(arr, out, scratch, count[0], qd.f32, DEPTH)
    # ... more top-level phases ...
```

### `qd.algorithms.select(arr, flags, out, num_out, scratch)`

Stream compaction. Copy every `arr[i]` whose corresponding `flags[i]` is `1` into a dense prefix of `out`, in stable input order, and write the count of selected elements to `num_out[0]`. Flags must be exactly `0` or `1` - see the constraints below.

```python
import quadrants as qd

N = 100_000
inp     = qd.field(qd.f32, shape=N)
flags   = qd.field(qd.i32, shape=N)         # caller fills with 0 / 1
out     = qd.field(qd.f32, shape=N)         # large enough for worst case
num_out = qd.field(qd.i32, shape=1)
scratch = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.select_scratch_slots(N), 1)))

# ... fill inp + flags via a separate kernel ...

qd.algorithms.select(inp, flags, out=out, num_out=num_out, scratch=scratch)

# Only out[0 : count] is meaningful; copy out the count host-side explicitly:
count = int(num_out.to_numpy()[0])
selected = out.to_numpy()[:count]
```

Constraints:

- **Dtypes:** `arr.dtype` is any scalar dtype in `{qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64}` *or* any `qd.types.struct(...)` / `qd.Struct.field({...})` composite (e.g. libuipc `Vector2i` / `Vector3i` / `Vector4i` / `LinearBVHAABB`-style structs). The scatter is `dst[idx] = src[i]`, which lowers per-field, so the algorithm is dtype-agnostic - no scratch reinterpretation needed for wider or composite element types.
- **`flags`:** 1-D `qd.i32` tensor with the same shape as `arr`. **Every entry must be exactly `0` or `1`** (`1` selects). The algorithm prefix-sums `flags` directly as counts, so non-0/1 values produce wrong indices and a wrong `num_out` count - the caller is responsible for normalization, no implicit normalization pass is performed. `flags` is caller-built - populate it with a kernel applying whatever predicate you want, writing exactly `1` for selected and `0` otherwise.
- **`out`:** 1-D tensor, same dtype as `arr`, with `len(out) >= len(arr)` so the worst-case all-selected run is safe. Only `out[0 : num_out[0]]` carries meaningful data on return; the tail is left untouched (whatever was in `out` before the call remains).
- **`num_out`:** 1-element `qd.i32` tensor. Same explicit-host-hop rule: do `int(num_out.to_numpy()[0])` after the call to get the count as a Python scalar.
- **`scratch`:** caller-owned 1-D `qd.u32` tensor of `select_scratch_slots(N)` slots (always `u32`, regardless of `arr.dtype`). See [Scratch space](#scratch-space).

Algorithm: the textbook scan-based compaction, emitted as a fixed-depth staircase inside a single kernel launch (the host entry derives the minimal depth from `N`, like `reduce` / `exclusive_scan`, then launches one `@qd.kernel`).

1. **Exclusive scan of `flags`** into the caller's `u32` scratch, producing per-element write indices. Same staircase phases as `exclusive_scan_add` (out-of-place: `flags` stays intact for the scatter / count, the indices land in `scratch[0:N]` and the partials above them).
2. **Scatter:** a phase reads each `(arr[i], flags[i], indices[i])` and, if the flag is set, writes `out[indices[i]] = arr[i]`. No races by construction of the exclusive scan over 0 / 1 flags.
3. **Count tail:** a one-thread phase computes `indices[N-1] + flags[N-1]` and stores it in `num_out[0]`.

Composing the func inside your own kernel (qipc-style): call `select_func(arr, flags, out, num_out, scratch, n, DEPTH)` at the **top level** of your `@qd.kernel`. Like the other `_func` forms, `n` is the live count read **on-device** (e.g. `count[0]`) and `DEPTH` is the compile-time phase count (the compaction handles any count `<= 256**DEPTH`). Unlike `reduce_*_func` / `exclusive_scan_*_func` there is **no `DTYPE` argument** - the scatter `out[idx] = arr[i]` lowers per-field, so `select_func` works for scalar *and* struct element dtypes unchanged. Size `scratch` with `select_scratch_slots(capacity_n)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow.

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
    qd.algorithms.select_func(arr, flags, out, num_out, scratch, count[0], DEPTH)
    # ... more top-level phases ...
```

Scratch footprint: `select_scratch_slots(N)` ≈ `N` u32 slots (one write index per input element). See [Scratch space](#scratch-space).

### `qd.algorithms.radix_sort(...)` / `radix_sort_func(...)`

Ascending in-place LSB radix sort over a 1-D tensor of 32-bit or 64-bit scalar keys (`u32` / `i32` / `f32` / `u64` / `i64` / `f64`), with optional lock-step permutation of a `values` tensor (key-value sort). Same design B split as the other algorithms - a friendly host entry plus a composable func:

- `radix_sort(keys, tmp_keys, scratch, *, values=None, tmp_values=None, end_bit=None, log256_max_n=None)` - the friendly **host (Python) entry**. Validates inputs, derives the pass count / scan depth / twiddle and the device-resident count, validates `scratch`, and launches the sort as one capturable kernel chain. Pass **both** `values` and `tmp_values` for a key-value sort, or neither for keys-only; `end_bit` defaults to the key width and `log256_max_n` to the minimal depth for `N`.
- `radix_sort_func(keys, tmp_keys, values, tmp_values, scratch, N, KEY_DTYPE, HAS_VALUES, END_BIT, LOG256_MAX_N)` - a `@qd.func`, called at the **top level** of your own `@qd.kernel` (the qipc path) so the sort composes with your other phases into one compiled kernel / captured graph. Here `N` is a 0-d `i32` ndarray (the count lives **on-device**, so one captured graph replays for any count `<= 256 ** LOG256_MAX_N`), the compile-time flags (`KEY_DTYPE`, `HAS_VALUES`, `END_BIT`, `LOG256_MAX_N`) are passed explicitly, and **for a keys-only sort you pass `keys` / `tmp_keys` again in the `values` / `tmp_values` slots** as placeholders (every value access is `HAS_VALUES`-guarded). The func does **no** host-side validation or scratch-sufficiency check (a DtoH would defeat graph capture).

```python
import quadrants as qd

N = 100_000
keys     = qd.field(qd.f32, shape=N)
tmp_keys = qd.field(qd.f32, shape=N)   # workspace; contents on return are garbage
# ... fill keys ...

scratch = qd.field(qd.u32, shape=max(qd.algorithms.radix_sort_scratch_slots(N), 1))

# Keys-only: the host entry derives end_bit (32 for f32), depth, and the device-resident N for you.
qd.algorithms.radix_sort(keys, tmp_keys, scratch)
# keys is now ascending; tmp_keys holds intermediate state.

# Key-value sort: pass both values and tmp_values.
values     = qd.field(qd.i32, shape=N)
tmp_values = qd.field(qd.i32, shape=N)
# ... fill values (e.g. with original indices) ...
qd.algorithms.radix_sort(keys, tmp_keys, scratch, values=values, tmp_values=tmp_values)
# keys ascending; values permuted so values[k] corresponds to keys[k].
```

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
    qd.algorithms.radix_sort_func(keys, tmp, keys, tmp, scratch, n, qd.u32, False, 32, D)
    # ... more top-level phases ...
```

Arguments:

- `keys`: 1-D tensor. Sorted **in place**. For `radix_sort`, pass `qd.field`, `qd.ndarray`, or `qd.Tensor`.
- `tmp_keys`: ping-pong workspace, same shape & dtype as `keys`, distinct buffer. Contents on return are intermediate and should be considered garbage.
- `scratch`: required caller-owned 1-D `qd.u32` tensor used as the per-pass tile-histogram + scan workspace. Size it with `qd.algorithms.radix_sort_scratch_slots(N[, log256_max_n])` (the footprint is dtype-independent — tile histograms are `u32` regardless of key width). The host `radix_sort` validates it for you and raises `InsufficientScratchError` if it is too small. See [Scratch space](#scratch-space).
- `values`, `tmp_values` (`radix_sort` keyword args; positional for the func): the key-value buffers (any supported scalar dtype, independent of the key dtype, same shape as `keys`, distinct from each other). For the host `radix_sort`, pass **both** (key-value) or **neither** (keys-only); for `radix_sort_func`, pass `keys` / `tmp_keys` here as placeholders for a keys-only sort and set `HAS_VALUES=False`.
- `end_bit` (`radix_sort` keyword; `END_BIT` template for the func): number of low key bits to sort. Defaults (host) to the full key width (32 for 4-byte keys, 64 for 8-byte). Must be a positive multiple of `8` that yields an even number of digit passes so the result lands back in `keys`. Pass a smaller value when the high bits are known to be zero (e.g. `16` for keys `< 2**16`) to save passes.
- `log256_max_n` (`radix_sort` keyword; `LOG256_MAX_N` template for the func): scan depth `D`; the emitted sort handles any element count `<= 256**D`, so a graph captured for a given `D` is reusable across all such counts. The host `radix_sort` defaults it to the minimal depth for `N`; size `scratch` with the same `D`.

Func-only arguments (passed explicitly to `radix_sort_func`; the host `radix_sort` derives them):

- `N`: 0-d `i32` ndarray (`shape=()`) holding the element count **on-device**. The host entry builds and fills this from `keys.shape[0]`.
- `KEY_DTYPE`: the key element dtype (one of the supported set). The host entry reads this off `keys`; the func takes it explicitly because an `ndarray` kernel argument exposes no `.dtype` inside the kernel.
- `HAS_VALUES`: compile-time bool — whether `values` / `tmp_values` are real buffers (`True`) or placeholders (`False`). The host entry sets it from whether you passed `values`.

Constraints:

- **Dtypes:** the key dtype and value dtype are each independently one of `{qd.u32, qd.i32, qd.f32, qd.u64, qd.i64, qd.f64}`. Narrower scalar dtypes (`qd.i16`, `qd.f16`, ...) and struct dtypes raise `NotImplementedError` at compile time. 8-byte keys run 8 digit passes per sort; 4-byte keys run 4. Scratch footprint is the same for both widths (the per-tile histograms are `u32` regardless).
- **Aliasing:** `keys` and `tmp_keys` must be distinct buffers; same for real `values` / `tmp_values`. The host `radix_sort` rejects aliased buffers with a `ValueError`; `radix_sort_func` does not check (passing the same buffer corrupts the sort).
- **Stability:** stable sort - equal keys keep their original input order in the output.
- **NaN handling (f32):** matches `numpy.sort` (NaNs land at the end). NaNs are not tested separately and should not be relied on for ordering invariants beyond `numpy.sort`.

Implementation:

- Classical LSB radix sort with 8-bit digits, four passes for `u32` / `i32` / `f32`. Each digit pass is three internal kernels:
  1. **Histogram** - every block computes its per-digit count into shared memory, then publishes the 256-bin tile histogram to the shared u32 scratch (digit-major layout: `tile_histograms[d * num_blocks + b]`).
  2. **Scan** - in-place exclusive scan over the flat tile_histograms buffer. The digit-major layout makes a single 1-D scan enough to produce per-(digit, block) global offsets.
  3. **Scatter** - each block ranks its keys via `block.radix_rank_match_atomic_or` (wave32 + wave64 clean), looks up the per-(digit, block) global offset from the scan output, and scatters keys (and values, if provided) to the destination buffer.
- After each pass we swap `keys` ↔ `tmp_keys`. Four passes is even, so the sorted keys end up back in `keys`.
- Signed-integer (`i32` / `i64`) and floating-point (`f32` / `f64`) keys are mapped to a sortable unsigned representation (`u32` / `u64`) before the first pass and mapped back after the last pass via in-place "twiddle" kernels (signed: XOR sign bit; float: flip sign bit on positives, flip all bits on negatives - the standard sortable-key transform). `u32` / `u64` keys are sorted directly with no twiddle.

Scratch footprint: `num_blocks * 256 + ...` u32 slots (where `num_blocks = ceil(N / 256)`), plus the scan staircase. Call `qd.algorithms.radix_sort_scratch_slots(N, LOG256_MAX_N)` to get the exact slot count (pure host arithmetic, no device round-trip). See [Scratch space](#scratch-space).

### `qd.algorithms.reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs, scratch)`

Collapse every **consecutive run of equal keys** into a single output entry `(unique_key, sum_of_values_in_run)`. Keys that compare equal but are separated by other keys form separate runs. For a global per-key sum, sort by key first (e.g. with `qd.algorithms.radix_sort`) and then reduce-by-key.

```python
import quadrants as qd

N = 100_000
keys_in    = qd.field(qd.i32, shape=N)       # sorted by key beforehand
values_in  = qd.field(qd.f32, shape=N)
keys_out   = qd.field(qd.i32, shape=N)       # capacity = N (worst case: all unique)
values_out = qd.field(qd.f32, shape=N)
num_runs   = qd.field(qd.i32, shape=1)
scratch    = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.reduce_by_key_scratch_slots(N), 1)))

# ... fill keys_in + values_in ...

qd.algorithms.reduce_by_key_add(
    keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs,
    scratch=scratch,
)

count    = int(num_runs.to_numpy()[0])
uniq_k   = keys_out.to_numpy()[:count]
sums     = values_out.to_numpy()[:count]
```

Arguments:

- `keys_in`: 1-D tensor of `u32` / `i32` / `f32`. Pass a `qd.field`, `qd.ndarray`, or `qd.Tensor`.
- `values_in`: 1-D tensor of `u32` / `i32` / `f32`, same shape as `keys_in`.
- `keys_out`: 1-D tensor of the same dtype as `keys_in`, with `len(keys_out) >= len(keys_in)` so the worst-case-all-unique run is safe. Only `keys_out[0 : num_runs[0]]` carries meaningful data on return; the tail is untouched.
- `values_out`: 1-D tensor of the same dtype as `values_in`, same length requirement. The first `num_runs[0]` slots are overwritten; the tail past that prefix is left untouched.
- `num_runs`: 1-element `qd.i32` tensor receiving the number of runs. Same explicit-host-hop rule: do `int(num_runs.to_numpy()[0])` after the call to get the count as a Python scalar.
- `scratch`: caller-owned 1-D `qd.u32` tensor of `reduce_by_key_scratch_slots(N)` slots (always `u32`, regardless of key / value dtype). See [Scratch space](#scratch-space).

Constraints:

- **Dtypes (first land):** `keys_in.dtype` and `values_in.dtype` ∈ {`qd.i32`, `qd.u32`, `qd.f32`}. Other dtypes raise `NotImplementedError`.
- **Reduction:** only `add` is exposed for first land. `min` / `max` variants need `atomic_min` / `atomic_max` for `f32`, which has spottier cross-backend support; defer to a follow-up gated on real qipc usage.
- **f32 non-associativity:** the order of additions inside a run is set by hardware atomic ordering, not host order, so `f32` results are *not* bitwise-equal to a serial scan. Tests tolerate a small relative error.
- **NaN handling (f32 keys):** `NaN != NaN` is true, so each NaN-keyed element becomes its own run. Consistent with treating NaN as "different from everything", which matches the run-length-encoding spirit.

Algorithm: scan + scatter + atomic_add - no segmented-scan primitive needed. Emitted as a fixed-depth staircase inside a single kernel launch (the host entry derives the minimal depth from `N`, like the other algorithms, then launches one `@qd.kernel`).

1. **Head-flag pass.** `head_flags[i] = 1` if `i == 0` or `keys[i] != keys[i-1]`, else `0`. Written to the caller's `u32` scratch (bit-cast from `i32`).
2. **In-place exclusive scan** of `head_flags` (using the same staircase phases as `exclusive_scan_add`). After this, `scratch[i] = sum(head_flags[0:i])`.
3. **Zero-init `values_out[0:N]`.** The scatter uses `atomic_add`; slots must start at the additive identity `0`.
4. **Scatter.** For each `i`, recompute `head_flag(i)` from `keys[i]` / `keys[i-1]`, derive the run index `pos = scratch[i] + head_flag(i) - 1` (inclusive scan minus 1), and write `keys_out[pos] = keys[i]` + `atomic_add(values_out[pos], values[i])`.
5. **Count.** `num_runs[0] = scratch[N-1] + head_flag(N-1)`.

Composing the func inside your own kernel (qipc-style): call `reduce_by_key_add_func(keys_in, values_in, keys_out, values_out, num_runs, scratch, n, VALUE_DTYPE, DEPTH)` at the **top level** of your `@qd.kernel`. `n` is the live count read **on-device** (e.g. `count[0]`); `DEPTH` is the compile-time phase count (any count `<= 256**DEPTH`); `VALUE_DTYPE` is the values dtype - needed only to write the typed zero into `values_out` before the scatter's `atomic_add` (keys are handled generically). Size `scratch` with `reduce_by_key_scratch_slots(capacity_n)`. Never nest the call in ordinary runtime `for` / `if` / `while` control flow.

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
    qd.algorithms.reduce_by_key_add_func(keys_in, values_in, keys_out, values_out, num_runs, scratch, count[0], qd.f32, DEPTH)
    # ... more top-level phases ...
```

Scratch footprint: `reduce_by_key_scratch_slots(N)` ≈ `1.004 * N` u32 slots. See [Scratch space](#scratch-space).

### `qd.algorithms.parallel_sort(keys, values=None)`

> **Deprecated.** New code should call the LSB radix sort `qd.algorithms.radix_sort` (the host entry) or `qd.algorithms.radix_sort_func` (a `@qd.func`) instead. The radix sort is asymptotically `O(N log_radix N)` rather than `O(N log^2 N)`, is **stable** (odd-even merge sort is not), supports 32-bit and 64-bit scalar keys across CUDA / AMDGPU / Vulkan / Metal, and accepts `qd.field`, `qd.ndarray`, and `qd.Tensor` (`parallel_sort` is field-only). The only thing `parallel_sort` is competitive on is very small N (~4K and below); even there the radix path is comparable on modern hardware. To migrate, allocate `tmp_keys` of the same shape and dtype as `keys` plus a `u32` `scratch` buffer, then call `radix_sort` (see its section above for the full signature). `parallel_sort` is kept for one release cycle for backward compat and will be removed thereafter.

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

> **Deprecated.** New code should call `qd.algorithms.exclusive_scan_add(arr, out)` instead. `PrefixSumExecutor` is **inclusive**-only, **`i32`**-only, and **CUDA / Vulkan**-only; the new functional API covers `{i32, u32, f32, i64, u64, f64}` on every supported backend and runs the exclusive variant directly. To migrate from inclusive in-place to exclusive out-of-place, drop the `Executor` wrapper, allocate a distinct `out` field, and post-process if you actually need the inclusive form (`inclusive[i] = exclusive[i] + arr[i]`). `PrefixSumExecutor` is kept for one release cycle for backward compat and will be removed in a future release.

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
keys       = qd.field(qd.f32, shape=(N,))
tmp_keys   = qd.field(qd.f32, shape=(N,))
indices    = qd.field(qd.i32, shape=(N,))
tmp_idx    = qd.field(qd.i32, shape=(N,))

@qd.kernel
def init() -> None:
    for i in range(N):
        keys[i] = qd.random()
        indices[i] = i

init()
D = 1
while 256 ** D < N:
    D += 1
scratch = qd.field(qd.u32, shape=max(qd.algorithms.radix_sort_scratch_slots(N, D), 1))
nd = qd.ndarray(qd.i32, shape=())
nd.fill(N)
qd.algorithms.radix_sort(keys, tmp_keys, indices, tmp_idx, scratch, nd, True, 32, D)
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
