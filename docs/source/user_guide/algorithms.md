# Algorithms

Device-wide algorithms - primitives that consume and produce whole arrays, executed as one or more kernel launches under the hood. They sit one tier above grid-scope synchronization: they *use* block, subgroup, and grid primitives internally and expose a high-level entry point that the user calls from host (Python) code, not from inside a kernel.

## What's available

| Op                                                          | What it does                                                       | CUDA | AMDGPU | Vulkan | Metal |
|-------------------------------------------------------------|--------------------------------------------------------------------|------|--------|--------|-------|
| `qd.algorithms.device_reduce_add(input, *, out)`            | `out[0] = sum(input)` (two-or-more-pass tree reduction)            | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_reduce_min(input, id, *, out)`        | `out[0] = min(input)` (same recursion, caller-supplied identity)   | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_reduce_max(input, id, *, out)`        | `out[0] = max(input)` (same recursion, caller-supplied identity)   | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_exclusive_scan_add(input, *, out)`    | `out[i] = sum(input[0:i])` (three-pass Blelloch-style scan; 32-bit + 64-bit scalars) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_exclusive_scan_min(input, id, *, out)` | `out[i] = min(input[0:i])` (same pipeline, caller-supplied identity; 32-bit + 64-bit scalars) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_exclusive_scan_max(input, id, *, out)` | `out[i] = max(input[0:i])` (same pipeline, caller-supplied identity; 32-bit + 64-bit scalars) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_select(input, flags, *, out, num_out)` | Stream compaction: copy `input[i]` to a dense prefix of `out` for every `flags[i] != 0`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_radix_sort(keys, *, tmp_keys, values=None, tmp_values=None, end_bit=None)` | LSB radix sort for 32-bit or 64-bit scalar keys (optional key-value). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_reduce_by_key_add(keys_in, values_in, *, keys_out, values_out, num_runs)` | Collapse each consecutive run of equal keys into `(key, sum_of_values)`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.parallel_sort`                               | Odd-even merge sort (in-place, key or key-value). **Deprecated**: prefer `device_radix_sort`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.PrefixSumExecutor`                           | Inclusive in-place prefix sum (i32 only). **Deprecated**: prefer `device_exclusive_scan_add`. | yes  | no     | yes    | no    |

\* `device_reduce_*`, `device_exclusive_scan_*`, `device_select`, `device_radix_sort`, `device_reduce_by_key_add`, and `parallel_sort` run anywhere a Quadrants kernel runs; portability is inherited from the underlying block / subgroup primitives. AMDGPU and Metal coverage is exercised less heavily than CUDA / Vulkan; report any failures.

## Semantics

### `qd.algorithms.device_reduce_{add,min,max}(input, [identity,] *, out)`

Device-wide tree reduction over a 1-D tensor.

```python
import quadrants as qd

inp = qd.field(qd.f32, shape=N)
out = qd.field(qd.f32, shape=1)
# ... fill inp ...

qd.algorithms.device_reduce_add(inp, out=out)
total = float(out.to_numpy()[0])   # explicit device->host hop
```

Signatures:

- `device_reduce_add(input, *, out)` - sum reduction. Identity (`0` for the dtype) is derived automatically.
- `device_reduce_min(input, identity, *, out)` - min reduction. `identity` is **required** and must be a value `e` such that `min(e, x) == x` for every `x` in the dtype (e.g. `math.inf` for `f32` / `f64`, `2**31 - 1` for `i32`, `2**63 - 1` for `i64`, `2**32 - 1` for `u32`, `2**64 - 1` for `u64`).
- `device_reduce_max(input, identity, *, out)` - max reduction. `identity` is **required** and must be the dtype's negative extremum (e.g. `-math.inf` for `f32` / `f64`, `-2**31` for `i32`, `-2**63` for `i64`, `0` for `u32` / `u64`).

Arguments:

- `input`: 1-D tensor. Pass a `qd.field`, `qd.ndarray`, or `qd.Tensor` wrapper around either - the kernels are polymorphic via the `qd.Tensor` annotation.
- `out`: 1-element tensor with the same dtype as `input`. Caller-supplied so the call is fully asynchronous - there is no implicit device→host sync. To get a Python scalar, do `out.to_numpy()[0]` explicitly after the call. This makes the host hop visible at the call site rather than hidden inside the algorithm.

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes are staged through a shared `Field(u32)` scratch and 8-byte dtypes through a shared `Field(u64)` scratch; the byte budget is the same.
- **Shape:** `input` must be 1-D; `out.shape` must be `(1,)`. Both must share the same dtype.
- **Identity (min / max only):** mandatory. Calling `device_reduce_min` / `device_reduce_max` without an `identity` argument raises `TypeError`.
- **f32 / f64 non-associativity:** `device_reduce_add` on a floating-point dtype is not bitwise-reproducible across `N` changes, nor bitwise-equal to host `numpy.sum`. Tests tolerate a small relative error rather than asserting bitwise.

Implementation:

- Two-or-more-pass tree reduction. Each pass uses `BLOCK_DIM = 256` threads per block and reduces 256 elements per block via `block.reduce_{add,min,max}`. For `N <= 256` one pass suffices; for `N` up to `256^2 = 65536`, two passes; for larger `N`, additional intermediate passes are added until the reduction terminates in a single block.
- Per-block partials are written to a **shared scratch field**. There are actually three scratch fields under the hood (a `Field(u32)` for 4-byte dtypes, a `Field(u64)` for 8-byte integer dtypes and 8-byte reduce, and a `Field(f64)` for `f64` exclusive scan), all lazily allocated at first algorithm call and sized to the same byte budget (default 1 MB each, which covers `N` up to ~64M 4-byte elements or ~32M 8-byte elements). The u32 / u64 scratches are bit-cast on access so a single field per width backs every dtype that goes through them; the f64 scratch is used directly to side-step a known precision loss in `bit_cast(f64, u64)` on scan results.
- The last pass writes the final value to `out[0]` directly. The kernel launches are pipelined back-to-back; correctness relies on the kernel-boundary serialization that Quadrants provides between host-launched kernels.

If you scan or reduce on `N > ≈ 64M`, raise the scratch budget *before any algorithm runs*:

```python
from quadrants import _scratch
_scratch.set_scratch_bytes(4 << 20)   # 4 MB, before any qd.algorithms.* call
```

### `qd.algorithms.device_exclusive_scan_{add,min,max}(input, [identity,] *, out)`

Device-wide exclusive prefix scan over a 1-D tensor: `out[i]` holds the reduction (`sum` / `min` / `max`) of `input[0:i]`. `out[0]` is always the monoid identity.

```python
import quadrants as qd

N = 1_000_000
inp = qd.field(qd.f32, shape=N)
out = qd.field(qd.f32, shape=N)
# ... fill inp ...

qd.algorithms.device_exclusive_scan_add(inp, out=out)
# out[0] == 0.0; out[i] == sum(inp[0:i]) for i > 0.
```

Signatures:

- `device_exclusive_scan_add(input, *, out)` - exclusive sum. Identity (`0` for the dtype) is derived automatically.
- `device_exclusive_scan_min(input, identity, *, out)` - exclusive min. `identity` is **required** (e.g. `math.inf` for `f32` / `f64`, `2**31 - 1` for `i32`, `2**63 - 1` for `i64`).
- `device_exclusive_scan_max(input, identity, *, out)` - exclusive max. `identity` is **required** (e.g. `-math.inf` for `f32` / `f64`, `-2**31` for `i32`, `-2**63` for `i64`, `0` for unsigned).

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through the shared `Field(u32)` scratch, 8-byte integer dtypes (`i64` / `u64`) stage through the shared `Field(u64)` scratch, and `f64` stages through a separate shared `Field(f64)` scratch (see Implementation below); the byte budget is the same for all three.
- **Shape:** `input` and `out` must both be 1-D with the same shape and dtype.
- **No in-place scan:** `out` must be a distinct buffer from `input`. Calling with `out is input` raises `ValueError`. (The kernels do not protect against same-buffer aliasing; allocating one extra buffer once is cheap relative to the scan itself.)
- **Identity (min / max only):** mandatory. Passing `None` or omitting `identity` raises `TypeError`.
- **Float non-associativity:** the order of additions inside a scan tree is not the same as a left-to-right host scan, so `f32` / `f64` results are *not* bitwise-equal to `numpy.cumsum`. Tests tolerate a small relative error (scaled by dtype precision).

Implementation:

- Blelloch 1990 three-pass exclusive scan:
  1. **Pass 1** - per-block tile reduce into the shared scratch (one slot per block).
  2. **Pass 2** - exclusive-scan the partials buffer in place. For `N ≤ BLOCK_DIM²` (= 65536) a single block does this. For larger `N`, the driver recurses: another tile-reduce on the partials, a recursive scan, then a downsweep that applies the higher-level prefixes.
  3. **Pass 3** - per-block tile scan + add the block prefix from scratch. Each block re-reads its tile from the input, runs `block.exclusive_scan` to get per-thread tile prefixes, and adds its `block_prefix` from the scanned partials.
- `BLOCK_DIM = 256`. Total scratch usage at `N = 1M` is `4096 + 16 = 4112` slots (~16 KB for 4-byte dtypes, ~32 KB for 8-byte), well under the 1 MB default.
- `f64` uses a dedicated `Field(f64)` scratch (rather than reinterpreting the u64 scratch via `bit_cast`) because `bit_cast(scan_result_f64, u64)` currently loses precision in Quadrants; routing partials through an f64-typed buffer side-steps the cast on the write side entirely. Same byte budget as the u64 scratch.

### `qd.algorithms.device_select(input, flags, *, out, num_out)`

Stream compaction. Copy every `input[i]` whose corresponding `flags[i]` is non-zero into a dense prefix of `out`, in stable input order, and write the count of selected elements to `num_out[0]`.

```python
import quadrants as qd

N = 100_000
inp     = qd.field(qd.f32, shape=N)
flags   = qd.field(qd.i32, shape=N)         # caller fills with 0 / 1
out     = qd.field(qd.f32, shape=N)         # large enough for worst case
num_out = qd.field(qd.i32, shape=1)

# ... fill inp + flags via a separate kernel ...

qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)

# Only out[0 : count] is meaningful; copy out the count host-side explicitly:
count = int(num_out.to_numpy()[0])
selected = out.to_numpy()[:count]
```

Constraints:

- **Dtypes:** `input.dtype` is any scalar dtype in `{qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64}` *or* any `qd.types.struct(...)` / `qd.Struct.field({...})` composite (e.g. libuipc `Vector2i` / `Vector3i` / `Vector4i` / `LinearBVHAABB`-style structs). The scatter is `dst[idx] = src[i]`, which lowers per-field, so the algorithm is dtype-agnostic - no scratch reinterpretation needed for wider or composite element types.
- **`flags`:** 1-D `qd.i32` tensor with the same shape as `input`. Each entry is treated as a boolean (`!= 0` selects). `flags` is caller-built - populate it with a kernel applying whatever predicate you want.
- **`out`:** 1-D tensor, same dtype as `input`, with `len(out) >= len(input)` so the worst-case all-selected run is safe. Only `out[0 : num_out[0]]` carries meaningful data on return; the tail is left untouched (whatever was in `out` before the call remains).
- **`num_out`:** 1-element `qd.i32` tensor. Same explicit-host-hop rule: do `int(num_out.to_numpy()[0])` after the call to get the count as a Python scalar.

Algorithm: the textbook scan-based compaction.

1. **Exclusive scan of `flags`** into the shared `Field(u32)` scratch, producing per-element write indices. Same three-pass internals as `device_exclusive_scan_add`.
2. **Scatter:** one parallel kernel reads each `(input[i], flags[i], indices[i])` and, if the flag is set, writes `out[indices[i]] = input[i]`. No races by construction of the exclusive scan over 0 / 1 flags.
3. **Count tail:** one-thread kernel computes `indices[N-1] + flags[N-1]` and stores it in `num_out[0]`.

Scratch budget: at the default 1 MB, `N + ceil(N/256) + ... ≤ 262144`, so roughly `N ≤ 260_000` works out of the box. Raise the budget via `_scratch.set_scratch_bytes(...)` before any algorithm runs for larger inputs.

### `qd.algorithms.device_radix_sort(keys, *, tmp_keys, values=None, tmp_values=None, end_bit=None)`

Ascending in-place radix sort over a 1-D tensor of 32-bit or 64-bit scalar keys (`u32` / `i32` / `f32` / `u64` / `i64` / `f64`), with optional lock-step permutation of an `values` tensor (key-value sort).

```python
import quadrants as qd

N = 100_000
keys     = qd.field(qd.f32, shape=N)
tmp_keys = qd.field(qd.f32, shape=N)   # workspace; contents on return are garbage
# ... fill keys ...

qd.algorithms.device_radix_sort(keys, tmp_keys=tmp_keys)
# keys is now ascending; tmp_keys holds intermediate state.

# Key-value sort:
values     = qd.field(qd.i32, shape=N)
tmp_values = qd.field(qd.i32, shape=N)
# ... fill values (e.g. with original indices) ...

qd.algorithms.device_radix_sort(
    keys, tmp_keys=tmp_keys, values=values, tmp_values=tmp_values,
)
# keys ascending; values permuted so values[k] corresponds to keys[k].
```

Arguments:

- `keys`: 1-D tensor. Sorted **in place**. Pass `qd.field`, `qd.ndarray`, or `qd.Tensor`.
- `tmp_keys`: ping-pong workspace, same shape & dtype as `keys`, distinct buffer. Contents on return are intermediate and should be considered garbage.
- `values`: optional 1-D tensor of any supported scalar dtype (the value dtype is independent of the key dtype), same shape as `keys`. If provided, permuted in lock-step with the keys.
- `tmp_values`: required iff `values` is provided. Same shape & dtype as `values`, distinct buffer. Same workspace semantics as `tmp_keys`.
- `end_bit`: number of low bits of the key to consider. Defaults to the full key width (32 for 4-byte keys, 64 for 8-byte keys). Must be a positive multiple of `8` (the radix-digit width). An even number of digit passes is required so the result lands back in `keys`; with the default `end_bit` this is automatic. Pass a smaller value when the high bits are known to be zero (e.g. `end_bit=16` for keys with values `< 2**16`) to save passes.

Constraints:

- **Dtypes:** `keys.dtype` and `values.dtype` are each independently one of `{qd.u32, qd.i32, qd.f32, qd.u64, qd.i64, qd.f64}`. Narrower scalar dtypes (`qd.i16`, `qd.f16`, ...) and struct dtypes raise `NotImplementedError`. 8-byte keys run 8 digit passes per sort; 4-byte keys run 4. Scratch footprint is the same for both widths (the per-tile histograms are `u32` regardless).
- **Aliasing:** `keys` and `tmp_keys` must be distinct buffers; same for `values` / `tmp_values`. Calling with the same buffer raises `ValueError`.
- **Stability:** stable sort - equal keys keep their original input order in the output.
- **NaN handling (f32):** matches `numpy.sort` (NaNs land at the end). NaNs are not tested separately and should not be relied on for ordering invariants beyond `numpy.sort`.

Implementation:

- Classical LSB radix sort with 8-bit digits, four passes for `u32` / `i32` / `f32`. Each digit pass is three internal kernels:
  1. **Histogram** - every block computes its per-digit count into shared memory, then publishes the 256-bin tile histogram to the global `Field(u32)` scratch (digit-major layout: `tile_histograms[d * num_blocks + b]`).
  2. **Scan** - in-place exclusive scan over the flat tile_histograms buffer. The digit-major layout makes a single 1-D scan enough to produce per-(digit, block) global offsets.
  3. **Scatter** - each block ranks its keys via `block.radix_rank_match_atomic_or` (wave32 + wave64 clean), looks up the per-(digit, block) global offset from the scan output, and scatters keys (and values, if provided) to the destination buffer.
- After each pass we swap `keys` ↔ `tmp_keys`. Four passes is even, so the sorted keys end up back in `keys`.
- Signed-integer (`i32` / `i64`) and floating-point (`f32` / `f64`) keys are mapped to a sortable unsigned representation (`u32` / `u64`) before the first pass and mapped back after the last pass via in-place "twiddle" kernels (signed: XOR sign bit; float: flip sign bit on positives, flip all bits on negatives - the standard sortable-key transform). `u32` / `u64` keys are sorted directly with no twiddle.

Scratch budget: `num_blocks * 256 + ...` `u32` slots per pass (re-used across passes), where `num_blocks = ceil(N / 256)`. The default 1 MB scratch budget covers `N` up to **≈ 260_000**. For `N = 1M` (qipc's hot path), raise the budget to **5 MB** before any algorithm runs:

```python
from quadrants import _scratch
_scratch.set_scratch_bytes(5 << 20)   # 5 MB; covers N up to ~1.3M
```

A single-pass decoupled-lookback variant ("Onesweep") is a perf follow-up; the first land prioritizes simplicity and small LoC over peak throughput.

### `qd.algorithms.device_reduce_by_key_add(keys_in, values_in, *, keys_out, values_out, num_runs)`

Collapse every **consecutive run of equal keys** into a single output entry `(unique_key, sum_of_values_in_run)`. Keys that compare equal but are separated by other keys form separate runs. For a global per-key sum, sort by key first (e.g. with `qd.algorithms.device_radix_sort`) and then reduce-by-key.

```python
import quadrants as qd

N = 100_000
keys_in    = qd.field(qd.i32, shape=N)       # sorted by key beforehand
values_in  = qd.field(qd.f32, shape=N)
keys_out   = qd.field(qd.i32, shape=N)       # capacity = N (worst case: all unique)
values_out = qd.field(qd.f32, shape=N)
num_runs   = qd.field(qd.i32, shape=1)

# ... fill keys_in + values_in ...

qd.algorithms.device_reduce_by_key_add(
    keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs,
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

Constraints:

- **Dtypes (first land):** `keys_in.dtype` and `values_in.dtype` ∈ {`qd.i32`, `qd.u32`, `qd.f32`}. Other dtypes raise `NotImplementedError`.
- **Reduction:** only `add` is exposed for first land. `min` / `max` variants need `atomic_min` / `atomic_max` for `f32`, which has spottier cross-backend support; defer to a follow-up gated on real qipc usage.
- **f32 non-associativity:** the order of additions inside a run is set by hardware atomic ordering, not host order, so `f32` results are *not* bitwise-equal to a serial scan. Tests tolerate a small relative error.
- **NaN handling (f32 keys):** `NaN != NaN` is true, so each NaN-keyed element becomes its own run. Consistent with treating NaN as "different from everything", which matches the run-length-encoding spirit.

Algorithm: scan + scatter + atomic_add - no segmented-scan primitive needed.

1. **Head-flag pass.** `head_flags[i] = 1` if `i == 0` or `keys[i] != keys[i-1]`, else `0`. Written to the shared `Field(u32)` scratch (bit-cast from `i32`).
2. **In-place exclusive scan** of `head_flags` (using the same three-pass internals as `device_exclusive_scan_add`). After this, `scratch[i] = sum(head_flags[0:i])`.
3. **Zero-init `values_out[0:N]`.** The scatter uses `atomic_add`; slots must start at the additive identity `0`.
4. **Scatter.** For each `i`, recompute `head_flag(i)` from `keys[i]` / `keys[i-1]`, derive the run index `pos = scratch[i] + head_flag(i) - 1` (inclusive scan minus 1), and write `keys_out[pos] = keys[i]` + `atomic_add(values_out[pos], values[i])`.
5. **Count.** `num_runs[0] = scratch[N-1] + head_flag(N-1)`.

Scratch budget: `~1.004 * N` u32 slots. The default 1 MB scratch covers `N` up to ~260_000; raise via `_scratch.set_scratch_bytes(...)` for larger inputs.

### `qd.algorithms.parallel_sort(keys, values=None)`

> **Deprecated.** New code should call `qd.algorithms.device_radix_sort(keys, tmp_keys=..., values=..., tmp_values=...)` instead. `device_radix_sort` is asymptotically `O(N log_radix N)` rather than `O(N log^2 N)`, is **stable** (odd-even merge sort is not), supports 32-bit and 64-bit scalar keys across CUDA / AMDGPU / Vulkan / Metal, and accepts `qd.field`, `qd.ndarray`, and `qd.Tensor` (`parallel_sort` is field-only). The only thing `parallel_sort` is competitive on is very small N (~4K and below); even there the radix path is comparable on modern hardware. To migrate, allocate a `tmp_keys` field of the same shape and dtype as `keys`, then call `device_radix_sort`. `parallel_sort` is kept for one release cycle for backward compat and will be removed thereafter.

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

> **Deprecated.** New code should call `qd.algorithms.device_exclusive_scan_add(input, *, out)` instead. `PrefixSumExecutor` is **inclusive**-only, **`i32`**-only, and **CUDA / Vulkan**-only; the new functional API covers `{i32, u32, f32, i64, u64, f64}` on every supported backend and runs the exclusive variant directly. To migrate from inclusive in-place to exclusive out-of-place, drop the `Executor` wrapper, allocate a distinct `out` field, and post-process if you actually need the inclusive form (`inclusive[i] = exclusive[i] + input[i]`). `PrefixSumExecutor` is kept for one release cycle for backward compat and will be removed in a future release.

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
qd.algorithms.device_radix_sort(
    keys, tmp_keys=tmp_keys, values=indices, tmp_values=tmp_idx,
)
# keys is now ascending; indices[k] is the original index of the k-th smallest key.
# (Stable: ties between equal keys preserve their input-order indices.)
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
