# Algorithms

Device-wide algorithms are primitives that consume and produce whole arrays, executed as one or more kernel launches under the hood. They sit one tier above block and subgroup primitives: they *use* `block.reduce`, `block.exclusive_scan`, `block.radix_rank_match_atomic_or`, and `subgroup` reductions internally, and rely on the kernel-launch boundary (plus `atomic_add` in a few places) for cross-block synchronization rather than any in-kernel grid-scope barrier. The user calls them from host (Python) code, not from inside a kernel.

## What's available

| Op                                                          | What it does                                                       | CUDA | AMDGPU | Vulkan | Metal |
|-------------------------------------------------------------|--------------------------------------------------------------------|------|--------|--------|-------|
| `qd.algorithms.device_reduce_{add,min,max}(arr, out)`         | `out[0] = sum/min/max(arr)` (two-or-more-pass tree reduction; identity derived from `arr.dtype` for min / max) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_exclusive_scan_{add,min,max}(arr, out)` | `out[i] = sum/min/max(arr[0:i])` (three-pass Blelloch-style scan; 32-bit + 64-bit scalars; identity derived from `arr.dtype` for min / max) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_select(arr, flags, out, num_out)`     | Stream compaction: copy `arr[i]` to a dense prefix of `out` for every `flags[i] != 0`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_radix_sort(keys, tmp_keys, values=None, tmp_values=None, end_bit=None)` | LSB radix sort for 32-bit or 64-bit scalar keys (optional key-value). | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs)` | Collapse each consecutive run of equal keys into `(key, sum_of_values)`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.parallel_sort`                               | Odd-even merge sort (in-place, key or key-value). **Deprecated**: prefer `device_radix_sort`. | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.PrefixSumExecutor`                           | Inclusive in-place prefix sum (i32 only). **Deprecated**: prefer `device_exclusive_scan_add`. | yes  | no     | yes    | no    |

\* `device_reduce_*`, `device_exclusive_scan_*`, `device_select`, `device_radix_sort`, `device_reduce_by_key_add`, and `parallel_sort` run anywhere a Quadrants kernel runs; portability is inherited from the underlying block / subgroup primitives.

## Scratch space

Every device-wide algorithm in this module decomposes into "per-block partial → cross-block combine → finalize" passes (tree reduction, three-pass Blelloch scan, four-pass radix sort, scan-then-scatter compaction). The per-block partials need somewhere to live between kernel launches - that buffer is called **scratch**. Rather than ask each algorithm to allocate its own (forcing a `qd.field(...)` per call and undermining the no-implicit-allocation contract of the rest of the API), `qd.algorithms` shares a single set of module-level scratch fields across every call.

There are **two scratch fields**, one per element width that algorithm partials need to live in:

- `Field(u32)` - used by every 4-byte algorithm: `i32` / `u32` / `f32` reduce + scan, `device_select` indices, `device_reduce_by_key_add` flags + values, `device_radix_sort` tile histograms (regardless of key width). 4-byte values are `bit_cast` to / from `u32` on the way in and out.
- `Field(u64)` - used by every 8-byte algorithm: `i64` / `u64` / `f64` reduce + scan, `u64` radix-sort keys. Same `bit_cast` story, just at 8-byte width.

Sizing: each field defaults to **5 MB** (`DEFAULT_SCRATCH_BYTES = 5 << 20`). That covers `N` up to ~1.3M elements for `device_select` / `device_radix_sort` / `device_reduce_by_key_add` (`~N` u32 slots, qipc's hot path), and well past `N = 64M` for `device_reduce_*` / `device_exclusive_scan_*` (`~N / BLOCK_DIM` u32 slots, `BLOCK_DIM = 256`). The u64 field is sized to the same byte budget, so it covers half as many elements.

**Allocation is lazy.** A scratch field is only allocated on its first `get_scratch_*()` call from inside an algorithm. Programs that never touch `qd.algorithms.*` pay nothing; programs that only touch 4-byte algorithms never allocate the u64 buffer. (The default budget is therefore a per-field worst case, not a fixed cost: a 4-byte-only caller pays 5 MB, not 10 MB.)

**`qd.reset()` invalidates every scratch field** via an `impl.on_reset` hook, and resets the byte budget back to `DEFAULT_SCRATCH_BYTES`. The next algorithm call after a `qd.init()` reallocates against the fresh runtime at the default capacity. This keeps `qd.init` / `qd.reset` a "clean slate" - all runtime-scoped state (resource handles *and* config) goes away on reset, by design. Apps that need a persistent bump should call `set_scratch_bytes` immediately after each `qd.init`.

**Tuning the budget.** Call `quadrants._scratch.set_scratch_bytes(N)` before any algorithm runs (or before any algorithm runs after a `qd.reset()`). Pass a larger value to cover bigger `N`, or a smaller value to reduce the resident footprint on memory-constrained devices:

```python
from quadrants import _scratch
_scratch.set_scratch_bytes(20 << 20)   # 20 MB; covers N up to ~5M for device_select / radix sort
```

`set_scratch_bytes` raises `RuntimeError` if any scratch field has already been allocated in the current runtime cycle (re-`qd.init`-ing wipes that constraint). `scratch_bytes` must be a positive multiple of 8.

The per-algorithm sections below mention scratch only to call out per-algo footprint (so you can size the budget for a known `N`); the mechanics live here.

## Semantics

### `qd.algorithms.device_reduce_{add,min,max}(arr, out)`

Device-wide tree reduction over a 1-D tensor: `out[0]` holds `sum(arr)` / `min(arr)` / `max(arr)`. The monoid identity is derived from `arr.dtype` automatically (`0` for `add`; largest representable value for `min` - `+inf` for floats, `INT{32,64}_MAX` for signed ints, `UINT{32,64}_MAX` for unsigned; smallest representable value for `max` - `-inf` for floats, `INT{32,64}_MIN` for signed ints, `0` for unsigned), mirroring the `block.reduce_min` / `subgroup.reduce_min` typed wrappers which don't take an identity for the same reason.

```python
import quadrants as qd

inp = qd.field(qd.f32, shape=N)
out = qd.field(qd.f32, shape=1)
# ... fill inp ...

qd.algorithms.device_reduce_add(inp, out=out)
total = float(out.to_numpy()[0])   # explicit device->host hop
```

Arguments:

- `arr`: 1-D input tensor. Pass a `qd.field`, `qd.ndarray`, or `qd.Tensor` wrapper around either - the kernels are polymorphic via the `qd.Tensor` annotation.
- `out`: 1-element tensor with the same dtype as `arr`. Caller-supplied so the call is fully asynchronous - there is no implicit device→host sync. To get a Python scalar, do `out.to_numpy()[0]` explicitly after the call. This makes the host hop visible at the call site rather than hidden inside the algorithm.

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through the shared u32 scratch and 8-byte dtypes through the shared u64 scratch; see [Scratch space](#scratch-space) for the mechanics.
- **Shape:** `arr` must be 1-D; `out.shape` must be `(1,)`. Both must share the same dtype.
- **f32 / f64 non-associativity:** `device_reduce_add` on a floating-point dtype is not bitwise-reproducible across `N` changes, nor bitwise-equal to host `numpy.sum`. Tests tolerate a small relative error rather than asserting bitwise.

Implementation:

- Two-or-more-pass tree reduction. Each pass uses `BLOCK_DIM = 256` threads per block and reduces 256 elements per block via `block.reduce_{add,min,max}`. For `N <= 256` one pass suffices; for `N` up to `256^2 = 65536`, two passes; for larger `N`, additional intermediate passes are added until the reduction terminates in a single block.
- Per-block partials are written to the shared scratch field (u32 for 4-byte dtypes, u64 for 8-byte dtypes; see [Scratch space](#scratch-space)).
- The last pass writes the final value to `out[0]` directly. The kernel launches are pipelined back-to-back; correctness relies on the kernel-boundary serialization that Quadrants provides between host-launched kernels.

Scratch footprint: `ceil(N / BLOCK_DIM)` slots, where `BLOCK_DIM = 256`. Well under the 5 MB default for any reasonable `N` (`N = 1G` is ~4M slots); see [Scratch space](#scratch-space) if you need a different budget.

### `qd.algorithms.device_exclusive_scan_{add,min,max}(arr, out)`

Device-wide exclusive prefix scan over a 1-D tensor: `out[i]` holds the reduction (`sum` / `min` / `max`) of `arr[0:i]`. `out[0]` is always the monoid identity, which is derived from `arr.dtype` automatically (`0` for `add`; largest representable value for `min` - `+inf` for floats, `INT{32,64}_MAX` for signed ints, `UINT{32,64}_MAX` for unsigned; smallest representable value for `max` - `-inf` for floats, `INT{32,64}_MIN` for signed ints, `0` for unsigned), mirroring the `block.exclusive_min` / `subgroup.exclusive_min_tiled` typed wrappers.

```python
import quadrants as qd

N = 1_000_000
inp = qd.field(qd.f32, shape=N)
out = qd.field(qd.f32, shape=N)
# ... fill inp ...

qd.algorithms.device_exclusive_scan_add(inp, out=out)
# out[0] == 0.0; out[i] == sum(inp[0:i]) for i > 0.
```

Constraints:

- **Dtypes:** scalar `qd.i32`, `qd.u32`, `qd.f32`, `qd.i64`, `qd.u64`, `qd.f64`. Narrower / wider scalar dtypes (e.g. `qd.i16`, `qd.f16`) and struct dtypes raise `NotImplementedError`. 4-byte dtypes stage through the shared u32 scratch and 8-byte dtypes through the shared u64 scratch; see [Scratch space](#scratch-space) for the mechanics.
- **Shape:** `arr` and `out` must both be 1-D with the same shape and dtype.
- **No in-place scan:** `out` must be a distinct buffer from `arr`. Calling with `out is arr` raises `ValueError`. (The kernels do not protect against same-buffer aliasing; allocating one extra buffer once is cheap relative to the scan itself.)
- **Float non-associativity:** the order of additions inside a scan tree is not the same as a left-to-right host scan, so `f32` / `f64` results are *not* bitwise-equal to `numpy.cumsum`. Tests tolerate a small relative error (scaled by dtype precision).

Implementation:

- Blelloch 1990 three-pass exclusive scan:
  1. **Pass 1** - per-block tile reduce into the shared scratch (one slot per block).
  2. **Pass 2** - exclusive-scan the partials buffer in place. For `N ≤ BLOCK_DIM²` (= 65536) a single block does this. For larger `N`, the driver recurses: another tile-reduce on the partials, a recursive scan, then a downsweep that applies the higher-level prefixes.
  3. **Pass 3** - per-block tile scan + add the block prefix from scratch. Each block re-reads its tile from `arr`, runs `block.exclusive_scan` to get per-thread tile prefixes, and adds its `block_prefix` from the scanned partials.
- `BLOCK_DIM = 256`. Total scratch usage at `N = 1M` is `4096 + 16 = 4112` slots (~16 KB for 4-byte dtypes, ~32 KB for 8-byte), trivial relative to the 5 MB default. See [Scratch space](#scratch-space) for budget mechanics.

### `qd.algorithms.device_select(arr, flags, out, num_out)`

Stream compaction. Copy every `arr[i]` whose corresponding `flags[i]` is non-zero into a dense prefix of `out`, in stable input order, and write the count of selected elements to `num_out[0]`.

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

- **Dtypes:** `arr.dtype` is any scalar dtype in `{qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64}` *or* any `qd.types.struct(...)` / `qd.Struct.field({...})` composite (e.g. libuipc `Vector2i` / `Vector3i` / `Vector4i` / `LinearBVHAABB`-style structs). The scatter is `dst[idx] = src[i]`, which lowers per-field, so the algorithm is dtype-agnostic - no scratch reinterpretation needed for wider or composite element types.
- **`flags`:** 1-D `qd.i32` tensor with the same shape as `arr`. Each entry is treated as a boolean (`!= 0` selects). `flags` is caller-built - populate it with a kernel applying whatever predicate you want.
- **`out`:** 1-D tensor, same dtype as `arr`, with `len(out) >= len(arr)` so the worst-case all-selected run is safe. Only `out[0 : num_out[0]]` carries meaningful data on return; the tail is left untouched (whatever was in `out` before the call remains).
- **`num_out`:** 1-element `qd.i32` tensor. Same explicit-host-hop rule: do `int(num_out.to_numpy()[0])` after the call to get the count as a Python scalar.

Algorithm: the textbook scan-based compaction.

1. **Exclusive scan of `flags`** into the shared u32 scratch, producing per-element write indices. Same three-pass internals as `device_exclusive_scan_add`.
2. **Scatter:** one parallel kernel reads each `(arr[i], flags[i], indices[i])` and, if the flag is set, writes `out[indices[i]] = arr[i]`. No races by construction of the exclusive scan over 0 / 1 flags.
3. **Count tail:** one-thread kernel computes `indices[N-1] + flags[N-1]` and stores it in `num_out[0]`.

Scratch footprint: ~`N` u32 slots (one write index per input element). The default 5 MB scratch covers `N` up to ~1.3M (qipc's hot path lands here out of the box); bump the budget per [Scratch space](#scratch-space) for larger inputs.

### `qd.algorithms.device_radix_sort(keys, tmp_keys, values=None, tmp_values=None, end_bit=None)`

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
  1. **Histogram** - every block computes its per-digit count into shared memory, then publishes the 256-bin tile histogram to the shared u32 scratch (digit-major layout: `tile_histograms[d * num_blocks + b]`).
  2. **Scan** - in-place exclusive scan over the flat tile_histograms buffer. The digit-major layout makes a single 1-D scan enough to produce per-(digit, block) global offsets.
  3. **Scatter** - each block ranks its keys via `block.radix_rank_match_atomic_or` (wave32 + wave64 clean), looks up the per-(digit, block) global offset from the scan output, and scatters keys (and values, if provided) to the destination buffer.
- After each pass we swap `keys` ↔ `tmp_keys`. Four passes is even, so the sorted keys end up back in `keys`.
- Signed-integer (`i32` / `i64`) and floating-point (`f32` / `f64`) keys are mapped to a sortable unsigned representation (`u32` / `u64`) before the first pass and mapped back after the last pass via in-place "twiddle" kernels (signed: XOR sign bit; float: flip sign bit on positives, flip all bits on negatives - the standard sortable-key transform). `u32` / `u64` keys are sorted directly with no twiddle.

Scratch footprint: `num_blocks * 256 + ...` u32 slots per pass (re-used across passes), where `num_blocks = ceil(N / 256)`. The default 5 MB scratch covers `N` up to ~1.3M (qipc's hot path lands here out of the box); bump the budget per [Scratch space](#scratch-space) for larger inputs.

### `qd.algorithms.device_reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs)`

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

1. **Head-flag pass.** `head_flags[i] = 1` if `i == 0` or `keys[i] != keys[i-1]`, else `0`. Written to the shared u32 scratch (bit-cast from `i32`).
2. **In-place exclusive scan** of `head_flags` (using the same three-pass internals as `device_exclusive_scan_add`). After this, `scratch[i] = sum(head_flags[0:i])`.
3. **Zero-init `values_out[0:N]`.** The scatter uses `atomic_add`; slots must start at the additive identity `0`.
4. **Scatter.** For each `i`, recompute `head_flag(i)` from `keys[i]` / `keys[i-1]`, derive the run index `pos = scratch[i] + head_flag(i) - 1` (inclusive scan minus 1), and write `keys_out[pos] = keys[i]` + `atomic_add(values_out[pos], values[i])`.
5. **Count.** `num_runs[0] = scratch[N-1] + head_flag(N-1)`.

Scratch footprint: ~`1.004 * N` u32 slots. The default 5 MB scratch covers `N` up to ~1.3M; bump the budget per [Scratch space](#scratch-space) for larger inputs.

### `qd.algorithms.parallel_sort(keys, values=None)`

> **Deprecated.** New code should call `qd.algorithms.device_radix_sort(keys, tmp_keys, values=..., tmp_values=...)` instead. `device_radix_sort` is asymptotically `O(N log_radix N)` rather than `O(N log^2 N)`, is **stable** (odd-even merge sort is not), supports 32-bit and 64-bit scalar keys across CUDA / AMDGPU / Vulkan / Metal, and accepts `qd.field`, `qd.ndarray`, and `qd.Tensor` (`parallel_sort` is field-only). The only thing `parallel_sort` is competitive on is very small N (~4K and below); even there the radix path is comparable on modern hardware. To migrate, allocate a `tmp_keys` field of the same shape and dtype as `keys`, then call `device_radix_sort`. `parallel_sort` is kept for one release cycle for backward compat and will be removed thereafter.

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

> **Deprecated.** New code should call `qd.algorithms.device_exclusive_scan_add(arr, out)` instead. `PrefixSumExecutor` is **inclusive**-only, **`i32`**-only, and **CUDA / Vulkan**-only; the new functional API covers `{i32, u32, f32, i64, u64, f64}` on every supported backend and runs the exclusive variant directly. To migrate from inclusive in-place to exclusive out-of-place, drop the `Executor` wrapper, allocate a distinct `out` field, and post-process if you actually need the inclusive form (`inclusive[i] = exclusive[i] + arr[i]`). `PrefixSumExecutor` is kept for one release cycle for backward compat and will be removed in a future release.

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
