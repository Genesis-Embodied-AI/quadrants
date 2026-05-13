# Algorithms

Device-wide algorithms — primitives that consume and produce whole arrays, executed as one or more kernel launches under the hood. They sit one tier above grid-scope synchronization: they *use* block, subgroup, and grid primitives internally and expose a high-level entry point that the user calls from host (Python) code, not from inside a kernel.

## What's available

| Op                                                 | What it does                                                   | CUDA | AMDGPU | Vulkan | Metal |
|----------------------------------------------------|----------------------------------------------------------------|------|--------|--------|-------|
| `qd.algorithms.device_reduce_add(input, *, out)`   | `out[0] = sum(input)` (two-or-more-pass tree reduction)        | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_reduce_min(input, id, *, out)` | `out[0] = min(input)` (same recursion, caller-supplied identity) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.device_reduce_max(input, id, *, out)` | `out[0] = max(input)` (same recursion, caller-supplied identity) | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.parallel_sort`                      | Odd-even merge sort (in-place, key or key-value)               | yes  | yes\*  | yes    | yes\* |
| `qd.algorithms.PrefixSumExecutor`                  | Inclusive in-place prefix sum (i32 only)                       | yes  | no     | yes    | no    |

\* `device_reduce_*` and `parallel_sort` run anywhere a Quadrants kernel runs; portability is inherited from the underlying block / subgroup primitives. AMDGPU and Metal coverage is exercised less heavily than CUDA / Vulkan; report any failures.

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

- `device_reduce_add(input, *, out)` — sum reduction. Identity (`0` for the dtype) is derived automatically.
- `device_reduce_min(input, identity, *, out)` — min reduction. `identity` is **required** and must be a value `e` such that `min(e, x) == x` for every `x` in the dtype (e.g. `math.inf` for `f32`, `2**31 - 1` for `i32`, `2**32 - 1` for `u32`).
- `device_reduce_max(input, identity, *, out)` — max reduction. `identity` is **required** and must be the dtype's negative extremum (e.g. `-math.inf` for `f32`, `-2**31` for `i32`, `0` for `u32`).

Arguments:

- `input`: 1-D tensor. Pass a `qd.field`, `qd.ndarray`, or `qd.Tensor` wrapper around either — the kernels are polymorphic via the `qd.Tensor` annotation.
- `out`: 1-element tensor with the same dtype as `input`. Caller-supplied so the call is fully asynchronous — there is no implicit device→host sync. To get a Python scalar, do `out.to_numpy()[0]` explicitly after the call. This makes the host hop visible at the call site rather than hidden inside the algorithm.

Constraints:

- **Dtypes (first land):** `qd.i32`, `qd.u32`, `qd.f32`. Calls with `qd.i64` / `qd.f64` raise `NotImplementedError`; lifting that is on the roadmap and gated on extending `block.reduce` to those dtypes.
- **Shape:** `input` must be 1-D; `out.shape` must be `(1,)`. Both must share the same dtype.
- **Identity (min / max only):** mandatory. Calling `device_reduce_min` / `device_reduce_max` without an `identity` argument raises `TypeError`.
- **f32 non-associativity:** `device_reduce_add` on `f32` is not bitwise-reproducible across `N` changes, nor bitwise-equal to host `numpy.sum`. Tests tolerate a small relative error rather than asserting bitwise.

Implementation:

- Two-or-more-pass tree reduction. Each pass uses `BLOCK_DIM = 256` threads per block and reduces 256 elements per block via `block.reduce_{add,min,max}`. For `N ≤ 256` one pass suffices; for `N` up to `256² = 65536`, two passes; for larger `N`, additional intermediate passes are added until the reduction terminates in a single block.
- Per-block partials are written to a **shared scratch field** (single `Field(u32)`, allocated lazily at first algorithm call, default 1 MB which covers `N` up to ≈ 64M elements). The shared scratch is bit-cast on access so a single field backs every supported dtype.
- The last pass writes the final value to `out[0]` directly. The kernel launches are pipelined back-to-back; correctness relies on the kernel-boundary serialization that Quadrants provides between host-launched kernels.

If you scan or reduce on `N > ≈ 64M`, raise the scratch budget *before any algorithm runs*:

```python
from quadrants import _scratch
_scratch.set_scratch_bytes(4 << 20)   # 4 MB, before any qd.algorithms.* call
```

### `qd.algorithms.parallel_sort(keys, values=None)`

In-place sort. Reorders `keys` ascending; if `values` is provided, applies the same permutation to `values` (key-value sort). Both arguments must be 1-D `qd.field` — `parallel_sort` reaches into `snode.ptr.offset` internally, so `ndarray` is **not** supported and will fail at compile time with an `AttributeError`.

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
- **Stability.** Odd-even merge sort is *not* a stable sort — equal keys may be reordered relative to one another. If stability matters, encode tiebreakers into the keys (e.g. pack the original index into the low bits).
- **Memory.** Strictly in-place — no auxiliary buffers from the caller's perspective.
- **Performance characteristic.** Beats radix-style sorts for small N (roughly N ≲ 4K).

### `qd.algorithms.PrefixSumExecutor`

Inclusive in-place prefix sum (scan) over a 1-D `i32` field. Construct once with the array length, then call `.run(field)` to scan.

```python
psum = qd.algorithms.PrefixSumExecutor(N)
arr  = qd.field(qd.i32, shape=(N,))
# ... fill arr ...
psum.run(arr)
# arr now holds the inclusive prefix sum: arr[i] = sum(arr_original[0..=i]).
```

Constructor:

- `length: int` — the **fixed** number of elements the executor will scan on every `.run()` call. Internally allocates an auxiliary `qd.field(i32, shape=padded_length)` sized to the Kogge-Stone hierarchy (block size = 64).

`run(input_arr)`:

- `input_arr` must be a 1-D `qd.field(qd.i32, shape=(length,))` — its length must match the constructor's `length` exactly. `run()` always blits `length` elements between `input_arr` and the internal buffer; passing a shorter field results in out-of-bounds reads / writes (no runtime check today).
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
keys = qd.field(qd.f32, shape=(N,))
indices = qd.field(qd.i32, shape=(N,))

@qd.kernel
def init() -> None:
    for i in range(N):
        keys[i] = qd.random()
        indices[i] = i

init()
qd.algorithms.parallel_sort(keys, indices)
# keys is now ascending; indices[k] is the original index of the k-th smallest key.
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

- `qd.simt.block.*` — the block-scope reductions and shared-memory primitives that algorithm kernels build on.
- `qd.simt.subgroup.*` — `inclusive_add` and friends, what the per-block scan stage of `PrefixSumExecutor` actually calls.
- `qd.simt.grid.mem_fence()` — the grid-scope memory fence that decoupled-look-back scans (a more efficient alternative to Kogge-Stone) require.
- [parallelization](parallelization.md) — broader synchronization story, including how `qd.algorithms` operations compose with hand-written kernels.
