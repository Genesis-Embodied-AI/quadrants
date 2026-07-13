# Subgroup primitives

Subgroup operations let threads within the same subgroup (warp on NVIDIA, wave on AMD, subgroup on Vulkan / Metal) cooperate directly - exchanging register values, voting on predicates, scanning, and electing a leader - without going through shared memory or block barriers. They are the building block for fast in-warp data exchange and are used internally by `Tile16x16` and `Tile32x32` (see [tile](tile.md)).

Subgroup ops live under `qd.simt.subgroup` and are written so the same Python source compiles to the right vendor primitive on each backend. Calling a backend that has not implemented an op fails when the kernel is compiled, not silently at runtime.

This page is a high-level tour: what each op does, how the pieces fit together, and what to expect performance-wise. The complete per-op reference - exact signatures, dtype support, backend lowering, and caller contracts - is generated from the docstrings in the API reference: {py:mod}`quadrants.lang.simt.subgroup` (data movement, identification / control, voting, and lane masks), {py:mod}`quadrants.lang.simt.reductions` (reductions and scans), and {py:mod}`quadrants.lang.simt.sorting` (sorting).

## What's available

Every op is listed below, grouped by category, with the backends that currently lower it.

### Data movement

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                       |
|---------------------------------------------|------|--------|-------------------------|------------------------------|
| `subgroup.shuffle(value, index)`            | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_down(value, offset)`      | yes  | yes\*  | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_up(value, offset)`        | yes  | yes\*  | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_xor(value, mask)`         | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.broadcast(value, index)`          | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.broadcast_first(value)`           | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |

\* On AMDGPU, `shuffle_down` / `shuffle_up` go through `ds_bpermute` rather than a single-cycle cross-lane move - see [Performance notes](#performance-notes) for cycle counts.

`shuffle_xor` and `broadcast_first` are portable wrappers over `shuffle` / `broadcast` (`shuffle_xor(value, mask)` ≡ `shuffle(value, lane ^ mask)`; `broadcast_first(value)` ≡ `broadcast(value, qd.u32(0))`). They inline at compile time and run wherever the underlying op runs.

**SPIR-V caveat**: `broadcast(value, index)` requires `index` to be *dynamically uniform* (the same on every lane). Use `shuffle` if you need a per-lane-varying source lane. See the `broadcast` docstring for the full contract.

### Identification and control

| Op                                          | CUDA    | AMDGPU  | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|---------|---------|-------------------------|
| `subgroup.invocation_id()`                  | yes     | yes     | yes                     |
| `subgroup.group_size()`                     | yes     | yes     | yes                     |
| `subgroup.log2_group_size()`                | yes     | yes     | yes                     |
| `subgroup.elect()`                          | yes     | yes     | yes                     |
| `subgroup.sync()`                           | yes     | yes     | yes                     |
| `subgroup.mem_fence()`                      | yes\*\* | yes\*\* | yes                     |

Default scope: every op in this module operates over the **full active subgroup** (32 lanes on wave32, 64 on wave64) unless its name carries a `_tiled` suffix. The `_tiled` variants take an extra `log2_size` template argument and split the subgroup into independent `2**log2_size`-aligned tiles - see [Tiled variants](#tiled-variants).

Caller contract for every op on this page: call from **uniform control flow with all lanes active** - the whole subgroup must execute the call together. Calling from divergent control flow is implementation-defined on most backends.

\*\* `mem_fence()` lowers to a workgroup-scope fence on CUDA and AMDGPU. Both are over-strict for the subgroup-scope ask but correct: a workgroup-scope fence orders memory as observed by the whole workgroup, of which the subgroup is a strict subset.

Renames relative to the previous `qd.simt.subgroup` API:

- `subgroup.barrier()` → `subgroup.sync()` (matching `block.sync()`).
- `subgroup.memory_barrier()` → `subgroup.mem_fence()` (matching the planned `block.mem_fence()` and `grid.mem_fence()`).
- `subgroup.ballot_full_subgroup(predicate)` → `subgroup.ballot(predicate)`.
- Every `<op>(value, log2_size)` reduction / scan / vote that previously took `log2_size` directly is now `<op>_tiled(value, log2_size)`; the bare `<op>(value)` form is the full-subgroup convenience wrapper. **This is a breaking change** - call sites that hard-coded `log2_size = 5` (the old "full warp" idiom) need to either drop the argument or add the `_tiled` suffix to keep the old behavior on wave64.

The `barrier()` / `memory_barrier()` / `ballot_full_subgroup()` names remain as deprecated aliases that emit a `DeprecationWarning` on first use and forward to the new ones; they will be removed in a future release. The rest of this page uses the new names.

### Voting and predicate ops

`all_true(p)` / `any_true(p)` / `all_equal(v)` vote across the entire subgroup and broadcast the `i32` (`0` or `1`) result to every lane. The two `ballot` variants return a lane-per-bit mask: `ballot_first_n(predicate, n)` returns a `u32` covering lanes `[0, n)` (with `n` a compile-time constant `<= 32`), and `ballot(predicate)` returns a `u64` covering every lane in the subgroup (32 lanes on wave32, 64 on wave64). `lanemask_{lt,le,eq,gt,ge}(lane_id)` are closed-form `u32` lane-mask constants.

| Op                                            | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|-----------------------------------------------|------|--------|-------------------------|
| `subgroup.ballot_first_n(predicate, n)`       | yes  | yes    | yes                     |
| `subgroup.ballot(predicate)`                  | yes  | yes    | yes                     |
| `subgroup.{all,any}_true(predicate)`          | yes  | yes    | yes                     |
| `subgroup.all_equal(value)`                   | yes  | yes    | yes                     |
| `subgroup.lanemask_{lt,le,eq,gt,ge}(lane_id)` | yes  | yes    | yes                     |

`all_equal` uses the backend's native `==`, so for floats `NaN != NaN` and `+0.0 == -0.0`. Bit-cast to a same-width integer dtype first if you need bit-equality on floats.

### Reductions and scans

`reduce_*` returns the reduction in lane 0 (other lanes are undefined); `reduce_all_*` broadcasts the reduction to every lane in the subgroup. `inclusive_*` / `exclusive_*` produce the per-lane prefix; lane `i` ends up with the scan of `value[0..i]` (inclusive) or `value[0..i-1]` (exclusive). `segmented_reduce_*` resets the scan at every `head_flag != 0` and returns the per-segment inclusive reduction in every lane of that segment.

| Op                                                   | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                                                  |
|------------------------------------------------------|------|--------|-------------------------|---------------------------------------------------------|
| `subgroup.reduce_add(v)`                             | yes  | yes\*  | yes                     | any type supporting `+`                                 |
| `subgroup.reduce_all_add(v)`                         | yes  | yes    | yes                     | any type supporting `+`                                 |
| `subgroup.segmented_reduce_add(v, head_flag)`        | yes  | yes\*  | yes                     | any type supporting `+`                                 |
| `subgroup.reduce_{min,max}(v)`                       | yes  | yes\*  | yes                     | integer + float                                         |
| `subgroup.reduce_all_{min,max}(v)`                   | yes  | yes    | yes                     | integer + float                                         |
| `subgroup.segmented_reduce_{min,max}(v, head_flag)`  | yes  | yes\*  | yes                     | integer + float                                         |
| `subgroup.inclusive_{add,mul,min,max,and,or,xor}(v)` | yes  | yes\*  | yes                     | integer + float (`_and` / `_or` / `_xor`: integer only) |
| `subgroup.exclusive_{add,mul,min,max,and,or,xor}(v)` | yes  | yes\*  | yes                     | integer + float (`_and` / `_or` / `_xor`: integer only) |

\* On AMDGPU these go through `ds_bpermute` (~tens of cycles per step) - see [Performance notes](#performance-notes).

Every op above has a paired `_tiled` form that takes an extra `log2_size` template parameter and operates on independent `2**log2_size`-aligned tiles within the subgroup - see [Tiled variants](#tiled-variants). For reductions other than the ones listed above, build a sized helper on top of `shuffle_down` / `shuffle` following the same pattern as `reduce_add_tiled` / `reduce_all_add_tiled`.

Float NaN handling for `min` / `max` reductions is implementation-defined (CUDA and AMDGPU suppress NaN; some SPIR-V drivers propagate it). Avoid NaN inputs if you need a portable result.

### Sorting

In-register key/value sort across the subgroup, one `(key, value)` pair per lane. Pure `shuffle` - no shared memory, no barriers - fully unrolled at compile time.

| Op                                     | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                                                          |
|----------------------------------------|------|--------|-------------------------|-----------------------------------------------------------------|
| `subgroup.bitonic_sort_kv(key, value)` | yes  | yes    | yes                     | key & value: i32, u32, f32, f64, i64, u64 (independently typed) |

Returns `(key, value)` - assign with `key, value = subgroup.bitonic_sort_kv(key, value)`. Sorts ascending on the `(key, value)` lex tuple; ties on `key` break on ascending `value` (not a textbook-stable sort - equal-keyed lanes come back in ascending-`value` order, not in original-lane order). Tiled variant: `bitonic_sort_kv_tiled(key, value, log2_size)` runs the same sort independently on each `2**log2_size`-aligned tile - see [Tiled variants](#tiled-variants). See [Bitonic key/value sort example](#bitonic-keyvalue-sort-example) for the short-input (sentinel-padding) pattern; the docstring covers the float-NaN and stability caveats in full.

## Subgroup size

`group_size()` and `log2_group_size()` return the subgroup size (and its base-2 log) in effect for the current program, as plain Python `int`s. They are usable both inside `@qd.kernel` / `@qd.func` bodies - where the value is folded in as a compile-time constant - and from host scope (handy for setting `block_dim` / grid shapes that match the subgroup width). Because the value is a plain `int`, it can be passed as a `qd.template()` argument; that is how every full-subgroup op picks the right `log2_size` per backend without a runtime branch (each one is `<op>_tiled(v, log2_group_size())` under the hood).

| Backend                 | Subgroup size                                                             |
|-------------------------|--------------------------------------------------------------------------|
| CUDA                    | `32` (every sm_30+ NVIDIA arch)                                          |
| AMDGPU                  | `64` (every AMDGPU target is pinned to wave64, see [supported_systems](supported_systems.md)) |
| SPIR-V (Vulkan / Metal) | probed from the device at `qd.init()` (`32` on Metal and most Vulkan GPUs) |

The value is fixed for the lifetime of a program, so two kernels launched under the same `qd.init()` always see the same subgroup size.

## Tiled variants

Every reduce / scan / vote op in this module has a paired `_tiled` form that takes an extra `log2_size` template parameter and runs `group_size() / (2**log2_size)` independent reductions / scans / votes in parallel - one per `2**log2_size`-aligned tile within the subgroup. The base op is the special case where the tile is the whole subgroup (i.e. `log2_size = log2_group_size()`).

With `log2_size = k`, the subgroup splits into tiles of `2**k` consecutive lanes each, and each tile does its own reduction completely independently of every other tile. The caller arranges `2**k <= group_size()` so every tile is full; a smaller `k` simply gives more, narrower tiles. It does **not** mean "only the first tile is active".

| `log2_size`   | wave32 (CUDA / most Vulkan / Metal)        | wave64 (AMDGPU - see [supported_systems](supported_systems.md)) |
| ---           | ---                                        | ---                                |
| 5 (tile = 32) | 1 tile: lanes 0-31 (= base op)             | 2 tiles: 0-31, 32-63               |
| 4 (tile = 16) | 2 tiles: 0-15, 16-31                       | 4 tiles: 0-15, 16-31, 32-47, 48-63 |
| 3 (tile = 8)  | 4 tiles of 8                               | 8 tiles of 8                       |
| 0 (tile = 1)  | every lane is its own tile (no-op)         | same                               |

### Supported `_tiled` ops

| Tiled op                                                              | Result placement  |
|----------------------------------------------------------------------|-------------------|
| `subgroup.{all,any}_true_tiled(p, log2_size)`                        | broadcast-to-tile |
| `subgroup.all_equal_tiled(v, log2_size)`                             | broadcast-to-tile |
| `subgroup.reduce_add_tiled(v, log2_size)`                            | tile-local lane 0 |
| `subgroup.reduce_all_add_tiled(v, log2_size)`                        | broadcast-to-tile |
| `subgroup.segmented_reduce_add_tiled(v, head_flag, log2_size)`       | broadcast-to-tile |
| `subgroup.reduce_{min,max}_tiled(v, log2_size)`                      | tile-local lane 0 |
| `subgroup.reduce_all_{min,max}_tiled(v, log2_size)`                  | broadcast-to-tile |
| `subgroup.segmented_reduce_{min,max}_tiled(v, head_flag, log2_size)` | broadcast-to-tile |
| `subgroup.inclusive_{add,mul,min,max,and,or,xor}_tiled(v, log2_size)` | broadcast-to-tile |
| `subgroup.exclusive_{add,mul,min,max,and,or,xor}_tiled(v, log2_size)` | broadcast-to-tile |
| `subgroup.bitonic_sort_kv_tiled(key, value, log2_size)`              | broadcast-to-tile (every lane in the tile holds its sorted-position pair) |

- **Broadcast-to-tile forms**: every lane in each tile holds that tile's result. Lanes in different tiles hold different results (their own tile's).
- **Tile-local lane-0 forms**: only the *tile-local* lane 0 holds the reduction. That's lane 0 alone with `log2_size=5` on wave32, lanes 0 and 32 with `log2_size=5` on wave64, and so on. Other lanes hold partial reductions and should be treated as undefined. Use the `reduce_all_*_tiled` counterparts if you want every lane to see its tile's result.

`log2_size` is a `qd.template()` compile-time constant in `[0, log2_group_size()]` - `[0, 5]` on wave32 backends and `[0, 6]` on AMDGPU wave64. The caller must ensure `2**log2_size <= group_size()`; passing a larger value silently computes the wrong result on most backends, and there is no runtime check. Ops a backend does not support raise a `qd.static_assert` at compile time.

## Examples

### Broadcast lane 0 to all lanes example

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

### Ballot: count how many lanes satisfy a condition example

```python
@qd.kernel
def count_positive(a: qd.types.ndarray(dtype=qd.f32, ndim=1),
                   counts: qd.types.ndarray(dtype=qd.u32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        mask = subgroup.ballot_first_n(qd.i32(a[i] > 0.0), 32)
        if subgroup.invocation_id() == 0:
            counts[i // 32] = mask
```

After the kernel, `counts[g]` contains a bitmask of which lanes in group `g` had positive values. Use `popcount(mask)` on the host to get the count.

### Identity shuffle (each lane reads its own id) example

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

### Swap neighbors (xor pattern via explicit lane) example

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

### Arbitrary per-lane gather example

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

### Tree reduction with `shuffle_down` example

Classic warp-level sum of 4 values - after the second step, lane 0 of each group of 4 holds the total:

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

### Sum 32 lanes with `reduce_add_tiled` example

The same tree, packaged as a one-liner. Lane 0 of each group of 32 holds the total; other lanes hold partial sums:

```python
@qd.kernel
def sum32(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
          dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(src.shape[0]):
        total = subgroup.reduce_add_tiled(src[i], 5)
        if subgroup.invocation_id() == 0:
            dst[i // 32] = total
```

`5` is `log2_size`; `2**5 == 32` matches the block dim. The body of `reduce_add_tiled` unrolls at compile time into five `shuffle_down + add` pairs, so the generated code is identical to a hand-written tree reduction.

### Broadcast the sum to all lanes with `reduce_all_add_tiled` example

When every lane needs the reduction result - e.g. to normalize by the sum - use the butterfly variant. No follow-up broadcast needed:

```python
@qd.kernel
def normalize32(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        total = subgroup.reduce_all_add_tiled(a[i], 5)
        a[i] = a[i] / total
```

Every lane in each group of 32 sees the same `total`.

### Partial-subgroup reductions example

`log2_size` does not have to match the full subgroup. Sum groups of 8 with `reduce_add_tiled(v, 3)` or groups of 16 with `reduce_all_add_tiled(v, 4)`; the caller just ensures `2**log2_size <= group_size()` (so `log2_size <= 5` on CUDA / Metal / Vulkan-wave32, `<= 6` on AMDGPU wave64). Use the bare `reduce_add(v)` / `reduce_all_add(v)` form when you want "the whole subgroup" without hard-coding the limit.

### Bitonic key/value sort example

Sort up to 32 `(key, value)` pairs in registers, one per lane, with `bitonic_sort_kv`. The pattern below is the contact-pruning sort used by Genesis: each lane carries a packed link-pair id (`key`) and a contact index (`value`); after the sort, lane `i` holds the `i`-th smallest pair under the lex order `(key, value)`. Lanes past the real data (`lane >= n_con`) carry a sentinel key (`+inf` here) that the sort moves to the tail:

```python
@qd.kernel
def sort_contacts(keys: qd.types.ndarray(dtype=qd.f32, ndim=1),
                  idxs: qd.types.ndarray(dtype=qd.i32, ndim=1),
                  n_con: qd.i32):
    qd.loop_config(block_dim=32)
    for tid in range(32):
        # Load real data into the low n_con lanes; sentinel-pad the rest.  +inf compares greater than every real
        # key, so the sentinels drift to the high end of the sort.
        my_key = qd.f32(1.0e30)
        my_idx = qd.i32(-1)
        if tid < n_con:
            my_key = keys[tid]
            my_idx = idxs[tid]

        my_key, my_idx = subgroup.bitonic_sort_kv(my_key, my_idx)

        if tid < n_con:
            keys[tid] = my_key
            idxs[tid] = my_idx
```

After the kernel, `keys[0..n_con]` is sorted ascending and `idxs` is the matching permutation. The body unrolls at compile time into 30 shuffles + lex compares (for `log2_size = 5`, the wave32 default); no shared memory, no barriers.

Use `bitonic_sort_kv_tiled(k, v, log2_size)` directly to run multiple independent sorts per subgroup - e.g. `bitonic_sort_kv_tiled(k, v, 3)` runs `group_size() / 8` independent 8-element sorts in parallel. The tiles are `2**log2_size`-aligned within the subgroup and do not interact.

### Inclusive scan with `inclusive_add_tiled` example

```python
@qd.kernel
def cumsum(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        a[i] = subgroup.inclusive_add_tiled(a[i], 5)
```

After the call, lane `k` (within each group of 32) holds `a[group_start] + a[group_start+1] + ... + a[k]`. The `5` is `log2_size`; `2**5 == 32` matches the block dim. The body unrolls at compile time into five `shuffle_up + add` pairs. Use a smaller `log2_size` to scan over partial-subgroup groups (e.g. `inclusive_add_tiled(v, 3)` produces independent prefix sums in groups of 8).

## Performance notes

- Shuffles are register-to-register on CUDA (`__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`) and on SPIR-V where the GPU has hardware support - typically a handful of cycles, no memory traffic.
- On AMDGPU, `shuffle` / `shuffle_down` / `shuffle_up` go through `ds_bpermute` (roughly tens of cycles), so they cost more than the register-to-register shuffles on CUDA. On wave64 AMD GPUs, shuffles that cross the 32-lane half-boundary can cost a few extra cycles on some GPU architectures, but the underlying reads issue in parallel so latency stays close to a single shuffle.
- `shuffle_xor` and `broadcast_first` are wrappers over `shuffle` / `broadcast` and inline at compile time, so on every backend they cost exactly the same as the underlying op.
- Both `ballot_first_n` and `ballot` lower to a single hardware instruction on every backend. At `n == 32`, `ballot_first_n` skips the predicate-masking step entirely; at `n < 32` it adds one multiply on the predicate.
- `reduce_add` and `reduce_all_add` each issue exactly `log2_group_size()` shuffles and `log2_group_size()` adds per call (5 on wave32, 6 on AMDGPU wave64). No barriers, no shared memory, no launch overhead (they inline). The same holds for the `_tiled` form at any `log2_size`.
- Prefer `reduce_all_add` over `reduce_add` + broadcast when you need the result in every lane - same cost, one fewer shuffle.
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.
- The `inclusive_*` / `exclusive_*` scans cost `log2_group_size()` shuffle+op pairs (exclusive adds one extra shuffle + select), the same as a hand-rolled warp scan, on every backend. Quadrants uses a portable shuffle tree rather than a hardware scan intrinsic even where one is available, trading a small cost difference for predictable, uniform behavior across backends.
- `bitonic_sort_kv` runs `log2_size * (log2_size + 1)` shuffles - 30 for `log2_size = 5` (wave32), 42 for `log2_size = 6` (wave64) - all compile-time unrolled into the calling kernel; no shared memory, no barriers within the sort.

## Related

- [tile](tile.md) - `Tile16x16` and `Tile32x32` build on `subgroup.shuffle` to implement register-resident matrix tiles.
