# Subgroup primitives

Subgroup operations let threads within the same subgroup (warp on NVIDIA, wave on AMD, subgroup on Vulkan / Metal) cooperate directly - exchanging register values, voting on predicates, scanning, and electing a leader - without going through shared memory or block barriers. They are the building block for fast in-warp data exchange and are used internally by `Tile16x16` and `Tile32x32` (see [tile](tile.md)).

Subgroup ops live under `qd.simt.subgroup` and are written so the same Python source compiles to the right vendor primitive on each backend. Calling a backend that has not implemented an op fails at trace or codegen time.

## What's available

The full Python API is grouped here by category. The first column lists each op, the next three columns indicate which backend currently lowers it.

### Data movement

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                       |
|---------------------------------------------|------|--------|-------------------------|------------------------------|
| `subgroup.shuffle(value, index)`            | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_down(value, offset)`      | yes  | yes\*  | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_up(value, offset)`        | yes  | yes\*  | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.shuffle_xor(value, mask)`         | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.broadcast(value, index)`          | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |
| `subgroup.broadcast_first(value)`           | yes  | yes    | yes                     | i32, u32, f32, f64, i64, u64 |

\* AMDGPU `shuffle_down` / `shuffle_up` go through `ds_bpermute` rather than a single-cycle cross-lane move - see [Performance notes](#performance-notes) for the full lowering and cycle counts.

`shuffle_xor` and `broadcast_first` are portable `@qd.func` wrappers on top of `shuffle` / `broadcast` (`shuffle_xor(value, mask)` ≡ `shuffle(value, lane ^ mask)`; `broadcast_first(value)` ≡ `broadcast(value, qd.u32(0))`). They inline at compile time and run wherever the underlying op runs.

### Identification and control

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|------|--------|-------------------------|
| `subgroup.invocation_id()`                  | yes  | yes    | yes                     |
| `subgroup.group_size()`                     | yes  | yes    | yes                     |
| `subgroup.log2_group_size()`                | yes  | yes    | yes                     |
| `subgroup.elect()`                          | yes  | yes    | yes                     |
| `subgroup.sync()`                           | yes  | yes    | yes                     |
| `subgroup.mem_fence()`                      | yes\*\* | yes\*\* | yes                  |

Default scope: every op in this module operates over the **full active subgroup** (32 lanes on wave32, 64 on wave64) unless its name carries a `_tiled` suffix. The `_tiled` variants take an extra `log2_size` template argument and split the subgroup into independent `2**log2_size`-aligned tiles - see [Tiled variants](#tiled-variants) below for when and why you might want them.

Renames relative to the previous `qd.simt.subgroup` API:

- `subgroup.barrier()` → `subgroup.sync()` (matching `block.sync()`).
- `subgroup.memory_barrier()` → `subgroup.mem_fence()` (matching the planned `block.mem_fence()` and `grid.mem_fence()`).
- `subgroup.ballot_full_subgroup(predicate)` → `subgroup.ballot(predicate)`.
- Every `<op>(value, log2_size)` reduction / scan / vote that previously took `log2_size` directly is now `<op>_tiled(value, log2_size)`; the bare `<op>(value)` form is the full-subgroup convenience wrapper. **This is a breaking change** - call sites that hard-coded `log2_size = 5` (the old "full warp" idiom) need to either drop the argument or add the `_tiled` suffix to keep the old behaviour on wave64.

The `barrier()` / `memory_barrier()` / `ballot_full_subgroup()` names remain as deprecated aliases that emit a `DeprecationWarning` on first use and forward to the new ones; they will be removed in a future release. The rest of this page uses the new names.

\*\* `mem_fence()` lowers to a workgroup-scope fence on CUDA (`__threadfence_block()`, via `nvvm.membar.cta`) and AMDGPU (LLVM `fence syncscope("workgroup") seq_cst`). Both are over-strict for the subgroup-scope ask but are correct: a workgroup-scope fence orders memory as observed by the whole workgroup, of which the subgroup is a strict subset. A future change can tighten these to true wave-scope fences if a measurable cost shows up.

### Voting and predicate ops

`all_true(p)` / `any_true(p)` / `all_equal(v)` vote across the entire subgroup and broadcast the `i32` (`0` or `1`) result to every lane.

The two `ballot` variants are tile-less by construction: `ballot_first_n(predicate, n)` returns a `u32` covering lanes `[0, n)` (with `n` a compile-time constant `<= 32`), and `ballot(predicate)` returns a `u64` covering every lane in the subgroup (32 lanes on wave32, 64 on wave64).

| Op                                            | CUDA                                  | AMDGPU                | SPIR-V (Vulkan / Metal)         |
|-----------------------------------------------|---------------------------------------|-----------------------|---------------------------------|
| `subgroup.ballot_first_n(predicate, n)`       | yes                                   | yes                   | yes                             |
| `subgroup.ballot(predicate)`                  | yes (u32 zext'd to u64)               | yes (wave32 / wave64) | yes (uvec4 hi:lo packed to u64) |
| `subgroup.{all,any}_true(predicate)`          | yes (fast: maps to `__{all,any}_sync`) | yes                   | yes                             |
| `subgroup.all_equal(value)`                   | yes (fast: 1 shuffle + vote.all)      | yes                   | yes                             |
| `subgroup.lanemask_{lt,le,eq,gt,ge}(lane_id)` | yes                                   | yes                   | yes                             |

CUDA shortcut: `all_true` / `any_true` lower to a single `__all_sync(0xFFFFFFFF, p)` / `__any_sync(0xFFFFFFFF, p)` (one `vote.all` / `vote.any` instruction). Every other backend uses a portable `shuffle_xor` butterfly with no branch in the emitted IR.

`all_equal` always uses the broadcast-and-`all_true` form: every lane reads the value at the start of the subgroup via `shuffle`, compares it with its own value, and `all_true`-reduces the per-lane equality bit. Cost: `log2(subgroup_size) + 1` shuffles in the portable case, or `1 shuffle + 1 vote.all` on CUDA. We deliberately do *not* use `__match_all_sync` even on CUDA: it requires sm_70+, and it does bit-equality on floats, contradicting this op's documented `OpGroupNonUniformAllEqual` semantics (`NaN != NaN`, `+0.0 == -0.0`). Callers wanting bit-equality on floats should bit-cast to the same-width integer dtype before calling.

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

Every op above has a paired `_tiled` form that takes an extra `log2_size` template parameter and operates on independent `2**log2_size`-aligned tiles within the subgroup - see [Tiled variants](#tiled-variants).

The SPV-only no-arg reductions (`subgroup.reduce_mul` / `reduce_and` / `reduce_or` / `reduce_xor`, plus the original `reduce_add_tiled(value)` with no `log2_size`) have been removed in favour of the portable sized API. For reductions other than the ones listed above, build a sized helper on top of `shuffle_down` / `shuffle` following the same pattern as `reduce_add_tiled` / `reduce_all_add_tiled`.

## Semantics

All of these ops operate within a single subgroup: they do not move data through memory and do not synchronise across subgroups.

### `shuffle(value, index)`

Each lane returns the `value` held by the lane whose subgroup-local id equals `index`.

- `value` is a scalar in a register. Supported dtypes are 32-bit and 64-bit signed/unsigned ints and `f32`/`f64`. (64-bit types are split into two 32-bit shuffles on AMDGPU; CUDA dispatches to its native 64-bit helpers.)
- `index` is a `u32`. If `index` is out of range for the active subgroup the result is implementation-defined, so pass `subgroup.invocation_id()`-derived values or known-good lane ids.

### `shuffle_down(value, offset)`

Lane `i` returns the `value` held by lane `i + offset`. Lanes near the top of the subgroup - where `i + offset >= subgroup_size` - receive an implementation-defined value (typically their own `value`), so reduction patterns must only trust lane 0's final result, or mask out the out-of-range lanes.

- `value` and `offset` dtypes: same as `shuffle` above; `offset` is a `u32`.
- Maps to `__shfl_down_sync` on CUDA and `OpGroupNonUniformShuffleDown` on SPIR-V. On AMDGPU it is emulated with `ds_bpermute`; wave64 cross-half offsets (any `offset >= 32` for low-half lanes, or any non-zero `offset` for high-half lanes that lands across the SIMD32 boundary) go through the same `permlane64 + ds_bpermute + select` lowering as `shuffle` - see [AMDGPU wave64 cross-half lowering](#amdgpu-wave64-cross-half-lowering). These operations are added on both RDNA and CDNA.

### `shuffle_up(value, offset)`

Lane `i` returns the `value` held by lane `i - offset`. Lanes near the bottom of the subgroup - where `i - offset < 0` - receive an implementation-defined value (typically their own `value`), so the bottom `offset` lanes' results should be ignored or masked.

- Same dtype rules as `shuffle` / `shuffle_down`; `offset` is a `u32`.
- Maps to `__shfl_up_sync` on CUDA and `OpGroupNonUniformShuffleUp` on SPIR-V. On AMDGPU it is emulated with `ds_bpermute((lane - offset) * 4, value)`; wave64 cross-half cases go through the [AMDGPU wave64 cross-half lowering](#amdgpu-wave64-cross-half-lowering) (same `permlane64 + ds_bpermute + select` sequence as `shuffle` / `shuffle_down`). These operations are added on both RDNA and CDNA.

### `shuffle_xor(value, mask)`

Lane `i` returns the `value` held by lane `i ^ mask`. Convenient for butterfly patterns (used internally by `reduce_all_add`).

- Same dtype rules as `shuffle`; `mask` is a `u32`.
- Implemented portably as a `@qd.func` over `shuffle`: every backend that lowers `shuffle` therefore lowers `shuffle_xor` with no additional codegen path. Inlines at compile time into a single `shuffle(value, u32(invocation_id()) ^ mask)`.
- The XOR partner must be inside the active subgroup; behaviour outside that range is implementation-defined (same caveat as `shuffle`).

### `broadcast(value, index)`

Every lane in the subgroup returns the `value` held by the lane whose subgroup-local id equals `index`. Expresses intent ("read lane `index`") more directly than `shuffle(value, index)` and on backends with a dedicated broadcast may map to a cheaper instruction.

- Same dtype rules as `shuffle`.
- Maps to `__shfl_sync` on CUDA, `ds_bpermute` (plus a `permlane64`-driven cross-half select on wave64) on AMDGPU, and `OpGroupNonUniformBroadcast` on SPIR-V. See [AMDGPU wave64 cross-half lowering](#amdgpu-wave64-cross-half-lowering) for the wave64 mechanics. These operations are added on both RDNA and CDNA.
- **Important: on SPIR-V, `index` must be dynamically uniform** - the same value on every lane in the subgroup. Passing a per-lane varying `index` is undefined behavior, because `OpGroupNonUniformBroadcast` requires its `Id` operand to be dynamically uniform across the subgroup. On CUDA / AMDGPU, `index` may vary per lane and the call is identical to `shuffle(value, index)`. If you need a varying source lane, use `shuffle` directly.

### `broadcast_first(value)`

Every lane returns lane 0's `value`. Convenience wrapper for the common "read lane 0 from every lane" pattern.

- Same dtype rules as `broadcast`.
- Implemented portably as a `@qd.func` over `broadcast(value, qd.u32(0))`: every backend that lowers `broadcast` therefore lowers `broadcast_first`. The `0` index is trivially dynamically uniform, so the SPIR-V `OpGroupNonUniformBroadcast` requirement is satisfied. Inlines at compile time.

### Common to the data-movement ops

- All shuffles / broadcasts are issued under a full active mask on CUDA (`0xFFFFFFFF`). Call them from uniform control flow; calling from divergent control flow is undefined on most backends. (This means: every thread has to execute the shuffle.)
- Subgroup size is hard-coded per backend on the LLVM backends - 32 on CUDA, 64 on AMDGPU (wave64 is forced on every AMDGPU target including RDNA, see [supported_systems](supported_systems.md)) - so kernels can rely on those literal values. On SPIR-V it is queried at runtime via `subgroup.group_size()` (typically 32 on Vulkan compute on most GPUs).

### `invocation_id()`

Returns this lane's subgroup-local index - `0..subgroup_size - 1`. Used both as a lane id when computing a target lane for `shuffle` / `broadcast`, and as a per-lane identifier in cooperative algorithms.

- Returns `i32`.
- Available on every backend.

### `group_size()` / `log2_group_size()`

Return the subgroup size (and its base-2 log) in effect for the current `Program`, as plain Python `int`s. Callable from anywhere after `qd.init()` - both inside `@qd.kernel` / `@qd.func` bodies (where the int is folded into the IR as a literal) and from host scope (handy for setting up `block_dim` / grid shapes that match the subgroup width).

| Backend | Compile-time value |
|---|---|
| CUDA | `32` (hard-coded on every sm_30+ NVIDIA arch) |
| AMDGPU | `64` (every AMDGPU target is pinned to `+wavefrontsize64`) |
| SPIR-V (Vulkan / Metal) | `subgroupSize` probed from the live device at `qd.init()` |

Because the return type is a plain `int`, the value is usable as a `qd.template()` argument - that is how every full-subgroup op picks the right `log2_size` per backend without a runtime branch (each one is `<op>_tiled(v, log2_group_size())` under the hood; see [Tiled variants](#tiled-variants)). The value is fixed for the lifetime of the `Program`: Vulkan / Metal devices expose a single immutable `subgroupSize` and Quadrants never opts in to `VK_EXT_subgroup_size_control`, so two kernels launched under the same `qd.init()` always see the same number.

Per-backend lowering notes:

- **CUDA**: `Program::subgroup_size()` returns the constant `32` directly. Inside a kernel, that constant gets folded into the IR - no runtime warp-size query, no PTX instructions emitted for `group_size()` itself.
- **AMDGPU**: `Program::subgroup_size()` returns `64`. Quadrants pins every AMDGPU function to `+wavefrontsize64,-wavefrontsize32` (see [supported_systems](supported_systems.md)), so CDNA (gfx9xx, gfx940/942) keeps its native wave64 mode and RDNA (gfx10/11/12) - which would otherwise default to wave32 - is forced into wave64 too.
- **SPIR-V (Vulkan / Metal)**: probed once at device-creation time. Vulkan reads `VkPhysicalDeviceSubgroupProperties::subgroupSize`; Metal hard-codes `32` (every shipping Apple / AMD / Intel Mac GPU is 32-wide). The probed value is stored in `DeviceCapability::spirv_subgroup_size` and feeds both `Program::subgroup_size()` and the fe-ll cache key, so two SPIR-V devices with different subgroup widths get distinct cache entries.

`log2_group_size()` asserts the size is a power of two and returns `bit_length() - 1` (so `5` on every wave32 backend, `6` on AMDGPU). The assert is purely defensive - no shipping SPIR-V driver reports a non-power-of-two subgroup size, but the Vulkan spec permits it.

### Tiled variants

Every reduce / scan / vote op in this module has a paired `_tiled` form that takes an extra `log2_size` template parameter and runs `group_size() / (2**log2_size)` independent reductions / scans / votes in parallel - one per `2**log2_size`-aligned tile within the subgroup. The base op is the special case where the tile is the whole subgroup (i.e. `log2_size = log2_group_size()`).

With `log2_size = k`, the subgroup splits into tiles of `2**k` consecutive lanes each, and each tile does its own reduction completely independently of every other tile. The caller arranges `2**k <= group_size()` so every tile is full; a smaller `k` simply gives more, narrower tiles. It does **not** mean "only the first tile is active".

| `log2_size` | wave32 (CUDA / most Vulkan / Metal) | wave64 (AMDGPU - see [supported_systems](supported_systems.md)) |
| --- | --- | --- |
| 5 (tile = 32) | 1 tile: lanes 0-31 (= base op)             | 2 tiles: 0-31, 32-63 |
| 4 (tile = 16) | 2 tiles: 0-15, 16-31                       | 4 tiles: 0-15, 16-31, 32-47, 48-63 |
| 3 (tile = 8)  | 4 tiles of 8                               | 8 tiles of 8 |
| 0 (tile = 1)  | every lane is its own tile (no-op)         | same |

Why it composes exactly: the underlying `subgroup.shuffle` / `subgroup.shuffle_down` / `subgroup.shuffle_up` / `subgroup.shuffle_xor` ops address every lane by absolute lane id with no built-in notion of a tile. Tiling emerges from how the higher-level reductions / scans / votes *compose* those shuffles.

#### Supported `_tiled` ops

| Tiled op                                                             | Result placement  |
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

- **Broadcast-to-tile forms**: every lane in each tile holds that tile's result. Lanes in different tiles hold different results (their own tile's).
- **Tile-local lane-0 forms**: only the *tile-local* lane 0 holds the reduction. That's lane 0 alone with `log2_size=5` on wave32, lanes 0 and 32 with `log2_size=5` on wave64, lanes 0 / 16 / 32 / 48 with `log2_size=4` on wave64, etc. Other lanes hold partial reductions and should be treated as undefined. Use the `reduce_all_*_tiled` counterparts if you want every lane to see its tile's result.

`log2_size` is a `qd.template()` - a compile-time constant in `[0, log2_group_size()]`, i.e. `[0, 5]` on wave32 backends (CUDA, Vulkan / Metal, RDNA wave32) and `[0, 6]` on AMDGPU wave64 (every non-segmented `_tiled` op uses `shuffle_xor` / `shuffle_up` / `shuffle_down` butterflies that span the full wave at `log2_size == log2_group_size()`; `segmented_reduce_*_tiled` reaches `log2_size == 6` via a dedicated u64-bitmask path on wave64). The caller must ensure `2**log2_size <= group_size()`; passing a larger value silently computes the wrong result on most backends and there is no runtime check. Backends that do not support a given op (`reduce_add_tiled` and friends on `*` backends, see the per-op tables) raise a `qd.static_assert` at compile time.

#### Lowering

Each base op is a one-line wrapper around its `_tiled` form: `reduce_add(v)` is `return reduce_add_tiled(v, log2_group_size())` (and similarly for every other op). The wrappers are plain Python (not `@qd.func`), so `log2_group_size()` is resolved at compile time to a Python `int` that flows into the underlying `@qd.func`'s `template()` parameter. The result is the same IR as a hand-written `_tiled` call that hard-codes `log2_size = 5` on wave32 backends or `log2_size = 6` on AMDGPU - no `arch` branch in user code, no runtime overhead vs. the per-backend hand-tuned form.

`segmented_reduce_*_tiled` accepts `log2_size <= 6`, with a compile-time-selected u64-bitmask path for `log2_size == 6` (reachable only on AMDGPU wave64). The u32-bitmask path used for `log2_size <= 5` is bitwise identical to the historical implementation, so CUDA / Metal / Vulkan-wave32 callers see zero overhead from the wave64 support.

### `elect()`

Returns `1` on lane 0 of every subgroup and `0` on every other lane. Useful for "exactly one lane does X" patterns where you don't care which lane does it - e.g. emitting a single global write per subgroup.

- Implemented portably as a `@qd.func` wrapper: `i32(invocation_id() == 0)`. Inlines at compile time into a single compare + zero-extend on every backend.
- This narrows the SPIR-V `OpGroupNonUniformElect` semantics, which would otherwise be free to pick any *active* lane. Under the documented uniform-CF + all-lanes-active contract for `qd.simt.subgroup` the distinction is invisible (lane 0 is always active and is a legal choice), and pinning the elected lane down keeps the behaviour identical across backends.

### `sync()`

Subgroup-scope thread-converging barrier - every lane in the subgroup must reach the call before any lane proceeds.

- Lowers to:
  - **SPIR-V**: `OpControlBarrier(Subgroup, Subgroup, 0)`.
  - **CUDA**: `__syncwarp(0xFFFFFFFF)` (`nvvm.bar.warp.sync`). Reconverges lanes that may have diverged under independent thread scheduling on Volta+; under uniform CF on Pascal and earlier this is effectively a no-op but is still legal.
  - **AMDGPU**: `llvm.amdgcn.wave.barrier`. Acts as a compiler reordering barrier on GCN (where waves are lockstep) and as a real wave-scope hardware barrier on RDNA.
- Caller contract on every backend: call from uniform control flow with all lanes active. Calling from divergent control flow has implementation-defined behaviour (CUDA's `nvvm.bar.warp.sync` will deadlock if the mask does not match the active set; AMDGPU's `wave.barrier` is a no-op on most chips so divergent calls silently pass through).
- The legacy name `subgroup.barrier()` is still available as a deprecated alias. It forwards to `sync()` and emits a `DeprecationWarning` on first use; prefer the new name in new code.

### `mem_fence()`

Subgroup-scope memory fence - orders memory operations within the subgroup without requiring thread convergence.

- Lowers to:
  - **SPIR-V**: `OpMemoryBarrier(Subgroup, AcquireRelease | UniformMemory | WorkgroupMemory)`.
  - **CUDA**: `__threadfence_block()` (`nvvm.membar.cta`) - workgroup-scope, see the `**` footnote in the matrix above.
  - **AMDGPU**: LLVM `fence syncscope("workgroup") seq_cst` - workgroup-scope, same caveat.
- Caller contract on every backend: call from uniform control flow with all lanes active. Calling from divergent control flow has implementation-defined behaviour (same caveats as `sync()`).
- The legacy name `subgroup.memory_barrier()` is still available as a deprecated alias. It forwards to `mem_fence()` and emits a `DeprecationWarning` on first use; prefer the new name in new code.

### `reduce_add(value)`

Sums `value` across every lane in the subgroup via a `shuffle_down` tree. The result is valid in **lane 0**; other lanes hold partial sums and should be considered undefined. Tiled variant: `reduce_add_tiled(value, log2_size)` runs the same tree independently on each `2**log2_size`-aligned tile - see [Tiled variants](#tiled-variants).

- The reduction works on any type that supports `+` and `shuffle_down`; in practice this means i32, u32, f32, f64, i64, u64.
- Decorated with `@qd.func` and inlined into the calling kernel - there is no kernel-launch overhead and no separate symbol to link. The body unrolls into exactly `log2_group_size()` `shuffle_down + add` pairs in the calling kernel's IR, with no runtime loop overhead.

### `reduce_all_add(value)`

Same sum as `reduce_add`, but broadcast to **every lane** in the subgroup. Implemented as a butterfly using `shuffle` with `lane ^ mask`, `mask` stepping through `1, 2, 4, ..., 2**(log2_group_size()-1)`. Tiled variant: `reduce_all_add_tiled(value, log2_size)` - see [Tiled variants](#tiled-variants).

- Use this when every lane needs the reduction result (e.g. to divide by the sum, or to branch on it uniformly). It costs exactly the same number of shuffles as `reduce_add` but leaves the answer in all lanes, so it replaces a `reduce_add` + `shuffle`/broadcast pair.
- Uses `subgroup.shuffle` under the hood.

### `reduce_{min,max}(value)`

Min / max of `value` across every lane in the subgroup via a `shuffle_down` tree. Result valid in **lane 0**; other lanes hold partial mins / maxes. Use `reduce_all_min` / `reduce_all_max` if every lane needs the answer. Tiled variants: `reduce_min_tiled(value, log2_size)` / `reduce_max_tiled(value, log2_size)` - see [Tiled variants](#tiled-variants).

- Accepts integer (`i32`, `u32`, `i64`, `u64`) and float (`f32`, `f64`) dtypes. Lowers via `qd.min` / `qd.max`, which dispatch to the backend's native min/max intrinsic.
- Float NaN handling is implementation-defined: PTX uses `fminnm` / `fmaxnm` (NaN-suppressing), AMDGPU uses `llvm.minnum` / `llvm.maxnum` (NaN-suppressing), SPIR-V uses `OpFMin` / `OpFMax` (NaN-propagating in some drivers). Avoid NaN inputs if you need a portable result.
- Decorated with `@qd.func` and inlined into the calling kernel.

### `reduce_all_{min,max}(value)`

Same min / max as `reduce_min` / `reduce_max`, but broadcast to **every lane** in the subgroup via a `shuffle_xor` butterfly. Same number of shuffles as the lane-0 forms. Tiled variants: `reduce_all_min_tiled` / `reduce_all_max_tiled` - see [Tiled variants](#tiled-variants).

- Use over `reduce_min` + broadcast / `reduce_max` + broadcast when every lane needs the result (e.g. to subtract the subgroup min, or to clamp to the subgroup max).
- Same dtypes and float-NaN caveat as `reduce_min` / `reduce_max`.

### `segmented_reduce_{add,min,max}(value, head_flag)`

Per-lane inclusive scan under `+` / `min` / `max` that resets at every non-zero `head_flag`, across the entire subgroup. Lane `i` returns the scan of `value[head_below..i + 1]`, where `head_below` is the largest lane index `<= i` whose `head_flag` is non-zero. If no such lane exists, lane 0 is treated as an implicit head, so the result is the inclusive scan from lane 0 to lane `i`. Tiled variants: `segmented_reduce_add_tiled(value, head_flag, log2_size)` (and `_min` / `_max`) - see [Tiled variants](#tiled-variants).

- `value` is any type supporting the operator (`+` and `shuffle_up` for `_add`; `qd.min`/`qd.max` and `shuffle_up` for `_min`/`_max`). `head_flag` is any integer scalar; the lowering tests `head_flag != 0`, so non-binary truthy values (e.g. `7`, `42`) work.
- Implementation: one `subgroup.ballot(head_flag != 0)` to materialise a `u64` of head positions, then a Hillis-Steele inclusive scan bounded by `distance >= offset` where `distance = lane - segment_head`. A compile-time branch in `_segment_head_distance_tiled` picks between two paths:
  - **`log2_size <= 5`** - u32-bitmask path. Shifts the relevant 32-lane half of the ballot down to bits 0..31 and runs the bit-mask + `clz` arithmetic in half-local coordinates (`lane_in_half = lane - half_base`). Half-local `distance` equals absolute `lane - segment_head_abs` because both terms are offset by the same `half_base`, so the downstream `shuffle_up`'s `distance >= offset` guard still works in absolute terms. This is the only path on wave32 backends - it compiles to identical IR to the historical wave32-only implementation, so CUDA / Metal / Vulkan callers see no perf regression from the wave64 support.
  - **`log2_size == 6`** - u64-bitmask path. Works in absolute lane coordinates with the full `u64` ballot, an OR-injected virtual head at lane 0 to guarantee a non-zero `lower`, and a `clz(u64)` for the segment head. Costs one extra `u64` shift + `u64 clz` vs the u32 path; only reachable when `group_size() == 64` (i.e. AMDGPU), so the entire branch is dead-code-eliminated at every `log2_size <= 5` call site.
- No identity element is involved at all - the per-lane `distance >= offset` guard ensures the scan never reaches across a segment boundary, so a partner from another segment is never combined with the local value (i.e. the implementation doesn't need a "what to combine with at the segment head" sentinel the way `exclusive_min` / `exclusive_max` do for lane 0).
- Cost: `1 ballot + 1 clz + log2_group_size() shuffles + log2_group_size() ops`, fully unrolled into the calling kernel's IR. Same shape as `inclusive_add` / `inclusive_min` / `inclusive_max`, plus one ballot and one `clz` for the per-lane segment-head lookup.
- Float NaN handling for `_min` / `_max` is implementation-defined (same caveat as `reduce_min` / `reduce_max`): PTX uses `fminnm` / `fmaxnm`, AMDGPU uses `llvm.minnum` / `llvm.maxnum`, SPIR-V uses `OpFMin` / `OpFMax`. Avoid NaN inputs if you need a portable result.
- AMDGPU note (`*` in the table): `shuffle_up` goes through `ds_bpermute` (~50 cycle latency), same as the other reductions.

### `inclusive_{add,mul,min,max,and,or,xor}(value)`

Per-lane inclusive scan across the entire subgroup, under the binary operator named by the suffix. Lane `i` returns `v[0] op v[1] op ... op v[i]`. Tiled variants: `inclusive_*_tiled(value, log2_size)` run the same scan independently on each `2**log2_size`-aligned tile - see [Tiled variants](#tiled-variants).

- The body unrolls into exactly `log2_group_size()` `shuffle_up + op` pairs in the calling kernel's IR, with no runtime loop overhead.
- `_add`, `_mul`, `_min`, `_max` accept integer and float dtypes (`i32`, `u32`, `i64`, `u64`, `f32`, `f64`); `_and`, `_or`, `_xor` accept integer dtypes only.
- All seven share a single `@qd.func` Hillis-Steele scan helper (`_inclusive_scan_tiled`); each public op is a one-line wrapper that supplies the binary operator. The shuffle is in uniform CF (every lane participates); only the per-lane reduce step is conditional, which is allowed by the `shuffle_up` contract on every backend.
- Decorated with `@qd.func` and inlined into the calling kernel - there is no kernel-launch overhead and no separate symbol to link.
- AMDGPU note (`*` in the table): same `ds_bpermute` cost as `shuffle_up` - roughly tens of cycles per step × `log2_group_size()` steps. Hardware-accelerated `OpGroupNonUniformInclusiveScan` on SPIR-V is no longer used, even on backends that supported it (Vulkan, Metal); the trade-off is a uniform implementation across backends with predictable cost.

### `exclusive_{add,mul,min,max,and,or,xor}(value)`

Per-lane exclusive scan across the entire subgroup, under the binary operator named by the suffix. Lane `i` (with `i > 0`) returns `v[0] op v[1] op ... op v[i - 1]`. Lane 0 returns the operator's identity in `value`'s dtype. Tiled variants: `exclusive_*_tiled(value, log2_size)` - see [Tiled variants](#tiled-variants).

- The body unrolls into the inclusive scan (`log2_group_size()` shuffle+op pairs) plus one extra `shuffle_up` and a per-lane select.
- The lane-0 identity is auto-derived from `value`'s dtype at compile time, with no runtime cost:
  - `_add` / `_or` / `_xor`: `0` in `value`'s dtype (built as `value - value` / `value ^ value`).
  - `_mul`: `1` in `value`'s dtype (built as `value - value + 1`).
  - `_and`: all-bits-set in `value`'s dtype (built as `~(value ^ value)`).
  - `_min`: `+inf` for real dtypes, `np.iinfo(dtype).max` for integer dtypes.
  - `_max`: `-inf` for real dtypes, `np.iinfo(dtype).min` for integer dtypes (`0` for unsigned).
- `_add`, `_mul`, `_min`, `_max` accept integer and float dtypes; `_and`, `_or`, `_xor` accept integer dtypes only.
- The first five (`_add`, `_mul`, `_and`, `_or`, `_xor`) build the identity from pure arithmetic on `value` and stay inside a single shared `@qd.func` helper (`_exclusive_scan_tiled`). `_min` and `_max` cannot manufacture `+inf` / `INT_MAX` from arithmetic on a value of unknown dtype, so they are plain Python wrappers that introspect `value`'s dtype at compile time and emit a typed-constant identity Expr before calling the same shared helper - the identity is a compile-time constant in the generated IR, so the cost is identical to the other five ops.
- The shared `_exclusive_scan_tiled` helper runs the inclusive scan, shifts the result up by one lane via `shuffle_up`, and substitutes the identity at lane 0. The lane-0 substitution is required because `shuffle_up` with offset 1 is implementation-defined at lane 0 (and `OpGroupNonUniformShuffleUp` calls it undefined outright).
- AMDGPU performance note (`*` in the table): same `ds_bpermute` cost as `shuffle_up`. Cost is one inclusive scan plus one extra `shuffle_up` and a select.

### `ballot_first_n(predicate, n)`

Returns a `u32` bitmask whose bit `i` is set iff `i < n` AND lane `i`'s `predicate` is non-zero. Bits `>= n` are always zero.

- `predicate` is any integer scalar; the lowering tests `predicate != 0` internally, so non-binary truthy values work.
- `n` is a `qd.template()` compile-time constant in `[1, 32]`. Pass `n = 32` for "ballot over the 32 representable lanes" - the most common case, used internally by `segmented_reduce_*` and other ballot consumers.
- Backend lowering:
  - **CUDA**: `__ballot_sync(0xFFFFFFFF, predicate)`. Warps are always 32 lanes, so the `u32` result naturally packs every lane.
  - **AMDGPU**: `llvm.amdgcn.ballot.i64` followed by `trunc to i32`. Packs lanes 0..31 into the result; on wave64 lanes 32..63's predicates are explicitly discarded by the truncate, matching the `n <= 32` contract. The `i64 + trunc` form is a workaround for an LLVM AMDGPU isel bug - `ballot.i32` is documented as well-defined on wave64 (PR [llvm/llvm-project#71556](https://github.com/llvm/llvm-project/pull/71556)) but in practice still fails `Cannot select` on gfx942 in LLVM 20 / 22 for non-constant predicates. The workaround costs nothing - both forms produce the same single `v_cmp_*_e64` plus a low-half store.
  - **SPIR-V**: `OpGroupNonUniformBallot` returns a `uvec4`; we extract component 0, which by spec contains the ballot bits for lanes 0..31.
- For `n < 32` we mask the predicate by `lane < n` before issuing the ballot, so bits `[n, 32)` of the result are forced to zero regardless of those lanes' actual predicate values. At `n == 32` the masking is provably a no-op on every backend (lanes `>= 32` are either non-existent on wave32 or already not represented in the `u32` result on wave64), so the masking is elided at compile time and the call lowers to a single ballot intrinsic.
- Caller contract: uniform CF + all lanes active. Calling from divergent control flow has implementation-defined behaviour (CUDA's `__ballot_sync` will deadlock if the active mask doesn't match `0xFFFFFFFF`).
- Useful for stream compaction over the first 32 lanes, the wave32 path of `segmented_reduce_*` (which uses the u32-bitmask form internally for `log2_size <= 5`), and any pattern that wants `clz` / `popcount` / `ffs` over a per-lane predicate within a `u32`. For full-subgroup ballots on AMDGPU wave64 use `ballot` (returns a `u64`).

### `ballot(predicate)`

Returns a `u64` bitmask covering the entire subgroup. Bit `i` is set iff lane `i`'s `predicate` is non-zero, for all `i` in `[0, subgroup_size)`.

- On wave32 backends (CUDA, RDNA wave32, most Vulkan / Metal) the high 32 bits of the result are always zero, since lanes `>= 32` do not exist. On wave64 backends (AMDGPU CDNA, GFX9, RDNA explicit-wave64) all 64 bits are meaningful.
- Backend lowering:
  - **CUDA**: `__ballot_sync(0xFFFFFFFF, p)` then zero-extend the `i32` result to `i64`.
  - **AMDGPU**: `llvm.amdgcn.ballot.i64`. Returns the full 64-bit ballot on wave64; on wave32 the AMDGPU backend lowers it to the wave32 ballot zero-extended to 64 bits, so the API stays uniform across wavefront modes.
  - **SPIR-V**: extract components 0 and 1 of the `OpGroupNonUniformBallot` `uvec4` (lanes 0..31 and 32..63 respectively) and pack them into a `u64` via `u64(hi) << 32 | u64(lo)`.
- Use this when you need a subgroup-wide population count, prefix mask, or compaction that has to cover more than 32 lanes. Use `ballot_first_n(p, 32)` instead if you only care about the first 32 lanes - it's one register cheaper on wave64 and avoids the wider `u64` result type.
- Caller contract: uniform CF + all lanes active.

### `{all,any}_true(predicate)`

Per-lane AND-reduction (`all_true`) or OR-reduction (`any_true`) of `predicate != 0` across every lane in the subgroup. Returns `i32` (`0` or `1`), broadcast to every lane. Tiled variants: `all_true_tiled(predicate, log2_size)` / `any_true_tiled(predicate, log2_size)` - see [Tiled variants](#tiled-variants).

- `predicate` is any scalar dtype. The op compares `predicate != 0` at the start, so e.g. `subgroup.all_true(some_int_field[i])` is well-formed.
- CUDA shortcut: lowers to a single `__all_sync(0xFFFFFFFF, p)` / `__any_sync(0xFFFFFFFF, p)` via the `cuda_all_sync_i32` / `cuda_any_sync_i32` runtime helpers (one `vote.all` / `vote.any` instruction). The same shortcut is selected for `all_true_tiled` / `any_true_tiled` at `log2_size == 5`. The shortcut is selected at compile time via `qd.static()` on the active arch, so the IR contains exactly the intrinsic call and no branch.
- Portable fallback (every other backend, and CUDA for partial-warp tiles): `shuffle_xor` butterfly - `log2_group_size()` shuffles plus `log2_group_size()` ANDs (or ORs), fully unrolled into the calling kernel's IR. Same shape as `reduce_all_add`.

### `all_equal(value)`

Returns `i32(1)` on every lane iff every lane in the subgroup has the same `value` (under the backend's native `==`), else `i32(0)`. Tiled variant: `all_equal_tiled(value, log2_size)` - see [Tiled variants](#tiled-variants).

- `value` is any scalar dtype. Equality is the backend's native `==`: for floats this means `NaN != NaN` (a subgroup with any `NaN` returns `0`) and `+0.0 == -0.0`, matching SPIR-V `OpGroupNonUniformAllEqual`. Callers wanting bit-equality on floats should `qd.bit_cast` to the same-width integer dtype first.
- Implementation: each lane reads the value at lane 0 via `shuffle`, compares to its own `value`, and `all_true`-reduces the equality bit. Inherits the CUDA shortcut transitively from `all_true`.
- Cost: 1 shuffle + 1 `vote.all` on CUDA; 1 shuffle + `log2_group_size()` butterfly shuffles otherwise. We deliberately do *not* use `__match_all_sync` on CUDA: it requires sm_70+ and uses bit-equality for floats, which would contradict this op's documented semantics.

### `lanemask_{lt,le,eq,gt,ge}(lane_id)`

Closed-form `u32` lane-mask constants parametrised by a lane id. Bit `i` of the result follows the relation in the suffix:

| Op             | Bit `i` set iff | Closed form                            |
|----------------|-----------------|----------------------------------------|
| `lanemask_lt`  | `i <  lane_id`  | `(1 << lane_id) - 1`                   |
| `lanemask_le`  | `i <= lane_id`  | `lt(lane_id) \| eq(lane_id)`            |
| `lanemask_eq`  | `i == lane_id`  | `1 << lane_id`                         |
| `lanemask_gt`  | `i >  lane_id`  | `~le(lane_id)`                         |
| `lanemask_ge`  | `i >= lane_id`  | `~lt(lane_id)`                         |

- `lane_id` is any integer scalar. Pass `subgroup.invocation_id()` to get the classic CUDA built-in form (current lane's mask), or any other expression to query an arbitrary lane's mask. The op is pure arithmetic - no shuffle, no ballot - so per-lane-varying `lane_id` works the same as a uniform one.
- Returns `u32`. Bit 0 corresponds to lane 0, bit 31 to lane 31.
- Caller contract: `lane_id` must be in `[0, 31]` (matching the `u32` return type, which represents 32 lanes). Passing `lane_id == 32` triggers an undefined-behaviour shift on most backends.
- Implemented portably as a `@qd.func` over `<<`, `-`, `|`, `~`. Inlines at compile time into 1-3 ALU ops on every backend.
- AMDGPU CDNA wave64 caveat: only the low 32 lanes are representable in this op (the return type is `u32`). If you need a mask covering all 64 wave64 lanes, use `subgroup.ballot` instead - it returns a `u64` and includes lanes 32..63.

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

After the kernel, every lane in a subgroup holds the original value of its lane 0. `subgroup.broadcast(a[i], qd.u32(0))` is interchangeable here.

### Ballot: count how many lanes satisfy a condition

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

### Sum 32 lanes with `reduce_add_tiled`

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

`5` is `log2_size`; `2**5 == 32` matches the block dim. The body of `reduce_add_tiled` unrolls at compile time into five `shuffle_down + add` pairs, so the generated IR is identical to a hand-written tree reduction.

### Broadcast the sum to all lanes with `reduce_all_add_tiled`

When every lane needs the reduction result - e.g. to normalise by the sum - use the butterfly variant. No follow-up broadcast needed:

```python
@qd.kernel
def normalize32(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        total = subgroup.reduce_all_add_tiled(a[i], 5)
        a[i] = a[i] / total
```

Every lane in each group of 32 sees the same `total`.

### Partial-subgroup reductions

`log2_size` does not have to match the full subgroup. Sum groups of 8 with `reduce_add_tiled(v, 3)` or groups of 16 with `reduce_all_add_tiled(v, 4)`; the caller just ensures `2**log2_size <= group_size()` (so `log2_size <= 5` on CUDA / Metal / Vulkan-wave32, `<= 6` on AMDGPU wave64). Use the bare `reduce_add(v)` / `reduce_all_add(v)` form when you want "the whole subgroup" without hard-coding the limit.

### Inclusive scan with `inclusive_add_tiled`

```python
@qd.kernel
def cumsum(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        a[i] = subgroup.inclusive_add_tiled(a[i], 5)
```

After the call, lane `k` (within each group of 32) holds `a[group_start] + a[group_start+1] + ... + a[k]`. The `5` is `log2_size`; `2**5 == 32` matches the block dim. The body unrolls at compile time into five `shuffle_up + add` pairs. Use a smaller `log2_size` to scan over partial-subgroup groups (e.g. `inclusive_add_tiled(v, 3)` produces independent prefix sums in groups of 8).

### AMDGPU wave64 cross-half lowering

AMDGPU `ds_bpermute_b32` - the LDS-routed permute that Quadrants uses to lower `shuffle`, `shuffle_down`, and `shuffle_up` - has a hardware quirk on RDNA (gfx10/11/12, e.g. RX 7900 XTX): its lane-id operand is **SIMD32-scoped**. On a wave64 RDNA wave the 64 lanes execute as two SIMD32 clusters; `ds_bpermute` on those chips can only address lanes inside the requesting lane's own SIMD32 half. CDNA (gfx9xx, MI200/MI300) keeps the wave on a single SIMD64, so `ds_bpermute` there is wave-wide and the quirk does not exist.

To make wave64 `shuffle` / `shuffle_down` / `shuffle_up` behave consistently across RDNA and CDNA, Quadrants always lowers cross-half-capable shuffles through this 3-op sequence:

```
swapped = v_permlane64_b32 value         # swap the two SIMD32 halves of the wave
lo      = ds_bpermute_b32 (lane*4), value
hi      = ds_bpermute_b32 (lane*4), swapped
result  = ((target_lane ^ self_lane) & 32) ? hi : lo
```

The two `ds_bpermute_b32` reads run in parallel - one reads the original payload (correct when target is in the same SIMD32 half), the other reads the `permlane64`-swapped payload (correct when the target is in the other half) - and a per-lane select picks between them based on whether the target crosses the 32-lane boundary. On CDNA the cross-half branch is dead, but the cost is one extra `v_permlane64_b32` (still well-defined and free) and one `v_cndmask_b32` - no measurable hit. On RDNA wave64 this is the only correct lowering.

One subtlety worth knowing about (mostly for anyone reading the generated IR): the lane-id operand to `ds_bpermute` is wrapped in an empty `+v` inline-asm fence inside the runtime helper. Without that fence, LLVM's AMDGPU backend can decide a compile-time-constant or otherwise uniform lane-id is "uniform across the wave" and silently lower the call to a `v_readlane_b32`-style instruction that addresses lanes 0..31 **wave-globally** rather than SIMD32-locally. That would break cross-half shuffles whose target lane is a literal (`broadcast(v, 47)`, `shuffle(v, qd.u32(40))`, etc.). The fence costs zero - same instruction shape on every path - and pins the lowering to a real `ds_bpermute_b32` so the SIMD-local semantics our `permlane64` pairing relies on always hold.

## Performance notes

- Shuffles are register-to-register on CUDA (`__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`) and on SPIR-V where the GPU has hardware support - typically a handful of cycles, no memory traffic.
- AMDGPU `shuffle`, `shuffle_down`, and `shuffle_up` all go through `ds_permute` / `ds_bpermute` (LDS-routed, roughly tens of cycles). On wave64 the lowering issues two parallel `ds_bpermute_b32` reads plus a `v_permlane64_b32` swap and a per-lane select to handle cross-half shuffles correctly on RDNA - see [AMDGPU wave64 cross-half lowering](#amdgpu-wave64-cross-half-lowering). The two `ds_bpermute` reads issue in parallel, so the latency is the same as a single read; the `permlane64` and `cndmask` add a few extra cycles.
- `shuffle_xor` and `broadcast_first` are `@qd.func` wrappers over `shuffle` / `broadcast` and inline at compile time, so on every backend they cost exactly the same as the underlying op.
- Both `ballot_first_n` and `ballot` lower to a single hardware instruction on every backend - one cycle on CUDA (`__ballot_sync`), one instruction on AMDGPU (a single `v_cmp_*_e64` populating the wavefront-width SETCC, then a low-half store for `ballot_first_n`), and `OpGroupNonUniformBallot` on SPIR-V (extract one or two components of the result `uvec4`). At `n == 32` `ballot_first_n` elides the predicate-masking step entirely; at `n < 32` it inserts one extra multiply on the predicate.
- `reduce_add` and `reduce_all_add` both issue exactly `log2_group_size()` shuffles and `log2_group_size()` adds per call (5 on wave32, 6 on AMDGPU wave64). No barriers, no shared memory, no launch overhead (they inline). The same holds for the `_tiled` form at any `log2_size`.
- Pick `reduce_all_add` over `reduce_add + broadcast` when you need the result in every lane - same cost, one fewer shuffle.
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.
- All seven `inclusive_*` ops are `@qd.func` Hillis-Steele scans; cost is exactly `log2_group_size()` shuffle+op pairs, the same as a hand-rolled CUDA warp scan, on every backend. Hardware-accelerated `OpGroupNonUniformInclusiveScan` on SPIR-V is no longer used - the cost difference vs. a portable shuffle tree is small in practice, and the uniform implementation makes performance predictable across CUDA, AMDGPU, and SPIR-V.

## Related

- [tile](tile.md) - `Tile16x16` and `Tile32x32` build on `subgroup.shuffle` to implement register-resident matrix tiles.
