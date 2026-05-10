# Subgroup primitives

Subgroup operations let threads within the same subgroup (warp on NVIDIA, wave on AMD, subgroup on Vulkan / Metal) cooperate directly — exchanging register values, voting on predicates, scanning, and electing a leader — without going through shared memory or block barriers. They are the building block for fast in-warp data exchange and are used internally by `Tile16x16` (see [tile16](tile16.md)).

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

\* AMDGPU `shuffle_down` / `shuffle_up` (and therefore `reduce_add`, which is built on `shuffle_down`) are currently emulated via `ds_bpermute` (~50 cycle latency).

`shuffle_xor` and `broadcast_first` are portable `@qd.func` wrappers on top of `shuffle` / `broadcast` (`shuffle_xor(value, mask)` ≡ `shuffle(value, lane ^ mask)`; `broadcast_first(value)` ≡ `broadcast(value, qd.u32(0))`). They inline at trace time and run wherever the underlying op runs.

### Identification and control

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|------|--------|-------------------------|
| `subgroup.invocation_id()`                  | yes  | yes    | yes                     |
| `subgroup.group_size()`                     | yes  | yes    | yes                     |
| `subgroup.elect()`                          | yes  | yes    | yes                     |
| `subgroup.sync()`                           | yes  | yes    | yes                     |
| `subgroup.mem_fence()`                      | yes\*\* | yes\*\* | yes                  |

Naming note: two of the names above were recently renamed to align with the project's naming conventions across scopes:

- `subgroup.barrier()` has been renamed to `subgroup.sync()` (matching `block.sync()`).
- `subgroup.memory_barrier()` has been renamed to `subgroup.mem_fence()` (matching the planned `block.mem_fence()` and `grid.mem_fence()`).

The old names remain as deprecated aliases that emit a `DeprecationWarning` on first use and forward to the new ones; they will be removed in a future release. The rest of this page uses the new names.

\*\* `mem_fence()` lowers to a workgroup-scope fence on CUDA (`__threadfence_block()`, via `nvvm.membar.cta`) and AMDGPU (LLVM `fence syncscope("workgroup") seq_cst`). Both are over-strict for the subgroup-scope ask but are correct: a workgroup-scope fence orders memory as observed by the whole workgroup, of which the subgroup is a strict subset. A future change can tighten these to true wave-scope fences if a measurable cost shows up.

### Voting and predicate ops

`all_true` / `any_true` / `all_equal` take a `log2_size` template parameter and reduce over each `2**log2_size` group of consecutive lanes, broadcasting the `i32` (`0` or `1`) result to every lane in the group. Same shape as `reduce_all_add` / `inclusive_*` / `exclusive_*`. The two `ballot` variants are full-subgroup only: `ballot_first_n(predicate, n)` returns a `u32` covering lanes `[0, n)` (with `n` a compile-time constant `<= 32`), and `ballot_full_subgroup(predicate)` returns a `u64` covering every lane in the subgroup (32 lanes on wave32, 64 on wave64).

| Op                                          | CUDA          | AMDGPU | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|---------------|--------|-------------------------|
| `subgroup.ballot_first_n(predicate, n)`     | yes           | yes    | yes                     |
| `subgroup.ballot_full_subgroup(predicate)`  | yes (u32 zext'd to u64) | yes (wave32 / wave64) | yes (uvec4 hi:lo packed to u64) |
| `subgroup.all_true(predicate, log2_size)`   | yes (fast at `log2_size==5`) | yes | yes |
| `subgroup.any_true(predicate, log2_size)`   | yes (fast at `log2_size==5`) | yes | yes |
| `subgroup.all_equal(value, log2_size)`      | yes (fast at `log2_size==5`, transitively via `all_true`) | yes | yes |
| `subgroup.lanemask_lt(lane_id)`             | yes           | yes    | yes                     |
| `subgroup.lanemask_le(lane_id)`             | yes           | yes    | yes                     |
| `subgroup.lanemask_eq(lane_id)`             | yes           | yes    | yes                     |
| `subgroup.lanemask_gt(lane_id)`             | yes           | yes    | yes                     |
| `subgroup.lanemask_ge(lane_id)`             | yes           | yes    | yes                     |

CUDA shortcut: when `log2_size == 5` (full warp), `all_true` / `any_true` lower to a single `__all_sync(0xFFFFFFFF, p)` / `__any_sync(0xFFFFFFFF, p)` (one `vote.all` / `vote.any` instruction). The shortcut is selected at trace time via `qd.static()` on `impl.current_cfg().arch` and the compile-time `log2_size`, so partial-warp uses (and every other backend) cleanly fall back to a portable `shuffle_xor` butterfly with no branch in the emitted IR.

`all_equal` always uses the broadcast-and-`all_true` form: every lane reads the value at the start of its group via `shuffle`, compares it with its own value, and `all_true`-reduces the per-lane equality bit. Cost: `1 + log2_size` shuffles in the portable case, or `1 shuffle + 1 vote.all` on CUDA at full-warp. We deliberately do *not* use `__match_all_sync` even on CUDA: it requires sm_70+, and it does bit-equality on floats, contradicting this op's documented `OpGroupNonUniformAllEqual` semantics (`NaN != NaN`, `+0.0 == -0.0`). Callers wanting bit-equality on floats should bit-cast to the same-width integer dtype before calling.

### Reductions and scans

`reduce_add`, `reduce_all_add`, and all seven `inclusive_*` and `exclusive_*` ops take a `log2_size` parameter.

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                       |
|---------------------------------------------|------|--------|-------------------------|------------------------------|
| `subgroup.reduce_add(v, log2_size)`         | yes  | yes\*  | yes                     | any type supporting `+`      |
| `subgroup.reduce_all_add(v, log2_size)`     | yes  | yes    | yes                     | any type supporting `+`      |
| `subgroup.reduce_min(v, log2_size)`         | yes  | yes\*  | yes                     | integer + float              |
| `subgroup.reduce_max(v, log2_size)`         | yes  | yes\*  | yes                     | integer + float              |
| `subgroup.reduce_all_min(v, log2_size)`     | yes  | yes    | yes                     | integer + float              |
| `subgroup.reduce_all_max(v, log2_size)`     | yes  | yes    | yes                     | integer + float              |
| `subgroup.segmented_reduce_add(v, head_flag, log2_size)` | yes | yes\* | yes              | any type supporting `+`      |
| `subgroup.segmented_reduce_min(v, head_flag, log2_size)` | yes | yes\* | yes              | integer + float              |
| `subgroup.segmented_reduce_max(v, head_flag, log2_size)` | yes | yes\* | yes              | integer + float              |
| `subgroup.inclusive_add(v, log2_size)`      | yes  | yes\*  | yes                     | integer + float             |
| `subgroup.inclusive_mul(v, log2_size)`      | yes  | yes\*  | yes                     | integer + float             |
| `subgroup.inclusive_min(v, log2_size)`      | yes  | yes\*  | yes                     | integer + float             |
| `subgroup.inclusive_max(v, log2_size)`      | yes  | yes\*  | yes                     | integer + float             |
| `subgroup.inclusive_and(v, log2_size)`      | yes  | yes\*  | yes                     | integer                      |
| `subgroup.inclusive_or(v, log2_size)`       | yes  | yes\*  | yes                     | integer                      |
| `subgroup.inclusive_xor(v, log2_size)`      | yes  | yes\*  | yes                     | integer                      |
| `subgroup.exclusive_add(v, log2_size)`      | yes  | yes\*  | yes                     | integer + float             |
| `subgroup.exclusive_mul(v, log2_size)`      | yes  | yes\*  | yes                     | integer + float             |
| `subgroup.exclusive_min(v, log2_size, identity)` | yes | yes\* | yes                  | integer + float             |
| `subgroup.exclusive_max(v, log2_size, identity)` | yes | yes\* | yes                  | integer + float             |
| `subgroup.exclusive_and(v, log2_size)`      | yes  | yes\*  | yes                     | integer                      |
| `subgroup.exclusive_or(v, log2_size)`       | yes  | yes\*  | yes                     | integer                      |
| `subgroup.exclusive_xor(v, log2_size)`      | yes  | yes\*  | yes                     | integer                      |

The SPV-only no-arg reductions (`subgroup.reduce_mul` / `reduce_and` / `reduce_or` / `reduce_xor`, plus the original `reduce_add(value)` with no `log2_size`) have been removed in favour of the portable sized API. For reductions other than the ones listed above, build a sized helper on top of `shuffle_down` / `shuffle` following the same pattern as `reduce_add` / `reduce_all_add`.

## Semantics

All of these ops operate within a single subgroup: they do not move data through memory and do not synchronise across subgroups.

### `shuffle(value, index)`

Each lane returns the `value` held by the lane whose subgroup-local id equals `index`.

- `value` is a scalar in a register. Supported dtypes are 32-bit and 64-bit signed/unsigned ints and `f32`/`f64`. (64-bit types are split into two 32-bit shuffles on AMDGPU; CUDA dispatches to its native 64-bit helpers.)
- `index` is a `u32`. If `index` is out of range for the active subgroup the result is implementation-defined, so pass `subgroup.invocation_id()`-derived values or known-good lane ids.

### `shuffle_down(value, offset)`

Lane `i` returns the `value` held by lane `i + offset`. Lanes near the top of the subgroup — where `i + offset >= subgroup_size` — receive an implementation-defined value (typically their own `value`), so reduction patterns must only trust lane 0's final result, or mask out the out-of-range lanes.

- `value` and `offset` dtypes: same as `shuffle` above; `offset` is a `u32`.
- Maps to `__shfl_down_sync` on CUDA and `OpGroupNonUniformShuffleDown` on SPIR-V. On AMDGPU it is currently emulated with `ds_bpermute` (see the support matrix above).

### `shuffle_up(value, offset)`

Lane `i` returns the `value` held by lane `i - offset`. Lanes near the bottom of the subgroup — where `i - offset < 0` — receive an implementation-defined value (typically their own `value`), so the bottom `offset` lanes' results should be ignored or masked.

- Same dtype rules as `shuffle` / `shuffle_down`; `offset` is a `u32`.
- Maps to `__shfl_up_sync` on CUDA and `OpGroupNonUniformShuffleUp` on SPIR-V. On AMDGPU it is currently emulated with `ds_bpermute((lane - offset) * 4, value)` (same fast-path FIXME as `shuffle_down`).

### `shuffle_xor(value, mask)`

Lane `i` returns the `value` held by lane `i ^ mask`. Convenient for butterfly patterns (used internally by `reduce_all_add`).

- Same dtype rules as `shuffle`; `mask` is a `u32`.
- Implemented portably as a `@qd.func` over `shuffle`: every backend that lowers `shuffle` therefore lowers `shuffle_xor` with no additional codegen path. Inlines at trace time into a single `shuffle(value, u32(invocation_id()) ^ mask)`.
- The XOR partner must be inside the active subgroup; behaviour outside that range is implementation-defined (same caveat as `shuffle`).

### `broadcast(value, index)`

Every lane in the subgroup returns the `value` held by the lane whose subgroup-local id equals `index`. Expresses intent ("read lane `index`") more directly than `shuffle(value, index)` and on backends with a dedicated broadcast may map to a cheaper instruction.

- Same dtype rules as `shuffle`.
- Maps to `__shfl_sync` on CUDA, `ds_bpermute` on AMDGPU, and `OpGroupNonUniformBroadcast` on SPIR-V.
- **Important: on SPIR-V, `index` must be dynamically uniform** — the same value on every lane in the subgroup. Passing a per-lane varying `index` is undefined behavior, because `OpGroupNonUniformBroadcast` requires its `Id` operand to be dynamically uniform across the subgroup. On CUDA / AMDGPU, `index` may vary per lane and the call is identical to `shuffle(value, index)`. If you need a varying source lane, use `shuffle` directly.

### `broadcast_first(value)`

Every lane returns lane 0's `value`. Convenience wrapper for the common "read lane 0 from every lane" pattern.

- Same dtype rules as `broadcast`.
- Implemented portably as a `@qd.func` over `broadcast(value, qd.u32(0))`: every backend that lowers `broadcast` therefore lowers `broadcast_first`. The `0` index is trivially dynamically uniform, so the SPIR-V `OpGroupNonUniformBroadcast` requirement is satisfied. Inlines at trace time.

### Common to the data-movement ops

- All shuffles / broadcasts are issued under a full active mask on CUDA (`0xFFFFFFFF`). Call them from uniform control flow; calling from divergent control flow is undefined on most backends. (This means: every thread has to execute the shuffle.)
- Subgroup size varies by backend (32 on NVIDIA, 32 or 64 on AMD depending on wavefront mode, 32 in Vulkan compute on most GPUs). Use `subgroup.group_size()` to query at runtime on SPIR-V; on CUDA / AMDGPU use a compile-time constant.

### `invocation_id()`

Returns this lane's subgroup-local index — `0..subgroup_size - 1`. Used both as a lane id when computing a target lane for `shuffle` / `broadcast`, and as a per-lane identifier in cooperative algorithms.

- Returns `i32`.
- Available on every backend.

### `group_size()`

Returns the subgroup size in effect for the current launch as an `i32`.

- **CUDA**: lowers to a static `32` constant — the warp size is fixed on every supported NVIDIA architecture (sm_30+). The optimizer can fold it into address arithmetic, so calling `group_size()` is no more expensive than hard-coding `32`.
- **AMDGPU**: lowers to `llvm.amdgcn.wavefrontsize`, which the AMDGPU backend constant-folds to `32` (RDNA / wave32) or `64` (CDNA, GFX9, RDNA wave64) at codegen time based on the function's wavefront-mode target feature.
- **SPIR-V**: lowers to a load of the `OpSubgroupSize` builtin — a true runtime query, since on Vulkan compute the subgroup size can be 32 on most desktop GPUs but is permitted to be other powers of two.

### `elect()`

Returns `1` on lane 0 of every subgroup and `0` on every other lane. Useful for "exactly one lane does X" patterns where you don't care which lane does it — e.g. emitting a single global write per subgroup.

- Implemented portably as a `@qd.func` wrapper: `i32(invocation_id() == 0)`. Inlines at trace time into a single compare + zero-extend on every backend.
- This narrows the SPIR-V `OpGroupNonUniformElect` semantics, which would otherwise be free to pick any *active* lane. Under the documented uniform-CF + all-lanes-active contract for `qd.simt.subgroup` the distinction is invisible (lane 0 is always active and is a legal choice), and pinning the elected lane down keeps the behaviour identical across backends.

### `sync()` / `mem_fence()`

`sync()` is a subgroup-scope thread-converging barrier — every lane in the subgroup must reach the call before any lane proceeds. `mem_fence()` is a subgroup-scope memory fence: it orders memory operations within the subgroup without requiring thread convergence.

- `sync()` lowers to:
  - **SPIR-V**: `OpControlBarrier(Subgroup, Subgroup, 0)`.
  - **CUDA**: `__syncwarp(0xFFFFFFFF)` (`nvvm.bar.warp.sync`). Reconverges lanes that may have diverged under independent thread scheduling on Volta+; under uniform CF on Pascal and earlier this is effectively a no-op but is still legal.
  - **AMDGPU**: `llvm.amdgcn.wave.barrier`. Acts as a compiler reordering barrier on GCN (where waves are lockstep) and as a real wave-scope hardware barrier on RDNA.
- `mem_fence()` lowers to:
  - **SPIR-V**: `OpMemoryBarrier(Subgroup, AcquireRelease | UniformMemory | WorkgroupMemory)`.
  - **CUDA**: `__threadfence_block()` (`nvvm.membar.cta`) — workgroup-scope, see the `**` footnote in the matrix above.
  - **AMDGPU**: LLVM `fence syncscope("workgroup") seq_cst` — workgroup-scope, same caveat.
- Caller contract on every backend: call from uniform control flow with all lanes active. Calling either op from divergent control flow has implementation-defined behaviour (CUDA's `nvvm.bar.warp.sync` will deadlock if the mask does not match the active set; AMDGPU's `wave.barrier` is a no-op on most chips so divergent calls silently pass through).
- The legacy names `subgroup.barrier()` and `subgroup.memory_barrier()` are still available as deprecated aliases. They forward to `sync()` / `mem_fence()` and emit a `DeprecationWarning` on first use; prefer the new names in new code.

### `ballot(predicate)`

Each lane evaluates `predicate` (an `i32`; non-zero is true, zero is false) and the result is a `u32` bitmask where bit `i` is set if lane `i`'s predicate was non-zero.

- Returns a `u32`. Bit 0 corresponds to lane 0, bit 1 to lane 1, etc.
- On CUDA, maps to `__ballot_sync(0xFFFFFFFF, predicate)`. On SPIR-V, maps to `OpGroupNonUniformBallot` (component 0 of the uvec4 result). On AMDGPU, maps to the `ballot.i32` intrinsic.
- The result covers the first 32 lanes. On AMDGPU CDNA with 64-wide wavefronts only the low 32 bits are returned; the upper 32 lanes are not represented. This is consistent with the 32-bit return type.
- Must be called from uniform control flow (all active lanes must execute the ballot).

Ballot is a building block for warp-cooperative algorithms: population counts (`popcount(ballot(cond))` counts how many lanes satisfy `cond`), prefix masks, and lane compaction.

### `reduce_add(value, log2_size)`

Sums `value` across `2**log2_size` consecutive lanes via a `shuffle_down` tree. The result is valid **in lane 0** of each group; other lanes hold partial sums and should be considered undefined.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into exactly `log2_size` `shuffle_down + add` pairs in the calling kernel's IR, with no runtime loop overhead.
- `2**log2_size` must not exceed the active subgroup size on the target (32 on CUDA / Metal and on RDNA, 64 on CDNA). Passing a larger value produces implementation-defined results; it does not error.
- The reduction works on any type that supports `+` and `shuffle_down`; in practice this means i32, u32, f32, f64, i64, u64.
- Decorated with `@qd.func` and inlined into the calling kernel — there is no kernel-launch overhead and no separate symbol to link.

Lanes 1..`2**log2_size - 1` receive undefined-but-safe partial sums (they never touch out-of-range lanes because the tree shrinks each step), but only lane 0's result is meaningful for the caller.

### `reduce_all_add(value, log2_size)`

Same sum as `reduce_add`, but broadcast to **every lane** in each `2**log2_size` group. Implemented as a butterfly using `shuffle` with `lane ^ mask`, `mask` stepping through `1, 2, 4, ..., 2**(log2_size-1)`.

- Same `log2_size` template + size-cap contract as `reduce_add`.
- Use this when every lane needs the reduction result (e.g. to divide by the sum, or to branch on it uniformly). It costs exactly the same number of shuffles as `reduce_add` but leaves the answer in all lanes, so it replaces a `reduce_add` + `shuffle`/broadcast pair.
- Uses `subgroup.shuffle` under the hood.

### `reduce_min(value, log2_size)` / `reduce_max(value, log2_size)`

Min / max of `value` across `2**log2_size` consecutive lanes via a `shuffle_down` tree. Result valid in **lane 0** of each group; other lanes hold partial mins / maxes.

- Same `log2_size` template + size-cap contract as `reduce_add`. Body unrolls into exactly `log2_size` `shuffle_down + min` (or `max`) pairs.
- Accepts integer (`i32`, `u32`, `i64`, `u64`) and float (`f32`, `f64`) dtypes. Lowers via `qd.min` / `qd.max`, which dispatch to the backend's native min/max intrinsic.
- Float NaN handling is implementation-defined: PTX uses `fminnm` / `fmaxnm` (NaN-suppressing), AMDGPU uses `llvm.minnum` / `llvm.maxnum` (NaN-suppressing), SPIR-V uses `OpFMin` / `OpFMax` (NaN-propagating in some drivers). Avoid NaN inputs if you need a portable result.
- Decorated with `@qd.func` and inlined into the calling kernel.

### `reduce_all_min(value, log2_size)` / `reduce_all_max(value, log2_size)`

Same min / max as `reduce_min` / `reduce_max`, but broadcast to **every lane** in each `2**log2_size` group via a `shuffle_xor` butterfly. Same number of shuffles as the lane-0 forms.

- Use over `reduce_min` + broadcast / `reduce_max` + broadcast when every lane needs the result (e.g. to subtract the group min, or to clamp to the group max).
- Same dtypes, size-cap contract, and float-NaN caveat as `reduce_min` / `reduce_max`.

### `segmented_reduce_add(value, head_flag, log2_size)` / `segmented_reduce_min(...)` / `segmented_reduce_max(...)`

Per-lane inclusive scan under `+` / `min` / `max` that resets at every non-zero `head_flag`, scoped to `2**log2_size` consecutive lanes. Lane `i` returns the scan of `value[head_below..i + 1]`, where `head_below` is the largest lane index `<= i` (within the lane's `2**log2_size` group) whose `head_flag` is non-zero. If no such lane exists inside the group, the group's first lane is treated as an implicit head, so the result is the inclusive scan from `group_base` to `i`.

- `value` is any type supporting the operator (`+` and `shuffle_up` for `_add`; `qd.min`/`qd.max` and `shuffle_up` for `_min`/`_max`). `head_flag` is any integer scalar; the lowering tests `head_flag != 0`, so non-binary truthy values (e.g. `7`, `42`) work.
- `log2_size` is a `qd.template()` — a compile-time constant. `2**log2_size` must not exceed 32: the underlying `subgroup.ballot_first_n(p, 32)` returns a `u32` covering the first 32 lanes, so on AMDGPU CDNA wave64 only the low 32 lanes contribute.
- Implementation: one `subgroup.ballot_first_n(head_flag != 0, 32)` to materialise a `u32` of head positions, then a Hillis-Steele inclusive scan bounded by `distance >= offset` where `distance = lane - segment_head`. `segment_head` is computed via `31 - clz(effective_mask & ((1 << (lane + 1)) - 1))`, with an OR-injected virtual head at `group_base` to guarantee a non-zero `lower`. The mask + `clz` setup is shared between all three ops via a small internal helper.
- No identity argument is required (unlike `exclusive_min` / `exclusive_max`) because the per-lane `distance >= offset` guard ensures the scan never reaches across a segment boundary, so a partner from another segment is never combined with the local value.
- Cost: `1 ballot + 1 clz + log2_size shuffles + log2_size ops`, fully unrolled into the calling kernel's IR. Same shape as `inclusive_add` / `inclusive_min` / `inclusive_max`, plus one ballot and one `clz` for the per-lane segment-head lookup.
- Float NaN handling for `_min` / `_max` is implementation-defined (same caveat as `reduce_min` / `reduce_max`): PTX uses `fminnm` / `fmaxnm`, AMDGPU uses `llvm.minnum` / `llvm.maxnum`, SPIR-V uses `OpFMin` / `OpFMax`. Avoid NaN inputs if you need a portable result.
- AMDGPU note (`*` in the table): `shuffle_up` goes through `ds_bpermute` (~50 cycle latency), same as the other reductions.

### `inclusive_add` / `inclusive_mul` / `inclusive_min` / `inclusive_max` / `inclusive_and` / `inclusive_or` / `inclusive_xor`

Per-lane inclusive scan over `2**log2_size` consecutive lanes, under the binary operator named by the suffix. Lane `i` within each group of `2**log2_size` lanes returns `v[group_start] op v[group_start + 1] op ... op v[i]`.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into exactly `log2_size` `shuffle_up + op` pairs in the calling kernel's IR, with no runtime loop overhead.
- `2**log2_size` must not exceed the active subgroup size on the target (32 on CUDA / Metal / RDNA, 64 on CDNA). Passing a larger value produces implementation-defined results; it does not error.
- `_add`, `_mul`, `_min`, `_max` accept integer and float dtypes (`i32`, `u32`, `i64`, `u64`, `f32`, `f64`); `_and`, `_or`, `_xor` accept integer dtypes only.
- All seven share a single `@qd.func` Hillis-Steele scan helper (`_inclusive_scan`); each public op is a one-line wrapper that supplies the binary operator. The shuffle is in uniform CF (every lane participates); only the per-lane reduce step is conditional, which is allowed by the `shuffle_up` contract on every backend. Cross-group `shuffle_up` partners are masked off by the per-lane `lane_in_group >= offset` guard, so groups smaller than the full subgroup compose correctly.
- Decorated with `@qd.func` and inlined into the calling kernel — there is no kernel-launch overhead and no separate symbol to link.
- AMDGPU note (`*` in the table): same `ds_bpermute` cost as `shuffle_up` — roughly tens of cycles per step × `log2_size` steps. Hardware-accelerated `OpGroupNonUniformInclusiveScan` on SPIR-V is no longer used, even on backends that supported it (Vulkan, Metal); the trade-off is a uniform implementation across backends with predictable cost.

### `exclusive_add` / `exclusive_mul` / `exclusive_min` / `exclusive_max` / `exclusive_and` / `exclusive_or` / `exclusive_xor`

Per-lane exclusive scan over `2**log2_size` consecutive lanes, under the binary operator named by the suffix. Lane `i` (with `i > 0`) within each group of `2**log2_size` lanes returns `v[group_start] op v[group_start + 1] op ... op v[i - 1]`. Lane 0 of each group returns the operator's identity in `value`'s dtype.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into the inclusive scan (`log2_size` shuffle+op pairs) plus one extra `shuffle_up` and a per-lane select.
- `_add`, `_mul`, `_or`, `_xor`, `_and` infer the lane-0 identity from `value`'s dtype: `value - value` (zero), `value - value + 1` (one), and `~(value ^ value)` (all bits set) respectively.
- `_min` and `_max` take an explicit `identity` argument because there is no portable type-extreme literal that can be derived from `value` alone — pass `+∞` (or the dtype's max) for `_min`, `-∞` (or the dtype's min) for `_max`.
- All seven share a single `@qd.func` helper (`_exclusive_scan`) that runs the inclusive scan, shifts the result up by one lane via `shuffle_up`, and substitutes `identity` at lane 0 of each group. The lane-0 substitution is required because `shuffle_up` with offset 1 is implementation-defined at lane 0 (and `OpGroupNonUniformShuffleUp` calls it undefined outright).
- AMDGPU performance note (`*` in the table): same `ds_bpermute` cost as `shuffle_up`. Cost is one inclusive scan plus one extra `shuffle_up` and a select.

### `ballot_first_n(predicate, n)`

Returns a `u32` bitmask whose bit `i` is set iff `i < n` AND lane `i`'s `predicate` is non-zero. Bits `>= n` are always zero.

- `predicate` is any integer scalar; the lowering tests `predicate != 0` internally, so non-binary truthy values work.
- `n` is a `qd.template()` compile-time constant in `[1, 32]`. Pass `n = 32` for "ballot over the 32 representable lanes" — the most common case, used internally by `segmented_reduce_*` and other ballot consumers.
- Backend lowering:
  - **CUDA**: `__ballot_sync(0xFFFFFFFF, predicate)`. Warps are always 32 lanes, so the `u32` result naturally packs every lane.
  - **AMDGPU**: `llvm.amdgcn.ballot.i32`. Packs lanes 0..31 into the result; on wave64 lanes 32..63's predicates simply do not appear in the `i32` result, which matches the `n <= 32` contract.
  - **SPIR-V**: `OpGroupNonUniformBallot` returns a `uvec4`; we extract component 0, which by spec contains the ballot bits for lanes 0..31.
- For `n < 32` we mask the predicate by `lane < n` before issuing the ballot, so bits `[n, 32)` of the result are forced to zero regardless of those lanes' actual predicate values. At `n == 32` the masking is provably a no-op on every backend (lanes `>= 32` are either non-existent on wave32 or already not represented in the `u32` result on wave64), so the masking is elided at trace time and the call lowers to a single ballot intrinsic.
- Caller contract: uniform CF + all lanes active. Calling from divergent control flow has implementation-defined behaviour (CUDA's `__ballot_sync` will deadlock if the active mask doesn't match `0xFFFFFFFF`).
- Useful for stream compaction over the first 32 lanes, segmented reductions (which cap at 32-lane segments via `log2_size <= 5`), and any pattern that wants `clz` / `popcount` / `ffs` over a per-lane predicate within a `u32`.

### `ballot_full_subgroup(predicate)`

Returns a `u64` bitmask covering the entire subgroup. Bit `i` is set iff lane `i`'s `predicate` is non-zero, for all `i` in `[0, subgroup_size)`.

- On wave32 backends (CUDA, RDNA wave32, most Vulkan / Metal) the high 32 bits of the result are always zero, since lanes `>= 32` do not exist. On wave64 backends (AMDGPU CDNA, GFX9, RDNA explicit-wave64) all 64 bits are meaningful.
- Backend lowering:
  - **CUDA**: `__ballot_sync(0xFFFFFFFF, p)` then zero-extend the `i32` result to `i64`.
  - **AMDGPU**: `llvm.amdgcn.ballot.i64`. Returns the full 64-bit ballot on wave64; on wave32 the AMDGPU backend lowers it to the wave32 ballot zero-extended to 64 bits, so the API stays uniform across wavefront modes.
  - **SPIR-V**: extract components 0 and 1 of the `OpGroupNonUniformBallot` `uvec4` (lanes 0..31 and 32..63 respectively) and pack them into a `u64` via `u64(hi) << 32 | u64(lo)`.
- Use this when you need a subgroup-wide population count, prefix mask, or compaction that has to cover more than 32 lanes. Use `ballot_first_n(p, 32)` instead if you only care about the first 32 lanes — it's one register cheaper on wave64 and avoids the wider `u64` result type.
- Caller contract: uniform CF + all lanes active.

### `all_true(predicate, log2_size)` / `any_true(predicate, log2_size)`

Per-lane AND-reduction (`all_true`) or OR-reduction (`any_true`) of `predicate != 0` across `2**log2_size` consecutive lanes. Returns `i32` (`0` or `1`), broadcast to every lane in the group.

- `predicate` is any scalar dtype. The op compares `predicate != 0` at the start, so e.g. `subgroup.all_true(some_int_field[i], 5)` is well-formed.
- `log2_size` is a `qd.template()` — a compile-time constant. Caller must ensure `2**log2_size` does not exceed the active subgroup size (32 on CUDA / Metal / RDNA, 64 on CDNA).
- CUDA full-warp shortcut: when `log2_size == 5`, lowers to a single `__all_sync(0xFFFFFFFF, p)` / `__any_sync(0xFFFFFFFF, p)` via the `cuda_all_sync_i32` / `cuda_any_sync_i32` runtime helpers (one `vote.all` / `vote.any` instruction). The shortcut is selected at trace time via `qd.static()` on the active arch and `log2_size`, so the IR contains exactly the intrinsic call and no branch.
- Portable fallback (every other backend, and CUDA at `log2_size < 5`): `shuffle_xor` butterfly — `log2_size` shuffles plus `log2_size` ANDs (or ORs), fully unrolled into the calling kernel's IR. Same shape as `reduce_all_add`.

### `all_equal(value, log2_size)`

Returns `i32(1)` on every lane in each `2**log2_size` group iff every lane in the group has the same `value` (under the backend's native `==`), else `i32(0)`.

- `value` is any scalar dtype. Equality is the backend's native `==`: for floats this means `NaN != NaN` (a group with any `NaN` returns `0`) and `+0.0 == -0.0`, matching SPIR-V `OpGroupNonUniformAllEqual`. Callers wanting bit-equality on floats should `qd.bit_cast` to the same-width integer dtype first.
- Implementation: each lane computes `group_base = invocation_id() & ~(2**log2_size - 1)`, reads the value at `group_base` via `shuffle`, compares to its own `value`, and `all_true`-reduces the equality bit. Inherits the CUDA full-warp shortcut transitively from `all_true`.
- Cost: 1 shuffle + 1 `vote.all` on CUDA at `log2_size == 5`; 1 shuffle + `log2_size` butterfly shuffles otherwise. We deliberately do *not* use `__match_all_sync` on CUDA: it requires sm_70+ and uses bit-equality for floats, which would contradict this op's documented semantics.

### `lanemask_lt(lane_id)` / `lanemask_le(lane_id)` / `lanemask_eq(lane_id)` / `lanemask_gt(lane_id)` / `lanemask_ge(lane_id)`

Closed-form `u32` lane-mask constants parametrised by a lane id. Bit `i` of the result follows the relation in the suffix:

| Op             | Bit `i` set iff | Closed form                            |
|----------------|-----------------|----------------------------------------|
| `lanemask_lt`  | `i <  lane_id`  | `(1 << lane_id) - 1`                   |
| `lanemask_le`  | `i <= lane_id`  | `lt(lane_id) \| eq(lane_id)`            |
| `lanemask_eq`  | `i == lane_id`  | `1 << lane_id`                         |
| `lanemask_gt`  | `i >  lane_id`  | `~le(lane_id)`                         |
| `lanemask_ge`  | `i >= lane_id`  | `~lt(lane_id)`                         |

- `lane_id` is any integer scalar. Pass `subgroup.invocation_id()` to get the classic CUDA built-in form (current lane's mask), or any other expression to query an arbitrary lane's mask. The op is pure arithmetic — no shuffle, no ballot — so per-lane-varying `lane_id` works the same as a uniform one.
- Returns `u32`. Bit 0 corresponds to lane 0, bit 31 to lane 31.
- Caller contract: `lane_id` must be in `[0, 31]` (matching the `u32` return type, which represents 32 lanes). Passing `lane_id == 32` triggers an undefined-behaviour shift on most backends.
- Implemented portably as a `@qd.func` over `<<`, `-`, `|`, `~`. Inlines at trace time into 1–3 ALU ops on every backend.
- AMDGPU CDNA wave64 caveat: only the low 32 lanes are representable in this op (the return type is `u32`). If you need a mask covering all 64 wave64 lanes, use `subgroup.ballot_full_subgroup` instead — it returns a `u64` and includes lanes 32..63.

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

Classic warp-level sum of 4 values — after the second step, lane 0 of each group of 4 holds the total:

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

### Sum 32 lanes with `reduce_add`

The same tree, packaged as a one-liner. Lane 0 of each group of 32 holds the total; other lanes hold partial sums:

```python
@qd.kernel
def sum32(src: qd.types.ndarray(dtype=qd.f32, ndim=1),
          dst: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(src.shape[0]):
        total = subgroup.reduce_add(src[i], 5)
        if subgroup.invocation_id() == 0:
            dst[i // 32] = total
```

`5` is `log2_size`; `2**5 == 32` matches the block dim. The body of `reduce_add` unrolls at trace time into five `shuffle_down + add` pairs, so the generated IR is identical to a hand-written tree reduction.

### Broadcast the sum to all lanes with `reduce_all_add`

When every lane needs the reduction result — e.g. to normalise by the sum — use the butterfly variant. No follow-up broadcast needed:

```python
@qd.kernel
def normalize32(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        total = subgroup.reduce_all_add(a[i], 5)
        a[i] = a[i] / total
```

Every lane in each group of 32 sees the same `total`.

### Partial-subgroup reductions

`log2_size` does not have to match the full subgroup. Sum groups of 8 with `reduce_add(v, 3)` or groups of 16 with `reduce_all_add(v, 4)`; the caller just ensures `2**log2_size <= subgroup_size` (so up to 5 on CUDA / Metal / RDNA, up to 6 on CDNA).

### Inclusive scan with `inclusive_add`

```python
@qd.kernel
def cumsum(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    qd.loop_config(block_dim=32)
    for i in range(a.shape[0]):
        a[i] = subgroup.inclusive_add(a[i], 5)
```

After the call, lane `k` (within each group of 32) holds `a[group_start] + a[group_start+1] + ... + a[k]`. The `5` is `log2_size`; `2**5 == 32` matches the block dim. The body unrolls at trace time into five `shuffle_up + add` pairs. Use a smaller `log2_size` to scan over partial-subgroup groups (e.g. `inclusive_add(v, 3)` produces independent prefix sums in groups of 8).

## Performance notes

- Shuffles are register-to-register on CUDA (`__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`) and on SPIR-V where the GPU has hardware support — typically a handful of cycles, no memory traffic.
- AMDGPU `shuffle`, `shuffle_down`, and `shuffle_up` all go through `ds_permute` / `ds_bpermute` today (LDS-routed, roughly tens of cycles).
- `shuffle_xor` and `broadcast_first` are `@qd.func` wrappers over `shuffle` / `broadcast` and inline at trace time, so on every backend they cost exactly the same as the underlying op.
- Both `ballot_first_n` and `ballot_full_subgroup` lower to a single hardware instruction on every backend — one cycle on CUDA (`__ballot_sync`), one instruction on AMDGPU (`v_ballot_b32` / `v_ballot_b64`), and `OpGroupNonUniformBallot` on SPIR-V (extract one or two components of the result `uvec4`). At `n == 32` `ballot_first_n` elides the predicate-masking step entirely; at `n < 32` it inserts one extra multiply on the predicate.
- `reduce_add` and `reduce_all_add` both issue exactly `log2_size` shuffles and `log2_size` adds per call. No barriers, no shared memory, no launch overhead (they inline).
- Pick `reduce_all_add` over `reduce_add + broadcast` when you need the result in every lane — same cost, one fewer shuffle.
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.
- All seven `inclusive_*` ops are `@qd.func` Hillis-Steele scans; cost is exactly `log2_size` shuffle+op pairs, the same as a hand-rolled CUDA warp scan, on every backend. Hardware-accelerated `OpGroupNonUniformInclusiveScan` on SPIR-V is no longer used — the cost difference vs. a portable shuffle tree is small in practice, and the uniform implementation makes performance predictable across CUDA, AMDGPU, and SPIR-V.

## Related

- [tile16](tile16.md) — `Tile16x16` builds on `subgroup.shuffle` to implement register-resident 16×16 matrix tiles.
- `qd.simt.warp.*` — CUDA-only counterparts (`warp.all_nonzero`, `warp.any_nonzero`, `warp.unique`, `warp.ballot`, `warp.match_*`, `warp.active_mask`, ...). The voting ops (`all_nonzero` / `any_nonzero` / `unique`) overlap with the new portable `subgroup.{all_true, any_true}`; the rest stay CUDA-bound. Useful when you need explicit active-mask control or an op that has no portable equivalent yet.
