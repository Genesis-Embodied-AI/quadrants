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

### How `log2_size` windowing works

Every op below that takes a `log2_size` template parameter operates on independent `2**log2_size`-aligned windows that **tile the entire subgroup** — not just the first `2**log2_size` lanes. With `log2_size = k`, the subgroup splits into `group_size() / (2**k)` windows of `2**k` consecutive lanes each, and each window does its own reduction / scan / vote completely independently of every other window. The caller must arrange `2**k <= group_size()` so every window is full (a smaller `k` simply gives more, narrower windows; it does **not** mean "only the first window is active").

| `log2_size` | wave32 (CUDA / most Vulkan / Metal) | wave64 (AMDGPU — see [supported_systems](supported_systems.md)) |
| --- | --- | --- |
| 5 (window = 32) | 1 window: lanes 0–31 | 2 windows: 0–31, 32–63 |
| 4 (window = 16) | 2 windows: 0–15, 16–31 | 4 windows: 0–15, 16–31, 32–47, 48–63 |
| 3 (window = 8)  | 4 windows of 8 | 8 windows of 8 |
| 0 (window = 1)  | every lane is its own window (no-op) | same |

Why it composes exactly: the underlying `subgroup.shuffle` / `subgroup.shuffle_down` / `subgroup.shuffle_up` / `subgroup.shuffle_xor` ops are themselves **full-subgroup** — they address every lane by absolute lane id, with no built-in notion of a window. Windowing emerges from how the higher-level reductions / scans / votes *compose* those shuffles.

#### Result placement per window

- **Broadcast-to-all forms** — `all_true`, `any_true`, `all_equal`, `reduce_all_add`, `reduce_all_min`, `reduce_all_max`, `inclusive_*`, `exclusive_*`, `segmented_reduce_*`: every lane in each window holds the per-window result. Lanes in different windows hold different results (their own window's).
- **Window-local-lane-0 forms** — `reduce_add`, `reduce_min`, `reduce_max`: only the *window-local* lane 0 holds the reduction. That's lane 0 alone with `log2_size=5` on wave32, lanes 0 and 32 with `log2_size=5` on wave64, lanes 0 / 16 / 32 / 48 with `log2_size=4` on wave64, etc. Other lanes hold partial reductions and should be treated as undefined. Use `reduce_all_*` if you want every lane to see its window's result.

#### Picking `log2_size` for a "full subgroup" reduction

`log2_size` is compile-time and `group_size()` is a compile-time constant on every LLVM backend (32 on CUDA, 64 on AMDGPU — wave64 is forced on every AMDGPU target, see [supported_systems](supported_systems.md)). On SPIR-V it is a runtime value, typically 32 on Vulkan compute. For a portable "full-subgroup" reduction the typical pattern is `log2_size = 5` (32-lane window — fits in every subgroup); accept that AMDGPU then gives you two reductions per wave (one per 32-lane half) and stitch the halves together yourself with `shuffle` if needed. If you specifically need a 64-lane reduction on AMDGPU, use `log2_size = 6` gated on `arch == amdgpu` at trace time.

### Voting and predicate ops

All three take a `log2_size` template parameter and are **windowed**: they operate independently on each `2**log2_size`-aligned window that tiles the entire subgroup, broadcasting the `i32` (`0` or `1`) per-window result to every lane in that window (the broadcast-to-all forms from [How `log2_size` windowing works](#how-log2size-windowing-works)). With `log2_size = 5` on wave32 you get one vote per subgroup; with `log2_size = 5` on wave64 you get two independent votes (lanes 0–31 and lanes 32–63 vote separately, and lanes in each half hold their own half's result). Same shape as `reduce_all_add` / `inclusive_*` / `exclusive_*`.

| Op                                          | CUDA          | AMDGPU | SPIR-V (Vulkan / Metal) |
|---------------------------------------------|---------------|--------|-------------------------|
| `subgroup.all_true(predicate, log2_size)`   | yes (fast at `log2_size==5`) | yes | yes |
| `subgroup.any_true(predicate, log2_size)`   | yes (fast at `log2_size==5`) | yes | yes |
| `subgroup.all_equal(value, log2_size)`      | yes (fast at `log2_size==5`, transitively via `all_true`) | yes | yes |

CUDA shortcut: when `log2_size == 5` (full warp), `all_true` / `any_true` lower to a single `__all_sync(0xFFFFFFFF, p)` / `__any_sync(0xFFFFFFFF, p)` (one `vote.all` / `vote.any` instruction). The shortcut is selected at trace time via `qd.static()` on `impl.current_cfg().arch` and the compile-time `log2_size`, so partial-warp uses (and every other backend) cleanly fall back to a portable `shuffle_xor` butterfly with no branch in the emitted IR.

`all_equal` always uses the broadcast-and-`all_true` form: every lane reads the value at the start of its group via `shuffle`, compares it with its own value, and `all_true`-reduces the per-lane equality bit. Cost: `1 + log2_size` shuffles in the portable case, or `1 shuffle + 1 vote.all` on CUDA at full-warp. We deliberately do *not* use `__match_all_sync` even on CUDA: it requires sm_70+, and it does bit-equality on floats, contradicting this op's documented `OpGroupNonUniformAllEqual` semantics (`NaN != NaN`, `+0.0 == -0.0`). Callers wanting bit-equality on floats should bit-cast to the same-width integer dtype before calling.

### Reductions and scans

`reduce_add`, `reduce_all_add`, and all seven `inclusive_*` and `exclusive_*` ops take a `log2_size` parameter and are **windowed**: each op operates independently on every `2**log2_size`-aligned window that tiles the entire subgroup (see [How `log2_size` windowing works](#how-log2size-windowing-works)). `reduce_add` is a window-local-lane-0 form (only the window's first lane holds the reduction; other lanes are undefined); `reduce_all_add` / `inclusive_*` / `exclusive_*` are broadcast-to-all forms (every lane in each window holds that window's per-window scan result). With `log2_size = 5` on wave32 you get one reduction / scan per subgroup; with `log2_size = 5` on wave64 you get two independent reductions / scans (lanes 0–31 and lanes 32–63 reduce / scan separately).

| Op                                          | CUDA | AMDGPU | SPIR-V (Vulkan / Metal) | dtypes                       |
|---------------------------------------------|------|--------|-------------------------|------------------------------|
| `subgroup.reduce_add(v, log2_size)`         | yes  | yes\*  | yes                     | any type supporting `+`      |
| `subgroup.reduce_all_add(v, log2_size)`     | yes  | yes    | yes                     | any type supporting `+`      |
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

The SPV-only no-arg reductions (`subgroup.reduce_mul` / `reduce_min` / `reduce_max` / `reduce_and` / `reduce_or` / `reduce_xor`, plus the original `reduce_add(value)` with no `log2_size`) have been removed in favour of the portable sized API (`reduce_add(v, log2_size)` / `reduce_all_add(v, log2_size)`). For reductions other than sum, build a sized helper on top of `shuffle_down` / `shuffle` following the same pattern.

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
- Subgroup size is hard-coded per backend on the LLVM backends — 32 on CUDA, 64 on AMDGPU (wave64 is forced on every AMDGPU target including RDNA, see [supported_systems](supported_systems.md)) — so kernels can rely on those literal values. On SPIR-V it is queried at runtime via `subgroup.group_size()` (typically 32 on Vulkan compute on most GPUs).

### `invocation_id()`

Returns this lane's subgroup-local index — `0..subgroup_size - 1`. Used both as a lane id when computing a target lane for `shuffle` / `broadcast`, and as a per-lane identifier in cooperative algorithms.

- Returns `i32`.
- Available on every backend.

### `group_size()`

Returns the subgroup size in effect for the current launch as an `i32`. **For use inside `@qd.kernel` / `@qd.func` bodies only** — calling it from host Python raises.

- **CUDA**: lowers to a static `32` constant — the warp size is fixed on every supported NVIDIA architecture (sm_30+). The optimizer can fold it into address arithmetic, so calling `group_size()` is no more expensive than hard-coding `32`.
- **AMDGPU**: lowers to `llvm.amdgcn.wavefrontsize`, which the AMDGPU backend constant-folds to `64` at codegen time. Quadrants pins every AMDGPU function to `+wavefrontsize64,-wavefrontsize32` (see [supported_systems](supported_systems.md)), so CDNA (gfx9xx, gfx940/942) keeps its native wave64 mode and RDNA (gfx10/11/12) — which would otherwise default to wave32 — is forced into wave64 too. `group_size()` is therefore always 64 on AMDGPU.
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

### `reduce_add(value, log2_size)`

Sums `value` across `2**log2_size` consecutive lanes via a `shuffle_down` tree. The result is valid in the **window-local lane 0** of each group (see [How `log2_size` windowing works](#how-log2size-windowing-works)); other lanes hold partial sums and should be considered undefined.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into exactly `log2_size` `shuffle_down + add` pairs in the calling kernel's IR, with no runtime loop overhead.
- `2**log2_size` must not exceed the active subgroup size on the target (32 on CUDA / Metal, 64 on AMDGPU — wave64 is forced on every AMDGPU target). Passing a larger value silently computes the wrong sum and there is no runtime check: each iteration calls `shuffle_down(value, offset >= subgroup_size)`, which on CUDA returns the calling lane's own value, on AMDGPU wraps around the wave (offset is taken mod 64 inside `ds_bpermute`), and on SPIR-V is fully undefined per spec — so the corrupted result varies by backend and, on Vulkan, by driver.
- The reduction works on any type that supports `+` and `shuffle_down`; in practice this means i32, u32, f32, f64, i64, u64.
- Decorated with `@qd.func` and inlined into the calling kernel — there is no kernel-launch overhead and no separate symbol to link.

Lanes other than the window-local lane 0 receive undefined-but-safe partial sums (they never touch out-of-range lanes because the tree shrinks each step), but only the window-local lane 0's result is meaningful for the caller. On wave64 with `log2_size=5` this is two valid results (lane 0 and lane 32), one per 32-lane half — reading only lane 0 silently drops the upper-half sum. Use `reduce_all_add` if every lane needs the answer.

### `reduce_all_add(value, log2_size)`

Same sum as `reduce_add`, but broadcast to **every lane** in each `2**log2_size` group. Implemented as a butterfly using `shuffle` with `lane ^ mask`, `mask` stepping through `1, 2, 4, ..., 2**(log2_size-1)`.

- Same `log2_size` template + size-cap contract as `reduce_add`.
- Use this when every lane needs the reduction result (e.g. to divide by the sum, or to branch on it uniformly). It costs exactly the same number of shuffles as `reduce_add` but leaves the answer in all lanes, so it replaces a `reduce_add` + `shuffle`/broadcast pair.
- Uses `subgroup.shuffle` under the hood.

### `inclusive_add` / `inclusive_mul` / `inclusive_min` / `inclusive_max` / `inclusive_and` / `inclusive_or` / `inclusive_xor`

Per-lane inclusive scan over `2**log2_size` consecutive lanes, under the binary operator named by the suffix. Lane `i` within each group of `2**log2_size` lanes returns `v[group_start] op v[group_start + 1] op ... op v[i]`.

- `log2_size` is a `qd.template()` — a compile-time constant. The body unrolls into exactly `log2_size` `shuffle_up + op` pairs in the calling kernel's IR, with no runtime loop overhead.
- `2**log2_size` must not exceed the active subgroup size on the target (32 on CUDA / Metal, 64 on AMDGPU — wave64 is forced on every AMDGPU target). Passing a larger value produces implementation-defined results; it does not error.
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

### `all_true(predicate, log2_size)` / `any_true(predicate, log2_size)`

Per-lane AND-reduction (`all_true`) or OR-reduction (`any_true`) of `predicate != 0` across `2**log2_size` consecutive lanes. Returns `i32` (`0` or `1`), broadcast to every lane in the group.

- `predicate` is any scalar dtype. The op compares `predicate != 0` at the start, so e.g. `subgroup.all_true(some_int_field[i], 5)` is well-formed.
- `log2_size` is a `qd.template()` — a compile-time constant. Caller must ensure `2**log2_size` does not exceed the active subgroup size (32 on CUDA / Metal, 64 on AMDGPU — wave64 is forced on every AMDGPU target).
- CUDA full-warp shortcut: when `log2_size == 5`, lowers to a single `__all_sync(0xFFFFFFFF, p)` / `__any_sync(0xFFFFFFFF, p)` via the `cuda_all_sync_i32` / `cuda_any_sync_i32` runtime helpers (one `vote.all` / `vote.any` instruction). The shortcut is selected at trace time via `qd.static()` on the active arch and `log2_size`, so the IR contains exactly the intrinsic call and no branch.
- Portable fallback (every other backend, and CUDA at `log2_size < 5`): `shuffle_xor` butterfly — `log2_size` shuffles plus `log2_size` ANDs (or ORs), fully unrolled into the calling kernel's IR. Same shape as `reduce_all_add`.

### `all_equal(value, log2_size)`

Returns `i32(1)` on every lane in each `2**log2_size` group iff every lane in the group has the same `value` (under the backend's native `==`), else `i32(0)`.

- `value` is any scalar dtype. Equality is the backend's native `==`: for floats this means `NaN != NaN` (a group with any `NaN` returns `0`) and `+0.0 == -0.0`, matching SPIR-V `OpGroupNonUniformAllEqual`. Callers wanting bit-equality on floats should `qd.bit_cast` to the same-width integer dtype first.
- Implementation: each lane computes `group_base = invocation_id() & ~(2**log2_size - 1)`, reads the value at `group_base` via `shuffle`, compares to its own `value`, and `all_true`-reduces the equality bit. Inherits the CUDA full-warp shortcut transitively from `all_true`.
- Cost: 1 shuffle + 1 `vote.all` on CUDA at `log2_size == 5`; 1 shuffle + `log2_size` butterfly shuffles otherwise. We deliberately do *not* use `__match_all_sync` on CUDA: it requires sm_70+ and uses bit-equality for floats, which would contradict this op's documented semantics.

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

`log2_size` does not have to match the full subgroup. Sum groups of 8 with `reduce_add(v, 3)` or groups of 16 with `reduce_all_add(v, 4)`; the caller just ensures `2**log2_size <= subgroup_size` (so up to 5 on CUDA / Metal, up to 6 on AMDGPU).

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
- `reduce_add` and `reduce_all_add` both issue exactly `log2_size` shuffles and `log2_size` adds per call. No barriers, no shared memory, no launch overhead (they inline).
- Pick `reduce_all_add` over `reduce_add + broadcast` when you need the result in every lane — same cost, one fewer shuffle.
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.
- All seven `inclusive_*` ops are `@qd.func` Hillis-Steele scans; cost is exactly `log2_size` shuffle+op pairs, the same as a hand-rolled CUDA warp scan, on every backend. Hardware-accelerated `OpGroupNonUniformInclusiveScan` on SPIR-V is no longer used — the cost difference vs. a portable shuffle tree is small in practice, and the uniform implementation makes performance predictable across CUDA, AMDGPU, and SPIR-V.

## Related

- [tile16](tile16.md) — `Tile16x16` builds on `subgroup.shuffle` to implement register-resident 16×16 matrix tiles.
- `qd.simt.warp.*` — CUDA-only counterparts (`warp.all_nonzero`, `warp.any_nonzero`, `warp.unique`, `warp.ballot`, `warp.match_*`, `warp.active_mask`, ...). The voting ops (`all_nonzero` / `any_nonzero` / `unique`) overlap with the new portable `subgroup.{all_true, any_true}`; the rest stay CUDA-bound. Useful when you need explicit active-mask control or an op that has no portable equivalent yet.
