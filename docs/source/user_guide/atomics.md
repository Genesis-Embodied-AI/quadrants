# Atomics

Atomic read-modify-write operations on a single memory location. They do not synchronize threads; the only ordering they provide is the per-location atomicity of the read-modify-write itself. For cooperative ops across threads see the `qd.simt.block.*`, `qd.simt.subgroup.*`, and `qd.simt.grid.*` namespaces. Bit-counting helpers on integer registers (`qd.math.popcnt`, `qd.math.clz`) are documented in [math](math.md).

## What's available

All atomic ops follow the same shape: `qd.atomic_op(x, y)` performs `x = op(x, y)` atomically and returns the **old** value of `x`. `x` must be a writable memory target (a field element, ndarray element, or matrix slot); scalars and constant expressions are not allowed.

| Op             | Semantics                              | i32 | u32 | i64  | u64  | f32 | f64 |
|----------------|----------------------------------------|-----|-----|------|------|-----|-----|
| `atomic_add`   | `x += y`                               | yes | yes | yes† | yes† | yes | \*  |
| `atomic_sub`   | `x -= y`                               | yes | yes | yes† | yes† | yes | \*  |
| `atomic_mul`   | `x *= y`                               | yes | yes | yes† | yes† | yes | \*  |
| `atomic_min`   | `x = min(x, y)`                        | yes | yes | yes† | yes† | yes | \*  |
| `atomic_max`   | `x = max(x, y)`                        | yes | yes | yes† | yes† | yes | \*  |
| `atomic_and`   | `x &= y`                               | yes | yes | yes† | yes† | —   | —   |
| `atomic_or`    | `x \|= y`                              | yes | yes | yes† | yes† | —   | —   |
| `atomic_xor`      | `x ^= y`                            | yes | yes | yes† | yes† | —    | —    |
| `atomic_exchange` | `x = y`, return old `x`             | yes | yes | yes† | yes† | yes  | yes‡ |

\* `f64` atomic add / sub / mul / min / max is hardware-dependent: supported on CUDA sm_60+ for `add`, falls back to a CAS loop elsewhere or raises at codegen time on older targets and on backends that do not lower a CAS loop. Prefer `f32` on hot paths if portability matters.

† `i64` / `u64` atomic RMW is **not portable to Metal**. Metal Shading Language only exposes 64-bit atomics as `atomic_fetch_min` / `atomic_fetch_max` on `uint64`, starting at Apple GPU family 9 (M3 / A17 and newer); `atomic_add` / `sub` / `mul` and the bitwise family are unavailable on every Apple GPU. The Metal RHI today over-advertises `spirv_has_atomic_int64` (gated on Apple7 / Mac2 in `quadrants/rhi/metal/metal_device.mm`), so trying to use 64-bit integer atomics under Metal currently fails at pipeline create time with `RhiResult=-1` ("SPIR-V shader was rejected by the backend"). Use `i32` / `u32` if you need cross-Metal portability. CUDA, AMDGPU, and Vulkan with `VK_KHR_shader_atomic_int64` are unaffected.

‡ `atomic_exchange` on `f16`, on shared (`qd.simt.block.SharedArray`) float arrays, and on f64 in workgroup memory is not yet wired up. Global-memory `atomic_exchange` on every other dtype/backend combination listed above is supported; the SPIR-V path bitcasts through the corresponding uint type so no `spirv_has_atomic_float_*` capability is required.

There is no `atomic_cas` (compare-and-swap) exposed in Python today. `atomic_exchange` covers the unconditional-swap subset (which lowers to a single instruction on every backend); a true CAS would let you build arbitrary atomic RMW operations with a retry loop, but surfacing it requires extending `AtomicOpType` to return both the old value *and* a success flag — out of scope for the current API.

All atomic ops can be called on either global memory (fields, ndarrays) or block-shared memory (`qd.simt.block.SharedArray`). They are sequentially consistent on the location they touch; they are **not** memory fences for the rest of the address space — to publish other writes alongside an atomic, pair the atomic with `qd.simt.block.mem_sync()` (block scope) or `qd.simt.grid.memfence()` (device scope).

**Backend caveat for the fence-pair pattern.** Both fence helpers have current portability gaps that affect the patterns recommended on this page:

- `qd.simt.block.mem_sync()` is supported on CUDA and SPIR-V; on AMDGPU it raises `ValueError("qd.block.mem_sync is not supported for arch ...")` at trace time.
- `qd.simt.grid.memfence()` is fully implemented only on CUDA. On AMDGPU it currently links as a silent no-op (cross-block ordering will fail without any diagnostic); on SPIR-V it fails at codegen.

On AMDGPU specifically, neither fence-pair recipe works as documented yet; cross-platform code that needs an atomic plus a fence must restructure around the kernel-launch boundary or be CUDA-bound until the AMDGPU lowerings land.

## Semantics

### `qd.atomic_add(x, y)` — and the rest of the family

```python
old = qd.atomic_add(x, y)
# Effect:
#   tmp = load(x)
#   store(x, op(tmp, y))
#   old = tmp
# all three steps execute as a single atomic transaction on x.
```

Properties common to every `qd.atomic_*`:

- **Returns the old value**, not the new one. This matches CUDA's `atomicAdd` and is what enables building reservation patterns: `slot = qd.atomic_add(counter, 1)` gives every thread a unique index.
- **Per-location atomicity, no fence on the rest of memory.** Writes you issued before an atomic on `x` are not necessarily visible to other threads after they observe the new `x`. Pair the atomic with `qd.simt.block.mem_sync()` or `qd.simt.grid.memfence()` if you need that ordering.
- **Vector / matrix arguments fan out element-wise.** `qd.atomic_add(field_of_vec3, qd.Vector([1.0, 2.0, 3.0]))` issues three independent scalar atomic-adds, one per component. There is no all-or-nothing guarantee across the components.

### `qd.atomic_min(x, y)` / `qd.atomic_max(x, y)`

Atomically writes back `min(x, y)` (resp. `max(x, y)`). Returns the old value of `x`. Floating-point min/max use **`minNum` / `maxNum`-style** semantics: if exactly one input is `NaN`, the **non-`NaN`** value is written back. This matches the f16 path's use of LLVM `llvm.minnum` / `llvm.maxnum` intrinsics (`quadrants/codegen/llvm/codegen_llvm.cpp:1337-1342`) and the GPU-native paths (CUDA sm_80+ `atomicMin`/`atomicMax` for floats, SPIR-V `FMin` / `FMax`). The f32 / f64 CPU CAS-loop path (`quadrants/runtime/llvm/runtime_module/atomic.h::min_f32` / `max_f32`) uses naive `<` / `>` comparisons, which give asymmetric NaN behaviour depending on operand order — do not rely on a particular result when either input is `NaN` on the CPU backend. Behaviour when *both* inputs are `NaN` is backend-dependent across the board.

### `qd.atomic_and(x, y)` / `qd.atomic_or(x, y)` / `qd.atomic_xor(x, y)`

Bitwise atomics. Integer dtypes only — passing `f32` / `f64` raises a type error at trace time.

### `qd.atomic_sub(x, y)` / `qd.atomic_mul(x, y)`

Atomic subtract and atomic multiply. `atomic_sub` is supported natively on most backends; `atomic_mul` on integer types lowers to a CAS loop on hardware without a native multiply atomic and is intentionally not heavily optimised — prefer reducing to a different scheme on hot paths.

### `qd.atomic_exchange(x, y)`

Atomically writes `y` into `x` and returns the old value of `x`. Unlike the other `qd.atomic_*` ops the new value of `x` does **not** depend on its old value — `x` is unconditionally overwritten. The exchange always succeeds; there is no retry / failure path.

```python
old = qd.atomic_exchange(x, y)
# Effect:
#   tmp = load(x)
#   store(x, y)
#   old = tmp
# all three steps execute as a single atomic transaction on x.
```

Lowers to one native instruction on every backend (CUDA `atomicExch`, AMDGPU `buffer_atomic_swap` / `global_atomic_swap`, SPIR-V `OpAtomicExchange`, x86 `xchg`). Useful for take-ownership / hand-off patterns:

```python
my_old_task = qd.atomic_exchange(slot, NO_TASK)
if my_old_task != NO_TASK:
    process(my_old_task)
# Whatever was in `slot` is now mine to process; I left NO_TASK behind for the next worker.
# No retry needed — exchange always succeeds.
```

Vector / matrix arguments fan out per component, same as the rest of the `qd.atomic_*` family: a `qd.atomic_exchange(field_of_vec3, qd.Vector([...]))` issues three independent scalar exchanges, one per slot, with no all-or-nothing guarantee across the components.

## Performance and portability notes

- **Atomic contention is the silent killer of throughput.** The cost of `qd.atomic_add(counter, 1)` from every thread is dominated by serialization at the location, not by the per-thread arithmetic. If many threads hit the same slot, prefer a two-stage scheme: per-warp / per-block reduction first (`qd.simt.block.reduce` if available, or `qd.simt.subgroup.reduce_add`), then a single atomic per warp / block.
- **Pair atomics with the right fence scope.** A bare atomic only orders the location it touches. To make other writes visible to readers that observe the new atomic value, follow the atomic with a fence: block-scope (`qd.simt.block.mem_sync()`) for shared-memory publishing, or grid-scope (`qd.simt.grid.memfence()`) for cross-block coordination.
- **`f64` atomics fall off the fast path** on most backends; if you only need monotonic accumulation, consider Kahan summation in registers and a single atomic-add at the end of the block.
- **`atomic_mul` is generally a CAS loop** under the hood; don't put it on the hot path.

## Related

- [math](math.md) — `qd.math.*`, including the bit-counting helpers (`popcnt`, `clz`) commonly paired with atomics in select / compact patterns.
- `qd.simt.block.*` — block-scope barriers and memory fences (`qd.simt.block.mem_sync()`).
- `qd.simt.subgroup.*` — warp-scope reductions and shuffles, the recommended pre-aggregation step before an atomic.
- `qd.simt.grid.*` — device-scope memory fence (`qd.simt.grid.memfence()`).
- [parallelization](parallelization.md) — thread-synchronization patterns and how atomics fit into the broader synchronization story.
