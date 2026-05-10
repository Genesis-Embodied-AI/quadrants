# Atomics

Atomic read-modify-write operations on a single memory location. They do not synchronize threads; the only ordering they provide is the per-location atomicity of the read-modify-write itself. For cooperative ops across threads see the `qd.simt.block.*`, `qd.simt.subgroup.*`, and `qd.simt.grid.*` namespaces. Bit-counting helpers on integer registers (`qd.math.popcnt`, `qd.math.clz`) are documented in [math](math.md).

## What's available

All atomic ops follow the same shape: `qd.atomic_op(x, y)` performs `x = op(x, y)` atomically and returns the **old** value of `x`. `x` must be a writable memory target (a field element, ndarray element, or matrix slot); scalars and constant expressions are not allowed.

"int" below means any of `i32` / `u32` / `i64` / `u64`. "Floats" means any of `f16` / `f32` / `f64`. Unless otherwise noted, "native" means the op lowers to a single hardware atomic instruction (or its SPIR-V / LLVM-IR equivalent), and "CAS" means a software compare-and-swap loop emitted around a non-atomic compute.

| Op                                          | CUDA                                       | AMDGPU                                | SPIR-V (Vulkan / Metal)                                | CPU                              |
|---------------------------------------------|--------------------------------------------|---------------------------------------|--------------------------------------------------------|----------------------------------|
| `atomic_add`                                | int / f32 native; f64 native (sm_60+)      | int / f32 native; f64 hardware-dependent | int native; f16 / f32 / f64 capability-gated, else CAS | int / f32 / f64 native; f16 via CAS |
| `atomic_sub`                                | rewritten to `atomic_add(x, -y)` at IR-construction time — see note below | (same) | (same) | (same) |
| `atomic_mul`                                | CAS on every dtype                         | CAS                                   | CAS                                                    | CAS                              |
| `atomic_min`, `atomic_max`                  | int native; floats via CAS                 | int native; floats via CAS            | int native; floats via CAS                             | int native; floats via CAS       |
| `atomic_and`, `atomic_or`, `atomic_xor`     | int only (native)                          | int only (native)                     | int only (native)                                      | int only (native)                |

A few cross-cutting notes that the cells above abbreviate:

- **`atomic_sub` is not a separate op in the IR.** `quadrants/ir/frontend_ir.cpp::AtomicOpExpression::flatten` rewrites every `atomic_sub(x, y)` into `atomic_add(x, -y)` before codegen sees it, so per-backend support and per-dtype behaviour are exactly those of `atomic_add`.
- **CAS-loop ops are noticeably slower than native atomics**, especially under contention — every contending thread retries the load + compare-exchange until it wins. Prefer pre-aggregating into a register or shared array and issuing a single atomic at the end of the block where possible.
- **f16 floats always use a CAS loop** (no native f16 atomic on any backend except SPIR-V with the right capability bit).
- **On CPU, "native" does not guarantee a single machine instruction.** On x86 and other architectures without hardware float atomics, the compiler backend lowers native float `atomic_add` (and integer `min` / `max`) to a CAS loop in machine code. Under high contention the performance is similar to the explicit "CAS" entries; the difference is that "native" ops benefit from hardware acceleration where available.
- **SPIR-V capability bits** (`spirv_has_atomic_float_add`, `spirv_has_atomic_float64_add`, `spirv_has_atomic_float16_add`) decide whether `atomic_add` lowers to native `OpAtomicFAddEXT` or a uint-backed CAS — the dispatch happens per-call inside `quadrants/codegen/spirv/spirv_codegen.cpp`.
- **`i64` / `u64` atomic RMW is not portable to Metal.** Metal Shading Language only exposes 64-bit atomics as `atomic_fetch_min` / `atomic_fetch_max` on `uint64` (Apple GPU family 9+, M3 / A17); `atomic_add` / `sub` / `mul` and the bitwise family are unavailable on every Apple GPU. The Metal RHI today over-advertises `spirv_has_atomic_int64` (gated on Apple7 / Mac2 in `quadrants/rhi/metal/metal_device.mm`), so 64-bit integer atomics under Metal fail at pipeline create time with `RhiResult=-1`. Use `i32` / `u32` for Metal portability. CUDA, AMDGPU, and Vulkan with `VK_KHR_shader_atomic_int64` are unaffected.

There is no `atomic_cas` (compare-and-swap) exposed in Python today. The C++ runtime uses CmpXchg internally; surfacing it requires extending `AtomicOpType`.

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

Atomic subtract and atomic multiply. `atomic_sub` is rewritten to `atomic_add(x, -y)` at IR-construction time (`quadrants/ir/frontend_ir.cpp::AtomicOpExpression::flatten`), so its per-backend behaviour is identical to `atomic_add`. `atomic_mul` always lowers to a CAS loop — no LLVM AtomicRMW or SPIR-V `OpAtomic*` op corresponds to multiply — and is intentionally not heavily optimised; prefer reducing to a different scheme on hot paths.

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
