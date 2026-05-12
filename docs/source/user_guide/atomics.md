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

All atomic ops can be called on either global memory (fields, ndarrays) or block-shared memory (`qd.simt.block.SharedArray`). They are sequentially consistent on the location they touch; they are **not** memory fences for the rest of the address space — to publish other writes alongside an atomic, pair the atomic with `qd.simt.block.mem_fence()` (block scope) or `qd.simt.grid.memfence()` (device scope).

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
- **Per-location atomicity, no fence on the rest of memory.** Writes you issued before an atomic on `x` are not necessarily visible to other threads after they observe the new `x`. Pair the atomic with `qd.simt.block.mem_fence()` or `qd.simt.grid.memfence()` if you need that ordering.
- **Vector / matrix arguments fan out element-wise.** `qd.atomic_add(field_of_vec3, qd.Vector([1.0, 2.0, 3.0]))` issues three independent scalar atomic-adds, one per component. There is no all-or-nothing guarantee across the components.

### `qd.atomic_min(x, y)` / `qd.atomic_max(x, y)`

Atomically writes back `min(x, y)` (resp. `max(x, y)`); returns the old value of `x`. Float min/max are `minNum` / `maxNum`-style: if exactly one operand is `NaN`, the non-`NaN` operand wins.

| Backends                  | `f16`                                  | `f32`, `f64`                       | Both inputs `NaN`                          |
|---------------------------|----------------------------------------|------------------------------------|--------------------------------------------|
| CPU, CUDA, AMDGPU (LLVM)  | CAS over `llvm.minnum` / `llvm.maxnum` | LLVM `atomicrmw fmin` / `fmax`           | `NaN` (per LLVM `minnum` / `maxnum` spec)  |
| Vulkan, Metal (SPIR-V)    | capability-gated, usually unsupported  | CAS loop with GLSL `FMin` / `FMax`       | undefined per spec; `NaN` in practice      |

### `qd.atomic_and(x, y)` / `qd.atomic_or(x, y)` / `qd.atomic_xor(x, y)`

Bitwise atomics. Integer dtypes only — passing `f32` / `f64` raises a type error at trace time.

### `qd.atomic_sub(x, y)` / `qd.atomic_mul(x, y)`

Atomic subtract and atomic multiply. `atomic_sub` is rewritten to `atomic_add(x, -y)` at IR-construction time (`quadrants/ir/frontend_ir.cpp::AtomicOpExpression::flatten`), so its per-backend behaviour is identical to `atomic_add`. `atomic_mul` always lowers to a CAS loop — no LLVM AtomicRMW or SPIR-V `OpAtomic*` op corresponds to multiply — and is intentionally not heavily optimised; prefer reducing to a different scheme on hot paths.

## Performance and portability notes

- **Atomic contention is the silent killer of throughput.** The cost of `qd.atomic_add(counter, 1)` from every thread is dominated by serialization at the location, not by the per-thread arithmetic. If many threads hit the same slot, prefer a two-stage scheme: per-warp / per-block reduction first (`qd.simt.block.reduce` if available, or `qd.simt.subgroup.reduce_add`), then a single atomic per warp / block.
- **Pair atomics with the right fence scope.** A bare atomic only orders the location it touches. To make other writes visible to readers that observe the new atomic value, follow the atomic with a fence: block-scope (`qd.simt.block.mem_fence()`) for shared-memory publishing, or grid-scope (`qd.simt.grid.memfence()`) for cross-block coordination.
- **`f64` atomics fall off the fast path** on most backends; if you only need monotonic accumulation, consider Kahan summation in registers and a single atomic-add at the end of the block.
- **`atomic_mul` is generally a CAS loop** under the hood; don't put it on the hot path.

### Atomic visibility scope across backends

Every `qd.atomic_*` is emitted at **device-wide scope**: visible to all threads on the GPU executing the kernel, but not required to be coherent with the host CPU mid-kernel. The host only observes results once the kernel completes, at which point the launcher's stream-sync flushes everything regardless. Choosing device scope (rather than the strongest "system" scope) lets every backend lower the op to a single hardware atomic instruction instead of a software CAS retry loop, which matters for correctness as much as for speed: under heavy contention, a CAS loop on a non-converging op like `atomic_xor` can livelock.

You don't normally need to think about scope as a user. It's listed here so the per-backend behaviour is explicit:

| Backend                 | Scope spelling in the IR          |
|-------------------------|-----------------------------------|
| CPU (x86_64)            | LLVM `seq_cst` (System)           |
| CUDA (NVPTX)            | LLVM `seq_cst` (System)           |
| AMDGPU                  | LLVM `seq_cst syncscope("agent")` |
| Vulkan / Metal (SPIR-V) | SPIR-V `Scope = Device`           |

CPU and CUDA lower system-scope atomics directly to a single hardware instruction, so they leave the LLVM default alone. AMDGPU's LLVM backend, in contrast, refuses to use its native single-instruction atomics at system scope (it would have to add cache-flush instructions that don't exist for that op), and silently falls back to a CAS loop; setting `syncscope("agent")` is what unlocks the native `flat_atomic_xor` / `global_atomic_xor` / `flat_atomic_smin` / `flat_atomic_add_f32` / … SPIR-V backends spell the same idea with the `Device` scope token. The user-visible semantics are identical across all four.

### Native instruction vs CAS fallback

The tables below reflect what the in-tree LLVM emits today for Quadrants' default targets (x86_64; CUDA `sm_60+`; AMDGPU `gfx942` / MI300X at `syncscope("agent")`; Vulkan/Metal via SPIR-V). Older / different GFX generations are footnoted.

**Integer atomics** (`i32`, `u32`, `i64`, `u64`):

| Op                                         | CPU (x86_64) | CUDA | AMDGPU | Vulkan / Metal (SPIR-V) |
|--------------------------------------------|--------------|------|--------|-------------------------|
| `atomic_add`, `atomic_sub`                 | ✅           | ✅   | ✅     | ✅                      |
| `atomic_and`, `atomic_or`, `atomic_xor`    | ✅¹          | ✅   | ✅     | ✅                      |
| `atomic_min`, `atomic_max`                 | 🟡           | ✅   | ✅     | ✅                      |
| `atomic_mul`                               | 🟡           | 🟡   | 🟡     | 🟡                      |

**Floating-point atomics** (`f32`, `f64`):

| Op                         | CPU (x86_64) | CUDA | AMDGPU | Vulkan / Metal (SPIR-V) |
|----------------------------|--------------|------|--------|-------------------------|
| `atomic_add`, `atomic_sub` | 🟡           | ✅   | ✅²    | ✅³                     |
| `atomic_min`, `atomic_max` | 🟡           | 🟡   | 🟡²    | 🟡                      |
| `atomic_mul`               | 🟡           | 🟡   | 🟡     | 🟡                      |

Key:

- ✅ — single hardware atomic instruction (`lock`-prefixed x86, PTX `atom.*`, AMDGPU `flat_atomic_*`, or SPIR-V `OpAtomic*`).
- 🟡 — software `cmpxchg` / `cmpswap` retry loop.

`f16` atomics are CAS on every backend (Quadrants forces a CAS loop built from `llvm.minnum` / `llvm.maxnum` / `fadd`), and on Vulkan / Metal are additionally gated on `spirv_has_atomic_float16_*` device capabilities.

¹ `lock and` / `or` / `xor` are single-instruction on x86, but they don't expose the old value. When the `qd.atomic_*` return value is unused (the common case — fire-and-forget update) LLVM emits the single `lock` op. When the old value is consumed, x86 falls back to a `cmpxchg` loop.

² AMDGPU float-atomic support is GFX-dependent. Empirically with the bundled LLVM:
- `gfx942` (CDNA3 / MI300X, Quadrants' default AMDGPU target): `atomic_add` f32 / f64 are native (`flat_atomic_add_f32` / `_f64`), `atomic_min` / `max` f64 are native (`flat_atomic_min_f64` / `max_f64`); f32 min/max still expand to CAS.
- `gfx906`, `gfx90a`, `gfx1030`, `gfx1100`: all f32 / f64 float atomics expand to CAS.

³ SPIR-V float `atomic_add` lowers to `OpAtomicFAddEXT` when the matching `spirv_has_atomic_float{32,64}_add` capability is present on the device, and to a CAS loop with a GLSL.std.450 payload otherwise. Quadrants does not currently emit `OpAtomicFMinEXT` / `OpAtomicFMaxEXT`, so float min/max is always CAS on SPIR-V backends.

## Related

- [math](math.md) — `qd.math.*`, including the bit-counting helpers (`popcnt`, `clz`) commonly paired with atomics in select / compact patterns.
- `qd.simt.block.*` — block-scope barriers and memory fences (`qd.simt.block.mem_fence()`).
- `qd.simt.subgroup.*` — warp-scope reductions and shuffles, the recommended pre-aggregation step before an atomic.
- `qd.simt.grid.*` — device-scope memory fence (`qd.simt.grid.memfence()`).
- [parallelization](parallelization.md) — thread-synchronization patterns and how atomics fit into the broader synchronization story.
