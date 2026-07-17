# Atomics

Atomic read-modify-write operations on a single memory location. They do not synchronize threads; the only ordering they provide is the per-location atomicity of the read-modify-write itself. For cooperative ops across threads see the `qd.simt.block.*`, `qd.simt.subgroup.*`, and `qd.simt.grid.*` namespaces. Bit-counting helpers on integer registers (`qd.math.popcnt`, `qd.math.clz`) are documented in [math](math.md).

The companion read-side primitive `qd.volatile_load(target)` is documented at the end of this page (`### qd.volatile_load(target)`); it pairs with atomic stores in producer / consumer patterns where the reader must observe every update from another thread or block, and is the recommended approach for spin-wait loops.

## What's available

All atomic ops follow the same shape: `qd.atomic_op(x, y)` performs `x = op(x, y)` atomically and returns the **old** value of `x`. `x` must be a writable memory target (a field element, ndarray element, or matrix slot); scalars and constant expressions are not allowed.

"int" below means any of `i32` / `u32` / `i64` / `u64`. "Floats" means any of `f16` / `f32` / `f64`. Unless otherwise noted, "native" means the op lowers to a single hardware atomic instruction (or its SPIR-V / LLVM-IR equivalent), and "CAS" means a software compare-and-swap loop emitted around a non-atomic compute.

| Op                                          | CUDA                                       | AMDGPU                                | SPIR-V (Vulkan / Metal)                                | CPU                              |
|---------------------------------------------|--------------------------------------------|---------------------------------------|--------------------------------------------------------|----------------------------------|
| `atomic_add`                                | int / f32 native; f64 native (sm_60+)      | int / f32 native; f64 hardware-dependent | int native; Metal global f32 native (Apple7+/Mac2+); Vulkan f16 / f32 / f64 capability-gated, else CAS; Metal f16 / f64 / shared float via CAS | int / f32 / f64 native; f16 via CAS |
| `atomic_sub`                                | rewritten to `atomic_add(x, -y)` at IR-construction time — see note below | (same) | (same) | (same) |
| `atomic_mul`                                | CAS on every dtype                         | CAS                                   | CAS                                                    | CAS                              |
| `atomic_min`, `atomic_max`                  | int native; floats via CAS                 | int native; floats via CAS            | int native; floats via CAS                             | int native; floats via CAS       |
| `atomic_and`, `atomic_or`, `atomic_xor`     | int only (native)                          | int only (native)                     | int only (native)                                      | int only (native)                |
| `atomic_exchange`                           | int / float native (`atomicExch`)          | int / float native (`*_atomic_swap`)  | int native; f32 / f64 global via uint-bitcast `OpAtomicExchange`; f16, shared float, workgroup f64 deferred‡ | int / float native (`xchg`)      |
| `atomic_cas`                                | int native (`atomicCAS`)                   | int native (`*_atomic_cmpswap`)       | int native (`OpAtomicCompareExchange`); f32 / f64 rejected at compile time§                               | int native (`cmpxchg`)           |

A few cross-cutting notes that the cells above abbreviate:

- **`atomic_sub` is not a separate op in the IR.** `quadrants/ir/frontend_ir.cpp::AtomicOpExpression::flatten` rewrites every `atomic_sub(x, y)` into `atomic_add(x, -y)` before codegen sees it, so per-backend support and per-dtype behavior are exactly those of `atomic_add`.
- **CAS-loop ops are noticeably slower than native atomics**, especially under contention — every contending thread retries the load + compare-exchange until it wins. Prefer pre-aggregating into a register or shared array and issuing a single atomic at the end of the block where possible.
- **f16 floats always use a CAS loop** (no native f16 atomic on any backend except SPIR-V with the right capability bit).
- **On CPU, "native" does not guarantee a single machine instruction.** On x86 and other architectures without hardware float atomics, the compiler backend lowers native float `atomic_add` (and integer `min` / `max`) to a CAS loop in machine code. Under high contention the performance is similar to the explicit "CAS" entries; the difference is that "native" ops benefit from hardware acceleration where available.
- **SPIR-V capability bits** (`spirv_has_atomic_float_add`, `spirv_has_atomic_float64_add`, `spirv_has_atomic_float16_add`, plus the matching `spirv_has_shared_atomic_float*_add` bits for workgroup memory) decide whether `atomic_add` lowers to native `OpAtomicFAddEXT` or a uint-backed CAS — the dispatch happens per-call inside `quadrants/codegen/spirv/spirv_codegen.cpp`.
- **Metal float `atomic_add`.** On Apple7+ / Mac2+ the Metal RHI advertises `spirv_has_atomic_float` / `spirv_has_atomic_float_add`, so global / device-buffer `f32` `atomic_add` (and `atomic_sub`, rewritten to add) lowers to `OpAtomicFAddEXT` → MSL `atomic_fetch_add` on `device atomic_float`. That path requires MSL 3.0 (`MTLLanguageVersion3_0` + SPIRV-Cross `set_msl_version(3,0,0)`). Metal does **not** support threadgroup `atomic_float`, and has no native float min/max / f16 / f64 atomics, so shared float atomics and those dtypes stay on the uint CAS path. Vulkan still reads the same capability bits from `VK_EXT_shader_atomic_float*`.
- **`i64` / `u64` atomic RMW is not portable to Metal.** Metal Shading Language only exposes 64-bit atomics as `atomic_fetch_min` / `atomic_fetch_max` on `uint64` (Apple GPU family 9+, M3 / A17); `atomic_add` / `sub` / `mul` and the bitwise family are unavailable on every Apple GPU. The Metal RHI today over-advertises `spirv_has_atomic_int64` (gated on Apple7 / Mac2 in `quadrants/rhi/metal/metal_device.mm`), so 64-bit integer atomics under Metal fail at pipeline create time with `RhiResult=-1`. Use `i32` / `u32` for Metal portability. CUDA, AMDGPU, and Vulkan with `VK_KHR_shader_atomic_int64` are unaffected.

† `i64` / `u64` atomic RMW is **not portable to Metal**. Metal Shading Language only exposes 64-bit atomics as `atomic_fetch_min` / `atomic_fetch_max` on `uint64`, starting at Apple GPU family 9 (M3 / A17 and newer); `atomic_add` / `sub` / `mul` and the bitwise family are unavailable on every Apple GPU. The Metal RHI today over-advertises `spirv_has_atomic_int64` (gated on Apple7 / Mac2 in `quadrants/rhi/metal/metal_device.mm`), so trying to use 64-bit integer atomics under Metal currently fails at pipeline create time with `RhiResult=-1` ("SPIR-V shader was rejected by the backend"). Use `i32` / `u32` if you need cross-Metal portability. CUDA, AMDGPU, and Vulkan with `VK_KHR_shader_atomic_int64` are unaffected.

‡ `atomic_exchange` on `f16`, on shared (`qd.simt.block.SharedArray`) float arrays, and on f64 in workgroup memory is not yet wired up. Global-memory `atomic_exchange` on every other dtype/backend combination listed above is supported; the SPIR-V path bitcasts through the corresponding uint type so no `spirv_has_atomic_float_*` capability is required.

§ `atomic_cas` on `f32` / `f64` is rejected at compile time (raises `QuadrantsTypeError`). Integer CAS (`i32` / `u32` / `i64` / `u64`) is supported on every backend listed in the table above, with the same Metal caveat for `i64` / `u64` (†) as the rest of the 64-bit integer atomic family.

All atomic ops can be called on either global memory (fields, ndarrays) or block-shared memory (`qd.simt.block.SharedArray`). They are sequentially consistent on the location they touch; they are **not** memory fences for the rest of the address space - to publish other writes alongside an atomic, pair the atomic with `qd.simt.block.mem_fence()` (block scope) or `qd.simt.grid.mem_fence()` (device scope).

## Semantics

### `qd.atomic_add(x, y)` - and the rest of the family

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
- **Per-location atomicity, no fence on the rest of memory.** Writes you issued before an atomic on `x` are not necessarily visible to other threads after they observe the new `x`. Pair the atomic with `qd.simt.block.mem_fence()` or `qd.simt.grid.mem_fence()` if you need that ordering.
- **Vector / matrix arguments fan out element-wise.** `qd.atomic_add(field_of_vec3, qd.Vector([1.0, 2.0, 3.0]))` issues three independent scalar atomic-adds, one per component. There is no all-or-nothing guarantee across the components.

### `qd.atomic_min(x, y)` / `qd.atomic_max(x, y)`

Atomically writes back `min(x, y)` (resp. `max(x, y)`); returns the old value of `x`. Float min/max are `minNum` / `maxNum`-style: if exactly one operand is `NaN`, the non-`NaN` operand wins.

| Backends                  | `f16`                                  | `f32`, `f64`                       | Both inputs `NaN`                          |
|---------------------------|----------------------------------------|------------------------------------|--------------------------------------------|
| CPU, CUDA, AMDGPU (LLVM)  | CAS over `llvm.minnum` / `llvm.maxnum` | LLVM `atomicrmw fmin` / `fmax`           | `NaN` (per LLVM `minnum` / `maxnum` spec)  |
| Vulkan, Metal (SPIR-V)    | capability-gated, usually unsupported  | CAS loop with GLSL `FMin` / `FMax`       | undefined per spec; `NaN` in practice      |

### `qd.atomic_and(x, y)` / `qd.atomic_or(x, y)` / `qd.atomic_xor(x, y)`

Bitwise atomics. Integer dtypes only — passing `f32` / `f64` raises a type error at compile time.

### `qd.atomic_sub(x, y)` / `qd.atomic_mul(x, y)`

Atomic subtract and atomic multiply. `atomic_sub` is rewritten to `atomic_add(x, -y)` at IR-construction time (`quadrants/ir/frontend_ir.cpp::AtomicOpExpression::flatten`), so its per-backend behavior is identical to `atomic_add`. `atomic_mul` always lowers to a CAS loop - no LLVM AtomicRMW or SPIR-V `OpAtomic*` op corresponds to multiply - and is intentionally not heavily optimized; prefer reducing to a different scheme on hot paths.

### `qd.atomic_exchange(x, y)`

Atomically writes `y` into `x` and returns the old value of `x`. Unlike the other `qd.atomic_*` ops the new value of `x` does **not** depend on its old value - `x` is unconditionally overwritten. The exchange always succeeds; there is no retry / failure path.

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
# Whatever was in `slot` is now mine to process; I left NO_TASK behind for the next worker.  No retry needed - exchange
# always succeeds.
```

Vector / matrix arguments fan out per component, same as the rest of the `qd.atomic_*` family: a `qd.atomic_exchange(field_of_vec3, qd.Vector([...]))` issues three independent scalar exchanges, one per slot, with no all-or-nothing guarantee across the components.

### `qd.atomic_cas(x, expected, desired)`

Atomic compare-and-swap: writes `desired` into `x` if and only if `x` currently equals `expected`, and unconditionally returns the value originally at `x`. The user recovers whether the swap actually fired with one comparison:

```python
old = qd.atomic_cas(x, expected, desired)
# Effect:
#   tmp = load(x)
#   if tmp == expected: store(x, desired)
#   old = tmp
# all three steps (load, conditional store, return-old) execute as a single atomic transaction on x.

success = (old == expected)
```

This is the basic primitive on top of which arbitrary atomic read-modify-write operations can be built with a retry loop. Returning the prior value (rather than a `(prior, success)` pair) matches CUDA `atomicCAS` and SPIR-V `OpAtomicCompareExchange`; lowers to one native instruction on every backend (CUDA `atomicCAS`, AMDGPU `*_atomic_cmpswap`, SPIR-V `OpAtomicCompareExchange`, x86 `cmpxchg`).

CAS-loop pattern for ops the framework doesn't expose natively (e.g. atomic-max-of-some-derived-quantity):

```python
@qd.kernel
def cas_loop_max():
    # Atomically: x = max(x, candidate). The framework already has atomic_max for primitives, but the same
    # shape works for any reduction whose backend support is missing.
    for _attempt in range(MAX_RETRIES):
        cur = x[None]
        new = qd.max(cur, candidate)
        old = qd.atomic_cas(x[None], cur, new)
        if old == cur:
            break  # CAS landed; we're done.
        # Otherwise some other thread won the race; loop back and re-read.
```

Currently restricted to integer dtypes (`i32` / `u32` / `i64` / `u64`); float CAS is rejected at compile time. The Metal `i64` / `u64` caveat in the support table footnote applies here too. There is no shared-memory CAS path yet.

### `qd.volatile_load(target)`

Read `target` from memory with **volatile** semantics: the compiler is forbidden from caching, hoisting, or merging the load with prior reads of the same address. Strictly speaking this is not an atomic op (it does not modify the cell, and per-thread it is no more atomic than an ordinary load) — it lives on this page because it is the read-side counterpart to `qd.atomic_*` in producer / consumer patterns. Without it, a spin-wait loop reading the location written by another thread or block is undefined at the LLVM-IR / SPIR-V level.

```python
val = qd.volatile_load(target)
# Effect:
#   load(target) — but the load is guaranteed to actually go to memory on every call,
#   not be reused from a register or hoisted out of an enclosing loop.
```

`target` must be a global lvalue (a field or ndarray subscript); function-scope local arrays are rejected because a local cannot be observed by another thread. Bit-packed quant snodes are also rejected (per-field volatile semantics on a shared physical word are not meaningful).

| Backend          | Lowering                                                                                  |
|------------------|-------------------------------------------------------------------------------------------|
| CUDA             | LLVM `load volatile` → PTX `ld.volatile.global`.                                          |
| AMDGPU           | LLVM `load volatile` → unhoistable `global_load_*` (the optimizer is inhibited from forwarding / merging). |
| Vulkan / Metal   | SPIR-V `OpLoad` with the `Volatile` `MemoryAccess` mask, propagated through SPIRV-Cross to a re-read on every use in the generated MSL / GLSL. |
| CPU (x86_64)     | LLVM `load volatile` (the optimizer cannot hoist or merge it; the runtime cost is identical to an ordinary load on x86). |

Quadrants additionally suppresses the optimizations that would otherwise let an aliased rewrite slip past codegen:

- `cache_loop_invariant_global_vars` does not hoist a volatile load out of an enclosing loop.
- `simplify` does not replace a volatile load with the value of an earlier load of the same address.
- On CUDA, the `__ldg` / `!invariant.load` read-only-cache fast path is suppressed for volatile loads (the two attributes are contradictory).

#### Spin-wait pattern (the canonical use case)

The naive approach

```python
while flags[prev] == STATE_INVALID:
    pass
```

is undefined: the compiler may hoist `flags[prev]` out of the loop, turning the spin into an infinite loop. `qd.volatile_load` is the cheapest correct approach on every backend:

```python
while qd.volatile_load(flags[prev]) == STATE_INVALID:
    pass
```

The two older workarounds remain correct but pay a perf tax over the volatile load:

- `qd.simt.grid.mem_fence()` inside the loop body — drains the device-scope cache on every iteration. Order-of-magnitude more expensive than a volatile read on contemporary hardware. Also does **not** help on Metal / Vulkan-on-macOS (the Metal device-scope fence only orders atomic accesses; see [grid](grid.md)).
- `qd.atomic_or(flags[prev], 0)` — forces a memory round-trip via an atomic RMW. Pays for the read-modify-write hardware path even though we only want to read; contention with concurrent stores is worse than a plain volatile load.

`qd.volatile_load` works on every backend uniformly, has no contention overhead, and matches what CUDA / OpenCL programmers reach for in the same situation.

#### Pairing with the producer

A volatile load only orders the location it reads. To make other writes from the producer visible to the reader after the volatile load observes the new value, the producer must publish through an ordering primitive — either an atomic store (the Metal-portable choice; see the per-store atomic ops above), or a plain store followed by `qd.simt.grid.mem_fence()` (CUDA / AMDGPU / native Vulkan only).

The decoupled-look-back scan in [grid](grid.md) shows the full pattern.

## Performance and portability notes

- **Atomic contention is the silent killer of throughput.** The cost of `qd.atomic_add(counter, 1)` from every thread is dominated by serialization at the location, not by the per-thread arithmetic. If many threads hit the same slot, prefer a two-stage scheme: per-warp / per-block reduction first (`qd.simt.block.reduce` if available, or `qd.simt.subgroup.reduce_add`), then a single atomic per warp / block.
- **Pair atomics with the right fence scope.** A bare atomic only orders the location it touches. To make other writes visible to readers that observe the new atomic value, follow the atomic with a fence: block-scope (`qd.simt.block.mem_fence()`) for shared-memory publishing, or grid-scope (`qd.simt.grid.mem_fence()`) for cross-block coordination.
- **`f64` atomics fall off the fast path** on most backends; if you only need monotonic accumulation, consider Kahan summation in registers and a single atomic-add at the end of the block.
- **`atomic_mul` is generally a CAS loop** under the hood; don't put it on the hot path.

### Atomic visibility scope across backends

Every `qd.atomic_*` is emitted at **device-wide scope**: visible to all threads on the GPU executing the kernel, but not required to be coherent with the host CPU mid-kernel. The host only observes results once the kernel completes, at which point the launcher's stream-sync flushes everything regardless. Choosing device scope (rather than the strongest "system" scope) lets every backend lower the op to a single hardware atomic instruction instead of a software CAS retry loop, which matters for correctness as much as for speed: under heavy contention, a CAS loop on a non-converging op like `atomic_xor` can livelock.

You don't normally need to think about scope as a user. It's listed here so the per-backend behavior is explicit:

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

³ SPIR-V float `atomic_add` lowers to `OpAtomicFAddEXT` when the matching `spirv_has_atomic_float{,64,16}_add` capability is present on the device, and to a CAS loop with a GLSL.std.450 payload otherwise. Metal always advertises global f32 add on Apple7+/Mac2+ (see bullet above); Vulkan is capability-gated per `VK_EXT_shader_atomic_float*`. Shared float add needs `spirv_has_shared_atomic_float*_add`, which Metal never sets. Quadrants does not currently emit `OpAtomicFMinEXT` / `OpAtomicFMaxEXT`, so float min/max is always CAS on SPIR-V backends.

## Related

- [math](math.md) - `qd.math.*`, including the bit-counting helpers (`popcnt`, `clz`) commonly paired with atomics in select / compact patterns.
- `qd.simt.block.*` - block-scope barriers and memory fences (`qd.simt.block.mem_fence()`).
- `qd.simt.subgroup.*` - warp-scope reductions and shuffles, the recommended pre-aggregation step before an atomic.
- `qd.simt.grid.*` - device-scope memory fence (`qd.simt.grid.mem_fence()`); see [grid](grid.md).
- [parallelization](parallelization.md) - thread-synchronization patterns and how atomics fit into the broader synchronization story.
