# Atomics

Atomic read-modify-write operations on a single memory location. They do not synchronize threads; the only ordering they provide is the per-location atomicity of the read-modify-write itself. For cooperative ops across threads see the `qd.simt.block.*`, `qd.simt.subgroup.*`, and `qd.simt.grid.*` namespaces. Bit-counting helpers on integer registers (`qd.math.popcnt`, `qd.math.clz`) are documented in [math](math.md).

## What's available

All atomic ops follow the same shape: `qd.atomic_op(x, y)` performs `x = op(x, y)` atomically and returns the **old** value of `x`. `x` must be a writable memory target (a field element, ndarray element, or matrix slot); scalars and constant expressions are not allowed.

| Op             | Semantics                              | i32 | u32 | i64 | u64 | f32 | f64 |
|----------------|----------------------------------------|-----|-----|-----|-----|-----|-----|
| `atomic_add`   | `x += y`                               | yes | yes | yes | yes | yes | \*  |
| `atomic_sub`   | `x -= y`                               | yes | yes | yes | yes | yes | \*  |
| `atomic_mul`   | `x *= y`                               | yes | yes | yes | yes | yes | \*  |
| `atomic_min`   | `x = min(x, y)`                        | yes | yes | yes | yes | yes | \*  |
| `atomic_max`   | `x = max(x, y)`                        | yes | yes | yes | yes | yes | \*  |
| `atomic_and`   | `x &= y`                               | yes | yes | yes | yes | —   | —   |
| `atomic_or`    | `x \|= y`                              | yes | yes | yes | yes | —   | —   |
| `atomic_xor`   | `x ^= y`                               | yes | yes | yes | yes | —   | —   |

\* `f64` atomic add / sub / mul / min / max is hardware-dependent: supported on CUDA sm_60+ for `add`, falls back to a CAS loop elsewhere or raises at codegen time on older targets and on backends that do not lower a CAS loop. Prefer `f32` on hot paths if portability matters.

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

Atomically writes back `min(x, y)` (resp. `max(x, y)`). Returns the old value of `x`. Floating-point **`f16` / `f32` /
`f64`** min/max use **`minNum` / `maxNum`-style** semantics on the LLVM backends: if exactly one operand is `NaN`, the
**non-`NaN`** value is written back — this matches the **`f16`** path (CAS built from `llvm.minnum` / `llvm.maxnum`
equivalents in `codegen_llvm.cpp`), and the **`f32` / `f64`** path, which uses LLVM `atomicrmw fmin` / `fmax`
(`atomicMin` / `atomicMax` where CUDA exposes them, SPIR-V `FMin` / `FMax`). The C++ compare loops in
`runtime_module/atomic.h` (`min_f32`, `max_f32`, …) remain only for **CPU** bitcode that is never patched to `atomicrmw`;
GPU runtime modules rewrite those symbols to the same `atomicrmw` lowering as user `qd.atomic_*`. Behaviour when *both*
operands are `NaN` is backend-dependent.

### `qd.atomic_and(x, y)` / `qd.atomic_or(x, y)` / `qd.atomic_xor(x, y)`

Bitwise atomics. Integer dtypes only — passing `f32` / `f64` raises a type error at trace time.

### `qd.atomic_sub(x, y)` / `qd.atomic_mul(x, y)`

Atomic subtract and atomic multiply. `atomic_sub` is supported natively on most backends; `atomic_mul` on integer types lowers to a CAS loop on hardware without a native multiply atomic and is intentionally not heavily optimised — prefer reducing to a different scheme on hot paths.

## Performance and portability notes

- **Atomic contention is the silent killer of throughput.** The cost of `qd.atomic_add(counter, 1)` from every thread is dominated by serialization at the location, not by the per-thread arithmetic. If many threads hit the same slot, prefer a two-stage scheme: per-warp / per-block reduction first (`qd.simt.block.reduce` if available, or `qd.simt.subgroup.reduce_add`), then a single atomic per warp / block.
- **Pair atomics with the right fence scope.** A bare atomic only orders the location it touches. To make other writes visible to readers that observe the new atomic value, follow the atomic with a fence: block-scope (`qd.simt.block.mem_sync()`) for shared-memory publishing, or grid-scope (`qd.simt.grid.memfence()`) for cross-block coordination.
- **`f64` atomics fall off the fast path** on most backends; if you only need monotonic accumulation, consider Kahan summation in registers and a single atomic-add at the end of the block.
- **`atomic_mul` is generally a CAS loop** under the hood; don't put it on the hot path.

### Atomic visibility scope across backends

Every `qd.atomic_*` is emitted at **device-wide scope**: visible to all threads on the GPU executing the kernel, but not required to be coherent with the host CPU mid-kernel. The host only observes results once the kernel completes, at which point the launcher's stream-sync flushes everything regardless. Choosing device scope (rather than the strongest "system" scope) lets every backend lower the op to a single hardware atomic instruction instead of a software CAS retry loop, which matters for correctness as much as for speed: under heavy contention, a CAS loop on a non-converging op like `atomic_xor` can livelock.

You don't normally need to think about scope as a user. It's listed here so the per-backend behaviour is explicit:

| Backend | Scope spelling in the IR | Representative hardware lowering (`atomic_xor`; `f32`/`f64` float min/max) |
|---|---|---|
| CPU (x86_64) | LLVM `seq_cst` (System) | `lock xor`; float min/max via `atomicrmw fmin`/`fmax` |
| CUDA (NVPTX) | LLVM `seq_cst` (System) | `atom.xor.b32` — single PTX op |
| AMDGPU | LLVM `seq_cst syncscope("agent")` | `flat_atomic_xor` / `global_atomic_xor` — single instruction |
| Vulkan / Metal (SPIR-V) | SPIR-V `Scope = Device` | `OpAtomicXor` — single op |

CPU and CUDA lower system-scope atomics directly to a single hardware instruction, so they leave the LLVM default alone. AMDGPU's LLVM backend, in contrast, refuses to use its native single-instruction atomics at system scope (it would have to add cache-flush instructions that don't exist for that op), and silently falls back to a CAS loop; setting `syncscope("agent")` is what unlocks the native `flat_atomic_xor` / `global_atomic_xor`. SPIR-V backends spell the same idea with the `Device` scope token. The user-visible semantics are identical across all four.

## Related

- [math](math.md) — `qd.math.*`, including the bit-counting helpers (`popcnt`, `clz`) commonly paired with atomics in select / compact patterns.
- `qd.simt.block.*` — block-scope barriers and memory fences (`qd.simt.block.mem_sync()`).
- `qd.simt.subgroup.*` — warp-scope reductions and shuffles, the recommended pre-aggregation step before an atomic.
- `qd.simt.grid.*` — device-scope memory fence (`qd.simt.grid.memfence()`).
- [parallelization](parallelization.md) — thread-synchronization patterns and how atomics fit into the broader synchronization story.
