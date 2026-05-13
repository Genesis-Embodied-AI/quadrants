# Math

`qd.math` is the quadrants standard library of math helpers.

This page currently documents only the bit-counting helpers. The broader `qd.math` surface is exported and usable today but is not yet documented here.

## Bit operations

Single-thread integer-register operations. They do not access memory and do not synchronize threads - each thread independently transforms a value in its own register.

| Op                                       | CUDA                | AMDGPU                | SPIR-V (Vulkan / Metal)        | x64 (CPU)           |
|------------------------------------------|---------------------|-----------------------|--------------------------------|---------------------|
| `qd.math.popcnt(x)`                      | i32, u32, i64, u64  | i32, u32, i64, u64    | i32, u32, i64, u64 (`OpBitCount`) | i32, u32, i64, u64  |
| `qd.math.clz(x)`                         | i32, u32, i64, u64  | i32, u32, i64, u64    | i32, u32, i64, u64 (`FindUMsb`-based) \*  | i32, u32, i64, u64  |
| `qd.math.ffs(x)`                         | i32, u32, i64, u64  | i32, u32, i64, u64    | i32, u32, i64, u64 (`FindILsb`-based) \*  | i32, u32, i64, u64  |
| `qd.math.fns(mask, base, offset)`        | u32 (PTX `fns` instruction) | u32 (portable @qd.func) | u32 (portable @qd.func)        | u32 (portable @qd.func) |

All four ops **return `i32` on every backend**, regardless of input width. This matches CUDA libdevice (`__nv_popc` / `__nv_clz` / `__nv_ffs` all return `int`) and the natural width of the AMDGPU SALU bit-count / leading-zero instructions (`s_bcnt1_i32_b64`, `s_flbit_i32_b64`); SPIR-V and x64 truncate down to `i32` so the same kernel source has the same return type everywhere. The result is always non-negative and fits in 7 bits (`popcnt` ≤ 64, `clz` ≤ 64, `ffs` ≤ 64, `fns` ≤ 31 plus the `0xFFFFFFFF` not-found sentinel for the `u32` case).

\* On SPIR-V the 64-bit path (i64 / u64) for `clz` and `ffs` is synthesised from two ext-inst calls on the 32-bit halves plus an `OpSelect`, since `GLSL.std.450 FindUMsb` and `FindILsb` are both 32-bit-only. The runtime device must advertise the `Int64` SPIR-V capability (Vulkan: `shaderInt64`); this is the same precondition any other 64-bit op would impose. On unsupported integer widths (e.g. `i8`, `i16`, `u16`) `clz`, `popcnt` and `ffs` hit `QD_NOT_IMPLEMENTED` on every backend.

### `qd.math.popcnt(x)`

Counts set bits in `x` and returns an `i32`. On CUDA, lowers to `__nv_popc` for 32-bit inputs and `__nv_popcll` for 64-bit inputs (i32 / u32 / i64 / u64; narrower widths are unsupported). On AMDGPU, lowers to the portable `llvm.ctpop` intrinsic (same dtype set as CUDA), which the AMDGPU LLVM backend further lowers to a native bit-count instruction (e.g. `s_bcnt1_i32_b64` for the 64-bit SALU path). On SPIR-V, lowers to `OpBitCount`. On x64, lowers to `llvm.ctpop`. Every backend truncates to `i32` to match the unified return type; the truncation is free since 64-bit `popcnt` results never exceed 64.

### `qd.math.clz(x)`

Counts leading zero bits in `x` and returns an `i32`. For a 32-bit input, `clz(0) = 32`; otherwise the result is in `[0, 31]`. The count is over the unsigned bit pattern, so `clz(-1) == 0` and `clz(0x7FFFFFFF) == 1`. Signed and unsigned inputs lower to the same intrinsic on every backend (LLVM IR is signless for integers; SPIR-V `FindUMsb` is unsigned by definition), so `qd.math.clz(qd.u32(x))` and `qd.math.clz(qd.bit_cast(x, qd.i32))` are equivalent. On CUDA, lowers to `__nv_clz` (32-bit) and `__nv_clzll` (64-bit). On AMDGPU, lowers to the portable `llvm.ctlz` intrinsic with `is_zero_undef = false` (matching `clz(0) = bitwidth`). On SPIR-V, the 32-bit case lowers to `GLSL.std.450 FindUMsb` followed by `31 - FindUMsb`. The 64-bit case is synthesised from a hi/lo decomposition: shift the operand right by 32 to get the high i32 half, truncate for the low half, run `FindUMsb` on each, and select `31 - FindUMsb(hi)` if the high half is non-zero or `63 - FindUMsb(lo)` otherwise. `FindUMsb` returns `-1` on a zero input, so `clz(0) == 64` falls out naturally.

### `qd.math.ffs(x)`

Finds the lowest set bit in `x` and returns its **1-indexed** position as an `i32`, with `ffs(0) == 0` (matching the CUDA `__ffs` convention). Otherwise the result is in `[1, bitwidth(x)]`, e.g. `ffs(1) == 1`, `ffs(2) == 2`, `ffs(0x80000000) == 32`. The count is over the unsigned bit pattern, so `ffs(-1) == 1` regardless of input signedness. On CUDA, lowers to libdevice's `__nv_ffs` (32-bit) / `__nv_ffsll` (64-bit), which already encode the `ffs(0) == 0` contract. On CPU and AMDGPU, lowers to `llvm.cttz` with `is_zero_undef = false` plus an explicit `select` for the zero case (cttz returns bitwidth on zero, so `cttz + 1` would otherwise yield `bitwidth + 1`). On SPIR-V, the 32-bit case lowers to `GLSL.std.450 FindILsb` plus a `+1` and a zero-input select; the 64-bit case is synthesised from a hi/lo decomposition that consults the low half first (since "first" set bit means lowest-indexed) and falls back to the high half offset by 32 when the low half is zero.

### `qd.math.fns(mask, base, offset)`

Find the `|offset|`-th set bit in a 32-bit `mask`, scanning from bit position `base` in the direction implied by the sign of `offset`. Mirrors the CUDA `__nv_fns` / PTX `fns.b32` instruction; takes three `u32 / u32 / i32` operands and returns a `u32`. Returns `0xFFFFFFFF` when the requested set bit does not exist.

* `offset > 0`: scan upward (towards higher bit indices) starting at `base` (inclusive). Returns the bit position of the `offset`-th set bit found at or above `base`.
* `offset < 0`: scan downward (towards lower bit indices) starting at `base` (inclusive). Returns the bit position of the `|offset|`-th set bit found at or below `base`.
* `offset == 0`: returns `base` if `mask & (1 << base)` is non-zero, else `0xFFFFFFFF`.

On CUDA, lowers to a single `fns.b32` PTX instruction (available since SM 5.0) emitted via inline asm, since the slim libdevice we ship does not include `__nv_fns`. On CPU, AMDGPU and SPIR-V, lowers to a portable `@qd.func` that loops over the 32 bit positions; the body is fully unrolled by each backend's lowering pipeline.

The CUDA fast path is the typical use case for QIPC-style work-stealing patterns where a thread needs to identify the *n*-th cooperating peer in a `__ballot` mask without a serial scan.

## Examples

### Bitset population count

```python
@qd.kernel
def count_bits(masks: qd.types.NDArray[qd.u32, 1], total: qd.types.NDArray[qd.i32, 1]) -> None:
    n = 0
    for i in range(masks.shape[0]):
        n += qd.math.popcnt(masks[i])
    qd.atomic_add(total[0], n)
```

### Highest set bit (Morton-code depth)

```python
@qd.func
def msb(x: qd.i32) -> qd.i32:
    return 31 - qd.math.clz(x)
```

### Iterating set bits in a mask

```python
@qd.func
def lowest_set_bit_index(mask: qd.u32) -> qd.i32:
    # Returns -1 when mask == 0; otherwise the 0-indexed position of the lowest set bit.
    return qd.math.ffs(mask) - 1
```

## Related

- [atomics](atomics.md) - atomic read-modify-write operations on global / shared memory; commonly paired with bit-counting in select / compact patterns.
- `qd.bit_cast` - reinterprets a value's bit pattern as another dtype.
