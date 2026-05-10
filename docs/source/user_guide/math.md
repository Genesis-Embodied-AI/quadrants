# Math

`qd.math` is the quadrants standard library of math helpers.

This page currently documents only the bit-counting helpers. The broader `qd.math` surface is exported and usable today but is not yet documented here.

## Bit operations

Single-thread integer-register operations. They do not access memory and do not synchronize threads — each thread independently transforms a value in its own register.

| Op                  | CUDA                | AMDGPU                       | SPIR-V (Vulkan / Metal)                            |
|---------------------|---------------------|------------------------------|----------------------------------------------------|
| `qd.math.popcnt(x)` | i32, u32, i64, u64  | unsupported (codegen FIXME)  | any int (`OpBitCount`)                             |
| `qd.math.clz(x)`    | i32, u32, i64, u64  | i32, u32, i64, u64           | 32-bit only (`FindSMsb` / `FindUMsb`); 64-bit input is silently truncated |

On unsupported integer widths (e.g. `i8`, `i16`, `u16`) `popcnt` and `clz` both hit `QD_NOT_IMPLEMENTED`. CUDA `popcnt` is i32 / u32 / i64 / u64 only; AMDGPU `popcnt` is currently unsupported (FIXME below).

**FIXME (AMDGPU):** the AMDGPU `emit_extra_unary` override (`quadrants/codegen/amdgpu/codegen_amdgpu.cpp`) has a `clz` branch (added for `subgroup.segmented_reduce_*`) but no `popcnt` branch; `popcnt` falls through to `QD_NOT_IMPLEMENTED`. The test suite records this (`tests/python/test_unary_ops.py::test_popcnt` `xfail`s on AMDGPU; `::test_clz` no longer does).

The classic CUDA bit-tricks `__ffs` (find first set bit) and `__fns` (find n-th set bit in a mask) are not exposed.

### `qd.math.popcnt(x)`

Counts set bits in `x` and returns an `i32`. On CUDA, lowers to `__nv_popc` for 32-bit inputs and `__nv_popcll` for 64-bit inputs (i32 / u32 / i64 / u64 only; narrower widths and AMDGPU are unsupported). On SPIR-V, lowers to `OpBitCount`.

### `qd.math.clz(x)`

Counts leading zero bits in `x` and returns an `i32`. For a 32-bit input, `clz(0) = 32`; otherwise the result is in `[0, 31]`. On CUDA, lowers to `__nv_clz` for 32-bit inputs (i32 / u32) and `__nv_clzll` for 64-bit inputs (i64 / u64) — the underlying intrinsics are declared on signed types but operate on the bit pattern, so unsigned inputs route through the same intrinsic and no `bit_cast` is required. On AMDGPU, lowers to LLVM's `Intrinsic::ctlz` with `is_zero_undef = false` so `ctlz(0) == bitwidth` matches CUDA's behaviour; the LLVM intrinsic is polymorphic over integer types so all four widths (i32 / u32 / i64 / u64) work. On SPIR-V, lowers to GLSL.std.450 `FindSMsb` for signed inputs and `FindUMsb` for unsigned inputs, with `bitwidth - 1 - FindMSB` to convert MSB index into a leading-zero count; the implementation is hard-coded to 32 bits in the `bitwidth` constant, so 64-bit input silently truncates (FindMSB itself is GLSL.std.450 32-bit-only on most drivers anyway).

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

## Performance and portability notes

- `qd.math.popcnt` is supported on CUDA (i32 / u32 / i64 / u64) and SPIR-V (any integer width). AMDGPU is unsupported (FIXME above).
- `qd.math.clz` is supported on every backend for i32 / u32 / i64 / u64 inputs (no `bit_cast` workaround needed). The only remaining caveat is SPIR-V silently truncating 64-bit inputs to 32 bits — avoid u64 / i64 input on SPIR-V if the high half might be non-zero.

## Related

- [atomics](atomics.md) — atomic read-modify-write operations on global / shared memory; commonly paired with bit-counting in select / compact patterns.
- `qd.bit_cast` — reinterprets a value's bit pattern as another dtype.
