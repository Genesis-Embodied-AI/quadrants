# Math

`qd.math` is the quadrants standard library of math helpers.

This page currently documents only the bit-counting helpers. The broader `qd.math` surface is exported and usable today but is not yet documented here.

## Bit operations

Single-thread integer-register operations. They do not access memory and do not synchronize threads — each thread independently transforms a value in its own register.

| Op                  | What it returns                              | i32 | u32 | i64 | u64 |
|---------------------|----------------------------------------------|-----|-----|-----|-----|
| `qd.math.popcnt(x)` | Number of set bits in `x`                    | yes | yes | yes | yes |
| `qd.math.clz(x)`    | Number of leading zero bits in `x`           | yes | \*  | yes | \*  |

\* `qd.math.clz` on CUDA currently rejects unsigned 32- and 64-bit inputs (`QD_NOT_IMPLEMENTED`); cast through `qd.bit_cast(x, qd.i32)` / `qd.i64` as a workaround. On SPIR-V, `qd.math.clz` is hard-coded to 32-bit (`FindMSB`); 64-bit input is silently truncated.

The classic CUDA bit-tricks `__ffs` (find first set bit) and `__fns` (find n-th set bit in a mask) are not exposed; for a leading-zero count of a u32, the `bit_cast` workaround above is the canonical approach.

### `qd.math.popcnt(x)`

Counts set bits in `x` and returns an `i32`. Lowers to `__popc` / `__popcll` on CUDA, `OpBitCount` on SPIR-V, `__builtin_amdgcn_popcnt` on AMDGPU. Defined for all integer dtypes.

### `qd.math.clz(x)`

Counts leading zero bits in `x` and returns an `i32`. For a 32-bit input, `clz(0) = 32`; otherwise the result is in `[0, 31]`. Lowers to `__nv_clz` / `__nv_clzll` on CUDA, `FindMSB` on SPIR-V (with `bitwidth - 1 - FindMSB` to convert MSB index into leading-zero count), `__builtin_amdgcn_sffbh_i32` on AMDGPU. See the cross-backend caveats in the support table.

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

For `qd.u32` input on CUDA, cast first: `qd.math.clz(qd.bit_cast(x, qd.i32))`.

## Performance and portability notes

- `qd.math.popcnt` is fully cross-backend.
- `qd.math.clz` has the dtype caveats noted above. Tests that depend on `qd.math.clz` over u32 or u64 should `bit_cast` to the matching signed type for portability.

## Related

- [atomics](atomics.md) — atomic read-modify-write operations on global / shared memory; commonly paired with bit-counting in select / compact patterns.
- `qd.bit_cast` — reinterprets a value's bit pattern as another dtype, used as a workaround for the `clz` u32 / u64 caveats above.
