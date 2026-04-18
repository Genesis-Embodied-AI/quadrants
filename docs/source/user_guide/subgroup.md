# Subgroup shuffles

Subgroup shuffle operations let threads within the same subgroup (warp on NVIDIA, wave on AMD) read each other's register values directly, without using shared memory or barriers. They are the building block for fast in-warp data exchange — broadcasts, neighbour exchanges, permutations — and are used internally by `Tile16x16` (see [tile16](tile16.md)).

Shuffle ops live under `qd.simt.subgroup` and are written so the same Python source compiles to the right vendor primitive on each backend.

## What's available

| Op                             | CUDA | AMDGPU | SPIR-V (Vulkan) | dtypes                       |
|--------------------------------|------|--------|-----------------|------------------------------|
| `subgroup.shuffle(v, idx)`     | yes  | yes    | yes             | i32, u32, f32, f64, i64, u64 |

Other flavours (`shuffle_up`, `shuffle_down`, `shuffle_xor`) are exposed in the Python module but are not yet implemented across backends. Calling them will fail at codegen. Use `shuffle` with an explicit lane index in the meantime — every shuffle pattern can be expressed that way.

## Semantics

`shuffle(value, index)` — each lane returns the `value` held by the lane whose subgroup-local id equals `index`. It does not move data through memory and does not synchronise across subgroups — only the lanes within one subgroup participate.

- `value` is a scalar in a register. Supported dtypes are 32-bit and 64-bit signed/unsigned ints and `f32`/`f64`. (64-bit types are split into two 32-bit shuffles on AMDGPU; CUDA dispatches to its native 64-bit helpers.)
- `index` is a `u32`. If `index` is out of range for the active subgroup the result is implementation-defined, so pass `subgroup.invocation_id()`-derived values or known-good lane ids.
- The op is issued under a full active mask on CUDA (`0xFFFFFFFF`). Call it from uniform control flow; calling from divergent control flow is undefined on most backends. (this means: all threads have to execute the shuffle)

Subgroup size varies by backend (32 on NVIDIA, 32 or 64 on AMD, 32 in Vulkan compute on most GPUs).

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

After the kernel, every lane in a subgroup holds the original value of its lane 0.

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

## Performance notes

- Shuffles are register-to-register on CUDA (`__shfl_sync`) and on SPIR-V where the GPU has hardware support — typically a handful of cycles, no memory traffic.
- AMDGPU `shuffle` uses `ds_permute`/`ds_bpermute` (LDS-routed, roughly tens of cycles).
- 64-bit dtypes (`i64`, `u64`, `f64`) are emulated as two 32-bit shuffles on AMDGPU. Prefer 32-bit values when you have a choice.

## Related

- [tile16](tile16.md) — `Tile16x16` builds on `subgroup.shuffle` to implement register-resident 16x16 matrix tiles.
- `subgroup.invocation_id()` — returns this lane's subgroup-local index.
- `subgroup.size()` — returns the active subgroup size.
