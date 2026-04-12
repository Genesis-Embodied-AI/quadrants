# Subgroup Size Control

GPU threads execute in groups called **subgroups** (also known as warps on NVIDIA, wavefronts on AMD, or SIMD groups on Apple). The subgroup size determines how many threads run in lockstep and share shuffle/ballot operations.

By default, Quadrants requests a subgroup size of 32 on Vulkan. You can override this per-loop with `qd.loop_config(subgroup_size=N)`, or query the device's supported range at runtime.

## Querying the device's subgroup size range

```python
qd.init(arch=qd.vulkan)

min_sg = qd.simt.min_subgroup_size()
max_sg = qd.simt.max_subgroup_size()
print(f"Subgroup size range: [{min_sg}, {max_sg}]")
```

The returned values depend on the backend:

| Backend | min | max | Notes |
|---------|-----|-----|-------|
| CUDA    | 32  | 32  | Fixed warp size |
| Vulkan  | 8–32 | 32–128 | Device-dependent |
| Metal   | 32  | 32  | Fixed SIMD group width |
| AMDGPU  | 32+ | 32+ | Typically 32 or 64 |

## Setting subgroup size on a loop

Use `qd.loop_config(subgroup_size=N)` immediately before a top-level for-loop:

```python
@qd.kernel
def k(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    qd.loop_config(block_dim=32, subgroup_size=32)
    for i in range(32):
        out[i] = qd.simt.subgroup.invocation_id()
```

### Backend-specific validation

Invalid subgroup sizes are rejected at kernel launch with a `ValueError`:

| Backend | Valid sizes | Invalid sizes |
|---------|-----------|---------------|
| CUDA    | 32 only   | All others raise `ValueError` |
| Metal   | 32 only   | All others raise `ValueError` |
| AMDGPU  | 32, 64    | All others raise `ValueError` |
| Vulkan  | Any value in `[min_subgroup_size, max_subgroup_size]` | Out-of-range values cause a driver error |
| CPU     | Not supported | Any value raises `ValueError` |

### Example: verifying subgroup IDs

```python
import numpy as np
import quadrants as qd

qd.init(arch=qd.vulkan)

N = 32
out = qd.ndarray(dtype=qd.i32, shape=(N,))

@qd.kernel
def read_subgroup_ids(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    qd.loop_config(block_dim=N, subgroup_size=N)
    for i in range(N):
        result[i] = qd.simt.subgroup.invocation_id()

read_subgroup_ids(out)
ids = out.to_numpy()
print(ids)  # [0, 1, 2, ..., 31]
```

## When to use subgroup size control

Most users do not need to set subgroup size explicitly — the default of 32 matches the native warp size on NVIDIA GPUs. Reasons to change it:

- **Portable SIMT code** across Vulkan devices with different native subgroup sizes.
- **Shuffle/ballot width control** when your algorithm depends on a specific number of lanes participating in subgroup operations.
- **Debugging** subgroup-related issues by testing with different sizes on Vulkan.
