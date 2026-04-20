% Note for contributors: this page grows incrementally. Each PR in the
% tensor series (``) adds the section that
% documents whatever new user-visible behaviour landed in that PR.
% Sections describe only currently-shipped functionality.

# Tensors

Quadrants offers two underlying tensor implementations, [`qd.field`](#fields)
and [`qd.ndarray`](#ndarrays). They have different runtime/compile-time
trade-offs, and different physical memory layouts can suit different kernels.

The tensor API lets you pick both the **backend** and (in a future
release) the **physical layout** on a per-tensor basis at allocation time.
The rest of the system (kernels, fastcache, autograd) stays out of the way.

This page documents the user-facing API as it lands. See
[`tensor_types`](tensor_types.md), [`scalar_tensors`](scalar_tensors.md),
and [`matrix_vector`](matrix_vector.md) for the underlying tensor primitives.

## Choosing a backend: `qd.Backend`

`qd.Backend` is an `IntEnum` with two members:

| Member | Underlying type | When to prefer |
|---|---|---|
| `qd.Backend.FIELD` | `qd.field` | Faster at runtime; recompiles when any dimension size changes. Best for tensors whose shape is effectively static across a run. |
| `qd.Backend.NDARRAY` | `qd.ndarray` | Slightly slower at runtime but avoids recompilation when sizes change. Best for tensors whose shape varies frequently (dynamic batch sizes, growing buffers). |

```python
import quadrants as qd

qd.Backend.FIELD       # IntEnum member, value 0
qd.Backend.NDARRAY     # IntEnum member, value 1

int(qd.Backend.FIELD)  # 0
qd.Backend["FIELD"]    # qd.Backend.FIELD
qd.Backend(1)          # qd.Backend.NDARRAY
```

The choice is per tensor: a single program can freely mix backends.

Subsequent releases will use this enum to drive the `qd.tensor(...)` factory
and per-tensor layout selection.
