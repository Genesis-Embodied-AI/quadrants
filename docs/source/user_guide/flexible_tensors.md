% Note for contributors: this page grows incrementally. Each PR in the
% flexible-tensors series (`hp/tensor-stork-N`) adds the section that
% documents whatever new user-visible behaviour landed in that PR.
% Sections describe only currently-shipped functionality.

# Flexible tensors

Quadrants offers two underlying tensor implementations, [`qd.field`](#fields)
and [`qd.ndarray`](#ndarrays). They have different runtime/compile-time
trade-offs, and different physical memory layouts can suit different kernels.

The flexible-tensors API lets you pick both the **backend** and (in a future
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

## Allocating a tensor with `qd.tensor()`

`qd.tensor(dtype, shape, backend=...)` is a thin dispatcher over `qd.field` and
`qd.ndarray`. It selects the underlying allocator based on the `backend=`
keyword:

```python
import quadrants as qd

qd.init(arch=qd.x64)

a = qd.tensor(qd.f32, shape=(4, 5))                                 # field (default)
b = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY)     # ndarray

assert isinstance(a, qd.ScalarField)
assert isinstance(b, qd.Ndarray)
```

The default backend is `qd.Backend.FIELD` to match the long-standing Quadrants
default.

Any extra keyword arguments are forwarded verbatim to the underlying
`qd.field` or `qd.ndarray` call, so backend-specific options remain available:

```python
# Field-only kwargs like order= just pass through.
c = qd.tensor(qd.f32, shape=(4, 5), order="ji")
```

Passing a non-`Backend` value raises `ValueError`:

```python
qd.tensor(qd.f32, shape=(3,), backend="field")  # ValueError
```

Integer values (`0`, `1`) are accepted because `Backend` is an `IntEnum`, but
prefer the named members for clarity at call sites.

## Vector and matrix tensors

For tensors whose elements are vectors or matrices, use `qd.tensor_vec` and
`qd.tensor_mat`. They dispatch over `qd.Vector.field` / `qd.Vector.ndarray`
and `qd.Matrix.field` / `qd.Matrix.ndarray` respectively, with the same
`backend=` keyword:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# A 1-D tensor of 4 length-3 vectors, on the field backend (default).
v = qd.tensor_vec(3, qd.f32, shape=(4,))

# Same shape, on the ndarray backend.
u = qd.tensor_vec(3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)

# A 1-D tensor of 3 (2x2) matrices, on the field backend.
m = qd.tensor_mat(2, 2, qd.f32, shape=(3,))
```

These match the existing `qd.Vector.*` / `qd.Matrix.*` factories one-for-one;
`qd.tensor_vec` / `qd.tensor_mat` simply add the per-tensor `backend=` knob.

Subsequent releases will add a `qd.tensor_annotation(backend)` helper for
kernel argument typing, and a `layout=` keyword for per-tensor physical-memory
layout.
