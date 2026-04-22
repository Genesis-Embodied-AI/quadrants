# Tensors

Quadrants offers two underlying tensor implementations, `qd.field`
and `qd.ndarray`. They have different runtime/compile-time
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
| `qd.Backend.FIELD` | `qd.field` | Faster at runtime; recompiles when any dimension size changes. |
| `qd.Backend.NDARRAY` | `qd.ndarray` | Slightly slower at runtime but avoids recompilation when sizes change. |

The choice is per tensor: a single program can freely mix backends.

## Allocating a tensor with `qd.tensor()`

`qd.tensor(dtype, shape, backend=...)` is a thin dispatcher over `qd.field` and
`qd.ndarray`. It selects the underlying allocator based on the `backend=`
keyword:

```python
import quadrants as qd

qd.init(arch=qd.x64)

a = qd.tensor(qd.f32, shape=(4, 5))                                 # ndarray (default)
b = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.FIELD)       # field

assert isinstance(a, qd.Ndarray)
assert isinstance(b, qd.ScalarField)
```

The default backend is `qd.Backend.NDARRAY`: it avoids recompilation when
sizes change.

## Vector and matrix tensors

For tensors whose elements are vectors or matrices, use `qd.Vector.tensor`
or `qd.Matrix.tensor`. They dispatch over `qd.Vector.field` /
`qd.Vector.ndarray` and `qd.Matrix.field` / `qd.Matrix.ndarray` respectively,
with the same `backend=` keyword:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# A 1-D tensor of 4 length-3 vectors (ndarray backend, default).
v = qd.Vector.tensor(3, qd.f32, shape=(4,))

# Same shape, on the field backend.
u = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=qd.Backend.FIELD)

# A 1-D tensor of 3 (2x2) matrices, ndarray backend.
m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,))
```

## Gradients

`needs_grad=True` works on every tensor factory and on every
backend, by passing the keyword through to the underlying
`qd.field` / `qd.ndarray` call:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# Ndarray-backed primal + grad (default backend).
a = qd.tensor(qd.f32, shape=(4,), needs_grad=True)
assert a.grad is not None

# Same on the field backend.
b = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.FIELD, needs_grad=True)
assert b.grad is not None

# Kernels write through canonical indices on both primal and grad.
@qd.kernel
def write_grad(x: qd.Tensor):
    for i in range(4):
        x.grad[i] = i * 100.0

write_grad(a)
print(a.grad.to_numpy())   # [0., 100., 200., 300.]
```

Gradient buffers always share the canonical shape of the primal, on both
backends. The `needs_grad` keyword also passes through `qd.Vector.tensor`
and `qd.Matrix.tensor` for compound element types.

## Controlling physical layout

The `layout=` keyword lets you pick per-tensor:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# Default (canonical) layout: same order as the canonical shape.
a = qd.tensor(qd.f32, shape=(N, B))

# Transposed storage: axis 1 (batch) becomes the outer SNode, axis 0 inner.
b = qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))
```

`layout` is a tuple of `int` listing the **canonical axis index at each
successive memory-nesting level, outermost first**. It must be a permutation
of `range(len(shape))`. The canonical (logical) shape that you pass and that
`tensor.shape` returns is *not* affected by `layout`:

```python
b = qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))
assert b.shape == (N, B)        # canonical shape, unchanged
b[i, j] = ...                   # canonical indexing in kernels still works
```

Any permutation is supported, up to Quadrants' `quadrants_max_num_indices`
(currently 12). `layout=None` and the identity permutation
(`(0, 1, ..., N-1)`) are equivalent and forward no permutation to the
underlying allocator.

Quadrants rejects mismatched / invalid layouts up front:

```python
qd.tensor(qd.f32, shape=(4, 5), layout=(0, 1, 2))   # ValueError: wrong length
qd.tensor(qd.f32, shape=(4, 5), layout=(0, 0))      # ValueError: not a permutation
qd.tensor(qd.f32, shape=(4, 5), order="ji")         # TypeError: use layout=
```

## Interop with NumPy and PyTorch

Every Python-side accessor — `tensor.shape`, `tensor.layout`,
`tensor.to_numpy()`, `tensor.to_numpy(dtype=...)`,
`tensor.from_numpy(...)`, `tensor.to_torch(device=...)`,
`tensor.from_torch(...)`, `tensor.to_dlpack()` (and therefore anything
built on top of it like `torch.utils.dlpack.from_dlpack`) — returns the
**canonical view**: the shape you passed at allocation time, indexed in
canonical axis order.

`layout=` is purely an internal performance hint. The data lives in
permuted physical storage, but Python callers never have to reason
about that:

```python
a = qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))
assert a.shape == (N, B)                 # canonical
assert a.layout == (1, 0)                # introspectable
assert a.to_numpy().shape == (N, B)      # canonical view of the same data

# Round-trips work in canonical-shape terms.
src = np.zeros((N, B), dtype=np.float32)
a.from_numpy(src)
assert (a.to_numpy() == src).all()

# DLPack carries the canonical shape with permuted strides; the
# resulting torch tensor is a transposed view of the underlying buffer
# (no data movement until you call ``.contiguous()``).
import torch
t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
assert tuple(t.shape) == (N, B)

# ``to_torch`` / ``from_torch`` are equivalent on either backend.
out = a.to_torch()
assert tuple(out.shape) == (N, B)
a.from_torch(out)
```

The exact same surface is available on both backends — switching
`qd.tensor(..., backend=qd.Backend.FIELD/NDARRAY)` does not require
any other code change at the call site.

Gradient buffers behave identically: `a.grad.to_numpy()` returns the
canonical view of the gradient.

## Annotating kernel arguments: `qd.Tensor`

Kernel parameter annotations use `qd.Tensor` regardless of backend:

```python
import quadrants as qd

qd.init(arch=qd.x64)

@qd.kernel
def fill(x: qd.Tensor):
    for i in range(x.shape[0]):
        x[i] = i

a = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.FIELD)
b = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)

fill(a)   # field branch
fill(b)   # ndarray branch
```
