# Tensors

Quadrants offers two underlying tensor implementations, [`qd.field` and `qd.ndarray`](tensor_types.md). They have different runtime/compile-time trade-offs, and different physical memory layouts can suit different kernels.

The tensor API lets you pick both the **backend** and the **physical memory layout** on a per-tensor basis at allocation time. The rest of the system (kernels, fastcache, autograd) stays out of the way.

See [`tensor_types`](tensor_types.md), [`scalar_tensors`](scalar_tensors.md), and [`matrix_vector`](matrix_vector.md) for the underlying tensor primitives.

## Choosing a backend: `qd.Backend`

`qd.Backend` is an `IntEnum` with two members:

| Member | Underlying type | When to prefer |
|---|---|---|
| `qd.Backend.FIELD` | `qd.field` | Faster at runtime; recompiles when any dimension size changes. |
| `qd.Backend.NDARRAY` | `qd.ndarray` | Slower at runtime but avoids recompilation when sizes change. |

The choice is per tensor: a single program can freely mix backends.

## Allocating a tensor with `qd.tensor()`

`qd.tensor(dtype, shape, backend=...)` is a thin dispatcher over `qd.field` and `qd.ndarray`. It selects the underlying allocator based on the `backend=` keyword:

```python
import quadrants as qd

qd.init(arch=qd.x64)

a = qd.tensor(qd.f32, shape=(4, 5))                                 # ndarray (default)
b = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.FIELD)       # field

assert isinstance(a, qd.Tensor)
assert isinstance(b, qd.Tensor)
```

`qd.tensor()` (and the `qd.Vector.tensor` / `qd.Matrix.tensor` siblings) returns a `qd.Tensor` wrapper that uniformly forwards a fixed surface (`shape`, `dtype`, `layout`, `to_numpy`, `from_numpy`, `to_torch`, `from_torch`, `to_dlpack`, `fill`, `copy_from`, `grad`, host-side `__getitem__` / `__setitem__`, pickle) regardless of which backend it wraps. Drop down to the bare impl with `t._unwrap()` (returns the underlying `qd.Ndarray` or `qd.ScalarField`) only if you need a backend-specific knob.

The default backend is `qd.Backend.NDARRAY`: it avoids recompilation when sizes change.

## Vector and matrix tensors

For tensors whose elements are vectors or matrices, use `qd.Vector.tensor` or `qd.Matrix.tensor`. They dispatch over `qd.Vector.field` / `qd.Vector.ndarray` and `qd.Matrix.field` / `qd.Matrix.ndarray` respectively, with the same `backend=` keyword:

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

`needs_grad=True` works on every tensor factory and on every backend, by passing the keyword through to the underlying `qd.field` / `qd.ndarray` call:

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

Gradient buffers always share the canonical shape of the primal, on both backends. The `needs_grad` keyword also passes through `qd.Vector.tensor` and `qd.Matrix.tensor` for compound element types.

## Controlling physical layout

Tweaking the memory layout on a per-tensor basis is commonly used to improve runtime performance. In practice, tuning axis order is sufficient in most cases. For advanced users seeking finer-grained control over the memory layout, see the SNode API (`qd.root`).

The `layout=` keyword lets you pick per-tensor:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# Default (canonical) layout: same order as the canonical shape.
a = qd.tensor(qd.f32, shape=(N, B))

# Transposed storage: axis 1 (batch) becomes the outer SNode, axis 0 inner.
b = qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))
```

`layout` is a tuple of `int` listing the **canonical axis index at each successive memory-nesting level, outermost first**. It must be a permutation of `range(len(shape))`. The canonical (logical) shape that you pass and that `tensor.shape` returns is *not* affected by `layout`:

```python
b = qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))
assert b.shape == (N, B)        # canonical shape, unchanged
b[i, j] = ...                   # canonical indexing in kernels still works
```

Any permutation is supported, up to Quadrants' `quadrants_max_num_indices` (currently 12). `layout=None` and the identity permutation (`(0, 1, ..., N-1)`) are equivalent and forward no permutation to the underlying allocator.

Quadrants rejects mismatched / invalid layouts up front:

```python
qd.tensor(qd.f32, shape=(4, 5), layout=(0, 1, 2))   # ValueError: wrong length
qd.tensor(qd.f32, shape=(4, 5), layout=(0, 0))      # ValueError: not a permutation
qd.tensor(qd.f32, shape=(4, 5), order="ji")         # TypeError: use layout=
```

## Interop with NumPy and PyTorch

Every Python-side accessor — `tensor.shape`, `tensor.layout`, `tensor.to_numpy()`, `tensor.to_numpy(dtype=...)`, `tensor.from_numpy(...)`, `tensor.to_torch(device=...)`, `tensor.from_torch(...)`, `tensor.to_dlpack()` (and therefore anything built on top of it like `torch.utils.dlpack.from_dlpack`) — returns the **canonical view**: the shape you passed at allocation time, indexed in canonical axis order.

`layout=` is purely an internal performance hint. The data lives in permuted physical storage, but Python callers never have to reason about that:

```python
a = qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))
assert a.shape == (N, B)                 # canonical
assert a.layout == (1, 0)                # introspectable
assert a.to_numpy().shape == (N, B)      # canonical view of the same data

# Round-trips work in canonical-shape terms.
src = np.zeros((N, B), dtype=np.float32)
a.from_numpy(src)
assert (a.to_numpy() == src).all()

# DLPack carries the canonical shape with permuted strides; the resulting torch tensor is a transposed view of the underlying buffer (no data movement until you call ``.contiguous()``).
import torch
t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
assert tuple(t.shape) == (N, B)

# ``to_torch`` / ``from_torch`` are equivalent on either backend.
out = a.to_torch()
assert tuple(out.shape) == (N, B)
a.from_torch(out)
```

The exact same surface is available on both backends — switching `qd.tensor(..., backend=qd.Backend.FIELD/NDARRAY)` does not require any other code change at the call site.

Gradient buffers behave identically: `a.grad.to_numpy()` returns the canonical view of the gradient.

## Annotating kernel arguments: `qd.Tensor`

Kernel parameter annotations use `qd.Tensor` regardless of backend. The same class doubles as the wrapper class returned by `qd.tensor()`, so the annotation and the runtime values agree:

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

The kernel argument is unwrapped to the bare impl before the template-mapper / AST sees it, so kernel bodies still write `x[i, j]` and pay no per-call cost for the wrapper.

## Pickle

`qd.Tensor` objects are picklable on **both** backends, including under non-identity layouts. Round-trip (pickle then unpickle) preserves the canonical data, the dtype, the shape, and the layout:

```python
import pickle
import quadrants as qd

qd.init(arch=qd.x64)

a = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD, layout=(1, 0))
a.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))

restored = pickle.loads(pickle.dumps(a))
assert isinstance(restored, qd.Tensor)
assert restored.shape == (3, 4)
assert restored.layout == (1, 0)
assert (restored.to_numpy() == a.to_numpy()).all()
```

## Wrapping a bare tensor: `qd.wrap`

If you have a bare `qd.field` / `qd.ndarray` / `qd.Vector.field` / `qd.Matrix.field` / `qd.Vector.ndarray` / `qd.Matrix.ndarray` impl (e.g. from older code or library boundaries) and want the unified `qd.Tensor` surface around it, use `qd.wrap(impl)`. It picks the most specific subclass (`Tensor`, `VectorTensor`, `MatrixTensor`):

```python
import quadrants as qd

qd.init(arch=qd.x64)

a = qd.ndarray(qd.f32, shape=(4, 5))
t = qd.wrap(a)
assert isinstance(t, qd.Tensor)
assert t._unwrap() is a   # same underlying impl
```

`qd.wrap` is the only sanctioned way to construct a wrapper around a bare impl after the fact. The `qd.Tensor(impl)` constructor itself rejects double-wrapping so you can't accidentally end up with a `Tensor` containing a `Tensor`.

## Cross-backend `copy_from` is not supported

`tensor.copy_from(other)` requires both tensors to share the same backend. Mixed-backend copies are not supported:

```python
a = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.FIELD)
b = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)
a.copy_from(b)   # raises: cross-backend copy unsupported
```

If you genuinely need to move data across backends, route it through Torch: `a.from_torch(b.to_torch())`.

## Known asymmetry: real-dtype `.grad` stub on the field backend

For tensors of a real (`f32` / `f64`) dtype allocated **without** `needs_grad=True`, the field backend currently allocates a zombie gradient stub anyway, so `t.grad` returns a wrapper around it. The ndarray backend correctly reports `t.grad is None` in the same case:

```python
t_field = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.FIELD)
t_nd    = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)

t_field.grad   # currently a Tensor wrapper around a zombie field
t_nd.grad      # None
```

Use `needs_grad=True` if you intend to read `.grad`. Integer dtypes are symmetric (`grad is None` on both backends regardless of `needs_grad`).
