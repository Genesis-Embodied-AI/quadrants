% Note for contributors: this page grows incrementally. Each PR in the
% tensor series (``) adds the section that
% documents whatever new user-visible behaviour landed in that PR.
% Sections describe only currently-shipped functionality.

# Tensors

Quadrants offers two underlying tensor implementations, [`qd.field`](tensor_types.md#global-field)
and [`qd.ndarray`](tensor_types.md#ndarray). They have different runtime/compile-time
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

For tensors whose elements are vectors or matrices, use `qd.Vector.tensor`
or `qd.Matrix.tensor`. They sit alongside the existing `qd.Vector.field` /
`qd.Vector.ndarray` and `qd.Matrix.field` / `qd.Matrix.ndarray` factories,
adding the per-tensor `backend=` knob with the same `qd.Backend` enum:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# A 1-D tensor of 4 length-3 vectors, on the field backend (default).
v = qd.Vector.tensor(3, qd.f32, shape=(4,))

# Same shape, on the ndarray backend.
u = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)

# A 1-D tensor of 3 (2x2) matrices, on the field backend.
m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,))
```

These mirror `qd.tensor()` for compound element types; `qd.Vector.tensor`
and `qd.Matrix.tensor` simply add the per-tensor `backend=` knob to the
existing `qd.Vector.*` / `qd.Matrix.*` factory family.

## Gradients

`needs_grad=True` works on every tensor factory and on every
backend, by passing the keyword through to the underlying
`qd.field` / `qd.ndarray` call:

```python
import quadrants as qd

qd.init(arch=qd.x64)

# Field-backed primal + grad.
a = qd.tensor(qd.f32, shape=(4,), needs_grad=True)
assert a.grad is not None

# Same on the ndarray backend.
b = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.NDARRAY, needs_grad=True)
assert b.grad is not None

# Kernels write through canonical indices on both primal and grad.
@qd.kernel
def write_grad(x: qd.template()):
    for i in range(4):
        x.grad[i] = i * 100.0

write_grad(a)
print(a.grad.to_numpy())   # [0., 100., 200., 300.]
```

Gradient buffers always share the canonical shape of the primal, on both
backends. The `needs_grad` keyword also passes through `qd.Vector.tensor`
and `qd.Matrix.tensor` for compound element types.

## Controlling physical layout

Different GPU kernels run best with different physical memory layouts —
some prefer the batch dimension contiguous (innermost), others want it
outermost. The `layout=` keyword lets you pick per-tensor:

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

All permutations up to rank 4 are supported (rank limit is Quadrants'
`quadrants_max_num_indices`, currently 12). `layout=None` and the identity
permutation (`(0, 1, ..., N-1)`) are equivalent and forward no `order=` to
the underlying `qd.field`.

```{note}
**Public `qd.tensor(..., backend=NDARRAY, layout=...)` still raises
``NotImplementedError`` for non-identity layouts.** The underlying
infrastructure — the AST subscript rewrite that turns canonical kernel
indexing into permuted physical access on layout-tagged ndarrays — is in
place as of an earlier change and is exercised in the test suite via the internal
``_with_layout`` helper. The user-facing factory is unblocked in an earlier change
together with the layout-aware torch interop.
```

`layout=` composes naturally with `needs_grad=True`: the grad SNode
inherits the same physical permutation as the primal, and both expose the
canonical shape.

```python
g = qd.tensor(qd.f32, shape=(4, 5), layout=(1, 0), needs_grad=True)
assert g.grad.shape == (4, 5)   # canonical
```

Quadrants rejects mismatched / invalid layouts up front:

```python
qd.tensor(qd.f32, shape=(4, 5), layout=(0, 1, 2))   # ValueError: wrong length
qd.tensor(qd.f32, shape=(4, 5), layout=(0, 0))      # ValueError: not a permutation
qd.tensor(qd.f32, shape=(4, 5), order="ji")         # TypeError: use layout=
```

A subsequent release will add a single polymorphic `qd.Tensor` annotation
that accepts either backend in one kernel signature.
