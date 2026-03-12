# Python backend

The Python backend executes Quadrants kernels and functions as plain Python, using PyTorch tensors for storage. No compilation or GPU is required.

This is useful for:
- Debugging kernel logic with a standard Python debugger
- Running on systems without a GPU or native compiler
- Quick iteration without compilation overhead

## Requirements

PyTorch must be installed. No other dependencies beyond the standard Quadrants install are needed.

## Quick start

```python
import quadrants as qd

qd.init(qd.python)

a = qd.ndarray(qd.f32, shape=(10,))

@qd.kernel
def fill(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for i in range(a.shape[0]):
        a[i] = float(i) * 2.0

fill(a)
```

## Supported features

- `@qd.kernel` and `@qd.func` — executed directly as Python functions
- `qd.field()` — scalar, vector, and matrix fields
- `qd.ndarray()` — scalar, vector, and matrix ndarrays
- Struct fields (`qd.types.struct`)
- Atomics (`atomic_add`, `atomic_sub`, `atomic_mul`, `atomic_min`, `atomic_max`, `atomic_and`, `atomic_or`, `atomic_xor`)
- `qd.grouped(field)` — struct-for iteration over field indices
- `qd.ndrange()`
- `qd.static()`
- `qd.cast()`
- `qd.math.isnan()`, `qd.math.isinf()`
- Dtype constructors (`qd.f32(x)`, `qd.i32(x)`)
- `qd.Vector()`, `qd.Matrix()`

## Limitations

- **Single-threaded.** Kernels run sequentially; there is no parallelism.
- **Atomics are plain sequential ops.** Correct in single-threaded execution but not a test of real atomic semantics.
- **`loop_config()` and `sync()` are no-ops.**
- **No SNode trees.** Fields are flat PyTorch tensors, not backed by the SNode system.
- **Performance is not representative** of compiled backends. Do not use for benchmarking.
- **GPU-specific features** (shared memory, block-level intrinsics, etc.) are not available.

## Storage model

Fields and ndarrays are instances of `PyTensor`, a `torch.Tensor` subclass. You can interoperate with PyTorch directly:

```python
qd.init(qd.python)
a = qd.field(qd.math.vec3, shape=(10,))

# a is a torch.Tensor under the hood
import torch
assert isinstance(a, torch.Tensor)
assert a.size() == torch.Size([10, 3])
```

Each tensor has the following attributes matching the standard Quadrants field/ndarray convention:
- `_tc` — the tensor itself
- `_T_tc` — transposed view (first batch dim moved to front)
- `_np` — numpy view
- `_T_np` — transposed numpy view

### Batch shape vs real shape

The `.shape` property returns the **batch dimensions only**, consistent with how Quadrants kernels index fields. For example, a `vec3` field of shape `(10,)` reports `.shape == (10,)` even though the underlying tensor has size `(10, 3)`. Use `.size()` to get the full torch shape.

```python
f = qd.field(qd.math.vec3, shape=(10,))
f.shape     # torch.Size([10])    — batch dims, what kernels see
f.size()    # torch.Size([10, 3]) — real tensor shape
```
