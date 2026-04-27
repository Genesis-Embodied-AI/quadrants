# Numpy and Torch interop

Quadrants provides interop with both numpy and PyTorch. There are three mechanisms:
- **Copy-based**: convert data between quadrants fields/ndarrays and numpy arrays or torch tensors
- **Zero-copy via DLPack**: obtain a torch tensor or numpy array that aliases the underlying Quadrants memory
- **Direct pass-through**: pass torch tensors directly into kernels as ndarray arguments (zero-copy)

## Copy-based interop

### Fields

Fields support `to_numpy()`, `from_numpy()`, `to_torch()`, and `from_torch()`:

```python
import numpy as np
import quadrants as qd

qd.init(arch=qd.gpu)

f = qd.field(qd.f32, shape=(4, 4))

# numpy -> field
arr = np.ones((4, 4), dtype=np.float32) * 3.0
f.from_numpy(arr)

# field -> numpy
result = f.to_numpy()
print(result[0, 0])  # 3.0
```

With torch:

```python
import torch

f = qd.field(qd.f32, shape=(4, 4))

# torch -> field
t = torch.ones(4, 4, dtype=torch.float32) * 5.0
f.from_torch(t)

# field -> torch
result = f.to_torch(device="cpu")
print(result[0, 0])  # 5.0
```

### Ndarrays

Ndarrays support `to_numpy()` and `from_numpy()`:

```python
a = qd.ndarray(qd.i32, shape=(10,))

# numpy -> ndarray
arr = np.arange(10, dtype=np.int32)
a.from_numpy(arr)

# ndarray -> numpy
result = a.to_numpy()
```

### Shape requirements

The shape of the numpy array or torch tensor must match the shape of the field or ndarray exactly. For matrix and vector fields, the element dimensions are appended to the shape:

```python
# A field of 3x2 matrices with shape (4,) has numpy shape (4, 3, 2)
m = qd.Matrix.field(3, 2, qd.f32, shape=(4,))
arr = np.zeros((4, 3, 2), dtype=np.float32)
m.from_numpy(arr)
```

## Zero-copy interop via DLPack

`to_torch()` and `to_numpy()` accept a keyword-only `copy` argument that controls whether the returned tensor/array is an independent copy of the data or a zero-copy view that aliases the underlying Quadrants memory.

```python
f = qd.field(qd.f32, shape=(1024,))

view  = f.to_torch(copy=False)  # zero-copy view: aliases f's memory
clone = f.to_torch(copy=True)   # independent copy
auto  = f.to_torch()            # default: copy=None (see below)
```

Modifications via the view are visible to subsequent Quadrants kernel reads, and vice-versa. The view stays valid until the underlying storage is reallocated -- typically on `qd.init()` or `qd.reset()`, after which a fresh call to `to_torch()` / `to_numpy()` returns a new view.

### When zero-copy is available

Zero-copy uses [DLPack](https://github.com/dmlc/dlpack) and requires:

- a backend with DLPack support: `cpu` (`x64`/`arm64`), `cuda`, `amdgpu`, or `metal` (Vulkan is not supported);
- a DLPack-supported dtype: `i32`, `i64`, `f32`, `f64`, `u1` (other dtypes such as `f16`, `u8`, `u16` fall back to the kernel-copy path);
- on Apple Metal, `torch >= 2.9.2` for fields (required for DLPack `bytes_offset` on MPS; see [pytorch/pytorch#168193](https://github.com/pytorch/pytorch/pull/168193));
- 0-dim `ScalarField` instances are not zero-copyable on any backend (PyTorch DLPack `bytes_offset` limitation);
- `StructField` members are not zero-copyable yet (see [Struct fields](#struct-fields) below); `copy=False` raises on `StructField`.

Zero-copy `to_numpy()` additionally requires a CPU backend, because numpy arrays cannot reference GPU memory.

### Semantics of `copy`

| Value | Behaviour |
|---|---|
| `None` (default) | Zero-copy view if supported, otherwise an independent copy. Never raises for support reasons. |
| `False` | Zero-copy view, or `ValueError` if zero-copy is unsupported for this backend/dtype. |
| `True` | Always returns an independent copy (clones the zero-copy view if available; otherwise allocates and runs the kernel-copy path). |

`copy=True` always returns a buffer that is safe to mutate without affecting the field/ndarray.

### Examples

```python
import quadrants as qd

qd.init(arch=qd.cuda)

f = qd.field(qd.f32, shape=(1024,))
f.fill(1.0)

view = f.to_torch(copy=False)
view *= 2.0           # mutates f's underlying memory directly
qd.sync()             # not strictly required; safe pattern

print(f[0])           # 2.0
```

Round-trip with NumPy on a CPU backend:

```python
qd.init(arch=qd.cpu)

a = qd.ndarray(qd.i32, shape=(8,))
a.from_numpy(np.arange(8, dtype=np.int32))

view = a.to_numpy(copy=False)
view[0] = 100
print(a.to_numpy()[0])   # 100  (default copy=None still returns the same view here)
```

### Caching

Zero-copy views are cached on the source object: repeated calls to `to_torch()` / `to_numpy()` on the same field/ndarray return the same underlying tensor without rebuilding the DLPack capsule. The cache is invalidated automatically when the runtime is reinitialised or reset.

```python
v1 = f.to_torch(copy=False)
v2 = f.to_torch(copy=False)
assert v1.data_ptr() == v2.data_ptr()   # same view

qd.reset()
qd.init(arch=qd.cuda)
f2 = qd.field(qd.f32, shape=(1024,))
v3 = f2.to_torch(copy=False)            # fresh view; v1/v2 must not be used
```

### Axis layout with `layout=`

`to_torch()` and `to_numpy()` accept a keyword `layout: str = "natural"` that selects the axis order of the returned view. The view is materialised by a single `movedim` from Quadrants' native layout and **cached per layout** alongside the natural view, so repeated calls in a hot loop reduce to one attribute lookup.

| `layout=` | Result | When to use |
|---|---|---|
| `"natural"` (default) | Quadrants' native order (batch axis last). | Direct kernel-storage layout; matches `to_numpy` / `to_torch` historically. |
| `"batch_first"` | Last axis moved to the front. | Frameworks (Genesis, JAX-style sims, RL libraries) that expect `(batch, ...)`. |

```python
f = qd.field(qd.f32, shape=(7, 4096))   # 7 components, 4096 envs

natural     = f.to_torch(copy=False)                              # shape (7, 4096)
batch_first = f.to_torch(copy=False, layout="batch_first")        # shape (4096, 7)

assert natural.data_ptr() == batch_first.data_ptr()               # same underlying memory
```

Caching matters on CPU, where the Python-side `to_torch().movedim(...)` pattern is dispatch-heavy in a per-step loop:

```python
# Hot loop -- prefer this:
for _ in range(n_steps):
    dst = f.to_torch(copy=False, layout="batch_first")       # cached, one attribute lookup
    ...

# over the equivalent but slower:
for _ in range(n_steps):
    dst = f.to_torch(copy=False).movedim(-1, 0)              # fresh torch dispatch every step
    ...
```

`layout=` composes with `copy=True` (clones the cached layout view), `device=` (transfers the cached layout view), `keep_dims` (matrix fields only), and the lifetime / synchronisation rules described above.

For 0-D and 1-D fields, every layout collapses to the same tensor.

Only the named layouts are cached; the design keeps the cache size bounded and predictable. New named layouts can be added in future Quadrants releases without breaking call sites.

### Apple Metal: synchronisation

On Apple Metal, Quadrants and PyTorch MPS use separate Metal command queues. Quadrants kernel writes are made visible to the MPS-backed view via an automatic `qd.sync()` on every `to_torch()` / `to_numpy()` call. Cloning a view also synchronises MPS so that the clone sees the latest writes:

```python
qd.init(arch=qd.metal)
f = qd.field(qd.f32, shape=(64,))

run_kernel(f)            # queues writes on the Quadrants Metal stream
view = f.to_torch()      # qd.sync() runs internally; view sees the writes
copy = view.clone()      # torch.mps.synchronize() runs internally; copy is up-to-date
```

You do not need to call `qd.sync()` or `torch.mps.synchronize()` yourself when using `to_torch()` / `to_numpy()`.

### Lifetime caveats

A zero-copy view becomes invalid when the underlying Quadrants storage is freed. This happens on `qd.reset()` and `qd.init()`. Holding a `copy=False` tensor across either is undefined behaviour:

```python
view = f.to_torch(copy=False)
qd.reset()
view[0]                 # undefined: view aliases freed memory
```

If you need a tensor that outlives the runtime, use `copy=True` (or the default `None`, which produces a view but is bound to the same caveat once you opt into zero-copy).

### Struct fields

`StructField.to_torch()` and `StructField.to_numpy()` return a dictionary mapping each member name to an **independent copy** of that member's data. Struct fields use AOS (array-of-structures) cell layout: a `Struct.field({"a": i32, "b": f32}, shape=(N,))` stores `[a0, b0, a1, b1, ...]` in memory, with stride `sizeof(cell)` between consecutive `a`'s. Quadrants' C++ DLPack export does not currently emit cell-stride-aware views for individual members (it computes contiguous strides at the member dtype size, which would interleave neighboring members' bytes), so member views are forced to be copies until that is fixed.

```python
S = qd.types.struct(pos=qd.f32, vel=qd.f32)
sf = S.field(shape=(16,))

dicts = sf.to_torch()           # dict of independent copies (always)
dicts["pos"][0] = 1.0           # does NOT write through to sf.pos
```

`copy=False` is rejected with `ValueError`; `copy=None` and `copy=True` are accepted and both produce copies.

## Direct torch tensor pass-through

Torch tensors can be passed directly into kernels where `qd.types.ndarray()` parameters are expected. The kernel reads from and writes to the torch tensor directly:

```python
import torch
import quadrants as qd

qd.init(arch=qd.gpu)

@qd.kernel
def square(inp: qd.types.ndarray(), out: qd.types.ndarray()) -> None:
    for i in range(32):
        out[i] = inp[i] * inp[i]

x = torch.ones(32, dtype=torch.float32) * 3.0
y = torch.zeros(32, dtype=torch.float32)
square(x, y)
print(y[0])  # 9.0
```

This also works with CUDA tensors when running on a CUDA backend:

```python
x = torch.ones(32, dtype=torch.float32, device="cuda:0") * 3.0
y = torch.zeros(32, dtype=torch.float32, device="cuda:0")
square(x, y)
```

### Integration with torch.autograd

Since torch tensors can be passed directly into kernels, you can integrate Quadrants kernels into PyTorch's autograd system by wrapping them in a `torch.autograd.Function`:

```python
@qd.kernel
def forward_kernel(t: qd.types.ndarray(), o: qd.types.ndarray()) -> None:
    for i in range(32):
        o[i] = t[i] * t[i]

@qd.kernel
def backward_kernel(t_grad: qd.types.ndarray(), t: qd.types.ndarray(), o_grad: qd.types.ndarray()) -> None:
    for i in range(32):
        t_grad[i] = 2 * t[i] * o_grad[i]

class Sqr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        outp = torch.zeros_like(inp)
        ctx.save_for_backward(inp)
        forward_kernel(inp, outp)
        return outp

    @staticmethod
    def backward(ctx, outp_grad):
        outp_grad = outp_grad.contiguous()
        inp_grad = torch.zeros_like(outp_grad)
        (inp,) = ctx.saved_tensors
        backward_kernel(inp_grad, inp, outp_grad)
        return inp_grad

x = torch.tensor([2.0] * 32, requires_grad=True)
loss = Sqr.apply(x).sum()
loss.backward()
print(x.grad[0])  # 4.0
```

## Summary

| Method | Copies data? | Works with fields? | Works with ndarrays? |
|--------|-------------|-------------------|---------------------|
| `to_numpy()` / `from_numpy()` (default) | yes | yes | yes |
| `to_torch()` / `from_torch()` (default) | yes | yes | yes |
| `to_numpy(copy=False)` / `to_torch(copy=False)` | no (DLPack view) | yes | yes |
| Direct pass-through | no | no | yes (as kernel arg) |

The `copy` parameter is supported on `to_numpy()` and `to_torch()` for `ScalarField`, `MatrixField` (and `VectorField`), `StructField`, and all `Ndarray` types. See [Zero-copy interop via DLPack](#zero-copy-interop-via-dlpack) for the support matrix and lifetime rules.
