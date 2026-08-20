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

Quadrants' zero-copy interop has been designed with **PyTorch as the first-class user interface**: support, defaults, and supported-dtype/backend matrices are driven by what PyTorch can consume cleanly via DLPack. NumPy is supported on CPU backends as a free side benefit of the same DLPack capsule. Several of the limitations below (e.g. the Apple Metal `torch >= 2.9.2` requirement, or the 0-dim `ScalarField` carve-out) are inherited from PyTorch's current DLPack importer rather than from Quadrants itself.

`to_torch()` and `to_numpy()` accept a keyword-only `copy` argument that controls whether the returned tensor/array is an independent copy of the data or a zero-copy view that aliases the underlying Quadrants memory.

```python
f = qd.field(qd.f32, shape=(1024,))

view  = f.to_torch(copy=False)  # zero-copy view: aliases f's memory, or ValueError
auto  = f.to_torch(copy=None)   # zero-copy if possible, otherwise copy
clone = f.to_torch(copy=True)   # independent copy (default)
plain = f.to_torch()            # same as copy=True
```

When using `copy=False` or `copy=None` (when zero-copy succeeds), modifications via the view are visible to subsequent Quadrants kernel reads, and vice-versa. The view stays valid until the underlying storage is reallocated -- typically on `qd.init()` or `qd.reset()`, after which a fresh call to `to_torch(copy=False)` / `to_numpy(copy=False)` returns a new view.

### When zero-copy is available

Zero-copy uses [DLPack](https://github.com/dmlc/dlpack) and requires:

- a backend with DLPack support: `cpu` (`x64`/`arm64`), `cuda`, `amdgpu`, or `metal`. Vulkan is not supported: Vulkan-backed DLPack tensors are not processed by many well-known scientific computing libraries (torch, numpy, etc.);
- a DLPack-supported dtype: `i32`, `i64`, `f32`, `f64`, `u1` (other dtypes such as `f16`, `u8`, `u16` fall back to the kernel-copy path);
- on Apple Metal, `torch >= 2.9.2` for fields (required for DLPack `bytes_offset` on MPS; see [pytorch/pytorch#168193](https://github.com/pytorch/pytorch/pull/168193));
- 0-dim `ScalarField` instances are not zero-copyable on any backend (PyTorch DLPack `bytes_offset` limitation);
- members of an AOS `StructField` (the default `Struct.field(..., layout=Layout.AOS)`) are not zero-copyable yet (see [Struct fields](#struct-fields) below); members of an SOA `StructField` (`layout=Layout.SOA`) **are** zero-copyable individually.

Zero-copy `to_numpy()` additionally requires a CPU backend, because numpy arrays cannot reference GPU memory. Note: `Field.to_numpy(copy=False)` and `MatrixField.to_numpy(copy=False)` currently require torch to be installed, because the C++ `field_to_dlpack` checks the torch version internally. `Ndarray.to_numpy(copy=False)` does not require torch.

On **NumPy >= 2.1**, `to_numpy(copy=False)` returns a **writable** array (via a DLPack v1 capsule). On NumPy 1.26–2.0, the returned array is **read-only** because those versions only consume DLPack v0 capsules, which lack writability metadata. If you need writable zero-copy numpy views, upgrade to NumPy >= 2.1.

### Semantics of `copy`

| Value | Behavior |
|---|---|
| `True` (default) | Independent copy via kernel. |
| `None` | Zero-copy view via DLPack when available, otherwise falls back to a copy silently. |
| `False` | Zero-copy view via DLPack, or `ValueError` if zero-copy is unsupported for this backend/dtype. |

The default `copy=True` always returns a buffer that is safe to mutate without affecting the field/ndarray. Use `copy=None` when you want zero-copy as a best-effort optimization without having to handle exceptions — it gives you a view when possible and a safe copy otherwise.

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
view[0] = 100  # Requires NumPy >= 2.1 for the assignment to succeed
print(a.to_numpy()[0])   # 100
```

### Caching

Each call to `to_torch(copy=False)` builds a fresh DLPack capsule, but the returned tensor aliases the same underlying memory. Downstream frameworks (e.g. Genesis) may cache the returned view on the field object for hot-path reuse; Quadrants itself does not cache views internally.

```python
v1 = f.to_torch(copy=False)
v2 = f.to_torch(copy=False)
assert v1.data_ptr() == v2.data_ptr()   # same underlying memory
```

### Apple Metal: synchronization

On Apple Metal, Quadrants and PyTorch MPS use separate Metal command queues. Every `to_torch()` / `to_numpy()` call runs `qd.sync()` internally to flush the Quadrants queue. Additionally, `copy=True` (the default) calls `torch.mps.synchronize()` after the kernel copy. This is necessary because, on Metal, Quadrants and Torch do not share the same compute streams. `copy=False` does **not** call `torch.mps.synchronize()`:

```python
qd.init(arch=qd.metal)
f = qd.field(qd.f32, shape=(64,))

run_kernel(f)                       # queues writes on the Quadrants Metal stream
view = f.to_torch(copy=False)       # qd.sync() only
copy = f.to_torch(copy=True)        # qd.sync() + torch.mps.synchronize()
```

The reverse direction (PyTorch writes to a zero-copy view, then a Quadrants kernel reads from the same field) is **not** automatically synchronized. Because Quadrants and PyTorch MPS submit work to separate Metal command queues, a kernel launched immediately after a torch write may execute before the torch write has actually committed to memory:

```python
qd.init(arch=qd.metal)
f = qd.field(qd.f32, shape=(64,))

view = f.to_torch(copy=False)
view.zero_()                     # queued on the torch MPS stream
my_kernel(f)                     # may run BEFORE view.zero_() commits!

torch.mps.synchronize()          # required to flush the torch MPS stream first
my_kernel(f)                     # now safe
```

This is intentional: forcing a sync on every Quadrants kernel that touches a previously-zerocopied field would be very expensive in workloads that batch many torch ops and many kernels back-to-back. If you mutate fields from torch and then read them from a Quadrants kernel on Metal, call `torch.mps.synchronize()` once between the torch ops and the kernels.

**Shared command queue.** The synchronization overhead above can be eliminated entirely by passing PyTorch MPS's `MTLCommandQueue` to Quadrants at init time via `external_metal_command_queue`. Quadrants provides `quadrants.interop.get_mps_command_queue()` to extract the queue pointer at runtime. When both frameworks share the same queue, Metal guarantees command buffer ordering automatically. See [Shared Metal command queue](./metal_shared_queue.md) for the setup guide.

### Lifetime caveats

A zero-copy view becomes invalid when the underlying Quadrants storage is freed. This happens on `qd.reset()` and `qd.init()`. Holding a `copy=False` tensor across either is undefined behavior:

```python
view = f.to_torch(copy=False)
qd.reset()
view[0]                 # undefined: view aliases freed memory
```

The default `copy=True` produces an independent copy that is unaffected. Only `copy=False` views are affected by this caveat.

### Struct fields

`StructField.to_torch()` and `StructField.to_numpy()` return a dictionary mapping each member name to a tensor / array; the `copy` argument is propagated to each member, so zero-copy availability is decided per member. The relevant axis is the SNode layout chosen at construction:

- **AOS** (default `Struct.field(..., layout=Layout.AOS)`): all members share the struct cell, e.g. `Struct.field({"a": i32, "b": f32}, shape=(N,))` stores `[a0, b0, a1, b1, ...]` in memory, with stride `sizeof(cell)` between consecutive `a`'s. Quadrants' C++ DLPack export does not currently emit cell-stride-aware views for individual members (it computes contiguous strides at the member dtype size, which would interleave neighboring members' bytes), so AOS members fall back to a kernel copy and `copy=False` raises on each AOS member.
- **SOA** (`Struct.field(..., layout=Layout.SOA)`): each member sits in its own dense SNode subtree with contiguous storage, so members are zero-copyable individually under the usual backend / dtype rules. `copy=False` succeeds and returns aliasing views.

```python
S_aos = qd.Struct.field({"pos": qd.f32, "vel": qd.f32}, shape=(16,))   # AOS (default)
d_aos = S_aos.to_torch()                                                # dict of kernel copies
d_aos["pos"][0] = 1.0                                                   # does NOT write back

S_soa = qd.Struct.field({"pos": qd.f32, "vel": qd.f32}, shape=(16,),
                        layout=qd.Layout.SOA)
d_soa = S_soa.to_torch(copy=False)                                      # dict of zero-copy views
d_soa["pos"][0] = 1.0                                                   # writes through to S_soa.pos
```

### Raw DLPack export with `to_dlpack()`

All field and ndarray types expose a `to_dlpack()` method that returns a raw [DLPack](https://github.com/dmlc/dlpack) `PyCapsule`. This is the low-level primitive that `to_torch(copy=False)` and `to_numpy(copy=False)` are built on; use it when you need to feed Quadrants data into a framework that speaks DLPack directly (e.g. JAX, CuPy, or a custom C extension).

```python
qd.init(arch=qd.cpu)
f = qd.field(qd.f32, shape=(8,))
f.fill(1.0)

capsule = f.to_dlpack()                                   # v0 capsule ("dltensor")
t = torch.utils.dlpack.from_dlpack(capsule)                # zero-copy torch tensor
```

For NumPy, prefer `to_numpy(copy=False)` which handles the DLPack protocol adapter internally. If you need a raw v1 capsule for another consumer, use `f.to_dlpack(versioned=True)`. Note that `np.from_dlpack` does not accept raw `PyCapsule` objects — it requires an object exposing `__dlpack__()` and `__dlpack_device__()`, which `to_numpy(copy=False)` provides via an internal adapter.

The `versioned` parameter selects the DLPack protocol version:

| `versioned` | Capsule type | Capsule name | Use case |
|---|---|---|---|
| `False` (default) | `DLManagedTensor` (v0) | `"dltensor"` | `torch.utils.dlpack.from_dlpack`, CuPy, JAX, and other v0 consumers. |
| `True` | `DLManagedTensorVersioned` (v1) | `"dltensor_versioned"` | `np.from_dlpack` on NumPy >= 2.1 (v0 capsules produce read-only arrays on NumPy >= 2.0; v1 consumer support requires >= 2.1). |

The same backend, dtype, and layout restrictions that apply to `to_torch(copy=False)` / `to_numpy(copy=False)` apply here — `to_dlpack()` is the underlying mechanism. The caller is responsible for calling `qd.sync()` between modifying the field and consuming the capsule.

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
| `to_numpy(copy=None)` / `to_torch(copy=None)` | no when possible, yes otherwise | yes | yes |
| `to_numpy(copy=False)` / `to_torch(copy=False)` | no (DLPack view) | yes | yes |
| `to_dlpack()` | no (raw capsule) | yes | yes |
| Direct pass-through | no | no | yes (as kernel arg) |

The `copy` parameter is supported on `to_numpy()` and `to_torch()` for `ScalarField`, `MatrixField` (and `VectorField`), `StructField`, `qd.Tensor`, and all `Ndarray` types. See [Zero-copy interop via DLPack](#zero-copy-interop-via-dlpack) for the support matrix and lifetime rules.
