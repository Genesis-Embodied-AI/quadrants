# Numpy and Torch interop

Quadrants provides interop with both numpy and PyTorch. There are two mechanisms:
- **Copy-based**: convert data between quadrants fields/ndarrays and numpy arrays or torch tensors
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
| `to_numpy()` / `from_numpy()` | yes | yes | yes |
| `to_torch()` / `from_torch()` | yes | yes | no |
| Direct pass-through | no | no | yes (as kernel arg) |
