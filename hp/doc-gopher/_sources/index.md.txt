# Quadrants

Quadrants is a high-performance parallel programming framework for GPU and CPU computing. Write Python-like code that compiles to optimized GPU kernels for CUDA, Metal, and Vulkan backends.

```python
import quadrants as qd

qd.init(arch=qd.gpu)

@qd.kernel
def hello(a: qd.types.NDArray[qd.i32, 1]) -> None:
    for i in range(10):
        a[i] = i * 2

a = qd.ndarray(qd.i32, (10,))
hello(a)
```

## Features

- **Simple**: annotate Python functions with `@qd.kernel` to run on GPU
- **Fast**: automatic parallelization of top-level for loops across GPU threads
- **Portable**: supports CUDA, Metal, and Vulkan backends
- **Flexible**: ndarrays, fields, structs, atomics, shared memory

```{toctree}
:caption: Quadrants
:maxdepth: 2

user_guide/index
```
