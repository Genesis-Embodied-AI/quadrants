# BufferView

`BufferView` allows passing a sub-range of an ndarray to a kernel, with optional bounds checking in debug mode.

## Creating a view

Use Python slice notation on any 1D `qd.ndarray`:

```python
import quadrants as qd
import numpy as np

N = 64
data = qd.ndarray(qd.f32, shape=(N,))
data.from_numpy(np.arange(N, dtype=np.float32))

first_half = data[:32]     # offset=0,  size=32
second_half = data[32:]    # offset=32, size=32
middle = data[16:48]       # offset=16, size=32
last_eight = data[-8:]     # offset=56, size=8
```

Only 1D ndarrays are supported. Slices with a step other than 1 raise `ValueError`.

For programmatically computed offsets, use the constructor directly:

```python
view = qd.BufferView(data, offset=16, size=32)
```

## Slicing a view (subview)

A `BufferView` can be sliced again to create a narrower subview. All views share the same backing ndarray - offsets accumulate automatically:

```python
a = data[8:24]      # BufferView: offset=8,  size=16
b = a[4:12]         # BufferView: offset=12, size=8
c = b[:4]           # BufferView: offset=12, size=4
```

This forms a closed slicing chain: `ndarray` → slice → `BufferView` → slice → `BufferView`. Each step validates bounds against the parent's size. The `subview()` method provides the same functionality with explicit offset and size:

```python
b = a.subview(offset=4, size=8)   # equivalent to a[4:12]
```

## Kernel type annotation

Use `BufferView` as a parameter annotation on `@qd.kernel`. The element dtype can be specified explicitly or omitted:

```python
from quadrants import BufferView

# Explicit dtype - the annotation declares the expected element type:
@qd.kernel
def scale(v: BufferView[qd.f32], factor: qd.f32):
    for i in range(v.size):
        v[i] = v[i] * factor

# No dtype - Quadrants infers it from the ndarray passed at call time:
@qd.kernel
def scale_any(v: BufferView):
    for i in range(v.size):
        v[i] = v[i] * 2.0

scale(data[:32], 2.0)
scale_any(data[:32])  # dtype inferred as qd.f32 from data
```

Both forms are equivalent at runtime. Use `BufferView[dtype]` when you want the annotation to document the expected type; use plain `BufferView` when the dtype varies or is determined at initialization time.

`v.size` gives the number of elements in the view; `v.shape` gives the equivalent tuple `(size,)`. Subscript `v[i]` transparently accesses `data[offset + i]`.

## Debug mode: bounds checking and callstack diagnostics

With `debug=True`, every subscript on a `BufferView` is bounds-checked against `[0, size)`. An out-of-bounds access raises `QuadrantsAssertionError` with a message that includes the kernel name, thread ID, the index, the view's offset and size, and the full compilation-time callstack:

```python
qd.init(arch=qd.cpu, debug=True)

@qd.func
def writer(v: BufferView[qd.f32], idx: qd.i32):
    v[idx] = 99.0                  # OOB when idx >= size

@qd.kernel
def kernel(v: BufferView[qd.f32]):
    for i in range(v.size):
        if i == 0:
            writer(v, v.size)      # passes out-of-range index

N = 32
data = qd.ndarray(qd.f32, shape=(N,))
kernel(data[:16])
```

Output:

```
quadrants.lang.exception.QuadrantsAssertionError:
BufferView Out Of Range: kernel[kernel] tid=0, got index 16 (offset=0, size=16).
Callstack:
kernel (script.py:11)
  writer (script.py:7)
```

The callstack shows every function frame from the kernel down to the leaf function where the access occurred. Bounds checking has no cost when `debug=False` (the default).

## Limitations

- Only **1D** ndarrays are supported as the backing buffer.
- Ndarrays with `needs_grad=True` are not supported. BufferView will raise `TypeError` on construction.
