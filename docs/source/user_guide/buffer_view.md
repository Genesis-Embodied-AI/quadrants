# BufferView: safe sub-range access

`BufferView` provides a zero-copy, bounds-checked view into a sub-range
`[offset, offset+count)` of an ndarray, suitable for passing to kernels that
should only operate on a portion of a larger buffer.

## Creating a view

### Slice syntax (preferred)

Use Python's standard slice notation on any 1D `qd.ndarray`:

```python
import quadrants as qd
import numpy as np

N = 64
data = qd.ndarray(qd.f32, shape=(N,))
data.from_numpy(np.arange(N, dtype=np.float32))

first_half  = data[:32]     # offset=0,  count=32
second_half = data[32:]     # offset=32, count=32
middle      = data[16:48]   # offset=16, count=32
last_eight  = data[-8:]     # offset=56, count=8  (negative indices work)
```

Only 1D ndarrays are supported. `step` must be omitted or 1; `data[::2]`
raises a `ValueError`.

### Explicit constructor

For programmatically computed offsets use the constructor directly:

```python
view = qd.BufferView(data, offset=16, count=32)
```

## Kernel type annotation

Use `BufferView[dtype]` as the parameter annotation.
The view is automatically decomposed into `(ndarray, offset, count)` at
compile time and reassembled inside the kernel:

```python
from quadrants import BufferView

@qd.kernel
def scale(v: BufferView[qd.f32], factor: qd.f32):
    for i in range(v.count):
        v[i] = v[i] * factor

scale(data[:32], 2.0)
```

`v.count` gives the length of the view; `v[i]` transparently accesses
`data[offset + i]`.

## Using BufferView in `@qd.func`

`BufferView[dtype]` works as a type annotation on `@qd.func` as well,
enabling composable helper functions:

```python
@qd.func
def fill_view(v: BufferView[qd.f32], val: qd.f32):
    for i in range(v.count):
        v[i] = val

@qd.kernel
def kernel(v: BufferView[qd.f32]):
    fill_view(v, 0.0)

kernel(data[16:48])
```

## Debug mode: bounds checking and callstack diagnostics

With `debug=True`, every subscript on a `BufferView` is bounds-checked
against `[0, count)`. An out-of-bounds access raises a
`QuadrantsAssertionError` with a structured message that includes the
kernel name, thread ID, the bad index, the view's offset and count, and
the full compilation-time callstack:

```python
qd.init(arch=qd.cpu, debug=True)

@qd.func
def writer(v: BufferView[qd.f32], idx: qd.i32):
    v[idx] = 99.0                  # OOB when idx >= count

@qd.kernel
def kernel(v: BufferView[qd.f32]):
    writer(v, 16)                  # passes index 16 to a view of count=16

N = 32
data = qd.ndarray(qd.f32, shape=(N,))
kernel(data[:16])
```

Output:

```
quadrants.lang.exception.QuadrantsAssertionError:
BufferView Out Of Range: kernel[kernel] tid=0, got index 16 (offset=0, count=16).
Callstack:
kernel (script.py:11)
  writer (script.py:7)
```

The callstack shows every function frame from the kernel down to the
leaf function where the access occurred.

Bounds checking is only active when `debug=True`; it has no cost in
production mode.

## Limitations

- Only **1D** ndarrays are supported as the backing buffer.
- The slice step must be 1 (or omitted).
- `BufferView[dtype]` annotations are evaluated at kernel compilation time;
  the view itself is passed at the call site.
