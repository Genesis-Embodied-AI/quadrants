# Scalar tensors

A scalar tensor is a field with zero dimensions — it holds a single value. This is useful for accumulators, counters, or any kernel that needs to produce a single output value.

Scalar tensors can be used with both fields and ndarrays.

## Creating a scalar tensor

```python
# field
counter = qd.field(qd.i32, shape=())

# ndarray
counter = qd.ndarray(qd.i32, shape=())
```

## Initializing fields at allocation

Use `fill=` to give a field an initial value:

```python
gravity = qd.field(qd.f32, shape=(), fill=-9.81)
counter = qd.field(qd.i32, shape=(), fill=0)
weights = qd.field(qd.f32, shape=16, fill=1.0)
directions = qd.Vector.field(3, qd.f32, shape=8, fill=[0, 0, 1])
```

Field allocation is deferred until Quadrants needs the storage. The initializer runs once at that point, before the first read or write, including access through a kernel or a zero-copy export. Declaring multiple fields with `fill=` lets Quadrants allocate them together. This avoids splitting the allocation with a separate `.fill()` call after each declaration, which can exhaust Metal's buffer bindings when a kernel accesses many fields.

`fill=` also works with `qd.Matrix.field` and vector or matrix types passed to `qd.field`. A scalar initializes every component; a list, tuple, or Quadrants vector or matrix supplies per-component values. Mutable values are captured when the field is declared. Omitting `fill` or passing `None` preserves the existing allocation behavior. Only the field itself is initialized; its gradient and dual fields keep their default values. A field declared without `shape` must still be placed before use.

The `fill=` parameter applies to fields, not ndarrays.

## Accessing the value

Index a scalar tensor with `[()]`:

```python
@qd.kernel
def increment(a: qd.Template) -> None:
    a[()] += 1
```

## Example: summing with atomics

A common use case is accumulating a result across many threads. Since multiple threads may write to the same scalar, use atomic operations:

```python
import quadrants as qd

qd.init(arch=qd.gpu)

total = qd.field(qd.i32, shape=())

@qd.kernel
def sum_range(N: int) -> None:
    for i in range(N):
        qd.atomic_add(total[()], i)

total[()] = 0
sum_range(100)
print("sum:", total[()])  # 4950
```

## Reading from Python

From Python, you can read and write the value using the same `[()]` indexing:

```python
total[()] = 0       # write
print(total[()])    # read
```
