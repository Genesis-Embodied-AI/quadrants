# Scalar tensors

A scalar tensor is a field with zero dimensions — it holds a single value. This is useful for accumulators, counters, or any kernel that needs to produce a single output value.

Scalar tensors only work with fields, not ndarrays (an ndarray requires at least one dimension).

## Creating a scalar tensor

```python
counter = qd.field(qd.i32, shape=())
```

## Accessing the value

Index a scalar tensor with `[None]`:

```python
@qd.kernel
def increment(a: qd.Template) -> None:
    a[None] += 1
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
        qd.atomic_add(total[None], i)

total[None] = 0
sum_range(100)
print("sum:", total[None])  # 4950
```

## Reading from Python

From Python, you can read and write the value using the same `[None]` indexing:

```python
total[None] = 0       # write
print(total[None])    # read
```
