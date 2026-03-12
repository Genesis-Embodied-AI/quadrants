# Sub-functions

A `@qd.kernel` can call other functions, as long as those functions are annotated with `@qd.func`.

## qd.func

`@qd.func` is the standard annotation for a function that can be called from a kernel. `@qd.func` functions can also call other `@qd.func` functions.

`@qd.func` is **inlined** into the calling kernel at compile time. This means:
- There is no function call overhead
- The compiler can optimize across the call boundary
- Recursive calls are not supported

```python
@qd.func
def add(a: qd.i32, b: qd.i32) -> qd.i32:
    return a + b

@qd.kernel
def compute(a: qd.Template) -> None:
    for i in range(10):
        a[i] = add(i, 1)
```

### Passing fields and ndarrays

Fields and ndarrays can be passed to `@qd.func` functions:

```python
@qd.func
def increment(arr: qd.Template, idx: qd.i32) -> None:
    arr[idx] += 1

@qd.kernel
def compute(a: qd.Template) -> None:
    for i in range(10):
        increment(a, i)
```
