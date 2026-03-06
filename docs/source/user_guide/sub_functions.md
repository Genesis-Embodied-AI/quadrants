# Sub-functions

A `@qd.kernel` can call other functions, as long as those functions have an appropriate Quadrants annotation.

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

### Passing by reference

When using `qd.template()` type annotations, scalar arguments are passed **by reference** — modifications to the parameter inside the function are visible to the caller:

```python
@qd.func
def set_value(x: qd.template()) -> None:
    x = 42

@qd.kernel
def compute() -> None:
    a = 0
    set_value(a)
    # a is now 42
```

Without the `qd.template()` annotation, scalar arguments are passed by value — modifications are not visible to the caller.

## qd.real_func

`@qd.real_func` is like `@qd.func` but the function is **not inlined**. Instead, it is compiled as a separate function with a real call/return.

```python
@qd.real_func
def factorial(n: qd.i32) -> qd.i32:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

@qd.kernel
def compute(a: qd.Template) -> None:
    for i in range(10):
        a[i] = factorial(i)
```

Key differences from `@qd.func`:
- Supports **recursion**
- Has function call overhead (call/return)
- The compiler cannot optimize across the call boundary
- Only supported on `cpu` and `cuda` backends

### Limitations

- `@qd.real_func` does not support passing `dataclasses.dataclass` arguments
- `@qd.real_func` is experimental
