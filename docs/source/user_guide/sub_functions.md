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

## Restricting a func to the top level (`requires_top_level=True`)

Some functions contain top-level for-loops, and thus must not be called from within top-level for-loops.

To both enforce this, and to signal to potential users/AIs that a func must not be called from within a top-level for-loop, you can annotate a `qd.func` with `qd.func(requires_top_level=True)`. This will throws `QuadrantsSyntaxError` at compile time if used within a for-loop.

```python
@qd.func(requires_top_level=True)
def op(arr: qd.Template, n: qd.i32) -> None:
    for i in range(n):  # one of several top-level phase loops
        arr[i] = arr[i] + 1
```

The check is purely compile-time — it adds no runtime or GPU cost, and a correctly placed call compiles to exactly the same code as an unmarked func.

What counts as **top level** (allowed):

- Directly in the kernel body.
- Inside a `qd.static(...)` loop — these are unrolled at compile time, so the calls land at top level (see [static](static.md)).
- Directly inside a `while qd.graph_do_while(...):` body (see [graphs](graph.md)).

What is **rejected**: nesting the call inside a runtime `for`, `if`, or `while`.

```python
@qd.kernel
def good(arr: qd.Template, n: qd.i32) -> None:
    op(arr, n)                  # OK: top level

@qd.kernel
def also_good(arr: qd.Template, n: qd.i32) -> None:
    for _ in qd.static(range(2)):
        op(arr, n)              # OK: qd.static is compile-time

@qd.kernel
def bad(arr: qd.Template, n: qd.i32, flag: qd.i32) -> None:
    if flag > 0:
        op(arr, n)              # QuadrantsSyntaxError raised at compile time
```
