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

Some device-wide ops are built as a **sequence of phases**, where each phase is a top-level `for` loop that Quadrants offloads as its own GPU launch (the same top-level-loop model that [graphs](graph.md) capture). The boundary between two consecutive top-level loops acts as a grid-wide barrier: phase *k* finishes on **every** block, and its global-memory writes become visible, before phase *k+1* begins. Algorithms with cross-phase data dependencies (e.g. a histogram pass feeding a scan feeding a scatter) rely on this ordering for correctness.

Such a func is only correct when it is called at the **top level** of a kernel, so its phase loops stay at top level and keep those barriers. Nest the call inside ordinary runtime control flow and the phase loops are demoted to serial inner loops: the per-phase parallelism and the inter-phase barriers both collapse, and the result is **silently corrupted**.

To turn that misuse into a hard error, mark the func with `requires_top_level=True`:

```python
@qd.func(requires_top_level=True)
def op(arr: qd.Template, n: qd.i32) -> None:
    for i in range(n):  # one of several top-level phase loops
        arr[i] = arr[i] + 1
```

Quadrants then checks the **call site** at compile time (during tracing) and raises `QuadrantsSyntaxError` if the func is called from a non-top-level position. The check is purely compile-time — it adds no runtime or GPU cost, and a correctly placed call compiles to exactly the same code as an unmarked func.

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

This is intended for multi-phase, device-wide algorithms — reductions, scans, sort, and similar — where nesting would otherwise miscompile rather than fail loudly.
