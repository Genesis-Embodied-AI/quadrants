# qd.static

`qd.static()` evaluates expressions at compile time rather than at kernel runtime. It is similar to `constexpr` in C++.

This enables two main capabilities:
- **Compile-time branching**: eliminating branches entirely from the compiled kernel
- **Loop unrolling**: unrolling loops at compile time

Since the expressions are evaluated at compile time, the arguments to `qd.static()` must be known at compile time — they cannot depend on non-templated kernel parameters or other runtime values.

## Compile-time branching

```python
@qd.kernel
def compute(use_fast_path: qd.Template, a: qd.Template) -> None:
    for i in range(10):
        if qd.static(use_fast_path):
            a[i] = i * 2
        else:
            a[i] = i * 3 + 1

compute(True, my_field)
```

Because `use_fast_path` is a templated parameter, it is known at compile time. The compiler will eliminate the `if/else` entirely — the compiled kernel will contain only `a[i] = i * 2`, with no branch at all. Calling `compute(False, my_field)` would trigger a recompilation that keeps only the `else` branch.

Without `qd.static`, the condition would be evaluated at runtime for every thread, which is slower. In addition, the kernel will contain the code for both branches, which will increase register pressure, and likely reduce occupancy.

## Loop unrolling

```python
@qd.kernel
def compute(a: qd.Template) -> None:
    for i in qd.static(range(3)):
        a[i] = i * 10
```

This is compiled as if you had written:

```python
@qd.kernel
def compute(a: qd.Template) -> None:
    a[0] = 0
    a[1] = 10
    a[2] = 20
```

This is useful when the loop count is small and known at compile time, and you want to avoid loop overhead or enable further compiler optimizations.

## Interaction with parallelization

A top-level for loop is normally parallelized across GPU threads. Wrapping it in `qd.static()` changes the behavior: the loop is unrolled at compile time instead of being parallelized.

A `qd.static` `if` wrapping a top-level for loop does **not** prevent the for loop from being parallelized — the `if` is resolved at compile time, leaving the for loop as a top-level construct.

```python
@qd.kernel
def compute(enable_pass: qd.Template, N: int, a: qd.Template) -> None:
    if qd.static(enable_pass):
        for i in range(N):  # still parallelized
            a[i] += 1
```

## Compile-time error

If you pass a runtime value (e.g. as a non-templated kernel parameter) to `qd.static()`, you will get a compilation error:

```python
@qd.kernel
def compute(val: float) -> None:
    if qd.static(val > 0.5):  # ERROR: val is a runtime value
        pass
```

This will raise a `QuadrantsCompilationError` with a message indicating that the argument must be a compile-time constant.
