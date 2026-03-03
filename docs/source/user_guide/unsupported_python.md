# Python features in kernel scope

Quadrants kernels are compiled to GPU machine code, so only a subset of Python is supported inside `@qd.kernel` and `@qd.func` functions. This page documents what works, what doesn't, and common workarounds.

## Supported

The following Python features work inside kernels:

- **Arithmetic operators**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Comparison operators**: `<`, `<=`, `>`, `>=`, `==`, `!=`
- **Logical operators**: `and`, `or`, `not`
- **Bitwise operators**: `&`, `|`, `^`, `~`, `<<`, `>>`
- **Augmented assignment**: `+=`, `-=`, `*=`, etc. (note: these are [atomic](./quirks.md))
- **`if` / `elif` / `else`**
- **`for` loops** over `range()`, `qd.ndrange()`, `qd.grouped()`, or `qd.static()`
- **`while` loops**
- **`break` and `continue`** in non-parallelized loops (inner loops, `while` loops)
- **`assert`** (when `debug=True` is passed to `qd.init()`)
- **`print`** (for scalar values)
- **Local variables** (scalars, vectors, matrices)
- **Lists and tuples** (as compile-time containers, e.g. `[1, 2, 3]`)
- **List comprehensions** (evaluated at compile time)
- **F-strings** (limited: scalar values only)
- **`qd.Vector`, `qd.Matrix`** construction and operations
- **Calling `@qd.func`** and **`@qd.real_func`** functions
- **Type casts**: `qd.cast()`, `int()`, `float()`
- **Math functions**: `qd.sqrt()`, `qd.sin()`, `qd.cos()`, `qd.abs()`, `qd.max()`, `qd.min()`, etc.
- **Atomic operations**: `qd.atomic_add()`, `qd.atomic_sub()`, `qd.atomic_min()`, `qd.atomic_max()`

## Not supported

The following Python features will raise errors or produce unexpected behavior inside kernels:

### Statements

| Feature | Error |
|---------|-------|
| `try` / `except` / `finally` | Compilation error |
| `raise` | `Unsupported node "Raise"` |
| `with` | `Unsupported node "With"` |
| `yield` / `yield from` | `Unsupported node "Yield"` |
| `import` | `Unsupported node "Import"` |
| `del` | `Unsupported node "Delete"` |
| `global` / `nonlocal` | `Unsupported node "Global"` |
| Function definitions inside kernels | `Function definition is not allowed in 'qd.kernel'` |
| Class definitions | `Unsupported node "ClassDef"` |
| `for ... else` | `'else' clause for 'for' not supported` |
| `while ... else` | `'else' clause for 'while' not supported` |

### Expressions and operators

| Feature | Error |
|---------|-------|
| `is` / `is not` | `Operator "is" in Quadrants scope is not supported` |
| `in` / `not in` | Compilation error |
| Sets (`{1, 2, 3}`) | Compilation error |
| `lambda` | `Unsupported node "Lambda"` |

### Types and data structures

| Feature | Notes |
|---------|-------|
| Strings | No string manipulation; only literal strings in `print()` and f-strings |
| Dictionaries | Only as compile-time constants via `qd.static()` |
| Classes / objects | Cannot define or instantiate arbitrary Python classes |
| Generators | Not supported |
| `*args` unpacking | Only in a [narrow case](./unsupported_python.md#argument-unpacking) |

### Control flow restrictions

- **`return` inside non-static `if`/`for`**: not allowed in `@qd.func` (allowed in `@qd.real_func`)
- **`break` in the outermost (parallelized) for loop**: not allowed — the outermost loop is mapped to GPU threads
- **`break`/`continue` in a static `for` inside a non-static `if`**: not allowed
- **Reassigning kernel arguments**: kernel parameters are immutable; create a local copy instead

## Common workarounds

### Use `qd.static()` for compile-time Python logic

If you need Python-level logic (dictionaries, complex conditionals, etc.), wrap it in `qd.static()` so it's evaluated at compile time:

```python
options = {"mode_a": 1, "mode_b": 2}
selected = "mode_a"

@qd.kernel
def compute(a: qd.Template) -> None:
    val = qd.static(options[selected])
    a[0] = val
```

### Local copy for immutable arguments

```python
@qd.kernel
def compute(n: int, a: qd.Template) -> None:
    local_n = n  # can now modify local_n
    local_n += 1
```

### Use `@qd.real_func` for early returns

`@qd.func` does not support `return` inside runtime `if` statements. Use `@qd.real_func` instead (CUDA and CPU only):

```python
@qd.real_func
def safe_div(a: qd.f32, b: qd.f32) -> qd.f32:
    if b == 0.0:
        return 0.0
    return a / b
```

# Argument unpacking

Syntax like `some_function(*args)` is supported in a narrow case:
- `*args` must be the last argument in the function call
- `*args` must not contain `dataclasses.dataclass` objects
