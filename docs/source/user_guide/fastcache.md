# Fastcache

## What it is

Fastcache reduces the time it takes to load cached kernels when a new Python process starts.

The standard [offline cache](init_options.md#offline_cache) already persists compiled kernels to disk so they don't need to be recompiled from scratch on the next process run. However, loading a cached kernel still requires parsing the kernel's Python AST, transforming it into IR. For applications with many kernels this front-end overhead alone can take several seconds.

Fastcache bypasses that front-end work. It computes a cheap cache key from the kernel source text, argument types, and compiler config, and uses it to load the compiled artifact directly.

On a Genesis simulator benchmark (`single_franka_envs.py`, Ubuntu 24.04, NVIDIA 5090):

| Configuration | Process-start kernel load time |
|---|---|
| Standard offline cache only | 7.2 s |
| Standard offline cache + fastcache | 2.9 s |

## Why you want it

Applications with many kernels (physics simulators, RL environments) pay a multi-second cost every time a new Python process starts, even when all kernels are already cached on disk. Fastcache turns that into a fraction of a second.

## How to use it

### Enabling fastcache on a kernel

Add `fastcache=True` to the `@qd.kernel` decorator:

```python
import quadrants as qd

@qd.kernel(fastcache=True)
def my_kernel(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(a.shape[0]):
        b[i] = a[i] * 2.0
```

That's it. On the first call, the kernel compiles normally and the fastcache entry is written to disk. On subsequent calls (including in new Python sessions), the cached artifact is loaded directly.

### Deprecated alternatives

The `@qd.pure` decorator and `@qd.kernel(pure=True)` parameter are older spellings of the same feature. They still work but emit a deprecation warning and are scheduled for removal in v4.0.0:

```python
# Deprecated — use fastcache=True instead
@qd.pure
@qd.kernel
def my_kernel(a: qd.types.NDArray[qd.f32, 1]) -> None: ...

# Also deprecated
@qd.kernel(pure=True)
def my_kernel(a: qd.types.NDArray[qd.f32, 1]) -> None: ...
```

### Runtime configuration

Fastcache requires the offline cache to be enabled (which it is by default). Two `qd.init` options are relevant:

| Option | Default | Effect |
|---|---|---|
| `src_ll_cache` | `True` | Master switch for fastcache. Set to `False` to disable it globally. |
| `print_non_pure` | `False` | When `True`, prints the name of every kernel at call time that is *not* marked `fastcache=True`. Useful for finding kernels you forgot to annotate. |

```python
qd.init(arch=qd.gpu)
# Fastcache is on by default. To disable:
# qd.init(arch=qd.gpu, src_ll_cache=False)

# To find un-annotated kernels:
# qd.init(arch=qd.gpu, print_non_pure=True)
```

### Diagnostics

You can inspect whether fastcache was used for a specific kernel via the `src_ll_cache_observations` attribute on the kernel's primal:

```python
@qd.kernel(fastcache=True)
def my_kernel(x: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(x.shape[0]):
        x[i] += 1.0

my_kernel(some_array)

obs = my_kernel._primal.src_ll_cache_observations
print(obs.cache_key_generated)  # True if the cache key was computed
print(obs.cache_validated)      # True if a cached entry was found and source hashes matched
print(obs.cache_loaded)         # True if the compiled kernel was loaded from cache
print(obs.cache_stored)         # True if the compiled kernel was stored to cache
```

On the first run you'll see `cache_stored=True` but `cache_loaded=False`. On the second run (after `qd.init`), `cache_loaded=True`.

## Constraints

A kernel is eligible for fastcache only if all of the following hold:

### 1. All data flows through parameters

The kernel must receive every piece of data it operates on as an explicit parameter. It must **not** capture variables from the enclosing Python scope (closures over fields, ndarrays, or mutable globals). This is the core "purity" constraint — the compiled kernel's behavior must be fully determined by its arguments.

```python
a = qd.ndarray(qd.f32, (10,))

# Not eligible: captures `a` from enclosing scope
@qd.kernel(fastcache=True)
def bad_kernel() -> None:
    for i in range(10):
        a[i] = 0.0  # raises QuadrantsCompilationError

# Eligible: `a` is passed as a parameter
@qd.kernel(fastcache=True)
def good_kernel(a: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(a.shape[0]):
        a[i] = 0.0
```

Sub-functions called by the kernel are also checked — they must not capture external state either.

### 2. Supported parameter types

Fastcache supports the following parameter types:

| Type | Supported | Cache key includes |
|---|---|---|
| `qd.types.NDArray` (scalar, vector, matrix) | Yes | dtype, ndim, layout |
| `torch.Tensor` | Yes | dtype, ndim |
| `numpy.ndarray` | Yes | dtype, ndim |
| `dataclasses.dataclass` | Yes | field types recursively; field values if annotated with `add_value_to_cache_key` |
| `@qd.data_oriented` objects | Yes | member types and values recursively |
| `qd.Template` primitives (int, float, bool) | Yes | type and value (baked into kernel) |
| Non-template primitives (int, float) | Yes | type only |
| `enum.Enum` | Yes | name and value |
| `qd.field` / `ScalarField` / `MatrixField` | **No** | — |

If any parameter is of an unsupported type, fastcache is silently disabled for that call and the kernel falls back to normal compilation. A warning is logged at the `warn` level identifying the offending parameter.

### 3. Fields are not supported

Kernels that use `qd.field` as parameters cannot use fastcache because field memory layout and offsets are not captured in the cache key. Use `qd.ndarray` instead if you want fastcache.

### 4. Source code must be readable

Fastcache hashes the source code of the kernel and all sub-functions it calls. If the source file cannot be read at runtime (e.g. the kernel is defined in a frozen/compiled module, or the file has been deleted), fastcache cannot validate the cache and will fall back to normal compilation.

### 5. Cache invalidation

The cache is automatically invalidated when any of the following change:

- The **Quadrants version** (`quadrants.__version__`).
- The **source code** of the kernel function or any `@qd.func` it calls.
- The **argument types** (e.g. switching from `f32` to `f64`, or changing ndarray dimensionality).
- The **compiler configuration** (e.g. `arch`, `debug`, `opt_level`, `fast_math`).
- **Template parameter values** (since they are baked into the compiled kernel).

You do not need to manually clear the cache when making code changes — the hash mismatch causes a transparent recompilation.
