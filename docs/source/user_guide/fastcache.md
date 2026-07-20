# Fastcache

## What it is

Fastcache reduces the time it takes to load cached kernels when a new Python process starts.

The standard [offline cache](init_options.md#offline_cache) already persists compiled kernels to disk. However, loading a cached kernel still requires re-reading and re-analyzing the Python kernel source (parsing it and preparing it for the compiler) before the saved artifact can be matched and reused, which takes non-negligible time. For applications with many kernels this parsing overhead alone can take several seconds.

Fastcache bypasses that parsing and analysis work. It computes a cheap cache key from the kernel source text, argument types, and compiler config, and uses it to load the compiled artifact directly.

On a Genesis simulator benchmark (`single_franka_envs.py`, Ubuntu 24.04, NVIDIA 5090):

| Configuration | Process-start kernel load time |
|---|---|
| Other caches (no fastcache) | 4.6 s |
| + fastcache | 0.3 s |

## How to use it

### Enabling fastcache on a kernel

Fastcache requires the kernel to be *pure*: all data it operates on must be passed as explicit parameters, with nothing captured from the enclosing Python scope (see [Constraints](#constraints) below). Because not all kernels satisfy this, fastcache is opt-in - you assert purity by adding `fastcache=True` to the `@qd.kernel` decorator:

```python
import quadrants as qd

@qd.kernel(fastcache=True)
def my_kernel(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(a.shape[0]):
        b[i] = a[i] * 2.0
```

That's it. On the first call, the kernel compiles normally and the fastcache entry is written to disk. When the next Python process starts, the cached artifact is loaded directly.

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

## Constraints

A kernel is eligible for fastcache only if all of the following hold:

### 1. All data flows through parameters

The kernel must receive every piece of data it operates on as an explicit parameter. It must **not** capture variables from the enclosing Python scope (closures over ndarrays, mutable globals, or any other external state). This is the core "purity" constraint - the compiled kernel's behavior must be fully determined by its arguments.

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

Sub-functions called by the kernel are also checked - they must not capture external state either.

Reaching data through the members of a [`@qd.data_oriented`](compound_types.md#qddata_oriented) or [`dataclasses.dataclass`](compound_types.md#dataclassesdataclass) parameter is **not** capture, even though the Python source (e.g. `self.x` inside a data-oriented method) reads like member access rather than a parameter. The object is itself an explicit parameter; the compiler flattens it at compile time and handles each accessed member by kind:

- an **ndarray** member is passed as a real runtime kernel parameter (an external tensor, bound at launch);
- a **primitive** member (`int` / `float` / `bool`) is baked into the kernel as a compile-time constant - its value becomes part of the kernel's specialization and is folded into the fastcache key - unless the class is declared [`template_primitives=False`](compound_types.md#runtime-primitives-template_primitivesfalse), in which case it becomes a runtime scalar parameter instead;
- a **`qd.field`** member is a globally-allocated tensor referenced directly, not a parameter (and, being baked in, disables fastcache for the call - see [Supported parameter types](#2-supported-parameter-types) below).

None of these are free variables captured from the enclosing Python scope, which is what the purity check forbids.

**Exemptions:** The following may be accessed from the enclosing scope without violating purity:

| Allowed capture | Why |
|---|---|
| `enum.Enum` values (e.g. `MyEnum.VALUE`) | Named constants that are assumed not to vary between process runs. |
| `math` / `numpy` constants (e.g. `math.pi`) | Assumed stable across process runs. |
| Quadrants module attributes (e.g. `qd.simt.Tile16x16.SIZE`) | Part of the compiler's own API; assumed consistent with the Quadrants version hash. |

Other named constants (non-enum, non-module) captured from scope will raise a `QuadrantsCompilationError`, except for `UPPERCASE` names which emit a warning instead.

Wrapping a captured global in `qd.static(...)` does **not** exempt it from this check. `qd.static` only controls compile-time evaluation; it does not put the value into the cache key, so a `qd.static`-wrapped global is still flagged - though during the current transition period this emits a warning rather than raising. To use such a constant in a fastcache kernel, pass it as a parameter (template primitive, [`@qd.data_oriented`](compound_types.md#qddata_oriented) member, or dataclass field) or make it one of the allowed captures above.

### 2. Supported parameter types

Fastcache supports the following parameter types:

| Type | Supported | Cache key includes |
|---|---|---|
| `qd.types.NDArray` (scalar, vector, matrix) | Yes | dtype, ndim, layout |
| `torch.Tensor` | Yes | dtype, ndim |
| `numpy.ndarray` | Yes | dtype, ndim |
| [`dataclasses.dataclass`](compound_types.md#dataclassesdataclass) | Yes | member types recursively; member values if annotated with `FIELD_METADATA_CACHE_VALUE` (see [Appendix - compound-type cache keying](#compound-type-cache-keying)) |
| [`@qd.data_oriented`](compound_types.md#qddata_oriented) objects | Yes | member types recursively; primitive member types and values baked into kernel, unless declared `template_primitives=False`, in which case primitive members contribute their *type* only (see [Appendix - compound-type cache keying](#compound-type-cache-keying)) |
| `qd.Template` primitives (int, float, bool) | Yes | type and value (baked into kernel) |
| Non-template primitives (int, float, bool) | Yes | type only |
| `enum.Enum` | Yes | name and value |
| `qd.field` / [`ScalarField`](matrix_vector.md#vector-and-matrix-fields) / [`MatrixField`](matrix_vector.md#vector-and-matrix-fields) | **No** | - |

If any parameter is of an unsupported type, fastcache is disabled for that call and the kernel falls back to normal compilation. For `qd.field` / [`ScalarField`](matrix_vector.md#vector-and-matrix-fields) / [`MatrixField`](matrix_vector.md#vector-and-matrix-fields) arriving through a [qd.Tensor](tensor.md)-annotated parameter, this is silent - no warning is emitted. For other unsupported types, a warning is logged at the `warn` level identifying the offending parameter.

### 3. Source code must be available

Fastcache hashes the source code of the kernel and all sub-functions it calls. If the source file cannot be read at runtime (e.g. the kernel is defined in a frozen/compiled module, or the file has been deleted), fastcache cannot validate the cache and will fall back to normal compilation.

## Cache keying

Each compiled artifact is stored under a key derived from all of the following:

- The **Quadrants version** (`quadrants.__version__`).
- The **source code** of the kernel function or any `@qd.func` it calls.
- The **argument types** (e.g. switching from `f32` to `f64`, or changing ndarray dimensionality).
- The **compiler configuration** (e.g. `arch`, `debug`, `opt_level`, `fast_math`).
- **Template parameter values** (since they are baked into the compiled kernel).

When any of these change, the resulting key is different, so a new compilation occurs and a new entry is stored. Previous entries remain on disk - multiple cached versions coexist. You do not need to manually clear the cache when making code changes - the hash mismatch causes a transparent recompilation.

## Advanced

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

## Appendix

### Compound-type cache keying

As part of generating the fastcache cache key, fastcache hashes each kernel parameter. Compound types are hashed recursively. The headline rules:

**`@qd.data_oriented`:** each attribute in `vars(obj)` is hashed. For each child:

- `qd.ndarray` member - `(dtype, ndim, layout)` is included in the cache key. Element values are not.
- Primitive (`int` / `float` / `bool` / `enum.Enum`) member - value is baked into the kernel (same semantics as a `qd.Template` primitive). Two instances of the same class with different primitive member values get different cache entries. **Exception:** if the class is declared `@qd.data_oriented(template_primitives=False)`, primitive members are lifted to runtime scalar args rather than baked, so only their *type* contributes to the cache key - two instances with different primitive values share one cache entry and changing a value does not recompile.
- Nested `@qd.data_oriented` member - recurses.
- Nested `dataclasses.dataclass` member - recurses (with the dataclass rules below).
- `qd.field` member - fastcache is disabled for the entire kernel call. The kernel still runs via normal compilation; a warn-level log line is emitted.

**`dataclasses.dataclass`:** each declared member is hashed. For each member, only the *type* is included in the cache key by default - **not** the value. To include a member's value, annotate it:

```python
import dataclasses
from quadrants.lang._fast_caching import FIELD_METADATA_CACHE_VALUE

@dataclasses.dataclass
class SimConfig:
    num_layers: int = dataclasses.field(metadata={FIELD_METADATA_CACHE_VALUE: True})
    dt: float = dataclasses.field(metadata={FIELD_METADATA_CACHE_VALUE: True})
```

This is necessary whenever a member's *value* is baked into a compiled kernel, in some way, rather than just its type. Typically this is by using [@qd.static](static.md), within the kernel.

Note the asymmetry: `@qd.data_oriented` primitive member values are baked into the kernel automatically (same semantics as `qd.Template`); `dataclasses.dataclass` members contribute only their *type* to the cache key unless you opt in per-member.
