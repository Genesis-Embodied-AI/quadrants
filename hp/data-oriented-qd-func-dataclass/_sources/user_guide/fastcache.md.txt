# Fastcache

## What it is

Fastcache reduces the time it takes to load cached kernels when a new Python process starts.

The standard [offline cache](init_options.md#offline_cache) already persists compiled kernels to disk. However, loading a cached kernel still requires parsing the kernel's Python AST and transforming it into IR. For applications with many kernels this front-end overhead alone can take several seconds.

Fastcache bypasses that front-end work. It computes a cheap cache key from the kernel source text, argument types, and compiler config, and uses it to load the compiled artifact directly.

On a Genesis simulator benchmark (`single_franka_envs.py`, Ubuntu 24.04, NVIDIA 5090):

| Configuration | Process-start kernel load time |
|---|---|
| Other caches (no fastcache) | 4.6 s |
| + fastcache | 0.3 s |

## How to use it

### Enabling fastcache on a kernel

Fastcache requires the kernel to be *pure*: all data it operates on must be passed as explicit parameters, with nothing captured from the enclosing Python scope (see [Constraints](#constraints) below). Because not all kernels satisfy this, fastcache is opt-in — you assert purity by adding `fastcache=True` to the `@qd.kernel` decorator:

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

The kernel must receive every piece of data it operates on as an explicit parameter. It must **not** capture variables from the enclosing Python scope (closures over ndarrays, mutable globals, or any other external state). This is the core "purity" constraint — the compiled kernel's behavior must be fully determined by its arguments.

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

**Exemptions:** The following may be accessed from the enclosing scope without violating purity:

| Allowed capture | Why |
|---|---|
| `enum.Enum` values (e.g. `MyEnum.VALUE`) | Named constants that are assumed not to vary between process runs. |
| `math` / `numpy` constants (e.g. `math.pi`) | Assumed stable across process runs. |
| Quadrants module attributes (e.g. `qd.simt.Tile16x16.SIZE`) | Part of the compiler's own API; assumed consistent with the Quadrants version hash. |

Other named constants (non-enum, non-module) captured from scope will raise a `QuadrantsCompilationError`, except for `UPPERCASE` names which emit a warning instead.

### 2. Supported parameter types

Fastcache supports the following parameter types:

| Type | Supported | Cache key includes |
|---|---|---|
| `qd.types.NDArray` (scalar, vector, matrix) | Yes | dtype, ndim, layout |
| `torch.Tensor` | Yes | dtype, ndim |
| `numpy.ndarray` | Yes | dtype, ndim |
| `dataclasses.dataclass` | Yes | member types recursively; member values if annotated with `FIELD_METADATA_CACHE_VALUE` (see [Advanced — compound-type cache keying](#compound-type-cache-keying)) |
| `@qd.data_oriented` objects | Yes | member types recursively; primitive member types and values baked into kernel (see [Advanced — compound-type cache keying](#compound-type-cache-keying)) |
| `qd.Template` primitives (int, float, bool) | Yes | type and value (baked into kernel) |
| Non-template primitives (int, float, bool) | Yes | type only |
| `enum.Enum` | Yes | name and value |
| Anything else (Pydantic models, UUIDs, back-pointers, etc.) | Yes (degraded — type identity only) | `type(v).__qualname__` (see [Pruning-driven argument hashing](#pruning-driven-argument-hashing)) |
| `qd.field` / `ScalarField` / `MatrixField` at a kernel-read path | **No** | — |

For *recognised-but-unsupported* tensor-like types (`qd.field` / `ScalarField` / `MatrixField`) reached at a path the kernel actually reads, fastcache is disabled for that call and the kernel falls back to normal compilation. For these arriving through a `qd.Tensor`-annotated parameter, this is silent — no warning is emitted; for other annotations a warning identifies the offending parameter.

For *unrecognised* types reached at a kernel-read path, the hasher uses `type(v).__qualname__` as a stable type-identity hash and emits a one-shot `[FASTCACHE][UNKNOWN_TYPE]` warning per type. Kernel-unused members do not need to be recognised at all — the pruning narrowing skips them.

### 3. Source code must be available

Fastcache hashes the source code of the kernel and all sub-functions it calls. If the source file cannot be read at runtime (e.g. the kernel is defined in a frozen/compiled module, or the file has been deleted), fastcache cannot validate the cache and will fall back to normal compilation.

## Cache keying

Each compiled artifact is stored under a key derived from all of the following:

- The **Quadrants version** (`quadrants.__version__`).
- The **source code** of the kernel function or any `@qd.func` it calls.
- The **argument types at paths the kernel actually reads** (see [Pruning-driven argument hashing](#pruning-driven-argument-hashing) below).
- The **compiler configuration** (e.g. `arch`, `debug`, `opt_level`, `fast_math`).
- **Template parameter values** (since they are baked into the compiled kernel).

When any of these change, the resulting key is different, so a new compilation occurs and a new entry is stored. Previous entries remain on disk — multiple cached versions coexist. You do not need to manually clear the cache when making code changes — the hash mismatch causes a transparent recompilation.

### Pruning-driven argument hashing

Fastcache uses a **two-level cache**:

- **L1** (source + config only): stores the set of *flat names* the kernel actually reads — e.g. `__qd_state__qd_x` for a kernel that reads `state.x`. This is the kernel's pruning info, computed at compile time.
- **L2** (L1 + narrow argument hash): stores the compiled artifact under a key that only hashes the arg paths in the L1 pruning set.

The practical implication: **kernel-unused members do not affect the cache key**. If a `@qd.data_oriented` container has an opaque metadata member (e.g. a UUID, a Pydantic config, a back-pointer to a parent solver), that member is *not* hashed because the kernel cannot read it — including it in the hash would only cause spurious cache misses without any safety benefit. The kernel reading `self.dofs_state.x` does not have its cache key disturbed by changes to `self._uid` or `self.cfg`.

For paths the kernel *does* read but that contain an unrecognised type (a type fastcache has no explicit handling for in `quadrants/lang/_fast_caching/args_hasher.py`), the hasher falls back to a deterministic `type(v).__qualname__`-based string and emits a one-shot `[FASTCACHE][UNKNOWN_TYPE]` warning per type. This keeps the cache key stable across instances of the same opaque class while making any missed tensor-like registration impossible to overlook — if a future tensor type is added to Quadrants but not registered with the args hasher, the warning fires on the first call so the gap can be closed before stale cache results occur.

`qd.field` / `ScalarField` / `MatrixField` are *recognised-but-unsupported*: their shape/dtype would affect codegen but fastcache doesn't yet know how to safely include them, so encountering one at a kernel-read path disables fastcache for the call (with a warn-level diagnostic).

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

### Compound-type cache keying

The args hasher walks compound-type kernel parameters recursively. For each leaf member it decides what (if anything) contributes to the cache key. The headline rules:

**`@qd.data_oriented`:** the walker descends into `vars(obj)`. Pruning narrowing applies to *ndarray children only* (the kernel-compile path tracks every kernel-accessed ndarray's structural path in `struct_ndarray_launch_info`); other children are always walked. For each walked child:

- `qd.ndarray` member, kernel-read — `(dtype, ndim, layout)` is included in the cache key. Element values are not.
- `qd.ndarray` member, kernel-unused — *skipped*. Changes to its dtype/ndim/layout don't invalidate the cache.
- Primitive (`int` / `float` / `bool` / `enum.Enum`) member — value is baked into the kernel (same semantics as a `qd.Template` primitive). Two instances of the same class with different primitive member values get different cache entries. (Quadrants does not currently track per-primitive kernel-access info on data_oriented args, so primitive members are always hashed — see the safety note in [Pruning-driven argument hashing](#pruning-driven-argument-hashing).)
- Opaque member (Pydantic config, UUID, parent back-pointer, anything else fastcache doesn't recognise) — hashed by `type(v).__qualname__` only. The one-shot `[FASTCACHE][UNKNOWN_TYPE]` warning fires on first encounter so a missing tensor-like registration in `args_hasher` is impossible to overlook.
- Nested `@qd.data_oriented` member — recurses (with these same rules).
- Nested `dataclasses.dataclass` member — recurses (with the dataclass rules below, including full flat-name pruning).
- `qd.field` member reached at a kernel-read path — fastcache is disabled for the entire kernel call. The kernel still runs via normal compilation; a warn-level log line is emitted. A `qd.field` member at a kernel-*unused* path is simply skipped (no diagnostic).

**`dataclasses.dataclass`:** the walker descends into the declared members. For each member, only the *type* is included in the cache key by default — **not** the value. To include a member's value, annotate it:

```python
import dataclasses
from quadrants.lang._fast_caching import FIELD_METADATA_CACHE_VALUE

@dataclasses.dataclass
class SimConfig:
    num_layers: int = dataclasses.field(metadata={FIELD_METADATA_CACHE_VALUE: True})
    dt: float = dataclasses.field(metadata={FIELD_METADATA_CACHE_VALUE: True})
```

This is necessary whenever the compiled kernel depends on the member's *value* rather than just its type (for example, when the value is used as a loop bound that the compiler bakes into the generated code). Without the annotation, two `SimConfig` instances with different `num_layers` values would share a fastcache key, and the second instance would silently load a kernel compiled for the wrong value.

Note the asymmetry: `@qd.data_oriented` primitive members are baked into the kernel automatically (same semantics as `qd.Template`); `dataclasses.dataclass` members contribute only their *type* to the cache key unless you opt in per-member.
