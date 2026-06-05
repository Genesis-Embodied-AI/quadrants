# Compound types

## Overview

It can be useful to combine multiple ndarrays or fields together into a single struct-like object that can be passed into kernels, and into @qd.func's.

The following compound types are available:
- `@dataclasses.dataclass` — lightweight container of tensors and primitives; can contain ndarrays
- `@qd.data_oriented` — enables object-oriented programming with Quadrants: `@qd.kernel` can decorate instance methods, with tensors stored on `self` instead of being passed in as kernel arguments or held as global fields
- `@qd.dataclass` — for structures that are embedded into the kernel, and don't contain ndarrays

| property                            | `@dataclasses.dataclass`              | `@qd.data_oriented`                   | `@qd.dataclass`                     |
|-------------------------------------|:-------------------------------------:|:-------------------------------------:|:-----------------------------------:|
| Exists as a struct in the kernel    | no (members flattened to args)        | no (members flattened to args)        | yes (fixed memory layout)           |
| Can be used as tensor element type  | no                                    | no                                    | yes                                 |
| Members can be tensors (field, ndarray) | yes                               | yes                                   | no                                  |
| `@qd.kernel` instance methods       | no                                    | yes                                   | no                                  |
| `@qd.func` instance methods         | no                                    | yes                                   | yes                                 |
| Member declaration                  | type-annotated class fields           | live attributes (no annotations)      | type-annotated class fields         |
| Kernel-arg annotation               | `MyStruct` (the dataclass type)       | `qd.Template`                       | `MyStruct` (the struct type)        |

Note on the "Members can be tensors" row for `@qd.dataclass`: a `@qd.dataclass`'s members must be primitives, fixed vectors, or fixed matrices — not `qd.field` / `qd.ndarray`. However, *allocating* a `@qd.dataclass` as a tensor of structs in SoA layout (`MyStruct.field(shape=(N,), layout=qd.Layout.SOA)`) extrudes each member into its own length-`N` tensor — so the resulting *collection* effectively behaves like a struct of parallel tensors, even though the `@qd.dataclass` type itself doesn't have tensor-typed members. See the [`@qd.dataclass` section](#qddataclass-qdtypesstruct) below.

> ⚠️ **Deprecation: `@dataclasses.dataclass` instance passed via `qd.Template`.**
> Passing a `@dataclasses.dataclass` instance into a `qd.Template`-annotated kernel parameter is not supported and emits a `DeprecationWarning` at compile time. In a future release it will become an error.

See [Nesting compatibility](#nesting-compatibility) below for a per-container × per-member-type breakdown, including the constraints on the outer kernel-arg annotation and ndarray reassignment.

## How to choose a compound type?

It's of course very subjective, but some guidelines you could consider:

- if you are trying to write a python class that runs on the GPU => use a `@qd.data_oriented`
- if you are trying to write typed dataclasses, for passing data around between the `@data_oriented` classes, and between methods of the same `@data_oriented` class => use `@dataclasses.dataclass`es
- `@qd.dataclass` is used to create structured element types for field tensors. We also use it to create the Cholesky [tiles](tile.md).

## dataclasses.dataclass

`dataclasses.dataclass` allows you to create structs containing:
- ndarrays
- fields
- primitive types

These structs:
- can be passed into kernels (`@qd.kernel`) and sub-functions (`@qd.func`)
- can be combined with other parameters in the function signature
- do not affect runtime performance compared to passing elements directly as parameters
- can be nested (a dataclass can contain other dataclasses)

The members are read-only. However, ndarrays and fields are stored as references (pointers), so the contents of the ndarrays and fields can be freely mutated by the kernels and `@qd.func`s.

### Example

```python
import quadrants as qd
from dataclasses import dataclass

qd.init(arch=qd.gpu)

@dataclass
class MyStruct:
    a: qd.types.NDArray[qd.i32, 1]
    b: qd.types.NDArray[qd.i32, 1]

a = qd.ndarray(qd.i32, shape=(55,))
b = qd.ndarray(qd.i32, shape=(57,))

@qd.kernel
def k1(my_struct: MyStruct) -> None:
    my_struct.a[35] += 3
    my_struct.b[37] += 5

my_struct = MyStruct(a=a, b=b)
k1(my_struct)
print("my_struct.a[35]", my_struct.a[35])
print("my_struct.b[37]", my_struct.b[37])
```

Output:
```text
my_struct.a[35] 3
my_struct.b[37] 5
```

### Nesting

Dataclasses can contain other dataclasses:

```python
@dataclass
class Inner:
    x: qd.types.NDArray[qd.f32, 1]

@dataclass
class Outer:
    inner: Inner
    y: qd.types.NDArray[qd.f32, 1]

@qd.kernel
def k2(s: Outer) -> None:
    s.inner.x[0] = 1.0
    s.y[0] = 2.0
```

### Passing nested sub-structs to a `qd.func`

You can pass either a whole nested-dataclass argument or one of its sub-struct members to a `qd.func`. The callee declares the sub-struct's type as the parameter annotation; the caller writes the attribute access at the call site:

```python
@dataclass
class Inner:
    x: qd.types.NDArray[qd.f32, 1]

@dataclass
class Outer:
    inner: Inner
    y: qd.types.NDArray[qd.f32, 1]

@qd.func
def touch_inner(inner: Inner) -> None:
    inner.x[0] += 1.0

@qd.func
def touch_outer(s: Outer) -> None:
    s.y[0] += 10.0
    touch_inner(s.inner)        # call site inside a qd.func body

@qd.kernel
def k(s: Outer) -> None:
    touch_outer(s)              # whole-struct call
    touch_inner(s.inner)        # sub-struct call
```

Sub-struct passing supports:

- arbitrary nesting depth (`f(s.a.b.c)` where each level is a dataclass)
- positional and keyword call sites (`f(s.inner)` and `f(inner=s.inner)`)
- call sites both directly inside `@qd.kernel` bodies and inside other `@qd.func` bodies
- pruning of the sub-struct's leaf members that the callee never reads

Note: assigning a sub-struct to a local variable and then passing it (`t = s.inner; touch_inner(t)`) is **not** supported. Pass the attribute access directly at the call site.

### Frozen vs non-frozen

A `dataclasses.dataclass` may be either non-frozen (the default) or frozen (`@dataclass(frozen=True)`). Both work as kernel arguments, but **kernel launch is faster with `frozen=True`** (because it enables some optimizations that would otherwise not be possible). Recommend `frozen=True` unless you specifically need to rebind members after construction. Note that rebinding members after construction contradicts certain best practices; for example, it is typically incompatible with type linters such as pyright and mypy.

### Under the hood

A `dataclasses.dataclass` is a Python-only container. The compiler reads it at compile time and flattens its members into individual kernel parameters — the container itself has no memory layout and doesn't exist on the kernel side. Inside a kernel, tensor members are read-write through indexing (`s.x[i] = ...`), but the member *binding* itself (`s.x = other_tensor`) cannot be reassigned from inside a kernel.

## qd.data_oriented

`@qd.data_oriented` is designed for classes that define `@qd.kernel` methods as class members. It wraps these methods to correctly bind `self` during kernel compilation.

```python
@qd.data_oriented
class Simulation:
    def __init__(self, n):
        self.x = qd.field(qd.f32, shape=n)

    @qd.kernel
    def step(self):
        for i in self.x:
            self.x[i] += 1.0

sim = Simulation(100)
sim.step()
```

`@qd.data_oriented` objects can also be passed as `qd.Template` parameters to kernels defined outside the class, and they support nesting (one `@qd.data_oriented` struct containing another).

### Primitive members

Primitive members on `self` (e.g. `int`, `float`, `bool`, `enum.Enum`) are supported, but they are treated as **template values**: each distinct primitive value across instances triggers a new kernel compilation, with the value baked into the kernel IR.

```python
@qd.data_oriented
class Simulation:
    def __init__(self, n):
        self.n = n
        self.x = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def step(self):
        for i in range(self.n):
            self.x[i] += 1.0

Simulation(100).step()   # compiles kernel #1 with n=100 baked in
Simulation(200).step()   # compiles kernel #2 with n=200 baked in
```

### Tensor members

`@qd.data_oriented` classes may hold tensor members of any backend: `qd.field`, `qd.ndarray`, or `qd.Tensor`.

```python
@qd.data_oriented
class State:
    def __init__(self, n):
        self.n = n
        self.a = qd.field(qd.f32, shape=n)
        self.b = qd.ndarray(qd.f32, shape=(n,))
        self.c = qd.tensor(qd.f32, shape=(n,))

    @qd.kernel
    def step(self):
        for i in range(self.n):
            self.a[i] += 1.0
            self.b[i] += 1.0
            self.c[i] += 1.0

state = State(100)
state.step()
```

### `stable_members=True` (required for full launch speed)

By default, every kernel launch on a `@qd.data_oriented` instance walks the instance's ndarray-typed members to detect reassignment — if any member now points at a different `qd.ndarray` than at compile time (different dtype / ndim / layout), the cached compiled kernel is invalidated and a fresh one is compiled. This safety check adds ~1–2 µs per launch per ndarray member.

For a Genesis-style container with dozens of ndarray attrs called thousands of times per simulation step, those microseconds dominate. **If your `@qd.data_oriented` class allocates its ndarray members once in `__init__` and never reassigns them**, declare it stable:

```python
@qd.data_oriented(stable_members=True)
class Simulation:
    def __init__(self, n):
        self.x = qd.ndarray(qd.f32, shape=(n,))
        self.v = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def step(self):
        ...
```

Equivalent class-level form:

```python
@qd.data_oriented
class Simulation:
    _qd_stable_members = True
    ...
```

`stable_members=True` is required to hit full launch throughput on Genesis-scale containers — without it, the per-call reassignment walk caps kernel dispatch at a few hundred kHz regardless of how cheap the kernel body is.

**Violating the promise is undefined behaviour.** If you reassign an ndarray member after the first launch with a different dtype / ndim / layout, the previously-compiled kernel is silently reused on the new array's storage. Bit-reinterprets, wrong shapes, segfaults are all possible. If your ndarray members might genuinely change shape between calls, do not set this flag.

### Fastcache

`@qd.kernel(fastcache=True)` is supported on methods of `@qd.data_oriented` classes, but is disabled for fields; see [Appendix — compound-type cache keying](fastcache.md#compound-type-cache-keying) for more information.

### Under the hood

Like `dataclasses.dataclass`, a `@qd.data_oriented` object is Python-only — the compiler flattens it into individual kernel parameters and the object itself has no kernel-side representation. Unlike `dataclasses.dataclass` it needs no member annotations: the compiler reads the live instance's attributes directly. Primitive members are baked into the kernel as constants, so each distinct primitive value compiles a new specialised kernel.

## qd.dataclass / qd.types.struct

Unlike `@qd.data_oriented` and `@dataclasses.dataclass`, `@qd.dataclass` creates a struct type that is available *inside* the kernels themselves. The other two compound types only exist on the Python side, before compilation, and don't appear in compiled kernel code at all.

`@qd.dataclass` members can only be:

- primitives (`qd.f32`, `qd.i32`, `qd.bool`, etc.)
- fixed-size vectors (`qd.types.vector(N, dtype)`)
- fixed-size matrices (`qd.types.matrix(M, N, dtype)`)

A `qd.dataclass` is analogous to a C struct. All members, including the fixed-size vectors and fixed-size matrices, are laid out within the struct itself. They are not pointers to tensors allocated elsewhere. Changing the size of a vector or matrix changes thus the size of the qd.dataclass.

A consequence of a qd.dataclass containing all members within itself, rather than being pointers, is that a qd.dataclass cannot contain `qd.field` or `qd.ndarray` members.

A `@qd.dataclass` can be turned into a tensor of structs (e.g. `MyStruct.field(shape=(N,))`) with two possible memory layouts:

- **Struct-of-arrays (SoA)** (`qd.Layout.SOA`): extrudes each member of the struct into its own tensor of length `N`.
- **Array-of-structs (AoS)** (`qd.Layout.AOS`): the storage is an array of `N` struct cells laid out contiguously in memory. AoS is only available with `qd.field` backing.

```python
@qd.dataclass
class Particle:
    pos: qd.types.vector(3, qd.f32)
    vel: qd.types.vector(3, qd.f32)
    mass: qd.f32

# AOS layout: each element of `particles` is a (pos, vel, mass) cell contiguous in memory.
# Only possible because Particle is a StructType — `@qd.data_oriented` and
# `dataclasses.dataclass` containers can't be the element type of a tensor.
particles = Particle.field(shape=(N,), layout=qd.Layout.AOS)
```

Methods can be added to a `@qd.dataclass` and may be decorated with `@qd.func` so they can be called from kernels via `instance.method(...)` syntax (the call is inlined at compile time, like any other `@qd.func`).

```python
@qd.dataclass
class Particle:
    pos: qd.types.vector(3, qd.f32)
    vel: qd.types.vector(3, qd.f32)
    mass: qd.f32

    @qd.func
    def kinetic_energy(self):
        return 0.5 * self.mass * self.vel.dot(self.vel)

particles = Particle.field(shape=(N,))

@qd.kernel
def total_ke() -> qd.f32:
    total = 0.0
    for i in range(N):
        total += particles[i].kinetic_energy()
    return total
```

`qd.types.struct(name1=type1, ...)` is the function-form equivalent of `@qd.dataclass`: it builds the same `StructType` without a class body.

```python
vec3 = qd.types.vector(3, qd.f32)
Particle = qd.types.struct(pos=vec3, vel=vec3, mass=qd.f32)
particles = Particle.field(shape=(N,))
```

### Under the hood

Unlike the other two compound types, `@qd.dataclass` is a real kernel-side type with a fixed memory layout. Each instance is laid out contiguously in memory, members are stored by value, and a tensor of the struct can be allocated (`Particle.field(...)`). Storing by value is also why ndarrays can't be members — ndarrays are heap-allocated buffers with dynamic shape and don't fit into a fixed-size cell.

## Nesting compatibility

This table summarises which member types are allowed inside which container type. "yes" means the member is walked correctly when the container is passed to a kernel; "no" means the member is ignored or the combination raises an error.

| Container ↓ &nbsp;&nbsp;&nbsp; / &nbsp;&nbsp;&nbsp; Member → | `qd.ndarray` | `qd.field` | primitive | `dataclasses.dataclass` | `@qd.data_oriented` | `@qd.dataclass` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `dataclasses.dataclass`         | yes | yes | yes | yes | deprecated | yes |
| `@qd.data_oriented`             | yes | yes | yes | yes | yes      | yes |
| `@qd.dataclass`                 | no  | yes | yes | no  | no       | yes |

### Outer kernel-arg annotation

The outermost annotation you put on the kernel parameter determines how the container is walked:

| Annotation | Kernel-arg walker | Notes |
|---|---|---|
| `qd.types.NDArray[...]`           | ndarray slot                                       | leaf-level only |
| `MyDataclass` (dataclass type)    | per-member flatten using annotations               | recommended for `@dataclasses.dataclass`; needs every member to have a quadrants-typed annotation |
| `qd.Template`                   | value-driven walk of `vars(self)`                  | for `@qd.data_oriented` containers (and primitives). **Not** supported for `@dataclasses.dataclass` — see the [deprecation notice in Overview](#overview). |

Practical consequence:

- **`@qd.data_oriented` containers** must be passed via `qd.Template` (or be the `self` of a `@qd.kernel` method on a `@qd.data_oriented` class). Using a typed-dataclass annotation on the outermost arg errors.

### Reassigning ndarray members

For `@qd.data_oriented` containers passed via `qd.Template`, reassigning an ndarray member between kernel launches is supported, including changes to `dtype`, `ndim`, or layout. A new specialised kernel is compiled and cached for the new shape; subsequent launches with the original shape continue to use the original cached kernel. (For `@dataclasses.dataclass` containers — passed via the dataclass-type annotation — the member binding follows the standard dataclass mutability rules: frozen dataclasses can't rebind, non-frozen ones can, and a rebind triggers a fresh kernel arg setup on the next launch.)

### Restrictions

- **`@qd.dataclass` cannot contain `qd.ndarray` or `qd.field` members.** See the [`@qd.dataclass`](#qddataclass-qdtypesstruct) section above for the full list of allowed member types. (The function-form factory `qd.types.struct(...)` has the same restrictions.)
- **A typed-dataclass kernel-arg annotation cannot have a `@qd.data_oriented` member type** — errors clearly at compile time. Typed-dataclass kernel args are flattened from annotations, but `@qd.data_oriented` carries no per-member annotations, so its members can only be walked from the live instance, which only happens on the `qd.Template` path.
- **Declare all ndarray members on a `@qd.data_oriented` class in `__init__`.** The template-mapper caches the set of ndarray-attribute paths reachable from the first instance walked, per class. Adding *new* ndarray attributes on later instances of the same class is safe — the per-instance weakref in the spec key disambiguates them, and the compile-time walker registers all reachable ndarrays. But:
  - **Deleting an ndarray attribute** that was present on the first launch raises `AttributeError` on the next launch (the cached path still tries to `getattr` the missing attribute).
  - **Reassigning a post-first-walk ndarray attribute** (one not present on the first instance walked, then added later and re-assigned) to one with a different `dtype` / `ndim` is *not* detected by the in-memory invalidation tracker. The stale compiled kernel is silently reused, leading to bit-reinterpretation of the new array's storage.
