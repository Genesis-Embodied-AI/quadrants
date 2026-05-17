# Compound types

## Overview

It can be useful to combine multiple ndarrays or fields together into a single struct-like object that can be passed into kernels, and into @qd.func's.

The following compound types are available:
- `dataclasses.dataclass` — **recommended**
- `@qd.data_oriented` — for classes that define `@qd.kernel` methods
- `@qd.dataclass` (and its function-form equivalent `qd.types.struct(...)`) — legacy Quadrants `StructType`, field-only

| type                               | can be passed to qd.kernel? | can be passed to qd.func? | can contain ndarray? | can contain field? | can be nested? | supports differentiation? |
|------------------------------------|:---------------------------:|:-------------------------:|:--------------------:|:------------------:|:--------------:|:-------------------------:|
| `dataclasses.dataclass`            | yes                         | yes                       | yes                  | yes                | yes            | no [*1]                   |
| `@qd.data_oriented`               | yes                         | yes                       | yes                  | yes                | yes            | yes                       |
| `@qd.dataclass` / `qd.types.struct` | yes                       | yes                       | no                   | yes                | yes            | yes                       |

See [Nesting compatibility](#nesting-compatibility) below for a per-container × per-member-type breakdown, including the constraints on the outer kernel-arg annotation and ndarray reassignment.

## Recommendation

**Use `dataclasses.dataclass` for new code.** It supports both fields and ndarrays, can be nested, and uses standard Python — no Quadrants-specific decorator needed.

The other compound types exist for historical reasons.

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

### ndarray members

`@qd.data_oriented` classes may also hold `qd.ndarray` (and `qd.Vector.ndarray` / `qd.Matrix.ndarray`) members. Inside a `@qd.kernel`, `self.x[i]` reads and writes the element of the ndarray member at index `i`; `self.x.shape[d]` is the length along dimension `d`.

```python
@qd.data_oriented
class State:
    def __init__(self, n):
        self.x = qd.ndarray(qd.f32, shape=(n,))
        self.v = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def step(self):
        for i in range(self.x.shape[0]):
            self.x[i] += self.v[i]

state = State(100)
state.step()
```

Mixing `qd.field` and `qd.ndarray` members in the same class is also supported. Nested `@qd.data_oriented` (or nested `dataclasses.dataclass`) containers with ndarrays inside are walked recursively.

`@qd.kernel(fastcache=True)` is supported on `@qd.data_oriented` classes; see [Advanced — compound-type cache keying](fastcache.md#compound-type-cache-keying) for the cache-keying rules and what disables it.

## Nesting compatibility

This table summarises which member types are allowed inside which container type. "yes" means the member is walked correctly when the container is passed to a kernel; "no" means the member is ignored or the combination raises an error.

| Container ↓ &nbsp;&nbsp;&nbsp; / &nbsp;&nbsp;&nbsp; Member → | `qd.ndarray` | `qd.field` | primitive | `dataclasses.dataclass` | `@qd.data_oriented` | `@qd.dataclass` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `dataclasses.dataclass`         | yes | yes | yes | yes | yes [\*1] | yes |
| `@qd.data_oriented`             | yes | yes | yes | yes | yes      | yes |
| `@qd.dataclass`                 | no  | yes | yes | no  | no       | yes |

[\*1] A `dataclasses.dataclass` may *hold* a `@qd.data_oriented` member, but the **outer kernel-arg annotation** must be `qd.template()`, not the dataclass type itself. Passing a typed-dataclass kernel arg (`def k(s: Outer)`) whose member type is a `@qd.data_oriented` class raises a clear `QuadrantsSyntaxError` at compile time pointing you to `qd.template()`. The reason: typed-dataclass kernel args are flattened from annotations, but `@qd.data_oriented` carries no per-member annotations — its members are walked from the live instance, which only happens on the template path.

### Outer kernel-arg annotation

The outermost annotation you put on the kernel parameter determines how the container is walked:

| Annotation | Kernel-arg walker | Notes |
|---|---|---|
| `qd.types.NDArray[...]`           | ndarray slot                                       | leaf-level only |
| `MyDataclass` (dataclass type)    | per-member flatten using annotations               | needs every member to have a quadrants-typed annotation |
| `qd.template()`                   | value-driven walk of `vars(self)` / dataclass members | supports the full nesting matrix above |

Two practical consequences:

- **Containers with `@qd.data_oriented` anywhere in the tree** must be passed via `qd.template()` (or be the `self` of a `@qd.kernel` method on a `@qd.data_oriented` class). Using a typed-dataclass annotation on the outermost arg errors.
- **A non-frozen `dataclasses.dataclass`** can be passed via the typed-dataclass annotation, but cannot be the outer `qd.template()` arg — `qd.template()` uses the instance as a dict key inside the template-mapper and a non-frozen dataclass has `__hash__ = None`. Add `frozen=True` if you need to pass it as `qd.template()` (for example, when it holds `@qd.data_oriented` children).

### Reassigning ndarray members

For both `dataclasses.dataclass` and `@qd.data_oriented` containers passed via `qd.template()`, reassigning an ndarray member between kernel launches is supported, including changes to `dtype`, `ndim`, or layout. A new specialised kernel is compiled and cached for the new shape; subsequent launches with the original shape continue to use the original cached kernel.

### Restrictions

A few combinations are still unsupported:

- **`@qd.dataclass` (the Quadrants `StructType` decorator) cannot contain ndarrays.** This is a legacy field-only type. Use `dataclasses.dataclass` or `@qd.data_oriented` instead. (The function-form factory `qd.types.struct(...)` produces the same `StructType` and has the same restrictions.)
- **A typed-dataclass kernel-arg annotation cannot have a `@qd.data_oriented` member type** (see [\*1] above) — errors clearly at compile time.
- **An outer `qd.template()` arg of dataclass type must be `frozen=True`** — non-frozen dataclasses are unhashable and the template-mapper cannot use them as cache keys.
- **Declare all ndarray members on a `@qd.data_oriented` class in `__init__`.** The template-mapper caches the set of ndarray-attribute paths reachable from the first instance walked, per class. Adding *new* ndarray attributes on later instances of the same class is safe — the per-instance weakref in the spec key disambiguates them, and the compile-time walker registers all reachable ndarrays. But:
  - **Deleting an ndarray attribute** that was present on the first launch raises `AttributeError` on the next launch (the cached path still tries to `getattr` the missing attribute).
  - **Reassigning a post-first-walk ndarray attribute** (one not present on the first instance walked, then added later and re-assigned) to one with a different `dtype` / `ndim` is *not* detected by the in-memory invalidation tracker. The stale compiled kernel is silently reused, leading to bit-reinterpretation of the new array's storage.

## qd.dataclass / qd.types.struct

`@qd.dataclass` is a Quadrants-native `StructType` decorator. The function-form factory `qd.types.struct(name1=type1, ...)` produces the same `StructType`. Both can only contain fields and primitive types (and other `StructType` members), not ndarrays.

```python
@qd.dataclass
class Particle:
    pos: qd.types.vector(3, qd.f32)
    vel: qd.types.vector(3, qd.f32)
    mass: qd.f32
```
