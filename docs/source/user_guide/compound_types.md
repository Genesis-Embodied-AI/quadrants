# Compound types

## Overview

It can be useful to combine multiple ndarrays or fields together into a single struct-like object that can be passed into kernels, and into @qd.func's.

The following compound types are available:
- `dataclasses.dataclass` — **recommended**
- `@qd.data_oriented` — for classes that define `@qd.kernel` methods, cannot contain ndarrays
- `@qd.struct` / `@qd.dataclass` — legacy, field-only

| type                               | can be passed to qd.kernel? | can be passed to qd.func? | can contain ndarray? | can contain field? | can be nested? | supports differentiation? |
|------------------------------------|:---------------------------:|:-------------------------:|:--------------------:|:------------------:|:--------------:|:-------------------------:|
| `dataclasses.dataclass`            | yes                         | yes                       | yes                  | yes                | yes            | no [*1]                   |
| `@qd.data_oriented`               | yes                         | yes                       | no                   | yes                | yes            | yes                       |
| `@qd.struct`, `@qd.dataclass`     | yes                         | yes                       | no                   | yes                | yes            | yes                       |

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

You can pass either a whole nested-dataclass argument or one of its sub-struct fields to a `qd.func`. The callee declares the sub-struct's type as the parameter annotation; the caller writes the attribute access at the call site:

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
- pruning of the sub-struct's leaf fields that the callee never reads

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

## qd.struct / qd.dataclass

`@qd.struct` (and its alias `@qd.dataclass`) is a Quadrants-native struct type. It can only contain fields and primitive types, not ndarrays.

```python
@qd.dataclass
class Particle:
    pos: qd.types.vector(3, qd.f32)
    vel: qd.types.vector(3, qd.f32)
    mass: qd.f32
```

### Inheritance

A `@qd.dataclass` may subclass one or more other `@qd.dataclass` types. The subclass inherits the parent's members and `@qd.func` methods, and can add its own or override inherited ones:

```python
@qd.dataclass
class Body:
    pos: qd.types.vector(3, qd.f32)
    mass: qd.f32

    @qd.func
    def momentum(self):
        return self.mass * self.pos

@qd.dataclass
class ChargedBody(Body):
    charge: qd.f32        # added to inherited `pos` and `mass`

    @qd.func
    def charged_mass(self):
        return self.mass + self.charge
```

`ChargedBody` ends up with members `pos`, `mass`, `charge` (parent members first, in declaration order) and both `momentum` and `charged_mass` methods. A member or method declared on the subclass overrides the inherited one of the same name.
