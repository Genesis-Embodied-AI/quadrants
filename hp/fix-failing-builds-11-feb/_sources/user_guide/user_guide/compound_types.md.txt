# Compound types

# Overview

It can be useful to combine multiple ndarrays or fields together into a single struct-like object that can be passed into kernels, and into @ti.func's.

The following compound types are available:
- `@ti.struct`
- `@ti.dataclass` (effectively an alias of `@ti.struct`: uses same underlying class, and mechanism)
- `@ti.data_oriented`
- `dataclasses.dataclass`

| type                               | can be passed to ti.kernel? | can be passed to ti.func? | can contain ndarray? | can contain field? | can be mixed with other parameters? | supports differentiation? | can be nested? | caches arguments? | comments |
|------------------------------------|-----------------------------|---------------------------|----------------------|--------------------|-------------------------------------|---------------------------|----------------|-------------------|----------|
| `@ti.struct`, `@ti.dataclass`      |                         yes | yes                       |                   no |                yes | yes                                 | yes                       | yes            | no                |          |
| `@ti.data_oriented`                |yes                          | yes                       | no                   |  yes               |yes                                   | yes                       | no             | no                |          |
| `@dataclasses.dataclass`            | yes                         | yes                       | yes                  | yes                | yes                                   | yes                     |yes         | no                | recommended approach |

`@dataclass.dataclass` is the current recommended approach:
- supports both fields and ndarrays
- can be nested
- can be used in both kernel and func calls
- can be combined with other parameters, in a kernel or func call

# dataclasses.dataclass

dataclasses.dataclass - henceforth referred to as 'dataclass' in this doc - allows you to create heterogeneous structs containing:
- ndarray
- fields
- primitive types

This struct:
- can be passed into kernels (`@ti.kernel`)
- can be passed into sub-functions (`@ti.func`)
- can be combined with other parameters, in the function signature of such calls
- does not affect runtime performance, compared to passing in the elements directly, as parameters

The members are read-only. However, ndarrays and fields are stored as references (pointers), so the contents of the ndarrays and fields can be freely mutated by the kernels and ti.func's.

## Limitations:
- on Mac, can only be used with Fields, not with ndarray [*1]
- Passing python dataclasses to `@ti.real_func` it not supported currently
- automatic differentiation is not supported currently

Notes:
- [*1] technically can be used with ndarray, but in practice, the current implementation will result in exceeding the number of allowed kernel parameters

## Usage:

Example:

```
import gstaichi as ti
from dataclasses import dataclass

ti.init(arch=ti.gpu)

a = ti.ndarray(ti.i32, shape=(55,))
b = ti.ndarray(ti.i32, shape=(57,))

@ti.kernel
def k1(my_struct: MyStruct) -> None:
    my_struct.a[35] += 3
    my_struct.b[37] += 5

my_struct = MyStruct(a=a, b=b)
k1(my_struct)
print("my_struct.a[35]", my_struct.a[35])
print("my_struct.b[37]", my_struct.b[37])
```

Output:
```
my_struct.a[35] 3
my_struct.b[37] 5
```
