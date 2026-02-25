# Compound types

# Overview

It can be useful to combine multiple ndarrays or fields together into a single struct-like object that can be passed into kernels, and into @qd.func's.

The following compound types are available:
- `@qd.struct`
- `@qd.dataclass` (effectively an alias of `@qd.struct`: uses same underlying class, and mechanism)
- `@qd.data_oriented`
- `dataclasses.dataclass`

| type                               | can be passed to qd.kernel? | can be passed to qd.func? | can contain ndarray? | can contain field? | can be mixed with other parameters? | supports differentiation? | can be nested? | caches arguments? | comments |
|------------------------------------|-----------------------------|---------------------------|----------------------|--------------------|-------------------------------------|---------------------------|----------------|-------------------|----------|
| `@qd.struct`, `@qd.dataclass`      |                         yes | yes                       |                   no |                yes | yes                                 | yes                       | yes            | no                |          |
| `@qd.data_oriented`                |yes                          | yes                       | no                   |  yes               |yes                                   | yes                       | no             | no                |          |
| `@dataclasses.dataclass`            | yes                         | yes                       | yes                  | yes                | yes                                   | yes                     |yes         | no                | recommended approach |

`@dataclasses.dataclass` is the current recommended approach:
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
- can be passed into kernels (`@qd.kernel`)
- can be passed into sub-functions (`@qd.func`)
- can be combined with other parameters, in the function signature of such calls
- does not affect runtime performance, compared to passing in the elements directly, as parameters

The members are read-only. However, ndarrays and fields are stored as references (pointers), so the contents of the ndarrays and fields can be freely mutated by the kernels and qd.func's.

## Limitations:
- on Mac, can only be used with Fields, not with ndarray [*1]
- Passing python dataclasses to `@qd.real_func` is not supported currently
- automatic differentiation is not supported currently

Notes:
- [*1] technically can be used with ndarray, but in practice, the current implementation will result in exceeding the number of allowed kernel parameters

## Usage:

Example:

```
import quadrants as qd
from dataclasses import dataclass

qd.init(arch=qd.gpu)

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
```
my_struct.a[35] 3
my_struct.b[37] 5
```
