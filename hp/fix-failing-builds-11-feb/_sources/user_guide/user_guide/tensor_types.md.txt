# Tensor types

There are three core tensor types:
- ndarray (`ti.ndarray`)
- global field (`ti.field`, referenced as a global variable, from a kernel)
- field arg (`ti.field`, passed into a kernel as a parameter)

# Example of each tensor type

Let's first give an example of using each:

## NDArray

```
import gstaichi as ti

ti.init(arch=ti.gpu)

a = ti.ndarray(ti.i32, shape=(10,))

@ti.kernel
def f1(p1: ti.types.NDArray[ti.i32, 1]) -> None:
    p1[0] += 1
```

Note that the typing for NDArray is `ti.types.NDArray[data_type, number_dimensions]`

## Global field

```
import gstaichi as ti

ti.init(arch=ti.gpu)

a = ti.field(ti.i32, shape=(10,))

@ti.kernel
def f1() -> None:
    a[0] += 1
```
You can see that we access the global variable referencing the field directly from the kernel. No need to provide the field as a parameter.

## Field args

```
import gstaichi as ti

ti.init(arch=ti.gpu)

a = ti.field(ti.i32, shape=(10,))

@ti.kernel
def f1(p1: ti.Template) -> None:
    p1[0] += 1
```
In this case, we provide the field to the kernel via a parameter, with typing type of `ti.Template`.

# Comparison of tensor types

| Tensor type | Launch latency | Runtime speed |Resizable without recompile? [*1]|Encapsulation?[*2]|
|-------------|----------------|-------------|----------------------------|----------------|
| ndarray     | Slowest        | Slower      | yes | Yes |
| global field | Fastest       | Fast        | no | No |
| field arg.  | Medium          | Fast       | no | Yes |

- [*1] We'll discuss this in 'Under the covers' below
- [*2] Will be discussed in 'Encapsulation' below

Let's define each of these column headings.

## Under the covers summary

When running a kernel, two things need to happen:
- the kernel needs to be compiled
- the parameters need to be sent to the GPU
- the kernel launch need to be sent to the GPU

Compilation speed is not affected by the tensor type. However:
- field args and ndarrays both are passed in to the GPU as parameters, and hence increase launch latency
- ndarrays have more parameter processing than field args, and have the biggest launch latency

Each tensor type is bound to the compiled kernel in some way:
- global fields are permanently bound to the kernel
    - to use the kernel with a different tensor, you'd need to copy and paste the kernel, with a new name
- field args are permanently bound to the compiled kernel
    - however, as the typing `ti.Template` alludes to, you can call the kernel with different fields, and the kernel will be automatically recompiled to bind with the new field
- ndarrays are only bound by:
    - the data type (`ti.i32` vs `ti.f32` for example)
    - the number of dimensions
    - you cannot pass in an ndarray with different data type or number of dimensions into the kernel, however
    - ... no recompilatino is needed for:
         - resizing the ndarray, or
         - passing in a different ndarray, that matches data type and number of dimensions

## Encapsulation

Using global variables provides fairly poor encapsulation and re-use.

Both ndarrays and field args provide better encapsulation, and kernel re-use.

## launch latency vs runtime speed

For kernels that run for sufficiently long, the launch latency will be entirely hidden by the kernel runtime. Launch latency only affects performacne for very short kernels.

# Recommendations

- for maximum flexibility to resize tensors, use ndarrays
- for maximum runtime speed, with good encapsulation, use field args
- if the kernels are very short, for maximum speed you might need to use global fields, but this comes at the expense of good encapsulation
