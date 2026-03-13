# Tensor types

There are two core tensor types:
- ndarray (`qd.ndarray`)
- field (`qd.field`)

In addition, when used from a qd kernel, fields can be:
- referenced as a global variable
- passed into the kernel as a parameter

For the remainder of this doc, we will compare three approaches, which we will refer to as:
- "ndarrays": always passed into kernels as parameters
- "global fields": referenced from kernel as global variable
- "field args": passed into the parameter as a kernel

## Example of each tensor approach

Let's first give an example of using each:

### NDArray

```python
import quadrants as qd

qd.init(arch=qd.gpu)

a = qd.ndarray(qd.i32, shape=(10,))

@qd.kernel
def f1(p1: qd.types.NDArray[qd.i32, 1]) -> None:
    p1[0] += 1
```

Note that the typing for NDArray is `qd.types.NDArray[data_type, number_dimensions]`

### Global field

```python
import quadrants as qd

qd.init(arch=qd.gpu)

a = qd.field(qd.i32, shape=(10,))

@qd.kernel
def f1() -> None:
    a[0] += 1
```
You can see that we access the global variable referencing the field directly from the kernel. No need to provide the field as a parameter.

### Field args

```python
import quadrants as qd

qd.init(arch=qd.gpu)

a = qd.field(qd.i32, shape=(10,))

@qd.kernel
def f1(p1: qd.Template) -> None:
    p1[0] += 1
```
In this case, we provide the field to the kernel via a parameter, with typing type of `qd.Template`.

## Comparison of tensor approaches

| Tensor approach | Launch latency | Runtime speed |Resizable without recompile? [*1]|Encapsulation?[*2]|
|-------------|----------------|-------------|----------------------------|----------------|
| ndarray     | Slowest        | Slower      | yes | Yes |
| global field | Fastest       | Fast        | no | No |
| field arg  | Medium          | Fast       | no | Yes |

- [*1] We'll discuss this in 'Under the covers' below
- [*2] Will be discussed in 'Encapsulation' below

Let's define each of these column headings.

### Under the covers summary

When running a kernel, three things need to happen:
- the kernel needs to be compiled
- the parameters need to be sent to the GPU (contributes to kernel launch latency)
- the kernel launch needs to be sent to the GPU (contributes to kernel launch latency)

Looking at kernel launch latency:
- field args and ndarrays both are passed in to the GPU as parameters, and hence increase launch latency
- ndarrays have more parameter processing than field args, and have the biggest launch latency

Each tensor type is bound to the compiled kernel in some way:
- global fields are permanently bound to the kernel
    - to use the kernel with a different tensor, you'd need to copy and paste the kernel, with a new name
- field args are permanently bound to the compiled kernel
    - however, as the typing `qd.Template` alludes to, you can call the kernel with different fields, and the kernel will be automatically recompiled to bind with the new field
- ndarrays are only bound by:
    - the data type (`qd.i32` vs `qd.f32` for example)
    - the number of dimensions
- Each call with an ndarray of a different data type or number of dimensions, that hasn't already been compiled for, will trigger a recompile.
- However, no recompilation is needed for passing in a different ndarray, that matches data type and number of dimensions.

### Encapsulation

Using global variables provides fairly poor encapsulation and re-use.

Both ndarrays and field args provide better encapsulation, and kernel re-use.

### launch latency vs runtime speed

For kernels that run for sufficiently long, the launch latency will be entirely hidden by the kernel runtime. Launch latency only affects performance for very short kernels.

## Recommendations

- for maximum flexibility to resize tensors, use ndarrays
- for maximum runtime speed, with good encapsulation, use field args
- if the kernels are very short, for maximum speed you might need to use global fields, but this comes at the expense of poor encapsulation
