# Getting started

# Installation

## Pre-requisites
- a supported platform (MacOS Arm64, Linux x64, Windows x64), see [Supported systems](./supported_systems.md)
- a supported Python version installed, see [Supported systems](./supported_systems.md)
- optionally, a supported GPU, see [Supported systems](./supported_systems.md)

## Procedure
```
pip install gstaichi
```
## Sanity checking the installation
```
python -c 'import gstaichi as ti; ti.init(arch=ti.gpu)'
```
(should not show any error messages)

# A first GsTaichi kernel

Let's use a linear congruential generator - a pseudo-random number generator - since they are not easily possible for a compiler to optimize out. First, in normal python:

```
def lcg_np(B: int, lcg_its: int, a: npt.NDArray) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x
```
We are taking in a numpy array, of size B, looping over it. For each value in the array, wwe run 1000 iterations of LCG, then update the original value.

Let's write out the full code, including creating a numpy array, and timing this method:

```
import numpy as np
import numpy.typing as npt
import time


def lcg_np(B: int, lcg_its: int, a: npt.NDArray) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x

B = 16000
a = np.ndarray((B,), np.int32)

start = time.time()
lcg_np(B, 1000, a)
end = time.time()
print("elapsed", end - start)
```

You can find the full code also at [lcg_python.py](../../../../python/gstaichi/examples/lcg_python.py)

On a Macbook Air M4, this gives the following output:
```
# elapsed 5.552601099014282
```

Now let's convert it to gstaichi

Here is the function, written as a GsTaichi kernel:

```
@ti.kernel
def lcg_ti(B: int, lcg_its: int, a: ti.types.NDArray[ti.i32, 1]) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x
```

Yes, it's the same except:
- added `@ti.kernel` annotation
- changed type from `npt.NDArray` to `ti.types.NDArray[ti.i32, 1]`

Before we run this we need to import gstaichi, and initialize it:

```
import gstaichi as ti

ti.init(arch=ti.gpu)
```
The `arch` parameter lets you choose between `gpu`, `cpu`, `metal`, `cuda`, `vulkan`.
- using `ti.gpu` will use the first GPU it finds

We'll also need to create a gstaichi ndarray:
```
a = ti.ndarray(ti.i32, (B,))
```
By comparison with numpy array:
- the parameters are reversed
- we use `ti.i32` instead of `np.int32`

When we time the kernel we have to be careful:
- running the kernel function `lcg_ti()` starts the kernel
- ... but it does not wait for it to finish

We'll only wait for it to finish when we access data from the kernel, or we call an explicit synchronization function, like `ti.sync()`. So let's do that:
```
ti.sync()
end = time.time()
```

In addition, whilst it looks like we aren't using the gpu before this, in fact we are: when we create the NDArray, the ndarray needs to be created in GPU memory, and again this happens asychronously. So before calling start we also add ti.sync():

```
ti.sync()
start = time.time()
```

The full program then becomes:

```
import gstaichi as ti
import time


@ti.kernel
def lcg_ti(B: int, lcg_its: int, a: ti.types.NDArray[ti.i32, 1]) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x

ti.init(arch=ti.gpu)

B = 16000
a = ti.ndarray(ti.int32, (B,))

ti.sync()
start = time.time()

lcg_ti(B, 1000, a)

ti.sync()
end = time.time()
print("elapsed", end - start)
```

When run on a Macbook Air M4, the output is something like:
```
# [GsTaichi] version 1.8.0, llvm 15.0.7, commit 5afed1c9, osx, python 3.10.16
# [GsTaichi] Starting on arch=metal
# elapsed 0.04660296440124512
```
Around 120x faster.

On one of our linux boxes with a 5090 GPU, the results are:
- numpy: 6.90 seconds
- gstaichi: 0.0199 seconds
- => 346 times faster

## What does GsTaichi do with the kernel function?

- any top level for loops are parallelized across the GPU cores (or CPU, if you run on CPU)
    - in our case, there will be 16,000 threads
    - compared to just a single thread in the numpy case

## fields: even faster

GsTaichi ndarrays are easy to use, and flexible, but we can increase speed by another ~30% or so (depending on the kernel), by using fields.

The kernel above doesn't load or store data except at the start and end: it's just exercising the GPU APU. To see the difference between Taichi ndarray and GsTaichi field runtime speed, we need a kernel that does more loads and stores.

We'll do a simple kernel that copies from one tensor to another. To avoid simply measuring the latency to read and write from/to global memory, we'll read and write the same values repeatedly.

```
import argparse
import time
import gstaichi as ti

ti.init(arch=ti.gpu)

parser = argparse.ArgumentParser()
parser.add_argument("--use-field", action="store_true")
args = parser.parse_args()

use_field = args.use_field
if use_field:
    V = ti.field
    ParamType = ti.Template
else:
    V = ti.ndarray
    ParamType = ti.types.NDArray[ti.i32, 1]

@ti.kernel
def copy_memory(N: int, a: ParamType, b: ParamType) -> None:
    for n in range(N):
        b[n % 100] = a[n % 100]

N = 20_000
a = V(ti.i32, (100,))
b = V(ti.i32, (100,))

# warmup
copy_memory(N, a, b)

num_its = 1000

ti.sync()
start = time.time()
for it in range(num_its):
    copy_memory(N, a, b)
ti.sync()
end = time.time()
iteration_time = (end - start) / num_its * 1_000_000
print("iteration time", iteration_time, "us")
```

Here are the outputs, using a 5090 gpu, on ubuntu:
```
$ python doc/mem_copy.py
[GsTaichi] version 1.8.0, llvm 15.0.4, commit b4755383, linux, python 3.10.15
[GsTaichi] Starting on arch=cuda
iteration time 29.45709228515625 us

$ python doc/mem_copy.py --use-field
[GsTaichi] version 1.8.0, llvm 15.0.4, commit b4755383, linux, python 3.10.15
[GsTaichi] Starting on arch=cuda
iteration time 21.44002914428711 us
```
=> in this test, fields are around 28% faster than ndarrays.

Technical note: the exact ratio depends on the kernel. In addition it's possible to construct toy examples like this where ndarray appears to be faster than fields, but in many commonly used kernels, such as Genesis func narrow phase kernel, for collisions, we observe fields are around ~30% faster
