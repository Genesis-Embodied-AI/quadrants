# Synchronization

Since running code on GPUs is inherently multi-threaded, synchronization of writes and reads to memory is important.

# Why is this an issue?

- when one thread writes to global memory, the write doesn't take place immediately. It takes 10-150ns for the data to work its way across to global memory, which is not on the GPU die itself, but in other chips, albeit on the same GPU card.
    - think of it like sending a snail mail letter, whilst you work on other things, and lead your life
- after the store statement has executed, the kernel continues to execute other statements, whilst the data works its way to memory
```
@ti.kernel f1(a: ti.Template) -> None:
     a[0] = 5
     # .. execution continues here immediately
```

Now, if another thread tries to read a[0], it will not get the new written value, but some much older value.

To prevent this issue, we need some kind of synchronization.

Synchronization of writes and reads for global memory is hard, because it is so far away from the cores; and takes such a long time to write, and read.

# Possible approaches

There are a few options:

## Each thread never reads memory written by other threads

- this approach is the safest, and the fastest
- not always possible of course

As an example though, if one submits a batch of tasks, and each task is independent, then parallelizing over the 'batch size' dimension means that threads do not read memory written by other threads

```
@ti.kernel
def f1(batch_size: int, a: ti.Template) -> None:
    for i_b in range(batch_size):
        # each thread does work independently of other threads
```

## Use separate kernels for synchronization

Sometimes we might have multiple for loops, which, on their own are thread safe, however, one for loop must not begin until the previous one has finished.

```
@ti.kernel
def f1(batch_size: int, a: ti.Template) -> None:
    for i_b in range(batch_size):
        # each thread does work independently of other threads
    for i_b in range(batch_size):
        # each thread does work independently of other threads
```

The default behavior of GsTaichi for kernels with multiple top-level for loops is to run each top level for loop in a separate GPU kernel. And this ensures that writes to global memory from the first for loop are guaranteed to be visible to all threads in the second for loop.

## Use atomics

Atomics tend to be the main other approach used by GsTaichi engineers for synchronization.

```
@ti.kernel
def f1(a: ti.Template, b: ti.Template) -> None:
    ti.atomic_add(a[0], b[0])
```
The operation is blocking, and will not return until the addition has taken place
- this is thus very slow
- on the other hand, if multiple threads execute atomic adds, with the same destination location, atomic add guarantees that the resulting destination location will take into account all the values added to it, by all threads

## Use barriers and fences

Barriers and fences only work well for shared memory. Using shared memory is an advanced feature, not typically used by GsTaichi users. Can be done though. GsTaichi offers:

### Shared memory

```
ti.simt.block.SharedArray(data_type, shape)
ti.simt.block.sync()
```
See [tests/python/test_shared_array.py](../../../../tests/python/test_shared_array.py) for examples.
