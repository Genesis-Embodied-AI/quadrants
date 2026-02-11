# Parallelization

Each top-level for-loop will be parallelized, within a kernel
- adding a non-static `if` over the top of a for-loop will lead to the for-loop not being parallelized
- a for-loop inside one or more inlined @ti.func, and/or static `if` will be still be parallelized however

Each top-level for-loop will be launched as a separate GPU kernel, under the hood.

The top-level for-loop can be encapsulated in one or more of the following, and still be parallelized:
- if statements where the conditional is `ti.static`
- inline functions (`@ti.func`)

## Does GPU kernel launch latency matter?

Kernel launch can be done in parallel whilst the previously launched kernel is still running. This means that if the previously launched kernel takes longer to run than the launch time for the new kernel, then the kernel launch latency will be perfectly hidden.

It's important to try to make sure that the work done by each kernel is sufficient to hide the kernel launch latency, otherwise the launch latency will be a bottleneck to maximum performance.

If kernel launch latency is a bottleneck, then you can look into:
- getting each kernel to do more work, to increase the relative runtime of the kernel relative to launch time
- reducing kernel launch time

To reduce kernel launch time, the following options are available:
- field args launch more quickly than ndarray args
- global fields incur no launch latency
- reducing the number and complexity kernel parameters reduces the kernel launch latency

## Global memory

In CUDA, there are 3 main types of memory:
- registers: fast, but limited in storage capacity
- shared memory: slower than registers, but faster than global. Less limited than registers, but still limited compared to global memory
- global memory: slowest of all. High latency. But massive, typically ~10s of gigabytes at the time of writing

There are some additional types, but these are variations on global memory:
- constant memory: global memory, but which can be stored easily in cache
- local memory: storage which is private to each specific thread, but, unintuitively, is stored off-chip, and is as slow as global memory

GsTaichi gives access to shared memory, using `ti.stmt.SharedArray()`, but typically GsTaichi kernels use only global memory and register memory. You cannot directly request to use registers, but registers will be used to hold any local variables, within the limits of available registers. Fields, ndarrays, and other data, are stored in global memory. This holds some implications for synchronization.

## Thread synchronization

Typically, GsTaichi kernels use `atomic_` operations for synchronization. This is relatively easy and intuitive, and it works perfectly with global memory. The main downside is that `atomic` operations are slow, because they involve both global memory and thread synchronization, both of which are intrinsically slow, and combining them is slower still.

When using shared memory, there are various barriers and fences that can be used, to ensure that writes from all threads so far have completed, and now threads are free to read from memory written by other threads.

However, these fences and barriers do not work for global memory, at least, not across all thread blocks.

To synchronize writes across all threads in a kernel, you'll need to finish the current kernel, and launch a new kernel.

## Avoiding synchronization

If there is a way of partitioning data such that no thread ever needs to read data written by another thread, then there is no need for synchronization, and the kernels will run quickly.

## Maximizing GPU core utilization

A 4090 GPU has ~16,000 cores. A 5090 GPU has ~20,000 cores (a bit more in each case, but 16k and 20k is easier to remember). In GsTaichi, the top level for loop is parallelized over gpu threads:

```
@ti.kernel
def k1() -> None:
    for i_b in range(B):  # parallelized across B GPU threads
        # work done by each thread
```

In order to maximize the efficient usage of the GPU we want to ensure that as many cores are being used as possible. This means that ideally `B` should be at least the number of cores in the GPU.

## Is it better to put everything in one single for-loop, or to split into multiple top-level loops?

In order to ensure that the kernel runtime is longer than the kernel launch time, we want to do as much work as possible in each kernel launch. This implies fewer kernel launches, that each do more.

However, we might need to break into multiple launches in order to synchronize writes to global memory.

## Compromise

The recommendations above often are self-conflicting. For example, maximizing the number of cores being used might require using atomics for synchronization, which might make the kernels slower. Reducing kerenl launches similarly might require using atomics, which would make the kernels run more slowly. So it will not in general be possible to satisfy all the above guidelines. But, it's useful to be aware of the design choices above, and strive to achieve them. Exact choices for best performance will often be an empirical question.
