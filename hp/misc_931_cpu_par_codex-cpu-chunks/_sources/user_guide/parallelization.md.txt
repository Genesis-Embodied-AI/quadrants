# Parallelization

Each top-level for-loop will be parallelized, within a kernel. On a GPU, each top-level for-loop is launched as a separate GPU kernel. On the CPU, parallel loops run on CPU worker threads.

A top-level for-loop can be encapsulated in one or more of the following, and still be parallelized:
- `if` statements where the conditional is [`qd.static`](static.md)
- inline functions (`@qd.func`)

Note that adding a non-static `if` over the top of a for-loop will lead to the for-loop NOT being parallelized.

## CPU parallelization

On the CPU, a parallel top-level loop runs on a thread pool, a group of reusable worker threads. A worker is a CPU thread that executes assigned work. The runtime, the part of Quadrants that manages execution, partitions the loop iterations across scheduled tasks. A scheduled task is a group of work assigned to one worker. Each worker executes that task's iterations sequentially. Other workers can execute other tasks in parallel.

The number of tasks can differ from the number of workers. When there are more tasks than workers, workers take more tasks as they finish. A task is not permanently associated with a particular worker.

### Default CPU scheduling

By default, Quadrants creates one worker thread per system-reported logical core.

For a CPU `range()` loop, the compiler groups consecutive iterations into blocks. A block is a range of original iterations processed by one generated inner loop. The compiler creates one block per worker and schedules each block as a separate task.

For example, consider this loop with 12,000 iterations. Here `work(i)` stands for the original work at iteration index `i`:

```python
for i in range(12000):
    work(i)
```

With 12 workers, the default scheduling divides these iterations into 12 blocks of 1,000 iterations each. The following pseudocode shows the resulting work and its scheduling. Pseudocode illustrates the steps without requiring executable Python. Here `submit_task` means scheduling the indented work on an available worker. Each task keeps its own block index:

```text
for block_index in range(12):
    submit_task:
        start = block_index * 1000
        end = start + 1000

        # Compiled inner loop for this block:
        for i in range(start, end):
            work(i)
```

Block 0 runs original iteration indices 0 through 999. Block 1 runs indices 1,000 through 1,999. This continues through block 11, which runs indices 11,000 through 11,999. Each task executes its inner loop sequentially, while different tasks can run in parallel.

### make_cpu_multithreading_loop=False

Setting `make_cpu_multithreading_loop=False` in `qd.init(...)` changes how iterations are divided among tasks. The default mode creates one task per configured worker, with all loop iterations partitioned across these tasks. With `False`, Quadrants groups the iterations into fixed-size groups, 32 by default. The number of tasks is independent of the number of workers, and proportional to the number of loop iterations.

For the same 12,000-iteration loop, the default task size of 32 produces this pseudocode. Each `submit_task` schedules the indented work on an available worker and retains its own start and end indices:

```text
for task_start in range(0, 12000, 32):
    task_end = min(task_start + 32, 12000)
    submit_task:
        for i in range(task_start, task_end):
            work(i)
```

This creates 375 tasks of 32 iterations each. A pool of 12 workers processes them, with each worker taking another task when it finishes. The default mode instead creates 12 tasks of 1,000 iterations for this loop.

### Comparison of make_cpu_multithreading_loop False vs True

With `True` - the default - the number of tasks equals the configured worker count. For long loops, this groups many iterations into each task and keeps the number of scheduling operations small. This is efficient for very large numbers of iterations, and where the bodies of each loop take comparable compute time. For small loop bodies, an additional benefit is that the compiler can unroll the body, reducing loop control overhead.

The tradeoff is that each task is indivisible once it starts. If one task takes much longer than the others, its worker must finish that task's remaining iterations while the other workers may be idle.

With `False`, the task size stays fixed - same number of iterations per task - and the task count grows with the iteration count. When there are more tasks than workers, a worker that finishes one task can take another. This lets workers share an expensive region of the loop when that region spans several tasks.

The trade-off is that for large numbers of loop iterations, by default many tasks will be created, which will require more scheduling operations, introducing overhead.

How to choose?
- with large numbers of similar size small iteration bodies => use `True`
- with small numbers of unevenly sized large iteration bodies => use `False`
- other scenarios => empirical question

Select the mode with `qd.init(arch=qd.cpu, make_cpu_multithreading_loop=True)` or `qd.init(arch=qd.cpu, make_cpu_multithreading_loop=False)`. Omitting the argument selects `True`.

### Parameterizing make_cpu_multithreading_loop=True

Select this mode with `qd.init(arch=qd.cpu, make_cpu_multithreading_loop=True)`. It is also the default when the argument is omitted. The two settings that determine work partitioning are `cpu_max_num_threads` and `cpu_min_block_size`.

#### cpu_max_num_threads

`cpu_max_num_threads` sets the number of CPU worker threads in the thread pool. Its default is the system-reported number of logical cores. In this mode, it also sets the number of tasks created for each parallel `range()` loop.

The worker count is not capped to the number of logical cores. If it exceeds that count, the operating system shares CPU time among the worker threads.

#### cpu_min_block_size

`cpu_min_block_size` sets the minimum number of original iterations per block. Its default is 512, and its value must be at least 1.

For a nonempty loop, the block size is:

```text
block_size = max(ceil(iteration_count / cpu_max_num_threads), cpu_min_block_size)
```

Here `iteration_count` is the number of original loop iterations, and `ceil` means rounding up to an integer. The final nonempty block can be shorter than `block_size`. Blocks beyond the original loop range are empty.

For example, configure four workers:

```python
qd.init(
    arch=qd.cpu,
    make_cpu_multithreading_loop=True,
    cpu_max_num_threads=4,
    cpu_min_block_size=512,
)
```

For a loop with 1,000 iterations, dividing by four gives 250. The minimum of 512 raises the block size to 512. The four tasks therefore contain 512, 488, 0, and 0 iterations.

Changing only `cpu_min_block_size` to 1 makes the block size 250. All four tasks then contain 250 iterations. It does not create 1,000 single-iteration tasks: the number of tasks remains four.

Reducing the minimum can expose more parallel work when a short loop otherwise leaves blocks empty. It does not create more tasks than the configured worker count. Once all blocks contain useful work, lowering the minimum further may leave the partition unchanged.

The setting `default_cpu_block_dim`, which controls task size in the alternative mode, does not determine the original-iteration block size in this mode. Use `cpu_min_block_size` for that purpose.

### Parameterizing make_cpu_multithreading_loop=False

Select this mode with `qd.init(arch=qd.cpu, make_cpu_multithreading_loop=False)`. Worker count and task size are independent. The two settings are `cpu_max_num_threads` and `default_cpu_block_dim`.

#### cpu_max_num_threads

`cpu_max_num_threads` sets the number of CPU worker threads in the thread pool. Its default is the system-reported number of logical cores. It does not set the number of tasks in this mode.

For example, four workers can process 32 tasks. As a worker finishes one task, it takes another until all tasks are complete. Increasing the worker count does not change those tasks' iteration ranges.

#### default_cpu_block_dim

`default_cpu_block_dim` sets the number of original iterations per task when the loop does not specify its own block size. Its default is 32. The final task can contain fewer iterations.

For a nonempty loop without a loop-specific block size:

```text
task_count = ceil(iteration_count / default_cpu_block_dim)
```

For example:

```python
qd.init(
    arch=qd.cpu,
    make_cpu_multithreading_loop=False,
    cpu_max_num_threads=4,
    default_cpu_block_dim=32,
)
```

For a loop with 1,000 iterations, this creates 32 tasks: 31 tasks of 32 iterations and one task of eight. The four workers share those tasks.

Changing only `default_cpu_block_dim` to 250 creates four tasks of 250 iterations. Changing it to 16 creates 63 tasks: 62 tasks of 16 iterations and one task of eight. The worker count stays at four in both cases.

Smaller tasks let workers share uneven work more finely, but require more scheduling operations. Larger tasks reduce the number of scheduling operations, but a long-running task can leave other workers idle near the end of the loop.

To override the task size for one loop, place `qd.loop_config(block_dim=64)` immediately before that loop inside the kernel. In this mode, the loop then uses up to 64 original iterations per task instead of `default_cpu_block_dim`.

The setting `cpu_min_block_size` has no effect in this mode.

For either mode, compare repeated kernel calls after compilation has completed and check that the results agree. The separate setting `num_compile_threads` controls threads used to compile kernels; it does not set the number of workers that execute the loop.

### Inspecting block indices

`qd.block_idx()` works on CPU, CUDA, AMDGPU, Vulkan, and Metal. On CPU, it returns the index of the block executing the current iteration. Each block runs as one runtime task. Indices start at zero for each parallel loop execution. They identify blocks, not worker threads or execution order.

This function works with both values of `make_cpu_multithreading_loop`. With `True`, it identifies the compiler-generated block of original iterations. With `False`, it identifies the group whose size is controlled by `block_dim`.

```python
@qd.kernel
def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    for i in range(200):
        out[i] = qd.block_idx()
```

With four workers, `make_cpu_multithreading_loop=True`, and `cpu_min_block_size=1`, this records 50 occurrences of each index from 0 through 3. With `make_cpu_multithreading_loop=False` and the default block size of 32, it records indices 0 through 6. The last block contains eight iterations.

Nested serial loops retain the enclosing block's index. On CPU, code outside a scheduled block, including a top-level explicitly serialized loop, returns `0`. The function is available inside kernels and their called functions. Empty blocks execute no original iterations, so the example does not record them.

On GPUs, `qd.block_idx()` returns the hardware thread-block index. Vulkan and Metal call these blocks workgroups. A hardware block contains multiple GPU threads and can process several groups of original iterations. Its index stays the same when it processes another group. Serial GPU code runs in block zero.

The same kernel can call `qd.block_idx()` on every supported backend, but block sizes and iteration assignments may differ. The returned index is not a globally unique identifier.

### Requesting serial execution

To run a loop's iterations in order on one thread, place `qd.loop_config(serialize=True)` immediately before it:

```python
@qd.kernel
def k_serial() -> qd.i32:
    result = 0
    qd.loop_config(serialize=True)
    for i in range(200):
        result = (result * 3 + i) % 10007
    return result
```

The directive `qd.loop_config(parallelize=1)` also makes the following loop execute on one thread. These requests apply with either value of `make_cpu_multithreading_loop`. Setting `make_cpu_multithreading_loop=False` by itself does not request serial execution.

See [CPU loop scheduling in qd.init options](init_options.md#cpu-loop-scheduling) for an initialization example and [All options](init_options.md#all-options) for the configuration reference.

## Multi-dimensional parallelization with qd.ndrange

Since only top-level for loops are parallelized, nested for loops will run sequentially on each thread. To parallelize over multiple dimensions, use `qd.ndrange()` to flatten them into a single top-level loop:

```python
@qd.kernel
def process_image(image: qd.Template) -> None:
    for row, col, channel in qd.ndrange(height, width, 3):
        image[row, col, channel] = row + col
```

This launches `height * width * 3` threads in parallel, rather than only parallelizing the outer loop.

### Syntax

Each argument to `qd.ndrange` is either:
- an integer `n`, meaning `range(0, n)`
- a tuple `(start, end)`, meaning `range(start, end)`

```python
@qd.kernel
def compute(a: qd.Template) -> None:
    for i, j in qd.ndrange((2, 10), 5):
        a[i, j] = i * 10 + j
    # i ranges over 2..9, j ranges over 0..4
```

### qd.grouped with qd.ndrange

`qd.grouped()` packs the loop indices into a single vector, which is useful for writing dimension-independent code:

```python
@qd.kernel
def fill(a: qd.Template) -> None:
    for I in qd.grouped(qd.ndrange(4, 8, 16)):
        a[I] = I[0] + I[1] + I[2]
```

`I` is a `qd.Vector` with one element per dimension.

### Controlling iteration order with `axes=`

By default, `qd.ndrange(d0, d1, ..., dN-1)` makes the **last argument the innermost (fastest-varying) axis** in the flat parallel loop: adjacent flat threads differ in the last index. The `axes=` keyword lets you choose a different iteration-nesting order. It's a tuple of `int` listing the **canonical axis index at each successive iteration-nesting level, outermost first**, and must be a permutation of `range(N)` where `N` is the number of arguments to `qd.ndrange`:

```python
@qd.kernel
def k():
    # axis 1 is outermost (slowest-varying), axis 0 is innermost (fastest-varying)
    for i, j in qd.ndrange(M, N, axes=(1, 0)):
        ...
```

The yielded loop variables (`i`, `j`, ...) are still bound to canonical axes 0, 1, ... — only the visit order changes. `axes=None` (the default) and the identity permutation `(0, 1, ..., N-1)` are equivalent and reproduce the default last-argument-innermost order. Mismatched length and non-permutation values are rejected up front with `qd.QuadrantsSyntaxError`; non-integer entries with `qd.QuadrantsTypeError`.

`axes=` is independent of what's in the loop body: it controls the iteration order regardless of whether the body touches a `qd.field`, a `qd.ndarray`, a `qd.tensor`, a `qd.Vector` / `qd.Matrix` variant, or no tensor at all.

`axes=` is supported by both the plain and `qd.grouped` forms:

```python
for i, j in qd.ndrange(M, N, axes=(1, 0)):
    ...
for I in qd.grouped(qd.ndrange(M, N, axes=(1, 0))):
    # I[0] is still the canonical axis-0 index, regardless of axes
    ...
```

## Does GPU kernel launch latency matter?

Kernel launch can be done in parallel while the previously launched kernel is still running. This means that if the previously launched kernel takes longer to run than the launch time for the new kernel, then the kernel launch latency will be perfectly hidden.

It's important to try to make sure that the work done by each kernel is sufficient to hide the kernel launch latency, otherwise the launch latency will be a bottleneck to maximum performance.

If kernel launch latency is a bottleneck, then you can look into:
- getting each kernel to do more work, to increase the relative runtime of the kernel relative to launch time
- reducing kernel launch time

Reducing the number and complexity kernel parameters reduces the kernel launch latency. In addition:
- field args incur less launch latency than ndarray args
- global fields incur no parameter-related launch latency

For the underlying execution model - what a launch actually involves, where the latency comes from, and when reducing it helps - see [Performance](performance.md).

## Global memory

In CUDA, there are 3 main types of memory:
- registers: fast, but limited in storage capacity
- shared memory: slower than registers, but faster than global. Less limited than registers, but still limited compared to global memory
- global memory: slowest of all. High latency. But massive, typically ~10s of gigabytes at the time of writing

There are some additional types, but these are variations on global memory:
- constant memory: global memory, but which can be stored easily in cache
- local memory: storage which is private to each specific thread, but, unintuitively, is stored off-chip, and is as slow as global memory

Quadrants gives access to shared memory, using `qd.simt.block.SharedArray()`, but typically Quadrants kernels use only global memory and register memory. You cannot directly request to use registers, but registers will be used to hold any local variables, within the limits of available registers. Fields, ndarrays, and other data, are stored in global memory. This holds some implications for synchronization.

## Thread synchronization

Typically, Quadrants kernels use `atomic_` operations for synchronization. This is relatively easy and intuitive, and it works perfectly with global memory. The main downside is that `atomic` operations are slow, because they involve both global memory and thread synchronization, both of which are intrinsically slow, and combining them is slower still.

When using shared memory, there are various barriers and fences that can be used, to ensure that writes from all threads so far have completed, and now threads are free to read from memory written by other threads. The block-level primitives (`qd.simt.block.sync`, `qd.simt.block.mem_fence`, the predicate-reducing barriers, and `SharedArray` itself) are documented in [block](block.md), which also discusses the important distinction between a thread-converging barrier and a memory-only fence.

However, these block-scope fences and barriers do not work for synchronizing writes across blocks. For cross-block coordination through global memory, use `qd.simt.grid.mem_fence()` (a device-scope fence; see [grid](grid.md)) — or, if you need full cross-block thread synchronization, finish the current kernel and launch a new kernel.

## Avoiding synchronization

If there is a way of partitioning data such that no thread ever needs to read data written by another thread, then there is no need for synchronization.

## Maximizing GPU core utilization

A 4090 GPU has ~16,000 cores. A 5090 GPU has ~20,000 cores. In Quadrants, the top level for loop is parallelized over gpu threads:

```python
@qd.kernel
def k1() -> None:
    for i_b in range(B):  # parallelized across B GPU threads
        # work done by each thread
```

In order to maximize the efficient usage of the GPU we want to ensure that as many cores are being used as possible. This means that ideally `B` should be at least the number of cores in the GPU.

## Is it better to put everything in one single for-loop, or to split into multiple top-level loops?

In order to ensure that the kernel runtime is longer than the kernel launch time, we want to do as much work as possible in each kernel launch. This implies fewer kernel launches, that each do more.

However, we might need to break into multiple launches in order to synchronize writes to global memory.

## Compromise

The recommendations above often are self-conflicting. For example, maximizing the number of cores being used might require using atomics for synchronization, which might make the kernels slower. Reducing kernel launches similarly might require using atomics, which would make the kernels run more slowly. So it will not in general be possible to satisfy all the above guidelines. But, it's useful to be aware of the design choices above, and strive to achieve them. Exact choices for best performance will often be an empirical question.
