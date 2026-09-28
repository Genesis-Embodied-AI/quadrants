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

Setting `make_cpu_multithreading_loop=False` in `qd.init(...)` changes how iterations are divided among tasks. The default mode creates one block per configured worker, with one task per block. With `False`, Quadrants groups the iterations into fixed-size groups, 32 by default. The number of tasks is independent of the number of workers.

With `True`, the number of tasks equals the configured worker count. For long loops, this groups many iterations into each task and keeps the number of scheduling operations small.

This mode also lets the compiler optimize consecutive iterations together. It can use vector instructions, which process several values in one CPU instruction, and avoid repeating calculations that stay the same across iterations. These opportunities are especially useful for cheap, regular work, such as adding two arrays. The benefit depends on the loop body.

The tradeoff is that each task is indivisible once it starts. If one task takes much longer than the others, its worker must finish that task's remaining iterations while the other workers may be idle.

With `False`, the task size stays fixed and the task count grows with the iteration count. When there are more tasks than workers, a worker that finishes one task can take another. This lets workers share an expensive region of the loop when that region spans several tasks. More tasks also require more scheduling operations.

Use the default mode when iterations have similar costs and the blocks provide enough parallel work, especially for cheap loops that benefit from vector instructions and low scheduling overhead. Try `False` when iteration costs vary substantially. Its smaller tasks can improve work distribution, but they add scheduling operations and do not provide the same compiled inner loops for optimization. Compare execution times for your loop before choosing.

Select the mode with `qd.init(arch=qd.cpu, make_cpu_multithreading_loop=True)` or `qd.init(arch=qd.cpu, make_cpu_multithreading_loop=False)`. Omitting the argument selects `True`.

For example, with four workers and 1,000 iterations:

| Mode | Tasks | Iterations per task |
| --- | --- | --- |
| `True` (default) | 4 | With the default minimum block size of 512: 512, 488, 0, and 0. |
| `False` | 32 | With the default task size of 32: 31 tasks of 32 iterations and one of 8. |

With `False`, the work is divided as follows. Here `submit_task` means assigning the indented work to an available worker. Each task retains its own bounds:

```text
for task_start in range(0, 1000, 32):
    task_end = min(task_start + 32, 1000)
    submit_task:
        for i in range(task_start, task_end):
            work(i)
```

Both modes run tasks in parallel, and each worker executes its current task's iterations sequentially. With four workers and 32 tasks, workers take another task whenever they finish. Having more tasks than workers lets several workers share an expensive region of the loop if that region spans multiple tasks. Once a worker starts a task, other workers cannot take over part of that task.

### Default block sizing and optimization

To choose the block size, Quadrants divides the iteration count by the worker count and rounds up to an integer. It then takes the larger of that value and 512. The final nonempty block can be shorter. Blocks whose start would lie beyond the original range are empty.

For example, with 12 workers and 200 iterations, the default block size is 512. All 200 iterations fit in the first block. The other eleven blocks are empty, so only one worker performs useful work. With 12 workers and 12,000 iterations, the default block size is 1,000. All twelve blocks contain useful work.

The inner loop and its work are compiled together. This lets the compiler move calculations that do not change between iterations outside the loop. It also gives the compiler an opportunity to use vector instructions, which process several values in one CPU instruction.

### CPU scheduling parameters

Pass these parameters to `qd.init(...)` to change how CPU loops are compiled and scheduled.

#### cpu_max_num_threads

`cpu_max_num_threads` sets the number of CPU worker threads in the thread pool. Its default is the system-reported number of logical cores.

With default scheduling, this value also determines the number of generated blocks. It is not capped to the number of logical cores. If it exceeds that count, the operating system shares CPU time among the worker threads.

#### cpu_min_block_size

`cpu_min_block_size` sets the minimum number of original iterations per generated block. Its default is 512, and its value must be at least 1. The final nonempty block can be shorter, and some blocks can be empty.

For a nonempty range, the default scheduling mode chooses:

```text
block_size = max(ceil(iteration_count / cpu_max_num_threads), cpu_min_block_size)
```

Here `ceil` means rounding up to the next integer. For example:

```python
qd.init(arch=qd.cpu, cpu_max_num_threads=12, cpu_min_block_size=1)
```

A loop with 200 iterations then has eleven blocks of 17 iterations and one block of 13. A minimum of 1 allows smaller blocks; it does not require one iteration per block.

#### make_cpu_multithreading_loop

`make_cpu_multithreading_loop` selects the CPU scheduling mode. Its default is `True`.

With `True`, Quadrants uses compiled inner loops. The compiler runs the `make_cpu_multithreaded_range_for` transform. A transform is a compiler step that rewrites the loop before generating machine code. This transform creates the outer block loop and compiled inner loop described above.

With `False`, the `make_cpu_multithreaded_range_for` transform is not applied. The task count is the iteration count divided by the task size, rounded up, independently of the worker count. In this mode, `cpu_min_block_size` has no effect. Parallel execution remains enabled.

#### default_cpu_block_dim

`default_cpu_block_dim` supplies the number of original iterations per task when `make_cpu_multithreading_loop=False` and the loop does not specify a block size. Its default is 32.

With `make_cpu_multithreading_loop=True`, the compiler overrides the transformed outer loop's `block_dim`, the number of outer iterations grouped into each scheduled task. It sets this value to 1. Each outer iteration already processes an entire generated block, so this keeps each generated block independently schedulable. It does not mean one original iteration per task.

#### num_compile_threads

`num_compile_threads` controls the threads used to compile a kernel's internal tasks. Its default is 4. It does not set the number of workers that execute the compiled CPU loop.

### Comparing the CPU scheduling modes

| Setting | True: with the `make_cpu_multithreaded_range_for` transform | False: without the `make_cpu_multithreaded_range_for` transform |
| --- | --- | --- |
| `cpu_max_num_threads` | Sets the thread-pool size and determines the number of generated blocks. | Sets the thread-pool size; task count depends on loop length and block size. |
| `cpu_min_block_size` | Sets the minimum number of original iterations in a generated block. | Unused. |
| `default_cpu_block_dim` | Overridden for the transformed outer loop: its `block_dim` is set to 1. | Supplies the number of original iterations per task when no loop-specific block size is set. |

### Choosing a CPU scheduling configuration

For short loops with expensive iterations, reducing `cpu_min_block_size` can expose more parallel work. Larger blocks give the compiler more iterations to process together and require fewer nonempty tasks. The best size depends on the work inside the loop.

Uneven iteration costs also matter. If four workers receive four blocks and one block takes much longer than the others, the remaining workers cannot take over part of that running block. With more, smaller tasks, the workers can share the expensive region if it spans several tasks. Smaller tasks also require more scheduling operations.

The default transformed path currently creates one block per configured thread. Lowering `cpu_min_block_size` can reduce the number of empty blocks, but it does not create more blocks than `cpu_max_num_threads`. The untransformed path can create more tasks than workers. These policies trade task granularity against scheduling work and compiler optimization opportunities. Task granularity means how much work is grouped into each task.

When comparing configurations, time repeated kernel calls after compilation has completed. Check that the results agree. Separate compilation time from execution time so that the comparison measures the scheduling choice.

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
