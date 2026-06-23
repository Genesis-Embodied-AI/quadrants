# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

`graph=True` and `graph_do_while` run on every backend. They are *hardware accelerated* on CUDA (via CUDA graphs) and AMDGPU (via HIP graphs); `graph_do_while` additionally requires CUDA SM 9.0+ / Hopper for its hardware-accelerated path. On other backends, `graph=True` is silently ignored and the kernel runs via the normal launch path, and `graph_do_while` falls back to a host-side do-while loop that copies the condition value GPU → host each iteration (causing a pipeline stall). `qd.checkpoint` gating runs entirely on the device on every GPU backend; only the CPU backend uses host-side gating.

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `qd.graph_do_while` | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `qd.checkpoint` | GPU-side | GPU-side | GPU-side | GPU-side | GPU-side | host-side |
| `qd.graph_parallel_context` / `qd.graph_parallel` (parallel sections) | concurrent | concurrent | runs serially | runs serially | runs serially | runs serially |

AMDGPU `graph_do_while` falls back to a host-side loop because HIP does not currently expose conditional / while graph nodes (as of ROCm 7.2).

Nested and sibling `graph_do_while` loops (and mixing `graph_do_while` with top-level `for`-loops) are **experimental** for now — see [Nested loops and mixing with for-loops](#nested-loops-and-mixing-with-for-loops).

## Basic usage

Add `graph=True` to a `@qd.kernel` decorator:

```python
@qd.kernel(graph=True)
def my_kernel(
    x: qd.types.ndarray(qd.f32, ndim=1),
    y: qd.types.ndarray(qd.f32, ndim=1),
):
    for i in range(x.shape[0]):
        x[i] = x[i] + 1.0
    for i in range(y.shape[0]):
        y[i] = y[i] + 2.0
```

The top level for-loops will be compiled into a single graph. The parallelism is the same as before, but the launch latency much reduced.

The kernel is used normally — no other API changes are needed:

```python
x = qd.ndarray(qd.f32, shape=(1024,))
y = qd.ndarray(qd.f32, shape=(1024,))

my_kernel(x, y)  # first call: builds and caches the graph
my_kernel(x, y)  # subsequent calls: replays the cached graph
```

This works the same way on CUDA and AMDGPU. The cache is keyed per (compiled-kernel-specialization, launch-id), so different template instantiations (different field bindings, etc.) get their own cached graph.

### Restrictions

- **No struct return values.** Kernels that return values (e.g. `-> qd.i32`) cannot use graphs. An error is raised if `graph=True` is set on such a kernel.
- **Primal kernels only.** The `graph=True` flag is applied to the primal (forward) kernel only, not its adjoint. Autodiff kernels use the normal launch path.
- **Device-resident ndarrays.** Graph mode bakes device pointers into the cached graph, so all ndarray arguments must be on the GPU. Passing a host-resident ndarray raises an error.
- **`qd_stream` is incompatible** with `graph=True`. Choose one or the other.

### Passing different arguments

You can pass different ndarrays to the same kernel on subsequent calls. The cached graph is replayed with the updated arguments — no graph rebuild occurs:

```python
x1 = qd.ndarray(qd.f32, shape=(1024,))
y1 = qd.ndarray(qd.f32, shape=(1024,))
my_kernel(x1, y1)  # builds graph

x2 = qd.ndarray(qd.f32, shape=(1024,))
y2 = qd.ndarray(qd.f32, shape=(1024,))
my_kernel(x2, y2)  # replays graph with new array pointers
```

## GPU-side iteration with `graph_do_while`

For iterative algorithms (physics solvers, convergence loops), you often want to repeat the kernel body until a condition is met, without returning to the host each iteration. Use `while qd.graph_do_while(flag):` inside a `graph=True` kernel:

```python
@qd.kernel(graph=True)
def solve(x: qd.types.ndarray(qd.f32, ndim=1),
          counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        counter[()] = counter[()] - 1   # bare statement: runs every iteration

x = qd.ndarray(qd.f32, shape=(N,))
counter = qd.ndarray(qd.i32, shape=())
counter.from_numpy(np.array(10, dtype=np.int32))
solve(x, counter)
# x is now incremented 10 times; counter is 0
```

The argument to `qd.graph_do_while()` must be the name of a scalar `qd.i32` ndarray parameter. The loop body repeats while this value is non-zero.

- On CUDA SM 9.0+ (Hopper), this uses CUDA conditional while nodes — the entire iteration runs on the GPU with no host involvement.
- On older CUDA GPUs, AMDGPU, and non-GPU backends, it falls back to a host-side do-while loop (see the [backend support table](#backend-support)).

### Patterns

**Counter-based**: set the counter to N, decrement each iteration. The body runs exactly N times.

```python
@qd.kernel(graph=True)
def iterate(x: qd.types.ndarray(qd.f32, ndim=1),
            counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        counter[()] = counter[()] - 1
```

**Boolean flag**: set a `keep_going` flag to 1, have the kernel set it to 0 when a convergence criterion is met.

```python
@qd.kernel(graph=True)
def converge(x: qd.types.ndarray(qd.f32, ndim=1),
             keep_going: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(keep_going):
        for i in range(x.shape[0]):
            # ... do work ...
            pass
        if some_condition(x):       # bare `if` at the loop level is fine
            keep_going[()] = 0
```

### Do-while semantics

`graph_do_while` has **do-while** semantics: the kernel body always executes at least once before the condition is checked. The flag value must be >= 1 at launch time. Passing 0 with a kernel that decrements the counter will cause an infinite loop.

### ndarray vs field

The parameter used by `graph_do_while` MUST be an ndarray.

However, other parameters can be any supported Quadrants kernel parameter type.

### Nested loops and mixing with for-loops

> **Experimental.** Nested / sibling `graph_do_while` loops, and mixing `graph_do_while` with top-level `for`-loops, are experimental for now. Single-loop `graph_do_while` is the stable path.

`graph_do_while` loops can be **nested** inside one another, placed **side by side** (siblings), and freely **mixed with plain top-level for-loops**. Each loop has its own scalar `qd.i32` counter ndarray.

```python
@qd.kernel(graph=True)
def nested(x: qd.types.ndarray(qd.i32, ndim=1),
           outer: qd.types.ndarray(qd.i32, ndim=0),
           inner: qd.types.ndarray(qd.i32, ndim=0)):
    # A plain for-loop at the top level runs exactly once (like an ordinary graph=True kernel).
    for i in range(x.shape[0]):
        x[i] = x[i] + 100

    while qd.graph_do_while(outer):
        # Reset the inner counter at the start of each outer iteration, else it only runs on the first outer pass.
        inner[()] = 5                  # bare statement: runs every outer iteration
        while qd.graph_do_while(inner):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            inner[()] = inner[()] - 1
        outer[()] = outer[()] - 1
```

A `graph_do_while`-loop may only appear at the kernel top level or directly inside another `graph_do_while` body.

Note that `qd.func`'s are inlined, so you can freely factorize these structures across `qd.func` boundaries.

### Restrictions

- The counter ndarray may be swapped between calls: the cached graph reads each counter through an indirection slot that is refreshed on every launch, so passing a different ndarray (or alternating between several) replays the cached graph without rebuilding it.

### Caveats

On platforms without native device-side conditional graph nodes — currently CUDA pre-SM 9.0 and **AMDGPU** (HIP has no conditional / while node API as of ROCm 7.2) — the value of the `graph_do_while` parameter will be copied from the GPU to the host each iteration, in order to check whether we should continue iterating. This causes a GPU pipeline stall. For nested loops this host round-trip happens once per iteration of each loop level, and each loop-body task is replayed individually, so deeply nested loops on these backends pay correspondingly more host overhead (they remain correct, just slower than the CUDA SM 9.0+ native path). At the end of each loop iteration:
- wait for GPU async queue to finish processing
- copy condition value to hostside
- evaluate condition value on hostside
- launch new kernels for next loop iteration, if not finished yet

Note: the basic `graph=True` path (without `graph_do_while`) does **not** stall the host like this on either CUDA or AMDGPU — the entire kernel sequence runs as a single GPU-side graph replay.

Therefore on unsupported platforms, you might consider creating a second implementation, which works differently. e.g.:
- fixed number of loop iterations, so no dependency on gpu data for kernel launch; combined perhaps with:
- make each kernel 'short-circuit', exit quickly, if the task has already been completed; to avoid running the GPU more than necessary

## Checkpoints with `qd.checkpoint` *(experimental)*

> **Experimental.** `qd.checkpoint`, `qd.GraphStatus`, and `kernel.resume(from_checkpoint=...)` are experimental APIs. The shape of the public surface (the context-manager signature, the `@qd.kernel(checkpoints=True)` flag, the `GraphStatus` fields, the host-side resume loop, the error messages, and the cross-backend lowering details) may change in any future release without a deprecation cycle.

`qd.checkpoint` lets a graph kernel break partway through, surface a reason to the host, let the host fix things up, and resume from the same location on the next launch. An example use-case is an algorithm implemented as a graph that may need to allocate additional memory partway through, where the operations in the graph are in-place, and therefore cannot be rerun without changing/corrupting the output, and therefore for which simply retrying the whole graph from the start is not an option.

To use checkpoints:

1. Decorate the kernel with `@qd.kernel(graph=True, checkpoints=True)`.
2. Place `with qd.checkpoint(cp_id, yield_on=flag):` around any section of the body where you want to be able to pause and resume.

```python
from enum import IntEnum

class Stage(IntEnum):
    SIM = 0

@qd.kernel(graph=True, checkpoints=True)
def step(
    arr: qd.types.ndarray(qd.f32, ndim=1),
    overflow_flag: qd.types.ndarray(qd.i32, ndim=0),
    newton_cond: qd.types.ndarray(qd.i32, ndim=0),
):
    while qd.graph_do_while(newton_cond):
        for i in range(arr.shape[0]):
            # ...
            pass
        with qd.checkpoint(Stage.SIM, yield_on=overflow_flag):
            for i in range(arr.shape[0]):
                # ...
                pass
        for i in range(arr.shape[0]):
            # ...
            pass
```

The `cp_id` argument is the label you'll use to identify the checkpoint from the host (in `GraphStatus.checkpoint` and `kernel.resume(from_checkpoint=...)`). It must be an int literal or an `IntEnum` value; the framework preserves the value as-is, so `qd.checkpoint(Stage.SIM, ...)` round-trips as `Stage.SIM` rather than the raw int. Labels must be unique within a kernel.

### Yield mechanism

When the body of a checkpoint writes a non-zero value into `yield_on[()]`:

1. Everything after the yielding checkpoint in the same launch is skipped.
2. `qd.checkpoint` will exit any surrounding `qd.graph_do_while`.

The framework never writes into your `yield_on` buffer — you own it end-to-end. That means:

- Before the **first** launch, initialise it to `0` (a freshly allocated `qd.ndarray` is not guaranteed to be zeroed).
- :warning: Before each **resume** launch, reset it to `0` (otherwise the body of the same checkpoint sees the stale non-zero value and yields again on the same condition, looping forever).

### Host-side yield / resume loop

Kernels annotated with `checkpoints=True` return a `qd.GraphStatus` from every launch (including from `kernel.resume(...)`). The status carries two fields:

- `status.yielded` — `True` iff a checkpoint's `yield_on=` flag was non-zero during this launch.
- `status.checkpoint` — the `cp_id` label of the yielding checkpoint (or `None` when `yielded` is `False`).

Resume by calling `kernel.resume(..., from_checkpoint=label)`. Everything before `label` in source order is skipped on the resume launch; everything from `label` onward runs normally. The canonical host loop:

```python
overflow_flag[()] = 0  # initialise before the first launch
status = step(arr, overflow_flag, newton_cond)
while status.yielded:
    handle_overflow_for(status.checkpoint, ...)
    overflow_flag[()] = 0  # clear before resume, otherwise the same checkpoint yields again
    status = step.resume(arr, overflow_flag, newton_cond,
                         from_checkpoint=status.checkpoint)
```

### Resume where?

- execution starts from and including the checkpoint block that yielded
- you must therefore ensure that re-running this checkpoint will not break the algorithm
- for example make sure that any check that lead to the yield do not modify any data before yielding

### Restrictions

- Must be used inside `@qd.kernel(graph=True, checkpoints=True)`. Without the flag, `qd.checkpoint(...)` raises `QuadrantsSyntaxError` at compile time with a fix-it pointing at `checkpoints=True`.
- `cp_id` must be an int literal or an `IntEnum` value, and must be unique across the kernel.
- `yield_on=` must be a kernel parameter that is a 0-d `qd.types.ndarray(qd.i32, ndim=0)`; expressions are not supported.
- Checkpoints cannot be nested inside other checkpoints. Checkpoints inside a `qd.graph_do_while` body are fine.
- The body of a `with qd.checkpoint(...)` block cannot contain bare top-level statements (assignments, augmented assignments, or bare call/expression statements). Every top-level statement must be inside a `for`-loop (or other control-flow construct). A docstring as the first statement is allowed. Bare statements raise `QuadrantsSyntaxError` at compile time with a fix-it pointing at the explicit one-iteration `for`-wrap:

  ```python
  with qd.checkpoint(0, yield_on=flag):
      for _ in range(1):
          c[()] = c[()] + 1
      for i in range(arr.shape[0]):
          arr[i] = arr[i] + 1
  ```

The restriction is by design: each top-level statement inside a checkpoint becomes its own GPU task / graph node, so silently wrapping bare statements would hide a sequence of N field writes ballooning into N kernel launches. Forcing the user to write the `for`-wrap themselves keeps the lowering visible and gives a single obvious place to fuse multiple writes into one task by sharing a single wrapper.

## Performance

The hardware support for each feature (graph and graph do while) was documented in the table above. But what does this mean in practice? Let's look through some representative examples.

### qd.kernel

Before migrating to graph, we have for example:
```
@qd.kernel
def k1(a: qd.type.NDArray, b: qd.type.NDArray, c: qd.type.NDArray):
    for i in range(a.shape[0]):
        fn_a(a, i)
    for i in range(b.shape[0]):
        fn_b(b, i)
    for i in range(c.shape[0]):
        fn_b(c, i)
```
We have three top-level for loops, which we call 'offloaded tasks'. Each offloaded task is compiled into a separate GPU kernel. When we call `k1` from python, the c++ host-side code launches three gpu kernels.

We can migrate it to graph by adding `graph=True`:
```
@qd.kernel(graph=True)
def k1(a: qd.type.NDArray, b: qd.type.NDArray, c: qd.type.NDArray):
    for i in range(a.shape[0]):
        fn_a(a, i)
    for i in range(b.shape[0]):
        fn_b(b, i)
    for i in range(c.shape[0]):
        fn_b(c, i)
```

Results:
- on hardware-accelerated platforms, we only launch a single graph from the host, rather than 3 kernels
- on other platforms, there is no change: we still launch 3 gpu kernels: no change: not better, not worse

### A while loop, conditional on a device-side scalar tensor

Before migrating to graph we have for example:
```
@qd.kernel
def k1(a: qd.type.NDArray, cond: qd.type.NDArray):
    fn_1(a, cond)
    fn_2(a, cond)
    fn_3(a, cond)

while True:
    k1(a, cond)
    if cond[()] == 0:
        break
```

So, we have:
- a python host-side loop
- the kernel contains device code, which will run on the gpu
- each iteration, we copy the value of cond, from the gpu to the host, and check the value
- this causes a gpu pipeline stall:
    - first we wait for the entire default stream gpu work to complete/drain
    - then we wait for the value of cond to copy from the gpu to the host
    - then we run the python code to check the value of cond
    - if we continue the loop, we now have to run through the python and c++ machinery to prepare the gpu kernel launch
    - then launch the gpu kernels inside k1
    - together, these steps can cause a noticeable delay, reducing throughput speed

After migrating to graph with graph do while we have:

```
@qd.kernel(graph=True)
def k1(a: qd.type.NDArray, cond: qd.type.NDArray):
    while qd.graph_do_while(condition=cond):
        fn_1(a, cond)
        fn_2(a, cond)
        fn_3(a, cond)

k1(a, cond)
```
Now:
- on supported hardware, the cond evaluation takes place on the gpu
    - and we avoid the gpu pipeline stall
- on unsupported hardware, we still incur the pipeline stall, as before
    - note that there will be some small acceleration, because the condition evaluation and kernel launch will take place entirely from c++, bypassing python
    - no worse, incrementally better

### A fixed-size for loop

Before migrating to graph we have for example:
```
@qd.kernel
def k1(a: qd.type.NDArray):
    fn_1(a)  # assume these each launch a single offloaded task (gpu kernel)
    fn_2(a)
    fn_3(a)

for _ in range(num_its):
    k1(a)
```

In this case, we have `num_its` launches of the three gpu kernels in k1
- there is nothing on the host side that waits for anything to finish on the gpu-side
- there is kernel launch latency associated with:
    - running k1 from host-side python
    - launching the gpu kernels for each of fn_1, fn_2, fn_3 from host-side c++

After migrating to graph we have something like:
```
@qd.kernel(graph=True)
def k1(a: qd.type.NDArray, count: qd.type.NDArray):
    while qd.graph_do_while(count):
        fn_1(a)
        fn_2(a)
        fn_3(a)
        count[()] = count[()] - 1   # bare statement: runs every iteration

k1(a, count)
```
- on supported hardware, the entire loop runs on the gpu
    - there is a single host-side launch, of the graph, when we run the `k1(a, count)` qd.kernel function
- we have an additional kernel, in order to decrement count
    - this is true on both supported and unsupported hardware
- on unsupported hardware, we now have a gpu pipeline stall that we didn't have before
    - depending on the contents of the kernels, this might increase the per-iteration time by anything between 1% and 30% or so

The recommendation is to use the graph do while here anyway, if you need it for any platform, in order to ensure the code is compact and maintainable.

If you do want fixed-size for loops to run optimally on unsupported hardware platforms, we could add a specializd `qd.graph_range_for` function. This would:
- on graph-do-while-supported hardware: handle adding the additional increment kernel
- on graph-do-while-unsupported hardware: handle running the loop entirely on the host-side, to avoid adding a gpu pipeline stall

In practice, for our own kernels, i.e. in genesis-world, they largely fall under the do while formulation, see the previous section. However, also have some that used to be do while, but have been migrated to an optimized fixed-size, see next section.

### A while loop, conditional on a device-side scalar tensor, that has been optimized into a fixed-size for loop

In code that has been extensively optimized, but doesn't yet use graph do while, we might have code like:
```
@qd.kernel
def k1(cond: qd.i32, a: qd.types.NDArray):
    for j in range(a.shape[0]):  # off-loaded task (gpu kernel)
        if cond != 0:  # only runs main kernel body if cond != 0
            ....
    for j in range(a.shape[0]):  # off-loaded task (gpu kernel)
        if cond != 0:
            ....
    for j in range(a.shape[0]):
        if cond != 0:
            ....

    for _ in range(1):
        check_cond(cond)  # check whether we should continue


for i in range(MAX_ITER):
    k1(cond, a)
```
In this case:
- we run a fixed number of iterations
- we update cond on the gpu
- we run all the gpu kernels MAX_ITER times
    - but we short-circuit their contents if cond == 0
- this still causes all the gpu kernels to launch MAX_ITER times
    - but they quickly exit, and don't take up too much time

This formulation avoids any gpu pipeline stalls caused by checking a gpu value on the hostside, before launching more kernels

When we migrate this to graph do while, we get:

```
@qd.kernel(graph=True)
def k1(a: qd.type.NDArray, cond: qd.type.NDArray):
    while qd.graph_do_while(condition=cond):
        for j in range(a.shape[0]):  # off-loaded task (gpu kernel)
            ....  # no need for cond check
        for j in range(a.shape[0]):  # off-loaded task (gpu kernel)
            ....
        for j in range(a.shape[0]):
            ....

        check_cond(cond)  # bare @qd.func call: runs every iteration (keeps its inner loops parallel)

k1(a, cond)
```
Similar to earlier.

This is now optimal on gpu-do-while supported hardware, and the performance on unsupported hardware is similar to in the earlier section.

HOWEVER, for unsupported hardware the baseline has changed, so the gpu pipeline stalls are now relative to a pipeline stall free baseline.

The effect in reality is situation dependent:
- when MAX_ITER is relatively high, and many kernels are being launched un-necessarily, the graph do while approach might still be faster, even significantly faster
    - this could happen when we are using the MAX_ITER approach for consistency across many scenarios
        - and some scenarios are a poor fit for the MAX_ITER appraoch
- when MAX_ITER is actually a fairly close fit for the number of iterations really required, it is possible that the graph do while approach will be slower on unsupported platforms

In this case, our recommendation is:
- use graph do while anyway, if you need it on any platform
    - this will ensure your code is compact and maintainable
- if you need optimum 100% performance on unsupported platforms, then consider PRing onto quadrants an optimized graph implementation for your target platform
    - for example it could somehow run MAX_ITER iterations anyway, similar to the earlier hand-rolled version, but via the graph abstraction, hence allowing the code to be compact, cross-platform, and also optimally fast

## Parallel sections with `qd.graph_parallel_context` *(experimental)*

A `with qd.graph_parallel_context():` region lets you declare independent stages so the graph runs them concurrently.

`qd.graph_parallel_context` is honored by the graph builder so it composes with `graph=True` and `graph_do_while`.

```python
@qd.kernel(graph=True)
def step(...):
    while qd.graph_do_while(ncond):
        assemble_shared(...)                 # serial: feeds both parallel sections

        with qd.graph_parallel_context():    # fork: parallel sections run concurrently
            with qd.graph_parallel():            # point-triangle contacts
                pt_assemble(...)
                pt_hessian(...)
            with qd.graph_parallel():            # edge-edge contacts (independent of pt)
                ee_assemble(...)
                ee_hessian(...)
        # join: everything below waits for BOTH parallel sections to finish
        merge_hessians(...)
        precondition(...)
```

### Semantics

- **Fork / join.** Every parallel section in the region forks from the work that precedes the region. All parallel sections must finish before any work *after* the region begins (the join). On CUDA the join is a single empty graph node depending on every parallel section's last kernel.
- **Parallel sections are independent — you guarantee it.** Calls *within* a parallel section keep their program order, but calls in *different* parallel sections have no ordering. The parallel sections must be data-race free with respect to one another: no parallel section may read what another writes, and no two parallel sections may write the same memory. Quadrants does not check this; getting it wrong gives nondeterministic results.

### Restrictions (enforced at kernel compile time)

- `qd.graph_parallel_context` may contain only `with qd.graph_parallel():` blocks, optionally wrapped in `if qd.static(...)` (so an optional parallel section can be compiled in or out — e.g. enabling edge-edge contacts only when a feature flag is set).
- `qd.graph_parallel()` may appear only directly inside a `qd.graph_parallel_context()`.
- `qd.graph_parallel_context` cannot be nested, and a parallel section body must be straight-line task work — no `qd.graph_do_while`, `qd.checkpoint`, or nested `qd.graph_parallel_context` inside a parallel section (a `qd.graph_parallel_context` may, however, sit inside a `qd.graph_do_while` body, as shown above).

### Backend behavior

| backend | scheduling |
| --- | --- |
| CUDA | parallel sections run **concurrently** |
| AMDGPU / CPU / Vulkan / Metal | parallel sections run **serially** |

Because parallel sections are independent by construction, running them serially produces identical results — only the scheduling differs.
