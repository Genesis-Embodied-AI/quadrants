# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `graph_do_while` | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `qd.checkpoint` (skip + `yield_on=`) | GPU-side | GPU-side | GPU-side | GPU-side | GPU-side | host-side |

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

### Fields as arguments

When different fields are passed as template arguments, each unique combination of fields produces a separately compiled kernel with its own graph cache entry. There is no interference between them.


## GPU-side iteration with `graph_do_while`

For iterative algorithms (physics solvers, convergence loops), you often want to repeat the kernel body until a condition is met, without returning to the host each iteration. Use `while qd.graph_do_while(flag):` inside a `graph=True` kernel:

```python
@qd.kernel(graph=True)
def solve(x: qd.types.ndarray(qd.f32, ndim=1),
          counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(1):
            counter[()] = counter[()] - 1

x = qd.ndarray(qd.f32, shape=(N,))
counter = qd.ndarray(qd.i32, shape=())
counter.from_numpy(np.array(10, dtype=np.int32))
solve(x, counter)
# x is now incremented 10 times; counter is 0
```

The argument to `qd.graph_do_while()` must be the name of a scalar `qd.i32` ndarray parameter. The loop body repeats while this value is non-zero.

- On CUDA SM 9.0+ (Hopper), this uses CUDA conditional while nodes — the entire iteration runs on the GPU with no host involvement.
- On older CUDA GPUs, AMDGPU, and non-GPU backends, it falls back to a host-side do-while loop (see [Caveats](#caveats) and the [backend support table](#backend-support)).

### Patterns

**Counter-based**: set the counter to N, decrement each iteration. The body runs exactly N times.

```python
@qd.kernel(graph=True)
def iterate(x: qd.types.ndarray(qd.f32, ndim=1),
            counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(1):
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
        for i in range(1):
            if some_condition(x):
                keep_going[()] = 0
```

### Do-while semantics

`graph_do_while` has **do-while** semantics: the kernel body always executes at least once before the condition is checked. This matches the behavior of CUDA conditional while nodes. The flag value must be >= 1 at launch time. Passing 0 with a kernel that decrements the counter will cause an infinite loop.

### ndarray vs field

The parameter used by `graph_do_while` MUST be an ndarray.

However, other parameters can be any supported Quadrants kernel parameter type.

### Restrictions

- The same physical ndarray must be used for the counter parameter on every
  call. Passing a different ndarray raises an error, because the counter's
  device pointer is baked into the graph at creation time.

### Caveats

On platforms without native device-side conditional graph nodes — currently CUDA pre-SM 9.0 and **AMDGPU** (HIP has no conditional / while node API as of ROCm 7.2) — the value of the `graph_do_while` parameter will be copied from the GPU to the host each iteration, in order to check whether we should continue iterating. This causes a GPU pipeline stall. At the end of each loop iteration:
- wait for GPU async queue to finish processing
- copy condition value to hostside
- evaluate condition value on hostside
- launch new kernels for next loop iteration, if not finished yet

Note: the basic `graph=True` path (without `graph_do_while`) does **not** stall the host like this on either CUDA or AMDGPU — the entire kernel sequence runs as a single GPU-side graph replay.

Therefore on unsupported platforms, you might consider creating a second implementation, which works differently. e.g.:
- fixed number of loop iterations, so no dependency on gpu data for kernel launch; combined perhaps with:
- make each kernel 'short-circuit', exit quickly, if the task has already been completed; to avoid running the GPU more than necessary

## Checkpoints with `qd.checkpoint` *(experimental)*

> **Experimental.** `qd.checkpoint`, `qd.GraphStatus`, and `kernel.resume(from_checkpoint=...)` are experimental APIs. The shape of the public surface (the context-manager signature, the `GraphStatus` fields, the host-side resume loop, the error messages, and the cross-backend lowering details) may change in any future release without a deprecation cycle.

`qd.checkpoint()` marks a section of a graph kernel as a *skippable, optionally yieldable stage*. An example use-case is an algorithm implemented as a graph where you might need to allocate additional memory part-way through, the graph operations are in-place, and simply retrying the whole graph from the start is not an option. `qd.checkpoint` lets the kernel break at some point in the graph, surface the reason to the host, let the host fix things up, and resume from that point on the next launch.

```python
@qd.kernel(graph=True)
def step(
    arr: qd.types.ndarray(qd.f32, ndim=1),
    overflow_flag: qd.types.ndarray(qd.i32, ndim=0),
    newton_cond: qd.types.ndarray(qd.i32, ndim=0),
):
    while qd.graph_do_while(newton_cond):
        with qd.checkpoint():                       # cp_id 0
            for i in range(arr.shape[0]):
                # ...
                pass
        with qd.checkpoint(yield_on=overflow_flag): # cp_id 1 (can yield)
            for i in range(arr.shape[0]):
                # ...
                pass
        with qd.checkpoint():                       # cp_id 2
            for i in range(arr.shape[0]):
                # ...
                pass
```

Each `with qd.checkpoint(...)` block gets a `cp_id` assigned. You can use the `cp_id` to identify which checkpoint yielded and which checkpoint to resume from — see [Host-side yield / resume loop](#host-side-yield--resume-loop) below.

### Yield mechanism

If `yield_on=foo` is supplied, the body may write a non-zero value into `foo[()]` (for example, when a pre-allocated buffer is too small) to signal "the host needs to handle something before this checkpoint can complete". When that happens:

1. The framework records the checkpoint that yielded (first yielder in declaration order wins).
2. Every later checkpoint in the same launch is skipped.
3. `qd.checkpoint` will exit any surrounding `qd.graph_do_while`.
4. `foo[()]` is reset to `0`.

### Host-side yield / resume loop

Kernels with at least one `yield_on=` checkpoint return a `qd.GraphStatus` from every launch (and from `kernel.resume(...)`). The status carries two fields:

- `status.yielded` — `True` iff some `yield_on=` flag was non-zero during this launch.
- `status.checkpoint` — `cp_id` of the first (in declaration order) checkpoint that fired its flag, or `None` when `yielded` is `False`.

Resume by calling `kernel.resume(..., from_checkpoint=status.checkpoint)`. Every `qd.checkpoint` with `cp_id < from_checkpoint` is skipped on the resume launch; the rest run normally. The canonical host loop:

```python
status = step(arr, overflow_flag, newton_cond)
while status.yielded:
    handle_overflow_for(status.checkpoint, ...)
    status = step.resume(arr, overflow_flag, newton_cond,
                         from_checkpoint=status.checkpoint)
```

Kernels with `qd.checkpoint()` but no `yield_on=` keep their previous return contract (typically `None`) — the `GraphStatus` surface is opt-in via `yield_on=`.

### Restrictions

- Must be used inside `@qd.kernel(graph=True)`.
- `yield_on=` (when supplied) must be a kernel parameter that is a 0-d `qd.types.ndarray(qd.i32, ndim=0)`.
- Checkpoints cannot be nested inside other checkpoints.
- The body of a `with qd.checkpoint(...)` block cannot contain bare top-level statements (assignments, augmented assignments, or bare call/expression statements). Every top-level statement must be inside a `for`-loop (or other control-flow construct) so the compiler can lower it as its own offloaded task with the correct `cp_id`. A docstring as the first statement is allowed. Bare statements raise `QuadrantsSyntaxError` at compile time with a fix-it pointing at the explicit one-iteration `for`-wrap:

  ```python
  with qd.checkpoint():
      for _ in range(1):
          c[()] = c[()] + 1
      for i in range(arr.shape[0]):
          arr[i] = arr[i] + 1
  ```

  The restriction is by design: each top-level statement inside a checkpoint becomes its own GPU task / graph node, so silently auto-wrapping bare statements would hide a sequence of N field writes ballooning into N kernel launches. Forcing the user to write the `for`-wrap themselves keeps the lowering visible and gives a single obvious place to fuse multiple writes into one task by sharing a single wrapper.
