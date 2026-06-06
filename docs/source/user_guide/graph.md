# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

Both features run on every backend. They are *hardware accelerated* on CUDA (via CUDA graphs) and AMDGPU (via HIP graphs); `graph_do_while` additionally requires CUDA SM 9.0+ / Hopper for its hardware-accelerated path. On other backends, `graph=True` is silently ignored and the kernel runs via the normal launch path, and `graph_do_while` falls back to a host-side do-while loop that copies the condition value GPU → host each iteration (causing a pipeline stall — see [Caveats](#caveats)).

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `graph_do_while` | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |

AMDGPU `graph_do_while` falls back to the host-side loop because HIP does not currently expose conditional / while graph nodes (as of ROCm 7.2).

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

### The entire kernel body is the loop body

This is the single biggest gotcha in `graph_do_while`. Read it carefully — it will save you debugging time.

`while qd.graph_do_while(flag):` is **not** a Python control-flow scope. The Quadrants AST transformer flattens every top-level statement of the kernel into a single IR (offloaded tasks), and the runtime then wraps **that entire IR** in the conditional WHILE node. The runtime has no concept of "tasks that came before the `while`" or "tasks that came after the `while`" — there is just one flat task list, and *all of it* is the loop body.

Concretely:

```python
@qd.kernel(graph=True)
def looks_innocent(x: qd.types.ndarray(qd.f32, ndim=1),
                   c:  qd.types.ndarray(qd.i32, ndim=0)):
    for i in range(x.shape[0]):       # <-- INSIDE the loop! re-executes every iter
        x[i] = 0.0                    #     (resets x[i] to 0 before every body)
    while qd.graph_do_while(c):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(1):
            c[()] = c[()] - 1
    for i in range(x.shape[0]):       # <-- ALSO inside the loop! re-executes every iter
        x[i] = x[i] * 2.0
```

After this kernel returns, `x` is **not** "zero-init'd, incremented N times, then doubled once". Every iteration runs *all four* `for` blocks in source order. The pre-loop `for` zeros `x` back to 0 at the top of every iteration; the in-loop `for` then sets it to 1; the post-loop `for` doubles it to 2. After N iterations the loop terminates and you see `x == 2.0` everywhere, not `x == N * 2`.

This applies to **anything** that becomes an offloaded task: top-level `for` loops, direct array writes, `qd.checkpoint` blocks (on the [hp/graph-checkpoint branch](https://github.com/Genesis-Embodied-AI/quadrants/tree/hp/graph-checkpoint)), and so on. Variable assignments that don't lower to an offloaded task (compile-time constants, type hints) are unaffected.

#### The loop-carried-state idiom

Move the pre-loop init and post-loop writeback into **separate, non-graph** `@qd.kernel` functions, and let the `graph=True` kernel contain **only** the `while qd.graph_do_while(...):` block:

```python
@qd.kernel  # no graph=True -- runs once per frame
def seed(q_iter: qd.types.ndarray(qd.f32, ndim=1),
         q:      qd.types.ndarray(qd.f32, ndim=1)):
    for i in range(q.shape[0]):
        q_iter[i] = q[i]

@qd.kernel(graph=True)
def newton(q_iter: qd.types.ndarray(qd.f32, ndim=1),
           ncond:  qd.types.ndarray(qd.i32, ndim=0),
           # ...
          ):
    while qd.graph_do_while(ncond):
        # ... iterative work that updates q_iter; reads q_iter from
        # the previous iter (it carries because nothing outside this
        # `while` block resets it) ...
        pass

@qd.kernel  # no graph=True -- runs once per frame
def writeback(q:      qd.types.ndarray(qd.f32, ndim=1),
              q_iter: qd.types.ndarray(qd.f32, ndim=1)):
    for i in range(q.shape[0]):
        q[i] = q_iter[i]

# Per-frame, on the host:
ncond.fill(1)        # reset the do-while flag here, not in a pre-loop kernel block
seed(q_iter, q)      # one-shot init -- runs exactly once
newton(q_iter, ncond)
writeback(q, q_iter) # one-shot writeback -- runs exactly once
```

Why this works:
- `seed` and `writeback` are *separate kernel launches* (no `graph=True`), so they run exactly once per frame, not once per Newton iteration.
- `newton` is `graph=True` and contains only the `while qd.graph_do_while(...):` block, so the runtime wraps only the iterative work in the conditional WHILE.
- `q_iter` is only mutated *inside* the loop body, so its value at the start of iteration `k+1` is whatever the last task of iteration `k` left in global memory — it carries normally.
- The do-while flag (`ncond` here) is reset on the **host** between frames (`ncond.fill(1)`), not inside the kernel. If you reset it inside a pre-loop kernel block, that reset will re-execute every iteration and you'll get an infinite loop.

A frame-constant value that you compute from inputs at the top of every iteration (e.g. `q_tilde = bodies.q + g * dt**2`) *can* live in a pre-loop kernel block — it'll re-execute every iteration, but it reads stable inputs and produces the same value each time, so it's wasted work but not a correctness bug. Prefer hoisting it into a `seed`-style kernel anyway.

> **Heads up (future change):** we plan to make the AST transformer **reject** kernels that put any offloaded-task-producing statement outside the `while qd.graph_do_while(...):` block, with an error message pointing to the seed/writeback idiom. Adopt the idiom now and your kernels won't need to change.

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
