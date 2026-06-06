# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

Both features run on every backend. They are *hardware accelerated* on CUDA (via CUDA graphs) and AMDGPU (via HIP graphs); `graph_do_while` additionally requires CUDA SM 9.0+ / Hopper for its hardware-accelerated path. On other backends, `graph=True` is silently ignored and the kernel runs via the normal launch path, and `graph_do_while` falls back to a host-side do-while loop that copies the condition value GPU → host each iteration (causing a pipeline stall — see [Caveats](#caveats)).

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `graph_do_while` (single) | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `graph_do_while` (nested / sibling) | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |

AMDGPU `graph_do_while` falls back to the host-side loop because HIP does not currently expose conditional / while graph nodes (as of ROCm 7.2).

Nested and sibling `graph_do_while` loops (see [Nested loops](#nested-loops-and-mixing-with-for-loops)) work on every backend: hardware-accelerated as a single GPU-side graph on CUDA SM 9.0+, and via a host-side driver everywhere else (CPU, CUDA pre-SM 9.0, AMDGPU, Vulkan, Metal). On the host-fallback backends each loop-body pass is replayed from the host and the condition value is copied GPU → host between iterations (see [Caveats](#caveats)).

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

> **Reset the flag outside the loop body, never inside it.** The counter / `keep_going` flag must reach a terminating value *as a result of the loop body*. If you (re)set the flag to a non-zero value **inside** the `while qd.graph_do_while(...)` body — e.g. `for _ in range(1): counter[()] = N` placed within the loop — it is re-applied on every iteration and the loop never terminates. Do the reset before the loop (a run-once top-level `for`, see [Loop-carried state](#loop-carried-state)) or on the host between launches (`counter.fill(N)`).

### ndarray vs field

The parameter used by `graph_do_while` MUST be an ndarray.

However, other parameters can be any supported Quadrants kernel parameter type.

### Nested loops and mixing with for-loops

`graph_do_while` loops can be **nested** inside one another, placed **side by side** (siblings),
and freely **mixed with plain top-level for-loops**. Each loop has its own scalar `qd.i32` counter
ndarray. On CUDA SM 9.0+ the whole nest runs entirely GPU-side as one graph of conditional while
nodes; on the host-fallback backends the loop nest is driven from the host.

```python
@qd.kernel(graph=True)
def nested(x: qd.types.ndarray(qd.i32, ndim=1),
           outer: qd.types.ndarray(qd.i32, ndim=0),
           inner: qd.types.ndarray(qd.i32, ndim=0)):
    # A plain for-loop at the top level runs exactly once (like an ordinary graph=True kernel).
    for i in range(x.shape[0]):
        x[i] = x[i] + 100

    while qd.graph_do_while(outer):
        # Re-initialise the inner counter at the start of every outer iteration, otherwise the
        # inner loop only runs on the first outer pass.
        for _ in range(1):
            inner[()] = 5
        while qd.graph_do_while(inner):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for _ in range(1):
                inner[()] = inner[()] - 1
        for _ in range(1):
            outer[()] = outer[()] - 1
```

Important points for nested loops:

- **Re-arm inner counters.** An inner loop's counter is not reset automatically between outer
  iterations. Reset it at the top of the enclosing loop body (as `inner[()] = 5` above), or the inner
  loop will only run during the first outer iteration.
- **Each level needs its own counter ndarray.** Don't share one counter ndarray across two levels.

### Kernel structure restriction

When a kernel uses `qd.graph_do_while()` anywhere, **every top-level statement** — both in the kernel
body and inside each `graph_do_while` body — must be either a `for`-loop or a `qd.graph_do_while()`
`while`-loop. Bare statements (assignments, `if`, etc.) at these levels are rejected with a
`QuadrantsSyntaxError`. Wrap any such statement in a trivial loop:

```python
# Instead of:   counter[()] = counter[()] - 1
for _ in range(1):
    counter[()] = counter[()] - 1
```

This restriction keeps each offloaded task tagged with the exact loop level it belongs to. `for`-loops
and `graph_do_while` loops may otherwise be freely ordered, mixed, and nested. A `graph_do_while`
`while`-loop may only appear at the kernel top level or directly inside another `graph_do_while` body —
it cannot be placed inside a `for`-loop.

### Loop-carried state

A common pattern is an iterative solver that initializes some working state once, refines it across
iterations, then reads the result back. Be deliberate about **where** each piece lives, because only
statements *inside* the `while qd.graph_do_while(...)` body repeat:

- **One-time init / writeback** belong at the kernel top level (a top-level `for`-loop runs exactly
  once — see [Nested loops and mixing with for-loops](#nested-loops-and-mixing-with-for-loops)), or in
  a separate non-`graph` kernel.
- **Per-iteration work** belongs inside the `while` body.
- **Loop-carried state** (a value that iteration `k+1` reads from iteration `k`) just works: it is held
  in global memory and nothing outside the loop body resets it between iterations.

```python
@qd.kernel(graph=True)
def newton(q_iter: qd.types.ndarray(qd.f32, ndim=1),
           q:      qd.types.ndarray(qd.f32, ndim=1),
           ncond:  qd.types.ndarray(qd.i32, ndim=0)):
    # One-time seed: top-level for-loop, runs exactly once.
    for i in range(q.shape[0]):
        q_iter[i] = q[i]
    # Iterative refinement: repeats while ncond != 0. q_iter carries between iterations.
    while qd.graph_do_while(ncond):
        # ... update q_iter from its previous-iteration value ...
        for _ in range(1):
            ncond[()] = ncond[()] - 1
    # One-time writeback: top-level for-loop, runs exactly once.
    for i in range(q.shape[0]):
        q[i] = q_iter[i]
```

If you would rather keep the `graph=True` kernel focused on just the loop, the equivalent **seed /
iterate / writeback** split works too — move the init and writeback into their own non-`graph`
`@qd.kernel` functions and call them around the iterate kernel on the host. Either structure is fine;
the only hard rule is that the do-while flag must not be reset inside the loop body (see
[Do-while semantics](#do-while-semantics)).

### Restrictions

- The counter ndarray may be swapped between calls: the cached graph reads each counter through an
  indirection slot that is refreshed on every launch, so passing a different ndarray (or alternating
  between several) replays the cached graph without rebuilding it.

### Caveats

On platforms without native device-side conditional graph nodes — currently CUDA pre-SM 9.0, **AMDGPU** (HIP has no conditional / while node API as of ROCm 7.2), Vulkan, Metal, and CPU — the value of the `graph_do_while` parameter will be copied from the GPU to the host each iteration, in order to check whether we should continue iterating. This causes a GPU pipeline stall. For nested loops this host round-trip happens once per iteration of each loop level, and each loop-body task is replayed individually, so deeply nested loops on these backends pay correspondingly more host overhead (they remain correct, just slower than the CUDA SM 9.0+ native path). At the end of each loop iteration:
- wait for GPU async queue to finish processing
- copy condition value to hostside
- evaluate condition value on hostside
- launch new kernels for next loop iteration, if not finished yet

Note: the basic `graph=True` path (without `graph_do_while`) does **not** stall the host like this on either CUDA or AMDGPU — the entire kernel sequence runs as a single GPU-side graph replay.

Therefore on unsupported platforms, you might consider creating a second implementation, which works differently. e.g.:
- fixed number of loop iterations, so no dependency on gpu data for kernel launch; combined perhaps with:
- make each kernel 'short-circuit', exit quickly, if the task has already been completed; to avoid running the GPU more than necessary
