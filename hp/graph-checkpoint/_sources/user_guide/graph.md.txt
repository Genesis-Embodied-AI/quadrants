# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

Both features run on every backend. They are *hardware accelerated* on CUDA (via CUDA graphs) and AMDGPU (via HIP graphs); `graph_do_while` additionally requires CUDA SM 9.0+ / Hopper for its hardware-accelerated path. On other backends, `graph=True` is silently ignored and the kernel runs via the normal launch path, and `graph_do_while` falls back to a host-side do-while loop that copies the condition value GPU → host each iteration (causing a pipeline stall — see [Caveats](#caveats)).

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `graph_do_while` | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `qd.checkpoint` skip (IF gate) | hardware accelerated | runs unconditionally | host-orchestrated sub-graph | host gating | host gating | host-branch gating |
| `qd.checkpoint(yield_on=…)` + `kernel.resume()` | implemented | not yet | implemented (host-orchestrated) | implemented (host gating) | implemented (host gating) | implemented (host-branch gating) |

AMDGPU `graph_do_while` falls back to the host-side loop because HIP does not currently expose conditional / while graph nodes (as of ROCm 7.2). The same HIP-7.2 constraint also rules out indirect-dispatch graph nodes, so `qd.checkpoint` on AMDGPU is implemented by splitting the kernel into one HIP graph per contiguous run of same-`cp_id` tasks and having the host decide which sub-graphs to launch per call (see "Backend coverage notes" below).

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

> **Status:** Implemented end-to-end on every supported backend except pre-Hopper CUDA. The Python contract (`qd.checkpoint`, `GraphStatus`, `kernel.resume(from_checkpoint=…)`) is identical everywhere; only the backend lowering differs:
>
> - **CUDA SM 9.0+** (slices 1a–2): each checkpoint becomes a CUDA-graph IF conditional node; `yield_on=` checkpoints append a tiny device-side yield-check kernel and the WHILE-with-yield condition kernel reads `yield_signal` for early exit.
> - **CPU/x64** (slice 6): host-branch gating inside `KernelLauncher`; the host walks tasks in order and skips those whose `cp_id` sits below `resume_point` or follows a yield observed earlier in the launch.
> - **AMDGPU** (slice 4): host-orchestrated sub-graphs. The cached HIP graph is split into one sub-graph per contiguous run of same-`cp_id` tasks, and the launcher launches only the eligible sub-graphs per call (HIP 7.2 has no conditional-node / indirect-dispatch API). `graph_do_while + checkpoint` falls through the same streaming launcher the host loop uses, with the gating logic ported to the per-task loop.
> - **Vulkan / Metal** (slice 5): host gating inside `GfxRuntime::launch_kernel`. The runtime already records and submits a fresh command list per launch, so eligible vs. skipped tasks are decided at host-loop time before pipeline bind + dispatch are recorded; yield observation is a flush + wait_idle + `Device::readback_data` after the last task of each yielding checkpoint.
> - **CUDA < SM 9.0**: parser still accepts the construct but every body currently runs unconditionally; `GraphStatus.yielded` is always `False`. Treat `yield_on=` as advisory rather than authoritative here.

`qd.checkpoint()` marks a section of a graph kernel as a *skippable, optionally yieldable stage*. The intended use is the qipc-style re-entrant Newton loop, where each iteration is divided into a handful of named stages (assemble, broadphase, line search, ...) and the host may need to grow a pre-allocated buffer when a stage detects overflow.

```python
@qd.kernel(graph=True)
def step(
    arr: qd.types.ndarray(qd.f32, ndim=1),
    overflow_flag: qd.types.ndarray(qd.i32, ndim=0),
    newton_cond: qd.types.ndarray(qd.i32, ndim=0),
):
    while qd.graph_do_while(newton_cond):
        with qd.checkpoint():                       # cp_id 0: assemble
            for i in range(arr.shape[0]):
                # ...
                pass
        with qd.checkpoint(yield_on=overflow_flag): # cp_id 1: BVH (can yield)
            for i in range(arr.shape[0]):
                # ...
                pass
        with qd.checkpoint():                       # cp_id 2: line search
            for i in range(arr.shape[0]):
                # ...
                pass
```

Each `with qd.checkpoint(...)` block gets a `cp_id` assigned by declaration order (0, 1, 2, ... flat across the whole kernel, independent of whether the checkpoint is inside or outside a `qd.graph_do_while`).

### Yield mechanism (CUDA SM 9.0+, CPU/x64, AMDGPU, Vulkan, Metal)

If `yield_on=foo` is supplied, the body may write a non-zero value into the `foo` ndarray to signal "the host needs to handle something" (typically: pre-allocated buffer too small). At the end of the checkpoint body the framework injects a small yield-check step that:

1. Reads `*foo` and skips out early if it's `0`.
2. Atomically claims `yield_signal` with this checkpoint's `cp_id` (first yielder in declaration order wins).
3. Disables every later checkpoint in the same launch (subsequent gate steps see "I'm past the yield point" and short-circuit their IF bodies).
4. Resets `*foo` to `0` so the next launch starts clean without host intervention.

Combined with `qd.graph_do_while`, the framework also handles WHILE early-exit: as soon as a yield has been observed, the loop terminates. Without that the loop body would re-enter, see "post-yield" state in every gate, skip every checkpoint, never decrement the counter, and spin forever.

On CUDA the yield-check step and WHILE early-exit are tiny device kernels appended to each checkpoint's IF body and to the loop's condition kernel respectively. On every other backend the same logic is implemented in the host launcher (`KernelLauncher` for CPU + AMDGPU streaming, `GraphManager::launch_cached_checkpoint_graph` for AMDGPU sub-graph mode, `GfxRuntime::launch_kernel` for Vulkan / Metal): same per-launch lifecycle (`yield_signal` and `resume_point` reset on the first yield-capable launch), same first-yielder-wins semantics, same `*flag = 0` reset, same WHILE early-exit and per-iteration `resume_point` reset. The host-side paths read the `yield_on=` flag through a backend-appropriate D2H (`hipMemcpyDtoH` on AMDGPU, `Device::readback_data` on Vulkan / Metal, a plain pointer read on CPU); pinned-host yield flags are a future enhancement to remove the read-back stall.

### Host-side yield / resume loop

Kernels with at least one `yield_on=` checkpoint return a `qd.GraphStatus` from every launch (and from `kernel.resume(...)`). This is true on every backend that implements the gate (CUDA SM 9.0+ via IF nodes, AMDGPU via host-orchestrated sub-graphs or streaming gating, Vulkan / Metal via per-task host gating in `GfxRuntime`, CPU via host-branch gating). On backends listed in the support table as "not yet" (only pre-Hopper CUDA at present), the launch still returns a `GraphStatus` but `yielded` is always `False` because the bodies all run unconditionally, so the host loop simply terminates after one pass. The status carries two fields:

- `status.yielded` — `True` iff some `yield_on=` flag was non-zero during this launch.
- `status.checkpoint` — `cp_id` of the first (in declaration order) checkpoint that fired its flag, or `None` when `yielded` is `False`.

Resume by calling `kernel.resume(..., from_checkpoint=status.checkpoint)`. Every `qd.checkpoint` with `cp_id < from_checkpoint` is skipped on the resume launch; the rest run normally. The canonical host loop looks like:

```python
status = step(arr, overflow_flag, newton_cond)
while status.yielded:
    handle_overflow_for(status.checkpoint, ...)
    status = step.resume(arr, overflow_flag, newton_cond,
                         from_checkpoint=status.checkpoint)
```

Kernels with `qd.checkpoint()` but no `yield_on=` keep their previous return contract (typically `None`) — the `GraphStatus` surface is opt-in via `yield_on=`.

### Restrictions (enforced at kernel compile time)

- Must be used inside `@qd.kernel(graph=True)`.
- `yield_on=` (when supplied) must be the bare name of a kernel parameter that is a 0-d `qd.types.ndarray(qd.i32, ndim=0)`.
- Checkpoints cannot be nested inside other checkpoints. A checkpoint inside a `qd.graph_do_while` body is fine and is the expected pattern.
- Cannot be combined with `qd.stream_parallel()` in the same kernel.

### Authoring tip: every statement inside a checkpoint must live in a top-level `for` loop

The checkpoint gate (a CUDA-graph IF node on CUDA, host-orchestrated sub-graph selection on AMDGPU, host task-loop gating on Vulkan / Metal / CPU) routes work into a checkpoint's body based on whether the statement was lowered to a `range_for` task with the matching `checkpoint_id`. Bare scalar statements (e.g. `counter[()] -= 1`) fall into the offloader's pending-serial bucket, which currently loses the surrounding `checkpoint_id` and emits the work as a `serial` task with `checkpoint_id == -1`. Tasks with `cp_id == -1` run unconditionally, outside every checkpoint gate — meaning a yielding checkpoint won't actually skip them on subsequent checkpoints in the same launch.

The established workaround (used throughout the test suite) is to wrap such statements in a one-iteration `for` loop:

```python
with qd.checkpoint(yield_on=overflow_flag):
    for _ in range(1):
        counter[()] = counter[()] - 1
```

This forces the offloader to emit a `range_for` task that picks up the surrounding `checkpoint_id`. The runtime cost of an extra 1-iteration loop is negligible compared to the kernel-launch overhead the rest of the graph saves; the proper fix (propagating `checkpoint_id` through the serial bucket) is tracked separately.

### Backend coverage notes

- **CUDA SM 9.0+**: full path — every checkpoint becomes a CUDA-graph IF conditional node; yielding checkpoints additionally append a yield-check kernel and (inside `qd.graph_do_while`) a yield-aware condition kernel for early exit.
- **AMDGPU**: same Python contract; lowering is host-orchestrated sub-graphs. The cached HIP graph is split into one sub-graph per contiguous run of same-`cp_id` tasks (one per checkpoint, plus one for the cp_id=-1 unconditional batches). The launcher iterates batches in order and launches only the eligible ones (`cp_id >= resume_point` and no yield observed earlier in the launch). For yielding checkpoints, after launching the sub-graph the host stream-syncs, reads the `yield_on=` flag with `hipMemcpyDtoH`, and clears the flag back to 0 so the next launch starts fresh. `graph_do_while + checkpoint` falls through the streaming launcher (HIP 7.2 has no conditional graph nodes), which carries the same gating contract task-by-task.
- **Vulkan / Metal**: same Python contract; lowering is per-task host gating in `GfxRuntime::launch_kernel`. There is no pre-recorded compute graph on these backends, so eligible vs. skipped tasks are decided host-side at the task loop and skipped tasks are not bound / dispatched at all. Yield observation does `flush + wait_idle + Device::readback_data` after the last task of each yielding checkpoint, which costs a sync per yielding-checkpoint per launch; pinned-host yield flags would remove that stall and stay a future enhancement.
- **CPU (x64 + arm64)**: full host-branch gating — same Python contract, implemented in `KernelLauncher` rather than as device-side IF nodes. There is no graph object on CPU, so there is also no graph-build cost, and a yielding launch simply stops executing further checkpoint tasks. The launcher is arch-agnostic; the same code path covers Linux x86 (`qd.x64`) and Apple Silicon (`qd.arm64`). Useful for prototyping and for unit-testing yield/resume host loops without a GPU.
- **CUDA < SM 9.0**: parser accepts `qd.checkpoint`, but every body runs unconditionally and the kernel's `GraphStatus.yielded` is always `False`. Treat `yield_on=` flags as advisory rather than authoritative here.
