# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

Both features run on every backend. They are *hardware accelerated* on CUDA (via CUDA graphs) and AMDGPU (via HIP graphs); `graph_do_while` additionally requires CUDA SM 9.0+ / Hopper for its hardware-accelerated path. On other backends, `graph=True` is silently ignored and the kernel runs via the normal launch path, and `graph_do_while` falls back to a host-side do-while loop that copies the condition value GPU → host each iteration (causing a pipeline stall — see [Caveats](#caveats)).

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `graph_do_while` (single) | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `graph_do_while` (nested / sibling) | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `qd.checkpoint` skip (IF gate) | hardware accelerated (conditional graph node) | GPU-side (codegen prologue + flat graph) | GPU-side (codegen prologue + flat HIP graph) | GPU-side (gate shader + indirect dispatch) | GPU-side (gate shader + indirect dispatch) | host-branch gating |
| `qd.checkpoint(yield_on=…)` + `kernel.resume()` | implemented | implemented | implemented | implemented | implemented | implemented |
| `qd.graph_parallel` / `qd.branch` (concurrent branches) | concurrent (parallel streams) | concurrent (parallel streams) | runs serially (correct) | runs serially (correct) | runs serially (correct) | runs serially (correct) |

`qd.checkpoint` gating happens entirely on the device for every GPU backend. The exact device mechanism differs by backend (see "Backend coverage notes" below) but the Python contract (`qd.checkpoint`, `GraphStatus`, `kernel.resume(from_checkpoint=…)`) is identical.

AMDGPU `graph_do_while` still falls back to the host-side loop because HIP does not currently expose conditional / while graph nodes (as of ROCm 7.2). `qd.checkpoint` gating is fully GPU-side on AMDGPU via the same codegen prologue + flat HIP graph approach used on pre-Hopper CUDA — body kernels self-early-return when `*resume_point > cp_id` or `*yield_signal != -1`, and a pre-built yield-check kernel (bundled HSACO covering gfx90a / gfx942 / gfx1030 / gfx1100 / 1101 / 1102 / 1200 / 1201) atomically updates `yield_signal` inline. The streaming path (graph_do_while + checkpoint) uses the same prologue and yield-check kernel, just driven by a host-side do-while loop instead of a single graph launch.

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

The argument to `qd.graph_do_while()` must reference a scalar `qd.i32` ndarray: either a bare kernel parameter (e.g. `counter`) or a `@qd.data_oriented` member ndarray accessed through `self` (e.g. `self.counter`). The loop body repeats while this value is non-zero.

```python
@qd.data_oriented
class Solver:
    def __init__(self):
        self.x = qd.ndarray(qd.f32, shape=(N,))
        self.counter = qd.ndarray(qd.i32, shape=())

    @qd.kernel(graph=True)
    def solve(self):
        while qd.graph_do_while(self.counter):   # member ndarray as the loop condition
            for i in range(N):
                self.x[i] = self.x[i] + 1.0
            for i in range(1):
                self.counter[()] = self.counter[()] - 1
```

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

The condition used by `graph_do_while` MUST be an ndarray — either a bare kernel parameter or a `@qd.data_oriented` member ndarray (`self.flag`).

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
- **Checkpoints compose with any loop level.** `qd.checkpoint` blocks can sit inside any
  `graph_do_while` level — top-level, nested, or a sibling — as flat siblings, provided that level is
  not itself inside a checkpoint (a checkpoint may not transitively contain another checkpoint, even
  across a loop boundary — see [Checkpoints](#checkpoints-with-qdcheckpoint-experimental)). `cp_id`s
  are assigned as one flat sequence across the whole kernel regardless of nesting, and yield / resume
  is global: a yield in any level returns control to the host and resume skips every `cp_id` below the
  resume point.

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

## Checkpoints with `qd.checkpoint` *(experimental)*

> **Status:** Implemented end-to-end on every supported backend. The Python contract (`qd.checkpoint`, `GraphStatus`, `kernel.resume(from_checkpoint=…)`) is identical everywhere; only the device-side lowering differs:
>
> - **CUDA SM 9.0+**: each checkpoint becomes a CUDA-graph IF conditional node. The gate kernel (`_qd_checkpoint_if_gate`) reads `*resume_point` and writes the IF handle; `yield_on=` checkpoints append a tiny device-side yield-check kernel inside the IF body that atomic-CASes `yield_signal`. The WHILE-with-yield condition kernel reads `yield_signal` for early exit.
> - **CUDA SM < 9.0** (pre-Hopper: Volta / Turing / Ampere / Ada): pre-Hopper CUDA has no conditional graph nodes (CUDA 12.4+ feature, hardware-gated to Hopper+). Gating is done entirely on the GPU via a codegen prologue: every body kernel inside a `qd.checkpoint` carries a few-LOC LLVM-IR early-return at its entry that reads `RuntimeContext::checkpoint_resume_point_ptr` and `RuntimeContext::checkpoint_yield_signal_ptr` and self-skips when its `cp_id` should not run. The graph is flat (no conditional nodes); the yield-check kernel sits inline after each yielding checkpoint and self-gates with the same predicate. The same pre-built `_qd_checkpoint_yield_check` fatbin covers sm_75 / sm_80 / sm_86 / sm_89 / sm_90+.
> - **Vulkan / Metal**: GPU-side gating via indirect dispatch. Each body kernel is issued as an indirect dispatch reading its `dim3` from a small per-kernel device buffer; a SPIR-V gate shader (`checkpoint_gate_shader`) runs at the head of each checkpoint and writes either the real `dim3` or `(0, 0, 0)` based on `*resume_point` / `*yield_signal`. A SPIR-V yield-check shader (`checkpoint_yield_check_shader`) does the atomic-CAS on `yield_signal`. Vulkan uses `vkCmdDispatchIndirect`; Metal uses `dispatchThreadgroupsWithIndirectBuffer:`.
> - **CPU/x64**: host-branch gating inside `KernelLauncher`. The host walks tasks in order and skips those whose `cp_id` sits below `resume_point` or follows a yield observed earlier in the launch. CPU is the only backend that does host-side per-checkpoint decisions, which is fine here because there is no D2H copy involved.
> - **AMDGPU**: GPU-side via the same codegen-prologue + flat HIP graph approach used on pre-Hopper CUDA. HIP 7.2 has no conditional-node API and no indirect-dispatch primitive, so every cp_id >= 0 body kernel carries an LLVM-IR early-return prologue that reads `RuntimeContext::checkpoint_resume_point_ptr` / `checkpoint_yield_signal_ptr`, and a pre-built yield-check kernel (bundled HSACO via `scripts/build_checkpoint_yield_check_hsaco.py`) is inlined after each yielding checkpoint's body to atomically update `yield_signal`. The streaming path (graph_do_while + checkpoint) uses the same prologue and yield-check kernel.

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

### Yield mechanism (CUDA, Vulkan, Metal, CPU/x64)

If `yield_on=foo` is supplied, the body may write a non-zero value into the `foo` ndarray to signal "the host needs to handle something" (typically: pre-allocated buffer too small). `yield_on=` accepts either a bare scalar `qd.i32` ndarray kernel parameter (`yield_on=overflow_flag`) or a `@qd.data_oriented` member ndarray accessed through `self` (`yield_on=self.overflow_flag`). At the end of the checkpoint body the framework injects a small yield-check step that:

1. Reads `*foo` and skips out early if it's `0`.
2. Atomically claims `yield_signal` with this checkpoint's `cp_id` (first yielder in declaration order wins).
3. Disables every later checkpoint in the same launch (subsequent gate steps see "I'm past the yield point" and short-circuit their IF bodies).
4. Resets `*foo` to `0` so the next launch starts clean without host intervention.

Combined with `qd.graph_do_while`, the framework also handles WHILE early-exit: as soon as a yield has been observed, the loop terminates. For **nested** `graph_do_while` loops the yield propagates outward — every enclosing loop level also exits — so control returns to the host from the outermost loop regardless of which level the yielding checkpoint sat in. Without that the loop body would re-enter, see "post-yield" state in every gate, skip every checkpoint, never decrement the counter, and spin forever.

All GPU backends implement the yield-check as a tiny device kernel:
- **CUDA SM 9.0+**: `_qd_checkpoint_yield_check` kernel appended inside each checkpoint's IF conditional body; the conditional gate skips it when the checkpoint is skipped.
- **CUDA SM < 9.0**: the same `_qd_checkpoint_yield_check` kernel inserted inline after the checkpoint's last body kernel in the flat graph; it self-gates on `*resume_point` / `*yield_signal` so a skipped checkpoint's yield-check is a no-op.
- **Vulkan / Metal**: SPIR-V `checkpoint_yield_check_shader` indirect-dispatched with the same per-checkpoint `dim3` buffer the body uses, so a `(0, 0, 0)` dispatch (skipped checkpoint) no-ops.
- **CPU**: a host statement at the end of each checkpoint body that reads the `yield_on=` flag and atomic-CASes `yield_signal`.

Per-launch lifecycle is identical across backends: `yield_signal` and `resume_point` reset on the first yield-capable launch, first-yielder-wins, `*flag = 0` reset, WHILE early-exit, per-iteration `resume_point` reset.  
The aggregated `yield_signal` is read back to the host once per launch (`cuMemcpyDtoH` on CUDA, `Device::readback_data` on Vulkan / Metal, a plain pointer read on CPU). Pinned-host yield flags are a future enhancement to remove the read-back stall.

### Host-side yield / resume loop

Kernels with at least one `yield_on=` checkpoint return a `qd.GraphStatus` from every launch (and from `kernel.resume(...)`). This is true on every backend that implements the gate (CUDA SM 9.0+ via IF conditional nodes, CUDA SM < 9.0 via codegen prologue + flat graph, AMDGPU via codegen prologue + flat HIP graph, Vulkan / Metal via gate shader + indirect dispatch, CPU via host-branch gating). The status carries two fields:

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
- `yield_on=` (when supplied) must reference a 0-d `qd.types.ndarray(qd.i32, ndim=0)` — either a bare kernel parameter (`yield_on=flag`) or a `@qd.data_oriented` member ndarray (`yield_on=self.flag`).
- Checkpoints cannot be nested inside other checkpoints — **not even through an intervening `qd.graph_do_while`**. A `checkpoint → graph_do_while → checkpoint` chain is still a nested checkpoint and is rejected (otherwise bare work in the outer checkpoint would silently re-execute on resume). A checkpoint inside a `qd.graph_do_while` body — at the top level, or in a **nested / sibling** loop that is not itself inside a checkpoint — is fine and is the expected pattern; the checkpoints at every level form one flat `cp_id` sequence.
- Cannot be combined with `qd.stream_parallel()` in the same kernel.

### Authoring tip: every statement inside a checkpoint must live in a top-level `for` loop

The checkpoint gate (a CUDA-graph IF node on CUDA SM 9.0+, a codegen-emitted LLVM-IR early-return on CUDA SM < 9.0, an indirect-dispatch gate shader on Vulkan / Metal, a host task-loop branch on CPU) routes work into a checkpoint's body based on whether the statement was lowered to a `range_for` task with the matching `checkpoint_id`. Bare scalar statements (e.g. `counter[()] -= 1`) fall into the offloader's pending-serial bucket, which currently loses the surrounding `checkpoint_id` and emits the work as a `serial` task with `checkpoint_id == -1`. Tasks with `cp_id == -1` run unconditionally, outside every checkpoint gate — meaning a yielding checkpoint won't actually skip them on subsequent checkpoints in the same launch.

The established workaround (used throughout the test suite) is to wrap such statements in a one-iteration `for` loop:

```python
with qd.checkpoint(yield_on=overflow_flag):
    for _ in range(1):
        counter[()] = counter[()] - 1
```

This forces the offloader to emit a `range_for` task that picks up the surrounding `checkpoint_id`. The runtime cost of an extra 1-iteration loop is negligible compared to the kernel-launch overhead the rest of the graph saves; the proper fix (propagating `checkpoint_id` through the serial bucket) is tracked separately.

### Backend coverage notes

- **CUDA SM 9.0+**: every checkpoint becomes a CUDA-graph IF conditional node. The gate kernel (`_qd_checkpoint_if_gate`) reads `*resume_point` and writes the IF handle; yielding checkpoints additionally append a `_qd_checkpoint_yield_check` kernel inside the IF body, and (inside `qd.graph_do_while`) a yield-aware condition kernel for early exit.
- **CUDA SM < 9.0** (pre-Hopper: V100 / Turing / Ampere / Ada): every body kernel still launches as a regular CUDA graph node, but each cp_id >= 0 body kernel carries a codegen-emitted LLVM-IR prologue that reads `RuntimeContext::checkpoint_resume_point_ptr` and `RuntimeContext::checkpoint_yield_signal_ptr` (both populated by `CachedGraph::persistent_ctx` at graph-build time) and early-returns when the checkpoint should be skipped. The `_qd_checkpoint_yield_check` kernel is inserted inline after each yielding checkpoint's last body in the flat graph and self-gates with the same predicate. There are no conditional graph nodes in this path; the gating is purely the per-thread early-return inside the body kernel. The pre-built yield-check fatbin targets sm_75 / sm_80 / sm_86 / sm_89 in addition to sm_90+. To exercise this path on Hopper+ hardware for testing, set `QD_CUDA_FORCE_FLAT_CHECKPOINT_GRAPH=1`.
- **Vulkan / Metal**: same Python contract; lowering is GPU-side via indirect dispatch. Each body kernel is issued as an indirect dispatch reading its `dim3` from a small per-kernel device buffer; a SPIR-V `checkpoint_gate_shader` runs at the head of each checkpoint and writes either the real `dim3` or `(0, 0, 0)` based on `*resume_point` / `*yield_signal`. A SPIR-V `checkpoint_yield_check_shader` does the atomic-CAS on `yield_signal` and is itself indirect-dispatched (so it no-ops on skipped checkpoints). Vulkan uses `vkCmdDispatchIndirect`; Metal uses `dispatchThreadgroupsWithIndirectBuffer:`. One `Device::readback_data` of `yield_signal` happens at the end of the command-buffer submission to surface yields to the host.
- **CPU (x64 + arm64)**: full host-branch gating — same Python contract, implemented in `KernelLauncher` rather than as device-side IF nodes. There is no graph object on CPU, so there is also no graph-build cost, and a yielding launch simply stops executing further checkpoint tasks. The launcher is arch-agnostic; the same code path covers Linux x86 (`qd.x64`) and Apple Silicon (`qd.arm64`). Useful for prototyping and for unit-testing yield/resume host loops without a GPU.
- **AMDGPU**: GPU-side gating, mirror of the pre-Hopper CUDA path. Every cp_id >= 0 body kernel carries a codegen-emitted LLVM-IR prologue that reads `RuntimeContext::checkpoint_resume_point_ptr` / `checkpoint_yield_signal_ptr` and early-returns when its checkpoint should be skipped. A pre-built `_qd_checkpoint_yield_check` kernel (bundled HSACO produced by `scripts/build_checkpoint_yield_check_hsaco.py`, covering gfx90a / gfx942 / gfx1030 / gfx1100 / 1101 / 1102 / 1200 / 1201) self-gates with the same predicate and atomic-CASes `yield_signal` when the user's `yield_on=` flag reads non-zero. The HIP graph fast path inlines that kernel into a single flat graph; the streaming path (graph_do_while + checkpoint) launches the same kernel directly via `hipModuleLaunchKernel` after each yielding checkpoint's last body kernel. One per-launch HtoD of `resume_point` + init of `yield_signal=-1`, one post-launch DtoH of `yield_signal` per launch (or per iter for graph_do_while). No host-side per-checkpoint decisions.

## Concurrent branches with `qd.graph_parallel` *(experimental)*

`qd.checkpoint` and `graph_do_while` change *which* kernels run and *how many times*; `qd.graph_parallel` changes *how* a graph's kernels are scheduled relative to each other. By default the kernels captured in a `graph=True` kernel run as a single dependency chain (each waits for the previous one), even when they are completely independent. A `with qd.graph_parallel():` region lets you declare independent stages so the CUDA graph runs them on **parallel streams**.

This is the graph-compatible analogue of [`qd.stream_parallel()`](streams.md) (which only works for non-graph kernels): both express "these sequences are independent, run them concurrently", but `graph_parallel` is honoured by the CUDA graph builder so it composes with `graph=True` and `graph_do_while`.

```python
@qd.kernel(graph=True)
def step(...):
    while qd.graph_do_while(ncond):
        assemble_shared(...)                 # serial: feeds both branches

        with qd.graph_parallel():            # fork: branches run concurrently
            with qd.branch(name="pt"):       # point-triangle contacts
                pt_assemble(...)
                pt_hessian(...)
            with qd.branch(name="ee"):       # edge-edge contacts (independent of pt)
                ee_assemble(...)
                ee_hessian(...)
        # join: everything below waits for BOTH branches to finish
        merge_hessians(...)
        precondition(...)
```

### Semantics

- **Fork / join.** Every `qd.branch()` in the region forks from the work that precedes the region. All branches must finish before any work *after* the region begins (the join). On CUDA the join is a single empty graph node depending on every branch's last kernel.
- **Branches are independent — you guarantee it.** Calls *within* a branch keep their program order, but calls in *different* branches have no ordering. The branches must be data-race free with respect to one another: no branch may read what another writes, and no two branches may write the same memory. Quadrants does not check this; getting it wrong gives nondeterministic results, exactly like `qd.stream_parallel()`.
- **`name=` is optional** and used only as a label for profiling / graph introspection.

### Restrictions (enforced at kernel compile time)

- Must be used inside `@qd.kernel(graph=True)`.
- A region body may contain only `with qd.branch():` blocks, optionally wrapped in `if qd.static(...)` (so an optional branch can be compiled in or out — e.g. enabling edge-edge contacts only when a feature flag is set). A single-branch region is allowed and lowers to a plain chain (no fork/join overhead).
- `qd.branch()` may appear only directly inside a `qd.graph_parallel()` region.
- Regions cannot be nested, and a branch body must be straight-line task work — no `qd.graph_do_while`, `qd.checkpoint`, or nested `qd.graph_parallel` inside a branch (a region may, however, sit inside a `qd.graph_do_while` body, as shown above).

### Backend behaviour

| backend | result | scheduling |
| --- | --- | --- |
| CUDA (graph path) | correct | branches run **concurrently** on parallel streams |
| AMDGPU / CPU / Vulkan / Metal | correct | branches run **serially** (the concurrency tags are honoured only by the CUDA graph builder today) |

Because branches are independent by construction, running them serially on the other backends produces identical results — only the scheduling differs. `qd.graph_parallel` lowers onto the same internal concurrency-group mechanism as `qd.stream_parallel`, so non-graph fallbacks also fork the branches across streams.
