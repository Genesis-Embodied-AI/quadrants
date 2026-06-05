# Graph

Graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch.

## Backend support

Both features run on every backend. They are *hardware accelerated* on CUDA (via CUDA graphs) and AMDGPU (via HIP graphs); `graph_do_while` additionally requires CUDA SM 9.0+ / Hopper for its hardware-accelerated path. On other backends, `graph=True` is silently ignored and the kernel runs via the normal launch path, and `graph_do_while` falls back to a host-side do-while loop that copies the condition value GPU → host each iteration (causing a pipeline stall — see [Caveats](#caveats)).

| Feature | `qd.cuda` SM 9.0+ | `qd.cuda` < SM 9.0 | `qd.amdgpu` | `qd.metal` | `qd.vulkan` | `qd.cpu` |
| --- | --- | --- | --- | --- | --- | --- |
| `graph=True` | hardware accelerated | hardware accelerated | hardware accelerated | runs (no acceleration) | runs (no acceleration) | runs (no acceleration) |
| `graph_do_while` | hardware accelerated | host fallback | host fallback | host fallback | host fallback | host fallback |
| `qd.checkpoint` skip (IF gate) | hardware accelerated (conditional graph node) | GPU-side (codegen prologue + flat graph) | GPU-side (codegen prologue + flat HIP graph) | GPU-side (gate shader + indirect dispatch) | GPU-side (gate shader + indirect dispatch) | host-branch gating |
| `qd.checkpoint(yield_on=…)` + `kernel.resume()` | implemented | implemented | implemented | implemented | implemented | implemented |

`qd.checkpoint` gating happens entirely on the device for every GPU backend. The exact device mechanism differs by backend (see "Backend coverage notes" below) but the Python contract (`qd.checkpoint`, `GraphStatus`, `kernel.resume(from_checkpoint=…)`) is identical.

AMDGPU `graph_do_while` still falls back to the host-side loop because HIP does not currently expose conditional / while graph nodes (as of ROCm 7.2). `qd.checkpoint` gating is fully GPU-side on AMDGPU via the same codegen prologue + flat HIP graph approach used on pre-Hopper CUDA — body kernels self-early-return when `*resume_point > cp_id` or `*yield_signal != -1`, and a pre-built yield-check kernel (bundled HSACO covering gfx90a / gfx942 / gfx1030 / gfx1100 / 1101 / 1102 / 1200 / 1201) atomically updates `yield_signal` inline. The streaming path (graph_do_while + checkpoint) uses the same prologue and yield-check kernel, just driven by a host-side do-while loop instead of a single graph launch.

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

If `yield_on=foo` is supplied, the body may write a non-zero value into the `foo` ndarray to signal "the host needs to handle something" (typically: pre-allocated buffer too small). At the end of the checkpoint body the framework injects a small yield-check step that:

1. Reads `*foo` and skips out early if it's `0`.
2. Atomically claims `yield_signal` with this checkpoint's `cp_id` (first yielder in declaration order wins).
3. Disables every later checkpoint in the same launch (subsequent gate steps see "I'm past the yield point" and short-circuit their IF bodies).
4. Resets `*foo` to `0` so the next launch starts clean without host intervention.

Combined with `qd.graph_do_while`, the framework also handles WHILE early-exit: as soon as a yield has been observed, the loop terminates. Without that the loop body would re-enter, see "post-yield" state in every gate, skip every checkpoint, never decrement the counter, and spin forever.

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
- `yield_on=` (when supplied) must be the bare name of a kernel parameter that is a 0-d `qd.types.ndarray(qd.i32, ndim=0)`.
- Checkpoints cannot be nested inside other checkpoints. A checkpoint inside a `qd.graph_do_while` body is fine and is the expected pattern.
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
