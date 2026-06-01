# Automatic differentiation

Automatic differentiation (autodiff) computes the exact gradient of a kernel's output with respect to its inputs, without the user writing the derivative formulas by hand. Gradient-based optimizers then use this gradient to train neural networks, fit physical models to data, drive differentiable simulators, or solve inverse problems.

**Note.** Throughout this page, the *primal* is the value a kernel computes in its normal forward pass (the field value, the loss, whatever the kernel writes); the *adjoint* (or *gradient*) is the derivative of the final scalar output (typically a loss) with respect to that primal value, stored in the `.grad` field next to the primal.

Quadrants implements autodiff at compile time: when `.grad()` is requested, the compiler emits a companion kernel that runs on the same backend as the forward one and writes gradients into the primal fields' `.grad` companions. There is no Python-side tape, no per-op dispatch overhead, and no dependency on an external AD framework. Forward mode and reverse mode are available on every backend Quadrants targets: x64 / arm64 CPU, CUDA, AMDGPU, Metal, and Vulkan.

**Recommendation.** Reverse-mode AD through dynamic loops (described further down) is currently gated behind an opt-in `ad_stack_experimental_enabled=True` flag at `qd.init`. If you are using autodiff at all, we recommend enabling this flag as it is required for any reverse-mode kernel with a dynamic loop carrying a non-linear primal, and free for every other kernel. See [the cost breakdown](./init_options.md#ad_stack_experimental_enabled) for details.

Three mechanisms are supported:

- **[Reverse mode](#reverse-mode-autodiff)** - one scalar output, many inputs. One backward pass yields every input gradient. This is the usual training setup and the bulk of the page.
- **[Forward mode](#forward-mode-ad-via-qdadfwdmode)** - few inputs, many outputs. One forward pass yields every output derivative along a chosen input direction.
- **[Custom gradients](#overriding-the-compiler-generated-gradient)** - override the auto-generated gradient with a user-supplied one, typically to inject a closed-form analytic derivative or to checkpoint for memory.

[Dynamic loops](#autodiff-with-dynamic-loops) and the [validation checker](#global-data-access-rules-and-the-validation-checker) are covered further down for when the default path is not enough.

## Reverse-mode autodiff

**Problem.** You have one scalar output (typically a loss) and many inputs, and you want `d(loss) / d(input_i)` for every `i` in a single pass. This is the shape of every gradient-based training loop: PyTorch's `.backward()` runs reverse-mode autodiff, as do most differentiable-simulation frameworks. If you want to put a learned parameter inside a Quadrants kernel and have an optimizer tune it, this is the section.

**How Quadrants does it.** The compiler walks the forward kernel's operations in reverse order and applies the chain rule, accumulating contributions into adjoint (`.grad`) fields allocated next to each primal. Everything runs on the same backend as the forward kernel: no Python-side tape, no per-op dispatch, no external AD framework. The backward pass does more work than the forward pass: it runs every forward op, then for each op accumulates a gradient contribution back into the inputs' adjoints, usually via atomic writes.

**Workflow.**

1. Allocate an adjoint (`.grad`) buffer next to every primal field whose gradient you need.
2. Run the forward kernel.
3. Seed the loss gradient - typically `loss.grad[None] = 1.0`.
4. Call `kernel.grad()`.
5. Read the gradients from the `.grad` fields.

Self-contained example:

```python
import quadrants as qd

qd.init(arch=qd.gpu)

x = qd.field(qd.f32)
y = qd.field(qd.f32)
# Step 1: allocate each primal together with its adjoint (.grad).
qd.root.dense(qd.i, 16).place(x, x.grad)
qd.root.place(y, y.grad)

@qd.kernel
def compute():
    for i in x:
        y[None] += x[i] * x[i]

# Step 2: run the forward kernel.
for i in range(16):
    x[i] = float(i)
y[None] = 0.0
compute()

# Step 3: seed the loss gradient.
y.grad[None] = 1.0
# (clear input adjoints before the reverse pass so they do not accumulate)
for i in range(16):
    x.grad[i] = 0.0
# Step 4: run the reverse kernel.
compute.grad()

# Step 5: read gradients back from the .grad fields.
# x.grad[i] == 2 * x[i]
```

Notes:

- `place(x, x.grad)` allocates the adjoint alongside the primal. Without it, `kernel.grad()` raises at first use. Ndarrays take `needs_grad=True` instead.
- Adjoints must be cleared before each reverse pass; leftover values accumulate. `qd.ad.Tape` (below) does this automatically.
- `kernel.grad(...)` takes the same arguments as the forward kernel.
- Reverse-mode AD through a *dynamic* loop (one whose trip count is not known at compile time) needs an opt-in compiler pipeline called the *adstack*, gated behind `ad_stack_experimental_enabled=True` in `qd.init()`. This path will be enabled by default in a future release, once thoroughly tested in production; see [Autodiff with dynamic loops](#autodiff-with-dynamic-loops) and [Under the hood (advanced)](#under-the-hood-advanced) for the current status.

**Integer casts stop gradients.** Integers have no meaningful derivative, so the chain rule reads as zero upstream of any cast to integer - no error, just silently-zero gradients. This is a property of differentiation through quantization, not of Quadrants. Rules of thumb:

- keep differentiable variables in `qd.f32` / `qd.f64` through the full forward chain;
- casting *to* a float is safe - the downstream float section remains differentiable, and the cast itself contributes a unit factor to the chain rule;
- casting *back to* an integer stops the gradient at that point, so only do it at integer-indexing sites, after any arithmetic whose gradient you need.

### Recording a backward pass with `qd.ad.Tape`

Training loops typically chain several kernels - physics step, feature extraction, loss. Differentiating such a pipeline by hand means calling each `.grad()` in the correct reverse order, seeding the loss, and clearing adjoints on every iteration.

`qd.ad.Tape` automates this. Kernel calls inside a `with qd.ad.Tape(loss=...)` block are recorded; on exit the tape replays them in reverse, seeds `loss.grad[None] = 1.0`, and writes the input gradients back into the `.grad` fields. Adjoints are cleared on entry, which is the desired behavior for almost every training iteration.

```python
with qd.ad.Tape(loss=y):
    compute()
# x.grad is now populated.
```

`Tape` is the default choice as soon as the forward pass spans more than a single kernel. Use `kernel.grad()` directly for one-shot kernels, or when the loss is not a single scalar and you want to seed multiple adjoint entries by hand.

### Forward-mode AD via `qd.ad.FwdMode`

**Problem.** Reverse mode is efficient when there is one scalar output (a loss) and many inputs. In the opposite shape - few inputs, many outputs - reverse mode still works but costs one full backward pass per output. Forward mode is the symmetric alternative: one forward pass per *input direction* gives you the derivative of *every* output along that direction. Concrete example: you have one kinematic parameter of a robot and want to know how every joint position changes when you nudge it. One input, many outputs: forward mode wins.

**How Quadrants does it.** Instead of walking the kernel in reverse, the compiler emits a *dual* kernel that runs forward and carries a tangent vector alongside each primal value. You pick the input direction upfront (the "seed"), the kernel propagates it, and the result lands in a `.dual` companion field next to each primal. The mathematical output is a Jacobian-vector product.

**Workflow.** Allocate a `.dual` field next to the primal (via `qd.root.lazy_dual()` or `needs_dual=True`), pick your seed, enter the `qd.ad.FwdMode` context manager, and run the forward kernel inside it:

```python
qd.init(arch=qd.gpu)

x = qd.field(qd.f32, shape=5)
loss = qd.field(qd.f32, shape=5)
qd.root.lazy_dual()   # place x.dual and loss.dual next to the primals

for i in range(5):
    x[i] = float(i)

@qd.kernel
def compute():
    loss[1] += x[3] * x[4]

# Directional derivative at (0, 0, 0, 1, 1): d loss[1] / d x[3] + d loss[1] / d x[4].
with qd.ad.FwdMode(loss=loss, param=x, seed=[0, 0, 0, 1, 1]):
    compute()

# loss.dual[1] == x[3] + x[4] == 7
```

Rules:

- `param` must be a single `ScalarField`. Differentiating with respect to multiple fields requires one `FwdMode` pass per field.
- `seed` is a flat list matching the flattened shape of `param`. For a 0-D `param`, `seed` defaults to `[1.0]`.
- `loss` is a scalar field or a list of scalar fields; the result lands in `loss.dual`. Duals are cleared on entry and kernel autodiff modes are restored on exit.
- Forward mode does not use the adstack pipeline: no compile-time flag is required.

**Forward vs reverse, picking the right one.** The *Jacobian* is the matrix of partial derivatives of every output with respect to every input: entry `(i, j)` is `d(output_i) / d(input_j)`.

- Forward mode computes one *column* of the Jacobian per pass: pick one input direction, get every output's derivative along it. Wins when inputs are few and outputs are many (for example, one kinematic parameter of a robot, many joint positions to differentiate).
- Reverse mode computes one *row* per pass: pick one output, get every input's derivative. Wins when outputs are few and inputs are many (for example, a single scalar loss over millions of trainable parameters).

To build the full Jacobian, call either mode once per basis vector of the smaller side and stack the results: `FwdMode` once per input in forward mode (stack the `loss.dual` columns), `kernel.grad()` once per output in reverse mode (seed `loss.grad` one entry at a time and stack the `.grad` rows).

### Overriding the compiler-generated gradient

Sometimes you may want to write your own backwards kernel, for example:

- You already know a closed-form analytic gradient that is cheaper, more numerically stable, or easier to vectorize than the auto-generated one.
- The forward pass calls external code (for example a custom C/CUDA op) that the compiler cannot see through.
- You want to checkpoint: re-run part of the forward on the backward pass instead of keeping intermediates in memory.
- You want `qd.ad.Tape` to drive a section whose gradient is supplied by hand, while auto-differentiating everything around it.

**Workflow.**

1. Write your forward as a plain Python function that calls one or more kernels. Decorate it with `@qd.ad.grad_replaced`.
2. Write a second Python function that does whatever you want the reverse pass to do (call a hand-written gradient kernel, rerun the forward for checkpointing, etc.). Decorate it with `@qd.ad.grad_for(<forward-function>)` - pass the decorated forward function itself, not its name.
3. Call the forward inside a `qd.ad.Tape` block as usual. On exit, the tape runs your gradient function in place of the compiler-generated one.

```python
x = qd.field(qd.f32)
total = qd.field(qd.f32)
qd.root.dense(qd.i, 128).place(x)
qd.root.place(total)
qd.root.lazy_grad()

@qd.kernel
def accumulate(mul: qd.f32):
    for i in range(128):
        qd.atomic_add(total[None], x[i] * mul)

@qd.ad.grad_replaced
def forward(mul):
    accumulate(mul)
    accumulate(mul)   # called twice in the forward pass

@qd.ad.grad_for(forward)
def backward(mul):
    # Analytic gradient: d total / d x[i] == 2 * mul for every i.
    accumulate.grad(mul)

with qd.ad.Tape(loss=total):
    forward(4)
# x.grad[i] == 4 for every i
```

### Global data access rules and the validation checker

**Problem.** Reverse-mode AD reads the same globals the forward pass touched to compute gradients. If the forward pass reads a global and then overwrites it in the same launch, the reverse pass sees the post-write value and by default silently computes the wrong gradient - no error, no warning, just incorrect numbers. An opt-in runtime check (described below) catches this pattern, but it is off by default because the cost would be prohibitive in production.

**How Quadrants does it.** The compiler imposes a per-launch constraint: within a single kernel launch, a field or ndarray entry that has been read must not be written to afterward. The constraint is strictly per-launch, so different kernels can freely read and write the same entry. Kernel scalar arguments are not subject to this rule: they are function parameters, not globals, and the reverse pass does not need to re-read their original value.

**Workflow.** Keep reads and writes to the same global entry in separate kernel launches; when developing, opt into the runtime validation checker described below to catch accidental violations.

Here is a kernel that violates the rule:

```python
@qd.kernel
def bad():
    # Reads b[None] for loss, then overwrites b[None] -> invalid.
    loss[None] = x[1] * b[None]
    b[None] += 100
```

This is the "read then overwrite" pattern: `b[None]` is read, then written, inside the same launch. The reverse pass would need the original `b[None]` to compute `dloss/dx[1]`, but by then it has been clobbered.

To fix it, separate the read and the write into two distinct kernels. Each kernel launch becomes self-consistent: `compute_loss` only reads, `update_b` only writes, and the rule is obeyed because the constraint is per-launch.

```python
@qd.kernel
def compute_loss():
    loss[None] = x[1] * b[None]

@qd.kernel
def update_b():
    b[None] += 100

# Call them in order. Each launch reads or writes b[None], never both.
compute_loss()
update_b()
```

The pattern often hides inside in-place time-stepping updates like `x[i] = x[i] + dt * v[i]` when the same loop body reads `x[i]` earlier. The same fix applies (split into two kernels), or equivalently, double-buffer: have the update write into an `x_new` field and swap the references after the kernel returns.

**Runtime check.** To catch violations at runtime instead of letting the gradients come out silently wrong, drive the reverse pass through [`qd.ad.Tape`](#recording-a-backward-pass-with-qdadtape) and pass `validation=True`, with `qd.init(debug=True)` set. A violation raises `QuadrantsAssertionError` with the offending field name. Kernels wrapped in `qd.ad.grad_replaced` are exempt - their gradient is the user's responsibility.

```python
with qd.ad.Tape(loss=loss, validation=True):
    bad()  # raises QuadrantsAssertionError naming b as the offending field
```

## Autodiff with dynamic loops

**Problem.** Reverse-mode AD through a dynamic loop (one whose trip count is not known at compile time) needs to recover the primal value at each iteration when walking the loop backwards. Without that, the chain-rule steps read a stale value and the gradients come out silently wrong. Static-unrolled (`qd.static(range(...))`) loops are not affected because every iteration becomes its own inlined block at compile time.

**How Quadrants does it.** Quadrants provides a dedicated compiler pipeline for this, called the *adstack* (short for "(a)uto(d)iff (stack)"). It allocates a per-variable stack alongside each primal that is updated inside the loop. The forward pass pushes one entry per iteration. The reverse pass walks the stack from top down. At each reverse iteration it reads the current top entry, applies the chain-rule contributions of that iteration, then pops the entry once and steps to the iteration underneath. Enabling adstack costs extra per-thread memory and compile time, but some kernels need it.

**Workflow.** Enable the pipeline at init time and keep using the normal reverse-mode workflow: `qd.init(..., ad_stack_experimental_enabled=True)`.

**Note.** Running with adstack enabled when it is not strictly needed is safe, but not the other way around. Running without it when it is needed raises a `QuadrantsCompilationError` in most cases: the autodiff pass rejects a non-static range that would otherwise lose its primal. A few edge-case loop shapes still slip past that rejection and produce silently-wrong gradients; these are tracked and fixed in the autodiff pass as they surface. There is no automated detector for this case. If you suspect a kernel may be affected, a reasonable check is to enable adstack and re-run: if the gradients are unchanged, adstack was not needed.

Reverse-mode AD walks the forward kernel in reverse and applies the chain rule at every op. The chain-rule factor at each op is that op's derivative with respect to its input. For *non-linear* ops (`sin`, `cos`, `exp`, `sqrt`, `tanh`, `pow`, ...) that derivative depends on the input's primal, so the reverse pass needs the primal value that was there on the forward pass. For *linear* ops (addition, subtraction, multiplication by a constant) the derivative is itself a constant and no primal is needed. In a dynamic loop the forward pass writes a different primal at each iteration, so the reverse pass cannot simply re-read the latest value - it needs one per iteration. adstack provides exactly that: a per-iteration stash of the primal.

### Examples of dynamic loops that need it

- A *loop-carried variable* - one whose value is carried forward from each iteration into the next, e.g. `v = v * 0.95 + 0.01`. The rest of this document uses "loop-carried variable" in this sense. A loop-carried variable needs adstack when it feeds into a non-linear op. Example: `for i in range(n): total += qd.sin(v); v = v * 0.95 + 0.01` - `qd.sin(v)` is non-linear, so the reverse pass needs the `v` at every iteration to evaluate `cos(v)`.
- A loop-carried variable used as an index into a global field, e.g. `a[idx]` where `idx` mutates across iterations. The field load routes the incoming adjoint into `a.grad[idx]`, which requires knowing which `idx` ran each iteration.
- An `if` whose condition depends on a loop-carried variable. The reverse pass has to walk the same branch that ran on the forward pass, so it needs to know which branch each iteration took.

### Examples of dynamic loops that do not need it

- A loop whose body is purely linear - even if it updates a loop-carried variable. Example: `for i in x: total += x[i]`. The chain-rule step at `total += x[i]` has derivative `d(total)/d(x[i]) = 1`, a constant, so computing `dL/dx[i]` from the upstream `dL/dtotal` only requires multiplying by that constant - no primal replay. Same for `for i in range(n): total += a * x[i] + b` (all ops linear) and for a linear recurrence like `v = 0.95 * v + 0.01` read linearly downstream.
- A non-linear op whose input does not change across iterations, e.g. `for i in range(n): total += qd.sin(a) * x[i]` with `a` fixed. `qd.sin(a)` produces the same primal every iteration, and `a` itself is still in scope after the loop, so the reverse pass needs one copy of `a`, not one per iteration.

`qd.static(range(...))` loops are unrolled at compile time and never need the adstack either.

### Supported loop shapes

Quadrants supports many common loop constructs, but not every loop shape that compiles in the absence of adstack is currently handled. Loop shapes outside the supported set are rejected at compile time, with the error naming the offending source line. Typical fixes are to restructure the loop into one of the supported shapes, or to file a bug. Setting `QD_DUMP_IR=1` before compiling dumps the kernel IR for the first unresolved adstack into `/tmp/ir_adstack_unresolved/` so you can attach it to the report.

See [Appendix A: types of dynamic loops supported by reverse-mode AD](#appendix-a-types-of-dynamic-loops-supported-by-reverse-mode-ad) for the authoritative list.

### Under the hood (advanced)

*You do not need to read this section to use reverse-mode AD. Skip past it unless you hit an overflow error on SPIR-V, an out-of-memory error on GPU, or a compile error from the autodiff pass naming a loop in your kernel.*

#### Why peek-and-pop, not pop-per-use

The reverse pass reads the current top of stack as many times as that iteration's chain-rule contributions need, then pops the entry once. The number of reads per entry equals the number of downstream uses of the primal in that iteration. Popping after each read would not work: the same primal often feeds several chain-rule terms in one iteration - e.g. a `v` that appears in both a `sin(v)` and a subsequent `v * w` is needed for both adjoint terms - so popping after the first read would discard the value before the remaining terms could use it.

#### One adstack per variable

A dynamic loop does not have a single adstack. The compiler allocates one adstack per scalar value the reverse pass has to replay:

- one for each floating-point [loop-carried variable](#examples-of-dynamic-loops-that-need-it) (e.g. `v += qd.sin(u)`);
- one for each integer loop-carried variable (counters, or indices used to address a global field: `idx += step; total += a[idx]`);
- one for each *branch flag* - a per-iteration boolean the compiler emits internally whenever an `if` inside the loop body depends on a loop-carried variable. The flag records which branch ran so the reverse pass walks the matching one.

A kernel with four `f32` loop-carried variables and one integer loop counter therefore allocates five separate adstacks, each sized independently.

#### Launch-time sizing

Adstack sizing is automatic on every backend. You do not need to tune anything, and there is no user-facing size knob:

- For each loop-carried variable, the compiler works out a launch-time formula for that adstack's depth at compile time. The formula is composed from the bound shapes listed under [Supported loop shapes](#supported-loop-shapes), combined with `+`, `-`, `*`, and `max`.
- Right before every kernel dispatch, Quadrants evaluates that formula against the live field and ndarray state, and resizes the per-thread backing buffer accordingly.
- If a loop shape falls outside what the compiler knows how to bound, it raises a compile error naming the offending source location - there is no silent over-allocation.

The evaluation happens in different places depending on the backend, but the result is the same:

| Backend | Where the formula is evaluated | Why |
| --- | --- | --- |
| CPU / CUDA / AMDGPU | On the host, right before dispatch | The host can read every input the formula depends on. |
| Metal / Vulkan | In a tiny on-device compute shader run before the main kernel | The formula can depend on ndarray contents that live on the GPU, so the evaluation has to run where the data is. |

Either way, the per-thread stride and each adstack's offset / max-size land in a small buffer the main kernel reads on every push. The backing heap grows on demand to match the largest size any launch has needed so far, and is reused across subsequent launches - you do not need to reserve memory up front.

The sized result is cached per task and reused while the loop bounds are unchanged. Although changing loop bounds at runtime is possible, it comes with some limitations for performance reasons. See [What can go wrong](#what-can-go-wrong) for details.

The on-device sizer relies on two common hardware features (64-bit integer arithmetic and raw-pointer storage-buffer access). Every mainstream GPU from late 2018 onward supports both.

#### Manual override

`qd.init()` exposes two escape hatches:

- `ad_stack_size=N` (default `0`): forces every adstack to exactly `N` slots and bypasses the sizer. Leave at `0` in day-to-day use; positive `N` is for stress tests or working around a suspected sizer bug.
- `ad_stack_sparse_threshold_bytes=B` (default `100 MiB`): cutoff below which the gate-passing-count sizing of [Memory footprint](#memory-footprint) is skipped in favour of the eager `dispatched_threads * stride` heap. The sparse path saves memory but pays a per-launch reducer dispatch; below `B` of conservative heap, that overhead outweighs the savings. Set to `0` to always use the sparse path; lower it if the default still skips kernels you want shrunk.

#### Memory footprint

Each adstack is *typed*: all of its slots hold values of the same scalar type - `f32`, `f64`, `i32`, `i64`, `bool`, and so on - inherited from the loop-carried variable it was allocated for. Denote that type `T`.

Total memory across all adstacks is approximately:

```
num_threads * stack_size * bytes_per_slot * num_buffers
```

where each quantity means:

| Quantity | What it is |
| --- | --- |
| `num_threads` | Concurrent thread slots, regardless of logical ndrange. CPU: thread-pool size (~tens). GPU adstack-bearing kernels: capped at 65536 on all backends (131072 on SPIR-V range-for, i.e. `for i in range(N):`), tightened to the actual flat product when the iteration bound is compile-time known. Forward-only kernels keep the full ndrange. |
| `stack_size` | Per-launch capacity resolved by the sizer. Varies between launches - if an ndarray-bounded loop iterates 16 times at one dispatch and 1024 at another, `stack_size` tracks each. |
| `bytes_per_slot` | Depends on `T` and on the backend (see table below). |
| `num_buffers` | Number of adstacks the kernel allocates - one per loop-carried variable plus one per dependent branch flag (see [One adstack per variable](#one-adstack-per-variable)). |

The float heap is by far the main reverse-mode memory bottleneck because a typical kernel allocates many float-typed adstacks - one per floating-point loop-carried scalar, each storing both primal and adjoint. The total scales as `num_threads * stack_size * num_float_buffers * 8` bytes, dominating the integer / boolean heap. Advanced static IR analysis is used to further shrink the float adstack in some common gated-kernel shapes. When a runtime gate sits directly above the adstack-using body and compares a single field entry to a constant, the compiler counts the gate-passing iterations at launch time and sizes the float adstack to that count. So a workload whose gate matches 5% of iterations pays 5% of the float-adstack cost. See [Appendix B: gate-index shapes that capture vs fall back to the worst-case heap](#appendix-b-gate-index-shapes-that-capture-vs-fall-back-to-the-worst-case-heap) for the authoritative list of supported shapes.

Every adstack slot always stores a *primal* value - the forward-pass value the reverse pass pops to recover the chain-rule step. Floating-point adstacks additionally store an *adjoint* slot where the reverse pass accumulates chain-rule contributions. Integer / boolean adstacks do not need an adjoint slot.

Platform-specific notes:

- Even though integer / boolean adstacks do not need an adjoint slot, LLVM backends still carry one for codegen uniformity. SPIR-V backends trim it.
- SPIR-V stores `bool` slots using 4 bytes (32 bits), because SPIR-V does not specify a portable in-memory layout for booleans.

The resulting per-slot cost on each platform is:

| T | LLVM bytes/slot | SPIR-V bytes/slot |
| --- | --- | --- |
| f32 | 8 | 8 |
| f64 | 16 | 16 |
| i32 / u32 | 8 | 4 |
| i64 / u64 | 16 | 8 |
| bool | 2 | 4 |

Adstack buffers live on the device on GPU and in host RAM on CPU.

#### Avoiding OOM on GPU

A large `ndrange` combined with several loop-carried variables multiplies quickly. If the allocator returns out-of-memory on a legitimately deep reverse-mode kernel, remedies in order:

1. Reduce `num_buffers` - split the kernel, checkpoint manually, or fold two accumulators into one so the reverse pass has fewer loop-carried variables to replay.
2. Raise `device_memory_fraction` or `device_memory_GB` in `qd.init()` if the GPU has headroom.

## What can go wrong

### Adstack overflow

Surfaces as `QuadrantsAssertionError: Adstack overflow ...` at the next Quadrants Python entry. The message names the offending kernel + offload task and the most likely cause.

The two cases the runtime distinguishes:

- *Untracked tensor mutation between launches.* A tensor backing a data-dependent loop bound was written to outside Quadrants's tracking - typically a DLPack zero-copy mutation through a torch tensor sharing storage with a Quadrants ndarray, or a raw pointer write through a non-torch consumer. The cached adstack capacity was sized against the value before the mutation; if the mutation grew the bound, the next launch overflows. Workaround: route the write through a Quadrants API (`Ndarray.write` / `Ndarray.fill` / a kernel that writes the value). Alternatively, catch the exception and re-launch - Quadrants invalidates the cached bound on raise, so the retry runs against the live state. Kernel state may be inconsistent after an overflow; do not retry the same step without restarting from a clean state.
- *Sizer under-estimated the bound (Quadrants bug).* On unusually intricate nested loops - typically deeply nested `for i in range(arr[...])` with cumulative-index arithmetic - the sizer can compute a bound that is mathematically tighter than the actual push count. To file a bug: clear `/tmp/ir/`, rerun your script with `QD_DUMP_IR=1` set in the environment so Quadrants dumps the kernel IR there, then open an issue on the Quadrants repo with the contents of `/tmp/ir/` attached as a zip. Workaround: pass a generous `ad_stack_size=N` to `qd.init()` with `N` large enough to cover the real push count (bypasses the sizer).

### Out-of-memory before the kernel even runs

A reverse pass through many loop-carried variables at a large ndrange can ask the runtime for more adstack memory than the device can physically back, even when the sizer's number is correct. Surfaces as an allocator OOM at launch time. Remedies are the ones listed under *Avoiding OOM on GPU* above: fewer loop-carried variables, a smaller ndrange, manual checkpointing, or more device-memory headroom.

### Loop bounds backed by a mutated ndarray

A reverse-mode kernel with `for i in range(n[j])` requires `n[j]` to hold the same value at the forward call and at `.grad()`. If anything writes to `n[j]` between those two points - the differentiable kernel itself, or any other kernel call - the backward call will trigger an `Adstack overflow` exception or the computed gradient would come out silently wrong.

The safe rule: populate loop-bound ndarrays before the forward call and leave them untouched until `.grad()` returns. The reason for that is Quadrants' adstack sizer design: it reads the loop bound separately at each dispatch, which includes forward and backward calls. Tape-based eager AD like [PyTorch's autograd](https://pytorch.org/docs/stable/notes/autograd.html) is not affected, since the trip count is recorded as the forward runs and reused at backward time.

### Inner reverse-mode loop with a complex bound at very large extent

A reverse-mode kernel with two nested loops is in some cases limited to an outer-loop extent of at most `1 << 24`. In particular when the enclosed loop's trip count is an uncommon expression of the outer-loop variable, e.g. `for i in range(arr.shape[0]): ... for j in range(arr[i // 2]):`. See [Appendix C](#appendix-c-evaluation-of-the-enclosed-loops-bound-expression) for a complete walkthrough of the enclosed loop's bound expression and workarounds. When the limit applies and the outer extent exceeds it, the kernel raises `RuntimeError: ... iteration count ... exceeds the 16777216 guard` at launch.

## Performance characteristics

- **Compile time scales with loop nesting.** The adstack pipeline trades compile time for generality. Kernels with many loop-carried variables, nested dynamic loops, or large inner-loop bodies produce visibly slow compile times - seconds stretching into minutes. Budget compile time accordingly when migrating existing reverse-mode AD workloads.
- **SPIR-V backward passes can be an order of magnitude slower than the forward pass.** Reverse-mode AD reruns every forward op and additionally accumulates a gradient contribution per op, usually via atomic writes.

## Appendix A: types of dynamic loops supported by reverse-mode AD

The compiler recognises the following bound shapes for adstack-aware loops:

| Bound shape | Example |
| --- | --- |
| Integer constant | `for i in range(42):` |
| Scalar integer field (`i32` / `i64`) at a constant or loop index | `for i in range(n[None]):`, `for i in range(n[j]):` |
| Ndarray argument shape along any axis | `for i in range(arr.shape[1]):` |
| Scalar ndarray read at a constant or loop index, including multi-axis reads | `for i in range(arr[j]):`<br>`for i in range(arr[j, k]):` |
| Two-argument `range(start, stop)` whose `start` and `stop` are any of the bound shapes above | `for k in range(start[j], stop[j]):` |
| An enclosing `for i in range(...)`, struct-for, or ndrange index (also multi-index ndrange over ndarray shapes) | `for i in range(N): ... for j in range(i):`<br>`for i, j in qd.ndrange(arr.shape[0], arr.shape[1]):` |
| A cast of a loop index stashed earlier in the same body, used to index a field or ndarray | `i_l = qd.cast(outer_i, qd.i32)` then later `arr[i_l]` or `field[i_l]` |

Examples of constructs that are *not* currently handled:

```python
@qd.kernel
def k_nonlinear_bound(a):
    for i in range(qd.sqrt(a)):   # non-linear transform of a bound shape
        ...

@qd.kernel
def k_mixed_index(a, b):
    for i in range(a):
        for j in range(b):
            for k in range(a[i], b[j]):   # cross-axis indices into independent ndarrays
                ...

@qd.kernel
def k_data_dependent(a):
    for i in range(a.shape[0]):
        while a[i] < 10:              # bound that can only be known by running the loop body
            a[i] = a[i] + 1

@qd.kernel
def k_inner_struct_for(a, field):
    for i in range(a.shape[0]):
        for j in field:               # struct-for as the enclosed loop with reverse-mode pushes
            ...
```

## Appendix B: gate-index shapes that capture vs fall back to the worst-case heap

The compiler accepts the gate index shapes below as bijective and falls back to the worst-case heap `num_threads * stack_size` for the rest. Only the float adstack heap shrinks; integer / boolean adstacks stay at `num_threads * stack_size` because their pushes fire unconditionally for control-flow replay. The float heap grows on demand if a later launch's gate matches more iterations.

Patterns that capture:
  - **Linear range loop**: `for i in range(n): if field[i] > eps: ...`
  - **Multi-axis StructFor**: `for I, J, K in field3d: if field3d[I, J, K] > eps: ...`
  - **Multi-axis ndrange**: `for ii, jj, kk in qd.ndrange(*shape): if grid[ii, jj, kk] > eps: ...`
  - **Multi-axis split of a flat loop**: a gate whose axes are two or more sub-expressions of the loop variable that hold pairwise distinct values, e.g. `field2d[i // K, i % K]`, `field2d[i // K, i]`, `field3d[i // (K * L), (i // L) % K, i % L]`.
  - **Iterating axes plus a slice**: a kernel argument or constant on an extra axis, e.g. `grid[i, 0]`, `grid[arg, ii, jj, kk]`.

Patterns that fall back to the worst-case heap:
  - **Single-axis arithmetic on the loop variable**: `field[i % K]`, `field[i / 2]`, `field[i + 5]`, `field[2 * i]`, and similar.
  - **Multi-axis index with axes holding the same value**: `field[i % K, i % K]`, or any multi-axis gate where two or more axes evaluate to the same value; the joint mapping is many-to-one and would alias iterations onto a few cells.
  - **Multi-axis index folding onto a smaller subspace**: `field[i % K0, (i // K0) % K1]` with the loop trip count above `K0 * K1`; the joint axis space is smaller than the loop and iterations wrap around it.
  - **Constant-index gate**: `field[42]`, or any axis that is a literal constant.
  - **Kernel-argument index, no iterating axis**: `field[arg]` where every axis is launch-constant.
  - **Indirect index via runtime load**: `field[other_field[i]]`; the compiler cannot prove `other_field` is injective.

## Appendix C: evaluation of the enclosed loop's bound expression

This appendix details how the runtime computes the worst-case trip count of an enclosed reverse-mode loop and which expression shapes each evaluation path accepts. It backs the *Inner reverse-mode loop with a complex bound at very large extent* entry under [What can go wrong](#what-can-go-wrong).

Consider a reverse-mode kernel with two nested loops where the enclosed loop's iteration count depends on the outer loop variable through an arithmetic expression on an ndarray index:

```python
for i in range(arr.shape[0]):       # outer loop
    for j in range(arr[i // 2]):    # enclosed loop: for <var> in range(<bound expression>)
        ...
```

The enclosed loop's iteration count `arr[i // 2]` is what we call the enclosed loop's *bound expression*. It is a function of the outer-loop variable `i`: as `i` ranges over `[0, arr.shape[0])`, the bound expression evaluates to a different integer at each iteration. Reverse-mode autodiff needs the adstack sized for the worst case - the largest inner-loop trip count that will ever occur across the outer loop's full range, i.e. `max(arr[i // 2] for i in range(arr.shape[0]))`. For example, if `arr = [3, 5, 1]` and the outer loop runs `i` over `[0, 6)`:

| `i` | `i // 2` | bound expression `arr[i // 2]` |
| --- | --- | --- |
| 0 | 0 | 3 |
| 1 | 0 | 3 |
| 2 | 1 | 5 |
| 3 | 1 | 5 |
| 4 | 2 | 1 |
| 5 | 2 | 1 |

Quadrants computes that worst case at launch time - in this example, the max of the column above, 5 - and sizes the adstack accordingly: each outer iteration accommodates up to 5 pushes and the adstack never overflows. With deeper loop nests each enclosed loop's bound expression is reduced separately and the adstack is sized as the product of those maxes.

### Evaluation paths

The compiler picks one of two evaluation paths to compute the maximum based on the backend and the bound expression's structure:

- **Parallel (GPU only):** the maximum is computed with a tiny parallel reduction kernel for efficiency. The reducer accepts a common subset of bound expressions:
  - **Integer ndarray or field read** up to 32 bits wide, indexed by literal constants or outer-loop variables: `arr[i, j]`, `field[i]`.
  - **Shape term**: `arr.shape[k]`.
  - **Literal integer constant**: `42`.
  - **Arithmetic combinator**: any `+`, `-`, `*`, `max` of the above.
- **Sequential:** the fallback path, used whenever the parallel path doesn't support the bound expression. Quadrants walks the bound expression one outer-loop iteration at a time on a single thread (host-side on CPU, single-thread on-device kernel on GPU); the adstack is sized identically, only the upfront cost differs. This path accepts everything the parallel path does, plus:
  - **Arithmetic-indexed read**: `arr[i // 2]`, `arr[i % 4]`.
  - **Indirect / nested read**: `arr1[arr2[i]]`, `my_field[arr[i]]`.

### Nested loops

Quadrants supports arbitrarily nested loops. When the bound expression itself contains another enclosed loop whose own bound expression must be reduced first, the enclosing bound expression takes the parallel path only if every nested bound expression is also supported by the parallel path; otherwise it falls back to the sequential walk. This keeps the runtime from mixing parallel and sequential evaluators inside a single bound expression, which would otherwise force per-iteration kernel launches.

### Sequential walk cap

The sequential walk's outer loop is artificially capped at 2^24 = 16 777 216 iterations on GPU backends to keep the walk time bounded; past that the kernel raises `RuntimeError: ... iteration count ... exceeds the 16777216 guard`. In the example above, the iteration count of the enclosed loop takes the sequential path because of the `i // 2` index, so it would raise at launch on GPU backends if `arr.shape[0] > (1 << 24)`.

To circumvent this limitation, rewrite the bound expression to unlock the parallel path (e.g. precompute `bounds[i] = arr[i // 2]` into a persistent separate buffer, pass `bounds` in as an input, and use `for j in range(bounds[i]):`), or keep the outer loop count below 2^24.
