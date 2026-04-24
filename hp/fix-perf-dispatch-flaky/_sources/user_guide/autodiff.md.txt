# Automatic differentiation

Automatic differentiation (autodiff) computes the exact gradient of a kernel's output with respect to its inputs, without the user writing the derivative formulas by hand. Gradient-based optimizers then use this gradient to train neural networks, fit physical models to data, drive differentiable simulators, or solve inverse problems.

Throughout this page, the *primal* is the value a kernel computes in its normal forward pass (the field value, the loss, whatever the kernel writes); the *adjoint* (or *gradient*) is the derivative of the final scalar output (typically a loss) with respect to that primal value, stored in the `.grad` field next to the primal.

Quadrants implements autodiff at compile time: when `.grad()` is requested, the compiler emits a companion kernel that runs on the same backend as the forward one and writes gradients into the primal fields' `.grad` companions. There is no Python-side tape, no per-op dispatch overhead, and no dependency on an external AD framework. Forward mode, reverse mode, and the adstack pipeline for dynamic loops (described further down) are fully supported on every backend Quadrants targets: x64 / arm64 CPU, CUDA, AMDGPU, Metal, and Vulkan.

Three styles are supported:

- **Reverse mode** - one scalar output, many inputs. One backward pass yields every input gradient. This is the usual training setup and the bulk of the page.
- **Forward mode** - few inputs, many outputs. One forward pass yields every output derivative along a chosen input direction. See [Forward-mode AD via `qd.ad.FwdMode`](#forward-mode-ad-via-qdadfwdmode).
- **Custom gradients** - override the auto-generated gradient with a user-supplied one, typically to inject a closed-form analytic derivative or to checkpoint for memory. See [Overriding the compiler-generated gradient](#overriding-the-compiler-generated-gradient).

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
- Reverse-mode AD through a *dynamic* loop (one whose trip count is not known at compile time) needs an opt-in compiler pipeline called the *adstack*, gated behind `ad_stack_experimental_enabled=True` in `qd.init()`. Support for this path is still experimental and the set of loop shapes it accepts is still growing; see [Autodiff with dynamic loops](#autodiff-with-dynamic-loops) and [Under the hood: adstack capacity and memory](#under-the-hood-adstack-capacity-and-memory) for what works today and what does not.

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

**Problem.** Reverse-mode AD reads the same globals the forward pass touched to compute gradients. If the forward pass reads a global and then overwrites it in the same launch, the reverse pass sees the post-write value and silently computes the wrong gradient - no error, no warning, just incorrect numbers.

**How Quadrants does it.** The compiler imposes a per-launch constraint: within a single kernel launch, a field or ndarray entry that has been read must not be written to afterward. The constraint is strictly per-launch, so different kernels can freely read and write the same entry. Kernel scalar arguments are not subject to this rule: they are function parameters, not globals, and the reverse pass does not need to re-read their original value.

**Workflow.** Keep reads and writes to the same global entry in separate kernel launches; when developing, opt into the runtime checker described below to catch accidental violations.

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

**Runtime check.** Violations of the rule do not produce an error on their own - the gradients are just silently wrong. To get Quadrants to validate the rule at runtime, pass `validation=True` to `qd.ad.Tape` (with `qd.init(debug=True)` set). A violation raises `QuadrantsAssertionError` with the offending field name. Kernels wrapped in `qd.ad.grad_replaced` are exempt - their gradient is the user's responsibility.

## Autodiff with dynamic loops

**Problem.** Reverse-mode AD through a dynamic loop (one whose trip count is not known at compile time) needs to recover the primal value at each iteration when walking the loop backwards. Without that, the chain-rule steps read a stale value and the gradients come out silently wrong. Static-unrolled (`qd.static(range(...))`) loops are not affected because every iteration becomes its own inlined block at compile time.

**How Quadrants does it.** Quadrants provides a dedicated compiler pipeline for this, called the *adstack* (short for "(a)uto(d)iff (stack)"). It allocates a per-variable stack alongside each primal that is updated inside the loop. The forward pass pushes an entry each iteration; the reverse pass pops them back off in reverse order to recover the correct primal for every chain-rule step. adstack is opt-in because it costs extra per-thread memory and compile time, and because most kernels do not need it. Running with adstack enabled when it is not strictly needed is safe. Running without it when it is needed raises a `QuadrantsCompilationError` in most cases: the autodiff pass rejects a non-static range that would otherwise lose its primal. A few edge-case loop shapes still slip past that rejection and produce silently-wrong gradients; these are tracked and fixed in the autodiff pass as they surface, so if you see wrong-but-non-zero gradients through a dynamic loop with adstack disabled, turn it on and rerun as a sanity check.

**Workflow.** Enable the pipeline at init time and keep using the normal reverse-mode workflow: `qd.init(..., ad_stack_experimental_enabled=True)`. The flag is compile-time, so it must be set before the offending kernel compiles.

Reverse-mode AD walks the forward kernel in reverse and applies the chain rule at every op. The chain-rule factor at each op is that op's derivative with respect to its input. For *non-linear* ops (`sin`, `cos`, `exp`, `sqrt`, `tanh`, `pow`, ...) that derivative depends on the input's primal, so the reverse pass needs the primal value that was there on the forward pass. For *linear* ops (addition, subtraction, multiplication by a constant) the derivative is itself a constant and no primal is needed. In a dynamic loop the forward pass writes a different primal at each iteration, so the reverse pass cannot simply re-read the latest value - it needs one per iteration. adstack provides exactly that: a per-iteration stash of the primal.

### Examples of dynamic loops that need it

- A *loop-carried variable* - one whose value is carried forward from each iteration into the next, e.g. `v = v * 0.95 + 0.01`. The rest of this document uses "loop-carried variable" in this sense. A loop-carried variable needs adstack when it feeds into a non-linear op. Example: `for i in range(n): total += qd.sin(v); v = v * 0.95 + 0.01` - `qd.sin(v)` is non-linear, so the reverse pass needs the `v` at every iteration to evaluate `cos(v)`.
- A loop-carried variable used as an index into a global field, e.g. `a[idx]` where `idx` mutates across iterations. The field load routes the incoming adjoint into `a.grad[idx]`, which requires knowing which `idx` ran each iteration.
- An `if` whose condition depends on a loop-carried variable. The reverse pass has to walk the same branch that ran on the forward pass, so it needs to know which branch each iteration took.

### Examples of dynamic loops that do not need it

- A loop whose body is purely linear - even if it updates a loop-carried variable. Example: `for i in x: total += x[i]`. The chain-rule step at `total += x[i]` has derivative `d(total)/d(x[i]) = 1`, a constant, so computing `dL/dx[i]` from the upstream `dL/dtotal` only requires multiplying by that constant - no primal replay. Same for `for i in range(n): total += a * x[i] + b` (all ops linear) and for a linear recurrence like `v = 0.95 * v + 0.01` read linearly downstream.
- A non-linear op whose input does not change across iterations, e.g. `for i in range(n): total += qd.sin(a) * x[i]` with `a` fixed. `qd.sin(a)` produces the same primal every iteration, and `a` itself is still in scope after the loop, so the reverse pass needs one copy of `a`, not one per iteration.

`qd.static(range(...))` loops are unrolled at compile time and never need the adstack either.

### Under the hood: adstack capacity and memory

*You do not need to read this section to use reverse-mode AD. If a kernel exceeds its adstack capacity, Quadrants raises a Python exception at the next `qd.sync()` whose message recommends bumping `default_ad_stack_size` - that is usually enough. Read on only if you hit that overflow, want to understand why, or want to cap the memory footprint explicitly.*

**Tuning.** Two `qd.init()` knobs control adstack sizing, both measured in slots per adstack (not bytes):

- `default_ad_stack_size=N` (default `256`): the fallback capacity for loops whose trip count the compiler cannot prove statically. Every adstack whose size was not deducible shares this value. Prefer this knob - it only affects the branch where the compiler had to guess.
- `ad_stack_size=N` (default `0 = adaptive`): hard override forcing every adstack in the program to exactly `N` slots regardless of what the compiler proved. Use only for targeted experiments, for example stress-testing the runtime heap path.

**One adstack per variable.** A dynamic loop does not have a single adstack. The compiler allocates one for each [loop-carried variable](#examples-of-dynamic-loops-that-need-it) the reverse pass has to replay - whether the variable is a floating-point accumulator (`v += qd.sin(u)`), an integer counter (`count += 1`), an integer index used to address a global field (`idx += step; total += a[idx]`), or any other scalar type that carries state across iterations. The compiler also allocates one for each *boolean branch flag*: a per-iteration boolean it emits internally whenever an `if` inside the loop body depends on a loop-carried variable - the flag records which branch ran on each iteration so the reverse pass walks the matching one. A kernel with four `f32` loop-carried variables and one integer loop counter therefore allocates five separate adstacks, each sized independently by the rule below.

**Sizing rule.** A `K`-iteration dynamic loop consumes `K + 2` slots in each of its adstacks: one slot per forward iteration, plus two setup slots (one for the initial adjoint, one for the primal's starting value). `default_ad_stack_size` is a per-stack slot count, so size it at the worst-case trip count of the deepest unprovable dynamic loop in the program plus 2.

| Loop shape | Required `default_ad_stack_size` |
| --- | --- |
| single dynamic `for i in range(a[None])` | `a_max + 2` |
| nested `for i in range(a[None]): for j in range(b[None])` | `a_max * b_max + 2` |
| `qd.ndrange(n, m)` with field-derived `n`, `m` | `n_max * m_max + 2` |

At `max_n_dofs_per_entity = 16`, a `16 x 16` ndrange hits the default exactly (`256`).

**Memory footprint.** Each adstack is *typed*: all of its slots hold values of the same scalar type - `f32`, `f64`, `i32`, `i64`, `bool`, and so on - inherited from the loop-carried variable the adstack was allocated for. We denote that type `T`. With one scratch buffer per adstack (see above), the total memory cost depends on two further quantities. The first is the number of threads the kernel actually dispatches, which we call `num_threads`. On CPU that is the thread-pool size, typically tens. On GPU it is the full ndrange. The second is `bytes_per_slot`, which depends on `T` and on the backend; the two tables below work through the concrete values. Total memory across all buffers is then approximately:

```
num_threads * default_ad_stack_size * bytes_per_slot * num_buffers
```

On all backends, every adstack slot stores a *primal* value of type `T` - that is what the reverse pass pops to recover the forward-pass value at each chain-rule step.

In addition, LLVM backends (CPU / CUDA / AMDGPU) store an *adjoint* of type `T` alongside the primal in each slot, regardless of what `T` is. The adjoint slot is where the reverse pass accumulates chain-rule contributions; LLVM carries it even when `T` is an integer or boolean (where the reverse pass never writes to it) because the codegen uses a uniform two-element slot layout, which keeps push/pop branch-free at the cost of the unused adjoint slot for non-floating-point `T`. So `bytes_per_slot = 2 * sizeof(T)` on LLVM for every choice of `T`:

| T | LLVM bytes/slot |
| --- | --- |
| f32 | 8 |
| f64 | 16 |
| i32 / u32 | 8 |
| i64 / u64 | 16 |
| bool | 2 |

On SPIR-V backends (Metal / Vulkan) the slot layout is trimmed: adstacks whose `T` is an integer type (`i32`, `i64`, ...) only store the primal because the reverse pass does not accumulate integer adjoints, and per-thread on-chip memory is more constrained than on LLVM. So `bytes_per_slot = sizeof(T)` for integer `T` and `bytes_per_slot = 2 * sizeof(T)` for floating-point `T`. SPIR-V has no defined layout for `OpTypeBool`, so booleans are widened to i32 at storage time:

| T | SPIR-V bytes/slot |
| --- | --- |
| f32 | 8 |
| f64 | 16 |
| i32 / u32 | 4 |
| i64 / u64 | 8 |
| bool | 4 |

Adstack buffers live on the device on GPU and in host RAM on CPU. The buffer grows on demand to match the largest size any launch has needed so far and is reused across subsequent launches, so you do not need to reserve memory up front.

**Avoiding OOM on GPU.** A big `ndrange` combined with several loop-carried f32 variables adds up fast: `ndrange(1024, 1024)` with `default_ad_stack_size=256` and four f32 buffers allocates roughly `1024 * 1024 * 256 * 8 * 4 bytes ~= 8 GB`, enough to exhaust a consumer GPU. Doubling `default_ad_stack_size` doubles the backing buffer linearly, so it is the simplest knob to reach for on out-of-memory. Remedies, in order:

1. Drop `default_ad_stack_size` toward the real worst-case trip count of your dynamic loops.
2. Reduce the number of loop-carried variables the reverse pass has to replay (split the kernel, checkpoint manually, or fold two accumulators into one).
3. Raise `device_memory_fraction` or `device_memory_GB` in `qd.init()` if the GPU has headroom.

## Known limitations

- Adstack overflow is reported as a Python-level exception on every backend, but asynchronously: the offending kernel writes to a host-polled SSBO flag during execution, and the next `qd.sync()` (explicit, or implicit via a host read like `to_numpy()` / `to_torch()`) reads the flag and raises. This follows the same pattern as CUDA async errors so every launch does not pay a per-launch sync. If you want the exception to land exactly at the offending kernel rather than at the next sync, call `qd.sync()` right after the kernel, or enable `qd.init(debug=True)` on LLVM backends to poll after every launch.
- Adstack trades compile time for generality. Kernels with many loop-carried variables, nested dynamic loops, or large inner-loop bodies produce visibly slow compile times - seconds stretching into minutes, and on SPIR-V backends sometimes into the territory where the driver's shader compiler gives up. Budget compile-time accordingly when migrating existing reverse-mode AD workloads.
- Reverse-mode AD does not propagate gradients through integer casts or non-real operations. No error is raised; the gradient simply stops at the cast and silently reads as zero upstream. Cast to `qd.f32` / `qd.f64` before the differentiable section.
- Backward passes on non-trivial kernels run noticeably slower than the corresponding forward pass, sometimes by an order of magnitude on SPIR-V.
