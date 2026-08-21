# Optimization passes

When you call a `@qd.kernel` function, Quadrants first translates the kernel into an internal form then rewrites it, step by step, into something equivalent but cheaper to run. "Equivalent" here means that they yield the exact same observable behavior and produce the exact same outputs [*1] for all valid inputs, regardless of how they are restructured. This page explains, at a high level, what those rewrite steps (the *optimization passes*) do, which ones cost compile time, and how to control and inspect them. You do not need to understand any of this to use Quadrants but it helps when you are trading compile time against runtime, or debugging a surprising result.

[*1] To within floating-point precision, not bit exact. In particular, fast maths optimizations are fairly approximate.

## Key terms

Let's start by defining terms that will be used throughout the page and are necessary in order to be able to understand it.

- **Kernel** - a function you decorate with `@qd.kernel`. It is the unit that Quadrants compiles and launches.
- **IR (intermediate representation)** - the compiler's internal version of your kernel: a flat list of small, explicitly-typed instructions, sitting between your Python source and the final machine code. Every pass reads and rewrites the IR; none of it is something you write by hand.
- **Pass** - one transformation step over the IR. An *optimization* pass rewrites the IR into a form that produces the **same results** but runs faster or uses less memory. (Some passes are not optimizations but *lowering* steps - they translate high-level constructs into lower-level ones; this page focuses on the optimizations.)
- **Basic block** - a straight-line run of instructions with no branches into or out of the middle. Control flow (`if`, loops) connects blocks together.
- **Offloaded task** - after the *offload* step, your kernel is split into one or more tasks, and each task becomes a single device launch: one GPU launch on a GPU backend, or one parallel loop on CPU. A simple kernel is usually one task; a kernel with, say, a short serial preamble followed by a big parallel loop becomes several tasks that run back to back.

## The compile pipeline at a glance

Compilation runs as a fixed sequence of stages. Optimization passes are interleaved with the lowering steps that gradually turn high-level IR into device code:

```
Python (AST = abstract syntax tree)
   |  lower to IR, type-check
   v
high-level IR  --> simplify --> (autodiff = automatic differentiation, if requested) --> simplify
   |
   v
offload  (split the kernel into offloaded tasks)
   |
   v
per-task IR  --> simplify --> lower memory access --> simplify
   |
   v
backend codegen  (translate IR into the device machine code your GPU runs)
```

The "simplify" boxes are all the same routine, invoked at several points. Most of the interesting optimization work happens inside it. The optional autodiff step, run only when you ask Quadrants for gradients, is covered in [Automatic differentiation](./autodiff.md).

## The simplify loop

Each "simplify" stage runs a small bundle of optimizations **repeatedly, until the IR stops changing** (a fixed point). Running to a fixed point matters because the passes feed each other: folding a constant can expose dead code, deleting that code can expose a common subexpression, and so on.

In the order they run each round:

| Pass | What it does |
|------|--------------|
| Extract constant | Lifts constant values out of larger expressions into standalone constant instructions, so the passes below can recognize and reuse them. |
| Unreachable-code elimination | Removes branches that can never be taken (e.g. the body of an `if` whose condition is always false). |
| Binary-op / algebraic simplification | Applies arithmetic identities: `x * 1 -> x`, `x + 0 -> x`, `x * 2 -> x + x`, and similar local rewrites over a short window of instructions ("peephole" optimizations). |
| Constant folding | Pre-computes expressions whose inputs are all known at compile time: `2 * 3 → 6`. |
| Dead-instruction elimination (**DIE**) | Drops instructions whose results are never used. Runs several times per round, after passes that tend to create newly-dead instructions. |
| Loop-invariant code motion (**LICM**) | Hoists a computation that produces the same value on every iteration out of the loop, so it runs once instead of N times. |
| Local simplify | Peephole cleanups within a block. |
| Common-subexpression elimination (**CSE**) | Finds an identical expression computed more than once and computes it a single time, reusing the result. |
| Control-flow-graph (**CFG**) optimization | Memory-focused optimizations that need a whole-task view; see the next section. Runs only once per stage (it is the most expensive pass). |

Two of these - CSE and CFG optimization - run only when `opt_level > 0` (the default is `1`).

## Control-flow-graph (CFG) optimization

A **control-flow graph** is a map of your kernel's basic blocks together with the branches connecting them. It lets the compiler answer questions of the form "if execution reaches *here*, what must already have happened?" - which is exactly what is needed to optimize reads and writes to memory. Two such optimizations run on the CFG:

- **Store-to-load forwarding** - if a value is written to a location and then read again before anything overwrites it, the read is replaced with the value directly, skipping the round trip through memory.
- **Dead-store elimination** - if a write is overwritten before anyone reads it, the write is removed.

Building and analyzing the CFG is the most expensive optimization in the pipeline, which is why it runs at most once per simplify stage rather than every round.

## Controlling the passes

All of these are compile-time options (fields of `CompileConfig`, the object that holds a kernel's compilation settings), so you set them at `qd.init(...)` (or via the matching `QD_<UPPERCASE_NAME>` environment variable). See [qd.init options](./init_options.md) for the full list and the environment-variable convention.

| Option | Default | Effect |
|--------|---------|--------|
| `cfg_optimization` | `True` | Turn the CFG optimization on/off. Turning it **off** makes compilation up to ~6× faster while costing ~1–5% of runtime speed - worth it when compile time dominates and the runtime delta is acceptable. |
| `opt_level` | `1` | `0` disables the two heavier passes (CSE and CFG optimization). |
| `advanced_optimization` | `True` | The fixed-point simplify loop above. Set to `False` to run just a single basic cleanup pass instead - much faster to compile, much less optimized. |
| `constant_folding` | `True` | Enables the constant-folding pass. |
| `fast_math` | `True` | Allows IEEE-relaxed floating-point rewrites (e.g. fusing a multiply and add). Covered in [qd.init options](./init_options.md#fast_math). |

For everyday use, leave them at their defaults - they are the best-supported and most reliable configuration. The most common deliberate change is `cfg_optimization=False` when iterating on a kernel whose compile time is in your way.

## Inspecting what the compiler did

These environment variables dump the IR so you can see the effect of each pass. Files are written to the directory set by the `debug_dump_path` option in `qd.init(...)` (default `/tmp/ir/`):

- `QD_DUMP_IR=1` - writes an IR snapshot at each major pipeline stage (after lowering, before/after each simplify, after offload).
- `QD_DUMP_SIMPLIFY=1` - writes an IR snapshot after every individual pass on every iteration of the simplify loop. Verbose, but it shows exactly which pass changed what.
- `QD_DUMP_CFG=1` - writes the control-flow graph itself. (This also forces the CFG pass to run over the whole kernel at once so the complete graph can be dumped.)

Setting `qd.init(print_ir=True)` prints the IR to the console at pipeline stages instead of writing files.

## Under the hood: per-task scoping

Once the kernel has been split into offloaded tasks, both CSE and the CFG optimization run over **one offloaded task's IR at a time**, never over the whole `qd.kernel` at once. This is both faster to analyze and safe: because each task is a separate device launch, a value held in a register in one task cannot survive into the next one, so there is never anything to deduplicate or forward across a task boundary. Anything written to global memory is treated as potentially read by a later task, so no store another task might need is dropped.

## Per-construct frontend compilation

The frontend stages above - the passes that turn your high-level kernel into offloaded tasks - can run either once over the whole kernel or, for eligible kernels, separately for each **top-level construct** (each independent top-level loop or serial run in your kernel). Compiling each construct in isolation is what will let a future cross-process cache reuse the unchanged constructs of a kernel you edited; today it produces the same offloaded tasks and the same results as the whole-kernel path, so the split is transparent.

Quadrants automatically falls back to the whole-kernel path whenever per-construct compilation would not be equivalent: [autodiff](autodiff.md) kernels, certain specialized kernels, and kernels where one construct's value depends on state another construct produced in a way that cannot be recomputed in isolation (for example a local variable that one top-level loop builds up over its iterations and another construct then reads, or a snapshot of a field that a later construct reads after an intervening write).

### Inspecting the split

You can see whether the split ran, and how many constructs it found, via the `per_offload_cache_observations` attribute on the kernel's compiled [primal](autodiff.md) (the forward kernel object, accessed as `._primal`):

```python
@qd.kernel
def my_kernel(x: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(x.shape[0]):
        x[i] += 1.0
    for i in range(x.shape[0]):
        x[i] += 2.0

my_kernel(some_array)

obs = my_kernel._primal.per_offload_cache_observations
print(obs.frontend_constructs_total)       # number of constructs the split compiled
print(obs.frontend_constructs_recompiled)  # how many were (re)compiled this time
print(obs.frontend_constructs_cache_hit)   # how many were reused (0 until the cross-process cache lands)
```

All three fields are `-1` when the split did not run for this compile - either because the kernel took the whole-kernel fallback above, or because the compiled kernel was served from a cache (the [offline cache](init_options.md#offline_cache) or [fastcache](fastcache.md)) so no frontend ran at all.
