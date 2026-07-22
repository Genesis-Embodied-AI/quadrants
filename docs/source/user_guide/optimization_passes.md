# Optimization passes

When you call a `@qd.kernel` function, Quadrants first translates the kernel into an internal form then rewrites it, step by step, into something equivalent but cheaper to run. "Equivalent" here means that they yield the exact same observable behavior and produce the exact same outputs [*1] for all valid inputs, regardless of how they are restructured. This page explains, at a high level, what those rewrite steps (the *optimization passes*) do, which ones cost compile time, and how to control and inspect them. You do not need to understand any of this to use Quadrants but it helps when you are trading compile time against runtime, or debugging a surprising result.

[*1] To within floating-point precision, not bit exact. In particular, fast maths optimizations are fairly approximate.

## Key terms

Let's start by defining terms that will be used throughout the page and are necessary in order to be able to understand it.

- **Kernel** - a function you decorate with `@qd.kernel`. It is the unit that Quadrants compiles and launches.
- **IR (intermediate representation)** - the compiler's internal version of your kernel: a flat list of small, explicitly-typed instructions, sitting between your Python source and the final machine code. Every pass reads and rewrites the IR; none of it is something you write by hand.
- **Pass** - one transformation step over the IR. An *optimization* pass rewrites the IR into a form that produces the **same results** but runs faster or uses less memory. (Some passes are not optimizations but *lowering* steps - they translate high-level constructs into lower-level ones; this page focuses on the optimizations.)
- **Basic block** - a straight-line run of instructions with no branches into or out of the middle. Control flow (`if`, loops) connects blocks together.
- **Offloaded task** - after the *offload* step, your kernel is split into one or more tasks, and each task becomes a single device launch: one GPU grid launch on a GPU backend, or one parallel loop on CPU. A simple kernel is usually one task; a kernel with, say, a short serial preamble followed by a big parallel loop becomes several tasks that run back to back.

## The compile pipeline at a glance

Compilation runs as a fixed sequence of stages. Optimization passes are interleaved with the lowering steps that gradually turn high-level IR into device code:

```
Python (AST)
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
backend codegen  (translate IR into device machine code, e.g. PTX/SASS via LLVM, or SPIR-V)
```

The "simplify" boxes are all the same routine (internally `full_simplify`), invoked at several points. Most of the interesting optimization work happens inside it. The optional autodiff step, run only when you ask Quadrants for gradients, is covered in [Automatic differentiation](./autodiff.md).

## The simplify loop

Each "simplify" stage runs a small bundle of optimizations **repeatedly, until the IR stops changing** (a fixed point). Running to a fixed point matters because the passes feed each other: folding a constant can expose dead code, deleting that code can expose a common subexpression, and so on.

In the order they run each round:

| Pass | What it does |
|------|--------------|
| Extract constant | Lifts constant values out of larger expressions into standalone constant instructions, so the passes below can recognize and reuse them. |
| Unreachable-code elimination | Removes branches that can never be taken (e.g. the body of an `if` whose condition is always false). |
| Binary-op / algebraic simplification | Applies arithmetic identities: `x * 1 → x`, `x + 0 → x`, `x * 2 → x + x`, and similar peephole rewrites. |
| Constant folding | Pre-computes expressions whose inputs are all known at compile time: `2 * 3 → 6`. |
| Dead-instruction elimination (**DIE**) | Drops instructions whose results are never used. Runs several times per round, after passes that tend to create newly-dead instructions. |
| Loop-invariant code motion (**LICM**) | Hoists a computation that produces the same value on every iteration out of the loop, so it runs once instead of N times. |
| Local simplify | Peephole cleanups within a block. |
| Common-subexpression elimination (**CSE**) | Finds an identical expression computed more than once and computes it a single time, reusing the result. |
| Control-flow-graph (**CFG**) optimization | Memory-focused optimizations that need a whole-task view; see the next section. Runs only once per stage (it is the most expensive pass). |

Two of these - CSE and CFG optimization - run only when `opt_level > 0` (the default is `1`).

**CSE is scoped per offloaded task.** Once the kernel has been split, common-subexpression elimination runs over one offloaded task's IR at a time rather than the whole kernel at once - the same per-task scoping described for [CFG optimization](#control-flow-graph-cfg-optimization) below, and for the same reason: each task is a separate device launch, so there is nothing to deduplicate across a task boundary. Scoping it this way keeps each run cheap and lets the per-task simplify stages run in parallel across the compiler's worker threads. One per-task CSE also runs once immediately after the offload split, before memory access is lowered, so that the memory-focused optimizations that follow see each global's reads and writes through a single shared access.

## Control-flow-graph (CFG) optimization

A **control-flow graph** is a map of your kernel's basic blocks together with the branches connecting them. It lets the compiler answer questions of the form "if execution reaches *here*, what must already have happened?" - which is exactly what is needed to optimize reads and writes to memory. Two such optimizations run on the CFG:

- **Store-to-load forwarding** - if a value is written to a location and then read again before anything overwrites it, the read is replaced with the value directly, skipping the round trip through memory.
- **Dead-store elimination** - if a write is overwritten before anyone reads it, the write is removed.

Building and analyzing the CFG is the most expensive optimization in the pipeline, which is why it runs at most once per simplify stage rather than every round.

**One CFG per offloaded task.** The CFG optimization is built and run separately for each offloaded task, over that task's IR alone - never over the whole `qd.kernel` at once. This is both faster to analyze and safe: because each task is a separate device launch, a value held in a register in one task cannot survive into the next one, so there is never anything to forward across a task boundary anyway. Anything written to global memory is treated as potentially read by a later task, so no store another task might need is dropped.

## Controlling the passes

All of these are fields of `CompileConfig`, so you set them at `qd.init(...)` (or via the matching `QD_<UPPERCASE_NAME>` environment variable). See [qd.init options](./init_options.md) for the full list and the environment-variable convention.

| Option | Default | Effect |
|--------|---------|--------|
| `cfg_optimization` | `True` | Turn the CFG optimization on/off. Turning it **off** makes compilation up to ~6× faster while costing ~1–5% of runtime speed - worth it when compile time dominates and the runtime delta is acceptable. |
| `opt_level` | `1` | `0` disables the two heavier passes (CSE and CFG optimization). |
| `advanced_optimization` | `True` | The fixed-point simplify loop above. Set to `False` to run just a single basic cleanup pass instead - much faster to compile, much less optimized. |
| `constant_folding` | `True` | Enables the constant-folding pass. |
| `fast_math` | `True` | Allows IEEE-relaxed floating-point rewrites (e.g. fusing a multiply and add). Covered in [qd.init options](./init_options.md#fast_math). |

For everyday use, leave them at their defaults - they are the best-supported and most reliable configuration. The most common deliberate change is `cfg_optimization=False` when iterating on a kernel whose compile time is in your way.

## Inspecting what the compiler did

These environment variables dump the IR so you can see the effect of each pass. Files are written to `debug_dump_path` (default `/tmp/ir/`):

- `QD_DUMP_IR=1` - writes an IR snapshot at each major pipeline stage (after lowering, before/after each simplify, after offload).
- `QD_DUMP_SIMPLIFY=1` - writes an IR snapshot after every individual pass on every iteration of the simplify loop. Verbose, but it shows exactly which pass changed what.
- `QD_DUMP_CFG=1` - writes the control-flow graph itself. (This also forces the CFG pass back onto the whole-kernel path so the complete graph can be dumped.)

Setting `qd.init(print_ir=True)` prints the IR to the console at pipeline stages instead of writing files.
