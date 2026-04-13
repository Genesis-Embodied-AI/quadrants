# qd.precise

`qd.precise(expr)` marks a floating-point expression as IEEE-strict. Every binary and unary FP op inside the wrapped subtree is evaluated in source order with no reassociation, no FMA contraction, and no algebraic simplification, regardless of the module-level `fast_math` setting. It is the moral equivalent of the `precise` keyword in MSL / HLSL.

## Why

Quadrants compiles kernels with `fast_math=True` by default. Under that mode the compiler is free to:

- **reassociate** FP ops (e.g. `(a + b) + c -> a + (b + c)`)
- **contract** mul-then-add into FMA
- **substitute approximations** for `sqrt`, `sin`, `cos`, `log`, `1/x`
- **algebraically simplify** (e.g. `a - a -> 0`, `a / a -> 1`)

This silently destroys compensated-arithmetic primitives (Dekker / Kahan 2Sum, Veltkamp split, double-single accumulators) whose entire correctness rests on the fact that `(a - aa) + (b - bb)` is non-zero under IEEE arithmetic. The traditional workaround is to flip the global `fast_math=False` switch, but that pays the perf cost everywhere, even when only a handful of lines need IEEE semantics.

`qd.precise(expr)` is the per-expression opt-in: keep `fast_math=True` globally for speed, and wrap the expressions that must be IEEE-exact.

## Basic usage

```python
@qd.func
def fast_two_sum(a, b):
    s = qd.precise(a + b)
    e = qd.precise(b - (s - a))   # would fold to 0 under fast-math without precise
    return s, e
```

Any expression value can be wrapped. The wrapper returns the same expression with every reachable FP op tagged as precise; at codegen time the tagged ops opt out of the optimizations above.

## What gets protected

`qd.precise` walks the wrapped expression tree and tags:

- Every `BinaryOp` (`+`, `-`, `*`, `/`, `%`, comparisons, bit ops on FP types)
- Every `UnaryOp` (`neg`, `sqrt`, `sin`, `cos`, `log`, `exp`, `rsqrt`, casts, bit_cast, ...)

The walker descends through `BinaryOp`, `UnaryOp`, and `TernaryOp` (e.g. `qd.select`) nodes, so wrapping a composite expression protects the inner ops too:

```python
# All three FP ops below are tagged: the outer sqrt, the inner add, and the inner mul.
r = qd.precise(qd.sqrt(a * a + b * b))

# Ternary is traversed through; the two branches and the condition's inner ops are tagged.
r = qd.precise(qd.select(cond, a + b, a - b))
```

## Where the walker stops

`qd.precise` does not descend into:

- Loads (ndarray indexing, field access)
- Constants
- `qd.func` call sites
- Atomic ops

Semantics inside a `qd.func` body are governed by that body's own ops. If you want IEEE-strict behavior inside a called function, wrap the relevant ops inside the function's body, not at the call site:

```python
@qd.func
def dot_precise(a, b, c, d):
    # Wrap inside the body, not at the caller.
    return qd.precise(a * b + c * d)

@qd.kernel
def k(...):
    r = dot_precise(x, y, z, w)   # inner ops are already precise
```

## Interaction with fast_math

`qd.precise` is a per-op override. It takes effect whether `fast_math` is on or off:

| Setting | Non-precise op | `qd.precise` op |
|---|---|---|
| `fast_math=True` | reassoc / contract / simplify | IEEE-strict |
| `fast_math=False` | IEEE-strict | IEEE-strict (redundant but harmless) |

The recommended workflow is to leave `fast_math=True` globally for throughput and reach for `qd.precise` only in the handful of spots that need IEEE behavior.

## Backend coverage

| Backend | Reassoc / contraction / algebraic folds | Approximate transcendentals (`sin` / `cos` / `log`) |
|---|---|---|
| CPU | LLVM FMF cleared | libc `sinf` is already correctly rounded |
| CUDA | LLVM FMF cleared | libdevice `__nv_<fn>f` (non-fast) selected |
| AMDGPU | LLVM FMF cleared | `__ocml_<fn>` already correctly rounded |
| Vulkan / MoltenVK | SPIR-V `NoContraction` decoration | best-effort: driver stdlib default (typ. 1-2 ULP) |
| Metal | SPIR-V `NoContraction` decoration | best-effort: driver stdlib default (typ. 1-2 ULP) |

On SPIR-V backends, `NoContraction` is defined by the spec to apply to arithmetic instructions only; most consumers ignore it on the `OpExtInst` calls used for transcendentals. The decoration is still emitted (it is harmless and future-proofs against downstream toolchains that start honoring it), but correctness of `qd.precise(qd.sin(x))` on Metal / Vulkan currently relies on the driver's default (non-fast-math) transcendental implementation being accurate enough for your use case.

## Example: Dekker 2Sum

A textbook compensated addition that computes `s + e = a + b` exactly in f32:

```python
@qd.func
def two_sum(a, b):
    s = qd.precise(a + b)
    bb = qd.precise(s - a)
    aa = qd.precise(s - bb)
    e = qd.precise((a - aa) + (b - bb))
    return s, e
```

Without the `qd.precise` wrappers, under `fast_math=True` the compiler recognizes `(a - (s - (s - a))) + (b - (s - a))` as algebraically zero and folds `e` to `0`. The wrappers prevent that fold, and `s + e` reproduces `a + b` to full precision.

## Caveats

- `qd.precise` is a scalar primitive. Passing a `Vector` / `Matrix` will raise. Apply it to individual components instead, or refactor your expression to use scalar ops inside.
- The tag is a property of the expression value, not the use site. If you alias a subexpression and then wrap one alias, both uses get IEEE semantics.
- Caching: a kernel that uses `qd.precise` has a different offline-cache key than a structurally-identical kernel without it, so the two can coexist without collisions.
