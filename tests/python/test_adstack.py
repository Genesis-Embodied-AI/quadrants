import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.misc import is_extension_supported

from tests import test_utils

# Diffable unary ops whose reverse formula accumulates a constant coefficient onto `stmt->operand` and therefore
# does not need per-iteration operand spilling on the adstack. Update this set and `unary_collections` in
# `quadrants/transforms/auto_diff.cpp` together when a new op lands in this class.
_KNOWN_LINEAR_UNARY_OPS = {"neg"}

_UNARY_OPS_PARAMS = [
    # Ops whose domain is all real values share a single `(step, offset)` pair that keeps the operand inside
    # the domain and, for `abs`/`sin`/`cos`, forces it to cross zero as `j` advances so the per-iteration
    # gradient sign actually varies. `abs` especially needs the sign-crossing: its derivative is piecewise-
    # constant, so a non-crossing operand would pass trivially. `exp`/`tanh` do not need the crossing (their
    # derivatives stay positive) but the same parameters work because their domain is all reals; they live
    # in this group on that basis alone.
    ("sin", 0.3, -0.4, 1e-4),
    ("cos", 0.3, -0.4, 1e-4),
    ("abs", 0.3, -0.4, 1e-4),
    ("tanh", 0.3, -0.4, 1e-4),
    ("exp", 0.3, -0.4, 1e-4),
    # Ops restricted to positive/subunit operands use a smaller step and zero offset to
    # stay inside their domain across every `x_val` and `n_iter` combination.
    # `tan` joins this positive-domain group because its singularity at pi/2 ~= 1.57 lies outside
    # the positive-path operand's reach for every `x_val` and `n_iter` combination.
    ("tan", 0.05, 0.0, 1e-4),
    ("log", 0.05, 0.0, 1e-4),
    ("sqrt", 0.05, 0.0, 1e-4),
    ("rsqrt", 0.05, 0.0, 1e-4),
    # asin/acos use a looser 1e-3 at f32 because native Vulkan asin/acos intrinsics on AMDGPU drift from the
    # CPU/PyTorch reference by ~1e-4 in single precision at n_iter=10. A per-iteration replay regression (the
    # one this test pins against) offsets the result by orders of magnitude, not parts per ten thousand, so
    # 1e-3 still catches it with plenty of margin. Other arches (CPU, Metal, CUDA) comfortably hit 1e-4 for
    # asin/acos too; the looser tolerance is just to keep the test green across drivers.
    ("asin", 0.05, 0.0, 1e-3),
    ("acos", 0.05, 0.0, 1e-3),
]


def _run_unary_loop_carried(qd_dtype, op_name, step, offset, x_val, n_iter, rel_tol):
    import torch

    qd_op = getattr(qd, op_name)
    torch_op = getattr(torch, op_name)
    torch_dtype = torch.float64 if qd_dtype == qd.f64 else torch.float32

    n = 4
    x = qd.field(qd_dtype, shape=n, needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in range(n_iter):
                a = x[i] + qd.cast(j, qd_dtype) * step + offset
                acc += qd_op(a)
            y[None] += acc

    for i in range(n):
        x[i] = x_val + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([x_val + i * 0.05 for i in range(n)], dtype=torch_dtype, requires_grad=True)
    y_t = torch.zeros((), dtype=torch_dtype)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch_dtype)
        for j in range(n_iter):
            a_t = x_t[i] + float(j) * step + offset
            acc_t = acc_t + torch_op(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    # Use `pytest.approx` directly rather than `test_utils.approx` so the caller's `rel_tol` is honoured as-is.
    # `test_utils.approx` floors `rel` to `max(rel, get_rel_eps())` which is >= 1e-6 on every backend (1e-4 on
    # Metal); that floor silently defeats the f64 variant's tight tolerance and makes it validate identical
    # precision to a loose f32 assertion. The backend-aware floor is not wanted here because the f64 variant
    # specifically exists to detect an f32-precision regression in an f64 backward pass.
    assert y[None] == pytest.approx(y_t.item(), rel=rel_tol)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=rel_tol)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("x_val", [0.001, 0.15, 0.26, 0.399])
@pytest.mark.parametrize("op_name,step,offset,tol", _UNARY_OPS_PARAMS)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_unary_loop_carried(op_name, step, offset, tol, x_val, n_iter):
    # Cross-check `d/dx sum_j op(x + j * step + offset)` against PyTorch autograd for a parametrized unary `op`.
    # Each op is sampled at interior values and at the edge of its domain: `x_val = 0.001` drives the positive-path
    # operand against 0 (log/sqrt gradients blow up there), `x_val = 0.399` drives it against 1 (asin/acos
    # gradient `1/sqrt(1 - x^2)` diverges), and `0.15` / `0.26` are interior samples. `0.26` (rather than `0.25`)
    # avoids `x[3] + offset = 0` at `j=0`, where `abs`'s derivative is undefined. The `(step, offset)` per op
    # keeps the operand inside the op's domain for every `x_val` and `n_iter`, while making the operand cross zero
    # for abs/sin/cos so the per-iteration sign actually varies.
    #
    # Extend the parametrize list below when a new unary op becomes differentiable through a dynamic loop: add
    # the op together with a `(step, offset)` pair that keeps the operand inside the op's domain. `n_iter = 1`
    # only covers the single-iteration path and will not on its own catch a regression in the multi-iteration
    # case; `abs` additionally needs the sign-crossing operand, otherwise its piecewise-constant derivative makes
    # the test pass trivially.
    #
    # Internal details: when reverse-mode AD walks a loop-variant unary op backwards it needs the exact forward
    # operand from each iteration to compute the gradient. If the op is not in the set the AD transform knows how
    # to back up per iteration, the operand falls back to a single slot overwritten each forward step, and the
    # reversed loop then reads the last-iteration value for every backward step - which at `n_iter >= 3` produces
    # a wrong gradient. That is the regression this test pins against: any unary op dropped from the supported set
    # causes the multi-iteration parametrize variants to fail.
    _run_unary_loop_carried(qd.f32, op_name, step, offset, x_val, n_iter, rel_tol=tol)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("x_val", [0.001, 0.15, 0.26, 0.399])
@pytest.mark.parametrize("op_name,step,offset,tol", _UNARY_OPS_PARAMS)
@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_unary_loop_carried_f64(op_name, step, offset, tol, x_val, n_iter):
    # f64 uses the same parametrize as f32 to keep ops in lockstep, but ignores the per-op f32 tolerance: f64
    # hits near-machine-precision on every backend, so a single tight global tolerance catches every drift.
    del tol
    _run_unary_loop_carried(qd.f64, op_name, step, offset, x_val, n_iter, rel_tol=1e-12)


@pytest.mark.needs_torch
@pytest.mark.parametrize(
    "op_name",
    # Pins MakeDual (forward mode) for every nonlinear unary op whose MakeAdjoint (reverse-mode) recompute audit was the
    # focus of PRs 1-4 of this chain: {tan, tanh, exp, log, sqrt, rsqrt}. Forward mode is argued safe-by-construction
    # because it runs in primal order, so the primal value is the current-iteration value by definition - there is no
    # stale-read hazard analogous to the reverse-mode one the audit targeted. This test pins that forward-order safety
    # argument per audited op so a future MakeDual refactor cannot regress it silently. The sign/absolute-value siblings
    # (abs, sin, cos, asin, acos) are covered by `test_adstack_unary_loop_carried` in the reverse-mode direction already
    # and were not part of the audit.
    ["tan", "tanh", "exp", "log", "sqrt", "rsqrt"],
)
@test_utils.test()
def test_unary_forward_mode_derivative(op_name):
    import torch

    N = 4
    x = qd.field(qd.f32, shape=N)
    loss = qd.field(qd.f32, shape=())
    qd.root.lazy_dual()

    qd_op = getattr(qd, op_name)
    torch_op = getattr(torch, op_name)

    @qd.kernel
    def kern():
        for i in x:
            loss[None] += qd_op(x[i])

    for i in range(N):
        x[i] = 0.1 * (i + 1)

    seed = [1.0] * N
    with qd.ad.FwdMode(loss=loss, param=x, seed=seed):
        kern()

    x_t = torch.tensor([0.1 * (i + 1) for i in range(N)], dtype=torch.float32, requires_grad=True)
    l_t = torch_op(x_t).sum()
    l_t.backward()
    expected = float(x_t.grad.sum().item())

    assert loss.dual[None] == pytest.approx(expected, rel=1e-5)


def test_unary_collections_audit():
    # Prevents drift between the Python unary op registry and the C++ `unary_collections` set in
    # `quadrants/transforms/auto_diff.cpp`. Every unary op whose `MakeAdjoint` branch accumulates onto
    # `stmt->operand` must be either in `unary_collections` (nonlinear: needs per-iteration operand spilling on
    # the adstack inside dynamic loops) or in the local `_KNOWN_LINEAR_UNARY_OPS` allow-list (reverse formula
    # uses a compile-time constant coefficient, so the single-slot spill path is correct for it). Forgetting to
    # classify a new diffable unary op falls back to the single-slot spill and produces silently wrong gradients
    # in dynamic loops.
    #
    # Four invariants are checked, all of them symmetric:
    #   (1) Every diffable-math op detected in MakeAdjoint is in `cpp_nonlinear` OR in `_KNOWN_LINEAR_UNARY_OPS`.
    #   (2) Every op in `cpp_nonlinear` has a matching diffable-math branch in MakeAdjoint.
    #   (3) Every op in `_KNOWN_LINEAR_UNARY_OPS` has a matching diffable-math branch in MakeAdjoint.
    #   (4) `cpp_nonlinear` and `_KNOWN_LINEAR_UNARY_OPS` are disjoint (an op cannot be both nonlinear and linear).
    src_path = pathlib.Path(__file__).resolve().parents[2] / "quadrants" / "transforms" / "auto_diff.cpp"
    src = src_path.read_text()

    cc_match = re.search(r"unary_collections\s*\{([^}]+)\}", src)
    assert cc_match is not None, "unary_collections not located in auto_diff.cpp"
    cpp_nonlinear = set(re.findall(r"UnaryOpType::(\w+)", cc_match.group(1)))

    make_adjoint_start = src.find("class MakeAdjoint")
    assert make_adjoint_start != -1, "class MakeAdjoint not located in auto_diff.cpp"
    adj_start = src.find("void visit(UnaryOpStmt *stmt) override", make_adjoint_start)
    assert adj_start != -1, "MakeAdjoint::visit(UnaryOpStmt*) not located in auto_diff.cpp"
    adj_end = src.find("void visit(", adj_start + 10)
    assert adj_end != -1, "next visitor method after MakeAdjoint::visit(UnaryOpStmt*) not found"
    adj_block = src[adj_start:adj_end]
    # Split the visitor's if/else-if chain into per-op segments, then classify each segment as "diffable math"
    # iff its body accumulates onto `stmt->operand`. Two accumulate entry points are recognised: the raw
    # `accumulate(stmt->operand, ...)` call (used by `neg` and `cast_value`) and the `acc(...)` lambda (used by
    # every nonlinear branch; `acc` wraps `accumulate_unary_operand_checked` which validates that the adjoint
    # formula does not read the forward `stmt` directly, and then delegates to `accumulate(stmt->operand, ...)`).
    # `cast_value` is excluded: its conditional accumulate is gated on real-typed elements and is orthogonal to
    # the `unary_collections` trade-off.
    #
    # Use `re.findall` rather than `re.search` so a future branch that ORs multiple `UnaryOpType::X` conditions
    # (e.g. `op_type == floor || op_type == ceil`) still classifies every op in the branch — capturing just the
    # first match would silently skip later ops in an OR chain.
    branches = re.split(r"\belse if\b", adj_block)
    diffable_math = set()
    for seg in branches:
        ops_in_seg = re.findall(r"UnaryOpType::(\w+)", seg)
        if not ops_in_seg:
            continue
        if "accumulate(stmt->operand" in seg or re.search(r"\bacc\(", seg):
            diffable_math.update(ops_in_seg)
    diffable_math.discard("cast_value")

    overlap = cpp_nonlinear & _KNOWN_LINEAR_UNARY_OPS
    assert not overlap, (
        f"Ops appear in BOTH `unary_collections` and `_KNOWN_LINEAR_UNARY_OPS`: {sorted(overlap)}. "
        f"An op must be classified as exactly one of (nonlinear, linear); otherwise the per-op audits below "
        f"silently cancel out."
    )

    missing = diffable_math - cpp_nonlinear - _KNOWN_LINEAR_UNARY_OPS
    assert not missing, (
        f"Diffable unary ops not classified as nonlinear or linear: {sorted(missing)}. "
        f"Add each one to `unary_collections` in quadrants/transforms/auto_diff.cpp (if nonlinear) or to "
        f"`_KNOWN_LINEAR_UNARY_OPS` in this file (if linear)."
    )
    stray_nonlinear = cpp_nonlinear - diffable_math
    assert not stray_nonlinear, (
        f"`unary_collections` lists ops with no matching accumulate branch (`accumulate(stmt->operand, ...)` or "
        f"`acc(...)`) in MakeAdjoint: {sorted(stray_nonlinear)}. Either remove them from `unary_collections` or "
        f"restore the MakeAdjoint branch."
    )
    stray_linear = _KNOWN_LINEAR_UNARY_OPS - diffable_math
    assert not stray_linear, (
        f"`_KNOWN_LINEAR_UNARY_OPS` lists ops with no matching accumulate branch in MakeAdjoint: "
        f"{sorted(stray_linear)}. Either remove them from `_KNOWN_LINEAR_UNARY_OPS` or restore the MakeAdjoint "
        f"branch; an entry here without a visitor means the op silently has no gradient implementation."
    )


@pytest.mark.xfail(
    reason="Reverse-mode NaN/Inf poisoning semantics is TBD. PyTorch propagates forward NaN into the backward graph "
    "(e.g. `log(-0.3).backward()` gives NaN), but Quadrants evaluates the reverse formula directly and returns a "
    "finite gradient. Either behaviour is defensible; picking a consistent rule is a separate design decision.",
    strict=True,
)
@pytest.mark.parametrize(
    "op_name,x_val",
    [
        # `(log, -0.3)` puts the operand outside the op's domain so the forward result is NaN. PyTorch backward
        # poisons the gradient with NaN; Quadrants currently evaluates `1 / operand = -3.333` verbatim and returns
        # that finite number, which this test documents as the expected-failure mode. The sqrt/asin/acos siblings
        # cannot be parametrized here under `xfail(strict=True)` because their reverse formulas divide by
        # `sqrt(<=0)` which is itself NaN, so Quadrants actually does return NaN there (test would XPASS and
        # `strict=True` treats an xpass as a failure).
        ("log", -0.3),
    ],
)
@test_utils.test()
def test_adstack_nan_propagation(op_name, x_val):
    # Pins the open question about reverse-mode NaN propagation. For each out-of-domain input the forward output
    # is NaN (both Quadrants and PyTorch agree). In reverse, PyTorch's backward pass propagates the NaN into the
    # gradient (`x.grad = NaN`) because a single NaN anywhere in the forward graph poisons every dependent grad.
    # Quadrants instead runs the analytical formula and returns a finite number. Which behaviour is "correct" is a
    # design call; the test is marked `xfail(strict=True)` so a deliberate change of semantics in either direction
    # forces a reviewer decision.
    qd_op = getattr(qd, op_name)

    x = qd.field(qd.f32, shape=(), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        y[None] = qd_op(x[None])

    x[None] = x_val
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[None] = 0.0
    compute.grad()

    assert math.isnan(x.grad[None])


@pytest.mark.xfail(
    reason="Reverse-mode NaN/Inf poisoning semantics is TBD (f64 variant). Same divergence as the f32 case: PyTorch "
    "propagates NaN in the backward graph; Quadrants runs `1 / operand` verbatim and returns a finite number.",
    strict=True,
)
@pytest.mark.parametrize("op_name,x_val", [("log", -0.3)])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64)
def test_adstack_nan_propagation_f64(op_name, x_val):
    # f64 counterpart of `test_adstack_nan_propagation`. The NaN-poisoning semantics question is dtype-independent
    # (PyTorch's backward graph propagates NaN regardless of dtype; Quadrants' reverse formula `1 / operand`
    # produces a finite number in both f32 and f64). Parametrizing over dtype ensures a future decision to change
    # semantics cannot accidentally fix one dtype and miss the other.
    qd_op = getattr(qd, op_name)

    x = qd.field(qd.f64, shape=(), needs_grad=True)
    y = qd.field(qd.f64, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        y[None] = qd_op(x[None])

    x[None] = x_val
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[None] = 0.0
    compute.grad()

    assert math.isnan(x.grad[None])


def _run_basic_gradient(qd_dtype, n_iter, rel_tol, approx=test_utils.approx, abs_tol=None):
    # Builds the kernel, runs forward + backward, and asserts a correct gradient. `approx` defaults to
    # `test_utils.approx` which is correct for f32 (its backend-specific floor kicks in); f64 callers must pass
    # `pytest.approx` to honor a tight `rel_tol=1e-14` that `test_utils.approx` would otherwise floor to 1e-6.
    # `abs_tol` is forwarded as `abs=` to the approx; f64 callers must pass `abs_tol=0` because pytest.approx's
    # default `abs=1e-12` would otherwise dominate for the expected magnitudes here (~0.6-0.95), making the
    # effective tolerance ~1e-12 absolute rather than 1e-14 relative and missing f32-narrowing regressions.
    # The adstack is structurally required here so the backward compiler can reverse the dynamic `range(n_iter)`
    # at all; the companion `test_adstack_basic_gradient_negative` pins that disabling the adstack raises
    # `QuadrantsCompilationError("non static range")`. Value-correctness of the per-iteration `v` spilled on the
    # adstack is NOT exercised by the linear body `v = v * 0.95 + 0.01` - see `test_adstack_basic_gradient`'s
    # docstring for the details.
    n = 4
    x = qd.field(qd_dtype, shape=n, needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter):
                v = v * 0.95 + 0.01
            y[None] += v

    x_vals = [0.1, 0.3, 0.5, 0.8]
    for i, v in enumerate(x_vals):
        x[i] = v
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute.grad()

    # `v = v * 0.95 + 0.01` iterated n_iter times gives v_final = 0.95**n_iter * x[i] + const, so
    # dv_final/dx[i] == 0.95**n_iter independent of x[i], and dy/dx[i] equals the same quantity.
    expected = 0.95**n_iter
    approx_kwargs = {"rel": rel_tol}
    if abs_tol is not None:
        approx_kwargs["abs"] = abs_tol
    for i in range(n):
        assert x.grad[i] == approx(expected, **approx_kwargs)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_basic_gradient(n_iter):
    # Smallest possible "does reverse-mode AD through a for-loop work at all" check. The kernel runs `n_iter`
    # iterations of `v = v * 0.95 + 0.01` per element and asserts that `dy/dx[i]` matches the analytical gradient
    # `0.95 ** n_iter` for every element.
    #
    # Internal details: the adstack is structurally required here so the backward compiler can reverse the
    # dynamic `range(n_iter)` at all - the companion `test_adstack_basic_gradient_negative` pins that disabling
    # the adstack raises `QuadrantsCompilationError` in exactly this kernel shape. Value-correctness of the
    # stored v, on the other hand, is NOT exercised: the loop body `v = v * 0.95 + 0.01` is linear, so the
    # backward chain `adj(v_prev) = 0.95 * adj(v_next)` only uses the compile-time constant 0.95 and never reads
    # v from the adstack. A broken push/load/pop that returned garbage for v would still produce the same exact
    # gradient. For push/load/pop value-correctness coverage, see `test_adstack_unary_loop_carried` (non-linear
    # unary ops in the loop body). `n_iter = 1` exercises the single-push adstack code path; `n_iter = 10`
    # exercises repeated push/pop under one forward invocation; multi-element coverage (n = 4) guards against
    # per-element accumulation bugs that a single-element variant would miss.
    _run_basic_gradient(qd.f32, n_iter=n_iter, rel_tol=1e-6)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_basic_gradient_f64(n_iter):
    # f64 counterpart of `test_adstack_basic_gradient`. Uses `pytest.approx` so the tight `rel_tol=1e-14` is honored;
    # `test_utils.approx` would floor it to `get_rel_eps()` (typically 1e-6) and silently pass an f32-precision
    # regression. `abs_tol=0` disables pytest.approx's default `abs=1e-12` floor, which would otherwise dominate for
    # the expected magnitudes `0.95**n_iter in [~0.6, 0.95]` and make the effective tolerance ~1e-12 absolute rather
    # than 1e-14 relative; that is still ~100x looser than f64 roundoff and would miss an f32-narrowing regression.
    _run_basic_gradient(qd.f64, n_iter=n_iter, rel_tol=1e-14, approx=pytest.approx, abs_tol=0)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@test_utils.test(ad_stack_experimental_enabled=False)
def test_adstack_basic_gradient_negative(n_iter):
    # Negative counterpart of `test_adstack_basic_gradient`: with the adstack disabled the backward compiler
    # cannot reverse a dynamic `range(n_iter)`, so `compute.grad()` raises `QuadrantsCompilationError("Cannot use
    # non static range in Backwards mode")` deterministically for every `n_iter`. Inlined rather than reusing
    # `_run_basic_gradient` because the shall-not-pass path never reaches the gradient assertion, so a shared
    # helper would carry a dead `rel_tol` argument down this branch.
    n = 4
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter):
                v = v * 0.95 + 0.01
            y[None] += v

    x_vals = [0.1, 0.3, 0.5, 0.8]
    for i, v in enumerate(x_vals):
        x[i] = v
    y[None] = 0.0
    compute()

    with pytest.raises(qd.QuadrantsCompilationError, match=r"non static range"):
        compute.grad()


@test_utils.test(
    require=qd.extension.adstack,
    ad_stack_size=4096,
)
def test_adstack_large_capacity_heap_backed():
    # Runs a backward pass with a deliberately huge (4096-slot) adstack and asserts the gradient comes out
    # correctly. With the old Function-scope storage the shader compile would fail because the per-thread private
    # memory footprint exceeded what Metal/MoltenVK accepts; with heap-backed storage the same kernel fits.
    #
    # Internal detail: the per-thread slice now lives in an SSBO sliced by `invoc_id * stride` instead of an
    # on-chip Array<f32, max_size>, so the per-thread shader footprint is O(1) regardless of `max_size`. Covers
    # the happy path of the heap-backed storage: allocation, push/pop with `count_var` still Function-scope,
    # LoadTop indexing into the slice, `AccAdjoint` back into the heap.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(128):
                y[None] += qd.sin(v)
                v = v + 1.0

    x[0] = 0.1
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    expected = sum(math.cos(0.1 + k) for k in range(128))
    assert x.grad[0] == pytest.approx(expected, rel=1e-4)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_mixed_f32_and_non_f32():
    # Runs a backward pass through a dynamic loop that carries both a float (`v`) and an integer counter (`j`), and
    # asserts the gradient on the float comes out correctly. Mixing the two types in one kernel exercises both
    # adstack storage paths at once.
    #
    # Internal detail: on SPIR-V, f32 adstacks live in BufferType::AdStackHeapFloat while i32/u1 adstacks share
    # BufferType::AdStackHeapInt (u1 reinterpreted as i32 via `ir_->cast(...)` at push/load). A codegen regression
    # in either path (e.g. pre-scan miscounting into the wrong stride, or the Push/Pop visitors routing to the
    # wrong heap_kind) would surface as a wrong gradient.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            j = 0
            for _ in range(5):
                y[None] += v * qd.cast(j + 1, qd.f32)
                j = j + 1
                v = v + 0.1

    x[0] = 1.0
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()
    # d y / d x[0] = sum_{k=0..4} (k+1): v at iter k is x[0] + 0.1*k, weight is k+1, so coefficient on x[0] is
    # sum_{k=0..4} (k+1) = 15. Five exactly-representable f32 accumulations so the result is exact up to a
    # handful of ULPs.
    assert x.grad[0] == pytest.approx(15.0, rel=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_many_non_f32_stacks_heap_backed():
    # Regression test for macOS Metal. Deeply nested reverse-mode kernels create one i32 adstack per dynamic loop
    # (to replay the counter) and one u1 adstack per data-dependent if (to replay the branch). Function-scope
    # storage makes the per-thread private-memory footprint grow linearly with the number of adstacks, and the
    # Apple MSL compiler rejects shaders past a few dozen slots; at ~100+ slots the pipeline creation XPC-times
    # out. The heap-backing path keeps Function-scope memory bounded so such kernels compile and run correctly.
    #
    # The test packs many sibling loops + ifs into a single kernel so the reverse pass allocates several i32 and u1
    # adstacks at once, then asserts the gradient is still correct. Correctness is the load-bearing assertion: a
    # mis-offset between adstacks on the shared int heap would alias one stack's primal slice onto another and
    # produce a wrong gradient. CPU backends store adstacks as regular memory and are unaffected; this primarily
    # guards the SPIR-V heap_int path.
    n = 3
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            # Six sibling dynamic loops + six data-dependent ifs inside each. Each dynamic for-loop creates an i32
            # adstack for its counter; each branch on a float-derived predicate creates a u1 adstack for the flag.
            # Written as explicit duplicated loops rather than a meta-for so the adstack count is unambiguous
            # (meta-loops with `qd.static` would unroll to one adstack per instantiation anyway, but expanding
            # manually keeps the intent auditable when this test is inspected during a failure).
            for a in range(4):
                if qd.cast(a, qd.f32) < v:
                    v = v * 1.1
                else:
                    v = v + 0.01
            for b in range(4):
                if v > qd.cast(b, qd.f32):
                    v = v * 1.05
                else:
                    v = v - 0.01
            for c in range(4):
                if v * 0.5 > qd.cast(c, qd.f32):
                    v = v + 0.1
                else:
                    v = v * 0.99
            for d in range(4):
                if v + qd.cast(d, qd.f32) > 0.0:
                    v = v * 1.02
                else:
                    v = v + 0.02
            for e in range(4):
                if v - qd.cast(e, qd.f32) > 0.0:
                    v = v * 0.97
                else:
                    v = v + 0.03
            for f in range(4):
                if v * 2.0 > qd.cast(f, qd.f32):
                    v = v + 0.05
                else:
                    v = v * 1.03
            y[None] += v

    x_vals = [1.0, 2.0, 3.0]
    for i in range(n):
        x[i] = x_vals[i]
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    # Finite-difference reference. A symbolic gradient would require tracking which branch each if took at every
    # iteration, which is exactly what the adstack replays for us; FD keeps the test oracle independent of the
    # code under test. `h = 1e-2` sits comfortably inside every branch-flip margin at the chosen `x_vals` (the
    # `v > cast(b, f32)` thresholds are integers and the local slope is O(1), so perturbation of 1e-2 never
    # crosses a branch) while keeping f32 cancellation roughly ULP-of-y / (2h) = ~1e-5 relative. Each branch is
    # affine in `v`, so the composite function is piecewise-affine in `x[i]`; FD central diff has zero
    # truncation error on that class and the only irreducible contribution is the rounding floor above.
    h = 1e-2
    for i in range(n):
        x[i] = x_vals[i] + h
        y[None] = 0.0
        compute()
        y_plus = y[None]
        x[i] = x_vals[i] - h
        y[None] = 0.0
        compute()
        y_minus = y[None]
        x[i] = x_vals[i]
        expected = (y_plus - y_minus) / (2.0 * h)
        assert x.grad[i] == pytest.approx(expected, rel=1e-3, abs=1e-4)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_rejects_unsupported_type():
    # Per-backend support matrix for loop-carried i8 in reverse-mode AD:
    #
    #   - SPIR-V (Metal / Vulkan) hard-rejects i8 loop-carried variables at codegen time: the adstack heap-backing
    #     only packs f32 and i32 (with u1 reinterpreted as i32); wider/other types have no home there. A
    #     Function-scope fallback is not offered because it is unusable for real workloads on Metal/MoltenVK
    #     (per-thread private-memory budget is exceeded) - silently falling back would paper over a
    #     correctness/perf cliff. The guard surfaces as a `RuntimeError` whose message names the supported type
    #     set. The match below pins the message so an accidental Function-scope fallback or a rename of the
    #     error string is caught.
    #   - LLVM (CPU / CUDA / AMDGPU) supports loop-carried i8 directly: the adstack heap there is byte-indexed
    #     and `AdStackAllocaStmt::size_in_bytes()` covers i8 at one byte per slot without any special case. The
    #     backward pass runs to completion and the forward output is what the test asserts.
    #
    # Skip on Vulkan drivers without shaderInt8: those reject i8 at the SPIR-V type gate (`spirv_ir_builder`
    # `QD_ERROR("Type i8 not supported")`) well before the adstack codegen guard fires, which would fail this
    # test with the wrong error message. Metal always reports spirv_has_int8=1 so it never skips here. The LLVM
    # backends do not consult `spirv_has_int8`, so the skip only applies on SPIR-V arches.
    arch = qd.lang.impl.get_runtime().prog.config().arch
    is_spirv = arch in (qd.metal, qd.vulkan)
    if is_spirv:
        caps = qd.lang.impl.get_runtime().prog.get_device_caps()
        if not caps.get(qd._lib.core.DeviceCapability.spirv_has_int8):
            pytest.skip("device lacks shaderInt8 - i8 is rejected at the SPIR-V type gate, not the adstack guard")

    # Internal detail: i8 is the probe type rather than f64 because Metal / MoltenVK rejects f64 at the
    # field-writer stage before the adstack codegen path is reached, whereas i8 is supported on both SPIR-V
    # backends (`spirv_has_int8`) and reaches the adstack guard. Any other type outside the SPIR-V supported set
    # (f16, i16, i64, u8, u16, u32, u64, f64) would do; i8 is the widest-supported of the SPIR-V-rejected
    # options and is also a plain supported type on every LLVM backend.
    x = qd.field(qd.i8)
    y = qd.field(qd.f32, shape=(), needs_grad=True)
    qd.root.dense(qd.i, 1).place(x)
    # needs_grad on the i8 field is meaningless (gradients do not flow through integer casts); the adstack
    # appears because the reverse pass replays the loop-carried i8 `v` value across the dynamic loop.

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(4):
                v = v + qd.cast(1, qd.i8)
            y[None] += qd.cast(v, qd.f32)

    x[0] = 0
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    if is_spirv:
        with pytest.raises(RuntimeError, match=r"f32, i32, and u1"):
            compute.grad()
    else:
        compute.grad()
        # Forward: v starts at 0, increments by 1 four times, writes 4 as f32 into y. No overflow expected.
        assert y[None] == pytest.approx(4.0, rel=1e-6)


def _overflowing_compute(n_elements=1, n_iter=64):
    # Shared kernel for the overflow tests. Builds `compute`, loads inputs, seeds the output gradient, and returns
    # `(compute, x, y)` so each test can drive the grad launch and read back assertions itself. `n_iter=64` + 2
    # adstack preamble pushes = 66 pushes, comfortably above the `ad_stack_size=32` override that every caller
    # places on the `@test_utils.test` decorator; `n_elements` controls how many threads run the overflowing loop
    # in parallel.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, n_elements).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter):
                y[None] += qd.sin(v)
                v = v + 1.0

    for i in range(n_elements):
        x[i] = 0.1 + 0.01 * i
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n_elements):
        x.grad[i] = 0.0
    return compute, x, y


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32)
def test_adstack_overflow_raises():
    # Runs a backward pass with a for-loop longer than the adstack can hold, and asserts the overflow surfaces as a
    # regular Python exception on the next `qd.sync()` - not a silent wrong gradient and not a process crash. This
    # is what users see when their differentiable kernel is too deep for the current `ad_stack_size`, and the error
    # message should tell them how to raise the capacity.
    #
    # Internal detail: both LLVM and SPIR-V defer the error to the next `qd.sync()` (same pattern as CUDA async
    # errors) so we do not pay a sync-per-launch. LLVM polls `runtime->adstack_overflow_flag` from
    # `LlvmProgramImpl::synchronize()` via `check_adstack_overflow()`; SPIR-V's gfx runtime raises via `QD_ERROR`
    # on sync. The test launches the overflowing grad kernel and calls `qd.sync()` inside the same `pytest.raises`
    # block so the deferred surfacing point is caught.
    compute, _, _ = _overflowing_compute()
    # On LLVM the runtime raises QuadrantsAssertionError (subclass of AssertionError) from
    # check_adstack_overflow; on SPIR-V the gfx runtime raises RuntimeError via QD_ERROR. We accept either,
    # matching only the message prefix.
    with pytest.raises((AssertionError, RuntimeError), match=r"[Aa]dstack overflow"):
        compute.grad()
        qd.sync()


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32)
def test_adstack_overflow_flag_resets_after_catch():
    # Once `check_adstack_overflow()` raises, the runtime must clear its overflow flag so a subsequent `qd.sync()`
    # (with no new overflowing grad launch in between) returns normally. Without the reset the user would see a
    # stale overflow exception every time they sync after the first one, which makes diagnosis and recovery
    # impossible.
    compute, _, _ = _overflowing_compute()
    with pytest.raises((AssertionError, RuntimeError), match=r"[Aa]dstack overflow"):
        compute.grad()
        qd.sync()
    # No new grad launch here - the flag must already be back to zero.
    qd.sync()


@test_utils.test(require=qd.extension.adstack, ad_stack_size=1024)
def test_adstack_large_capacity_resolves_overflow():
    # Same kernel shape as `test_adstack_overflow_raises`, but with `ad_stack_size=1024` explicitly passed to
    # `qd.init()`. Asserts that raising the capacity (rather than shrinking the loop) is a valid workaround and
    # that the backward pass runs to completion with a correct gradient. This is the remediation path the overflow
    # error message points users at.
    compute, x, _ = _overflowing_compute()
    compute.grad()
    qd.sync()

    # y += sin(v) iterated with v = x[0] + k for k = 0..63, so dy/dx[0] = sum_k cos(x[0] + k).
    expected = sum(math.cos(0.1 + k) for k in range(64))
    assert x.grad[0] == pytest.approx(expected, rel=1e-4)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=4096, offline_cache=False)
def test_adstack_heap_backed_exceeds_old_threadstack_budget():
    # Pins the LLVM heap-backed adstack: the cumulative per-thread adstack footprint may exceed the ~256 KB
    # secondary-thread stack budget that the old `create_entry_block_alloca` path enforced. Kernel has eight f32
    # loop-carried variables at `ad_stack_size=4096`, so the per-thread adstack total is `8 * (8 + 4096 * 8) =
    # 262,208 bytes` - 64 bytes past the old 262,144 byte ceiling. Before the heap-backing work the LLVM codegen
    # hard-errored this at compile time via `QD_ERROR_IF(ad_stack_fn_scope_bytes_ > kFnScopeAdStackBudgetBytes,
    # ...)`; SPIR-V compiled but overflowed the Metal / MoltenVK private-memory budget at shader-compile time.
    # Now both arches allocate the slice inside `runtime->adstack_heap_buffer` (LLVM) or the per-dispatch
    # SSBO (SPIR-V) and the kernel runs to completion with a correct gradient on every arch.
    #
    # `offline_cache=False` is load-bearing for the unfixed-tree check: with the cache on, a run that previously
    # succeeded against a heap-backed runtime would still produce the right gradient via the cached bitcode even
    # after the codegen changes are reverted. The test must force a fresh compile every run so the `QD_ERROR_IF`
    # on the unfixed tree actually fires and terminates the process.
    #
    # Internal details: each outer element `i` drives eight independent recurrences `a_k = a_k * 0.9 + x[i]` at
    # the same trip count (`n_iter`). The reverse pass pushes once for the initial value plus once per iteration,
    # so each adstack needs `n_iter + 2` slots; at `ad_stack_size=4096` we only use a few of those slots per
    # adstack but the slab still has to be allocated at full capacity. The gradient reduces to a closed form
    # `d/dx[i] sum_k (sum_j 0.9^j)` per recurrence, giving `dy/dx[i] = 8 * sum_j 0.9^j for j in 0..n_iter-1`.
    n = 4
    n_iter = 32
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            a3 = 0.0
            a4 = 0.0
            a5 = 0.0
            a6 = 0.0
            a7 = 0.0
            for _ in range(n_iter):
                a0 = a0 * 0.9 + x[i]
                a1 = a1 * 0.9 + x[i]
                a2 = a2 * 0.9 + x[i]
                a3 = a3 * 0.9 + x[i]
                a4 = a4 * 0.9 + x[i]
                a5 = a5 * 0.9 + x[i]
                a6 = a6 * 0.9 + x[i]
                a7 = a7 * 0.9 + x[i]
            y[None] += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7

    for i in range(n):
        x[i] = 0.1 * (i + 1)
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()
    qd.sync()

    geom = sum(0.9**j for j in range(n_iter))
    for i in range(n):
        assert x.grad[i] == pytest.approx(8.0 * geom, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32)
def test_adstack_overflow_multithreaded():
    # Multi-element field so several threads execute the overflowing grad body in parallel. Asserts the overflow
    # still surfaces as a single Python exception rather than deadlocking, crashing, or racing on the flag. Every
    # thread writes the same flag value (non-zero), so a race on the write is benign; this test pins that the
    # read side is also safe (one raise per sync regardless of how many threads flipped the bit).
    compute, _, _ = _overflowing_compute(n_elements=16)
    with pytest.raises((AssertionError, RuntimeError), match=r"[Aa]dstack overflow"):
        compute.grad()
        qd.sync()


def test_adstack_overflow_during_teardown_does_not_abort(tmp_path):
    # This test runs the kernel in a child process (not via `@test_utils.test`, which iterates arches), so it
    # cannot rely on the decorator's `require=qd.extension.adstack` skip. Guard manually: skip if the CPU backend
    # was not built with the adstack extension, matching what the sibling overflow tests get from the decorator.
    if not is_extension_supported(qd.cpu, qd.extension.adstack):
        pytest.skip("adstack extension not available on cpu")

    # If a user launches an overflowing grad kernel and never calls `qd.sync()` before the process exits, the
    # adstack-overflow flag is still set when Python interpreter teardown invokes `Program::finalize()`. The two
    # teardown syncs inside `Program::finalize()` must not re-raise a `QuadrantsAssertionError` into the
    # destructor path - doing so would terminate the process with `std::terminate()` instead of returning a clean
    # exit code. A subprocess runs the overflowing-grad kernel without calling `qd.sync()` at all and exits; this
    # test asserts that the child returns with exit code 0 rather than SIGABRT (-6) or any other non-zero code.
    #
    # Internal details: `Program::finalize()` invokes `program_impl_->pre_finalize()` before the two teardown
    # `synchronize()` calls. `LlvmProgramImpl::pre_finalize()` sets `finalizing_ = true` so
    # `LlvmProgramImpl::synchronize()` short-circuits `check_adstack_overflow()`. Note the flag must be set
    # *before* those syncs run - setting it only inside `LlvmProgramImpl::finalize()` (which is dispatched after
    # them) is too late. The subprocess is launched from a temp file because `python -c "<kernel>"` breaks
    # Quadrants' kernel source-inspect (`getsourcelines` cannot find the source of an inlined `-c` string); the
    # grad call is deliberately left unsynced so this is the teardown path, not the user-catch path.
    child_script = textwrap.dedent(
        """
        import quadrants as qd

        qd.init(arch=qd.cpu, ad_stack_experimental_enabled=True, ad_stack_size=32)

        x = qd.field(qd.f32)
        y = qd.field(qd.f32)
        qd.root.dense(qd.i, 1).place(x, x.grad)
        qd.root.place(y, y.grad)

        @qd.kernel
        def compute():
            for i in x:
                v = x[i]
                for _ in range(64):
                    y[None] += qd.sin(v)
                    v = v + 1.0

        x[0] = 0.1
        y[None] = 0.0
        compute()
        y.grad[None] = 1.0
        x.grad[0] = 0.0
        compute.grad()
        # Intentionally no qd.sync() and no try/except here: the adstack-overflow flag is left set when the
        # process exits, so teardown must swallow it via the `finalizing_` guard rather than re-raising.
        """
    )
    script_path = tmp_path / "overflow_teardown_child.py"
    script_path.write_text(child_script)
    # No `timeout=` on subprocess.run: pytest's own per-test timeout (`--timeout=...` in CI and locally) already
    # terminates the whole test if the child deadlocks. Adding a second timeout here would only duplicate that
    # safety net with a different failure mode (TimeoutExpired vs pytest-timeout's clean teardown).
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"child exited with {result.returncode}\n"
            f"stdout:\n{result.stdout.decode()}\n"
            f"stderr:\n{result.stderr.decode()}"
        )


@pytest.mark.parametrize("n_iter", [30, 100])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_near_capacity(n_iter):
    # Pins that a field-load-bounded reverse-mode loop sizes its adstack from the live field value at each
    # launch. Parametrized on both sides of the previous K+2=32 overflow boundary: `n_iter=30` would have
    # required 32 slots, `n_iter=100` would have required 102 slots. Both cases now run to completion with
    # the analytical gradient since the structural pre-pass captures the symbolic trip count and the host
    # launcher evaluates it per dispatch.
    #
    # Internal details: the trip count is loaded from a runtime field (`n_iter_fld`) rather than a Python
    # int constant so the `irpass::determine_ad_stack_size` structural pre-pass captures a `SizeExpr::FieldLoad`
    # (not a `Const`). The host evaluator in `LlvmRuntimeExecutor::publish_adstack_metadata` reads `n_iter_fld`
    # via `SNodeRwAccessorsBank`, recomputes the per-launch stride / offsets / max-sizes, and writes them into
    # the runtime metadata buffers that the LLVM codegen for `AdStack*` reads via `LLVMRuntime_get_adstack_*`.
    # Restricted to LLVM-CPU here: SPIR-V still bakes `max_size` as a codegen-time immediate (future work will
    # move SPIR-V onto the same per-launch metadata path, at which point the arch restriction can drop).
    # Companion to `test_adstack_overflow_raises` which still exercises the explicit `ad_stack_size=32` knob
    # (that path forces every adstack to exactly 32 slots and intentionally overflows).
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    n_iter_fld = qd.field(qd.i32, shape=())
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter_fld[None]):
                y[None] += qd.sin(v)
                v = v + 1.0

    x[0] = 0.1
    n_iter_fld[None] = n_iter
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0

    compute.grad()
    qd.sync()
    expected = sum(math.cos(0.1 + k) for k in range(n_iter))
    # `rel=1e-4` rather than 1e-5: n_iter=100 accumulates a ~1e-5 relative drift on AMD Vulkan (RADV) that the
    # tighter bound catches but that is within f32 accumulation noise for a 100-term oscillating cosine sum.
    assert x.grad[0] == test_utils.approx(expected, rel=1e-4)


def _run_sum_linear(
    qd_dtype, use_static_loop, use_varying_coeff, n_iter, rel_tol, approx=test_utils.approx, abs_tol=None
):
    n = 4
    x = qd.field(qd_dtype, shape=n, needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for a in qd.static(range(n_iter)) if qd.static(use_static_loop) else range(n_iter):
                if qd.static(use_varying_coeff):
                    y[None] += v * qd.cast(a + 1, qd_dtype)
                else:
                    y[None] += v

    x_vals = [0.1, 0.3, 0.5, 0.8]
    for i, v in enumerate(x_vals):
        x[i] = v
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    expected = sum(a + 1 for a in range(n_iter)) if use_varying_coeff else float(n_iter)
    approx_kwargs = {"rel": rel_tol}
    if abs_tol is not None:
        approx_kwargs["abs"] = abs_tol
    for i in range(n):
        assert x.grad[i] == approx(expected, **approx_kwargs)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("use_static_loop", [True, False])
@pytest.mark.parametrize("use_varying_coeff", [True, False])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_sum_linear(use_static_loop, use_varying_coeff, n_iter):
    # Linear accumulation `y = sum_j v * coeff_j` across all four combinations of (static-unrolled vs dynamic loop)
    # x (constant coefficient vs loop-index-varying coefficient), at three loop lengths. Replaces the earlier three
    # separate tests (`test_adstack_sum_fixed_coeff`, `test_adstack_sum_constant_coeffs`,
    # `test_adstack_sum_static_loop_correct`) with a single parametrized version so every branch of that truth
    # table is covered at each trip count.
    #
    # Internal details: this test deliberately does not mutate `v` inside the loop, so the reverse pass does not
    # require adstack replay of `v` to compute the right gradient - `v`'s per-iteration value is the same `x[i]`.
    # The point of this test is therefore not to stress the adstack (that is `test_adstack_basic_gradient`'s job)
    # but to prove that enabling the adstack extension does not silently regress linear reverse-mode AD for either
    # unrolled or dynamic loop shapes. No negative counterpart is included: for `use_static_loop=True` the inner
    # loop is unrolled and the backward kernel contains no dynamic range, so the adstack option does not change
    # the gradient; for `use_static_loop=False` disabling the adstack would raise `QuadrantsCompilationError`
    # (same compile-time rejection covered by `test_adstack_basic_gradient_negative`), which is out of scope here.
    _run_sum_linear(qd.f32, use_static_loop, use_varying_coeff, n_iter, rel_tol=1e-6)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("use_static_loop", [True, False])
@pytest.mark.parametrize("use_varying_coeff", [True, False])
@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_sum_linear_f64(use_static_loop, use_varying_coeff, n_iter):
    # f64 counterpart uses `pytest.approx` so the tight rel_tol is not floored by `test_utils.approx`, and
    # `abs_tol=0` disables pytest.approx's default `abs=1e-12` floor so the 1e-14 relative tolerance is actually
    # honored. Note: the expected gradients here are integers in `{1, 3, 6, 10, 55}` which are exactly representable
    # in both f32 and f64, so this test cannot catch an f32-narrowing regression of the backward pass regardless of
    # tolerance - the value coverage comes from `test_adstack_basic_gradient_f64` whose non-integer expected values
    # genuinely exercise the f64 precision floor. Kept here to preserve the shape coverage (static/dynamic inner
    # loop x varying/constant coefficient) at f64 since a future type-narrowing bug that depends on shape might
    # still surface through compile-time / structural differences between f32 and f64 codegen paths.
    _run_sum_linear(qd.f64, use_static_loop, use_varying_coeff, n_iter, rel_tol=1e-14, approx=pytest.approx, abs_tol=0)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_runtime_if_wrapping_loop_with_carried_var():
    # Pins the MakeAdjoint::visit(RangeForStmt) current_block-restore behaviour. Reverse-mode AD through a dynamic
    # for-loop with a loop-carried float, nested inside a runtime-guarded `if`, must emit its post-loop reverse
    # stmts (stack-underflow cleanup and the `accumulate x.grad[i]` on the initial-value stmt) as siblings *after*
    # the reversed for-loop, not inside its body. Without the save/restore of `current_block` around the per-stmt
    # iteration, these stmts land inside the body and the gradient is silently wrong
    # (e.g. `1 + 1 + 1 = 3.0` instead of `1 + 0.95 + 0.95**2 = 2.8525`). Compile-time-true `if` branches do not
    # trigger the pattern because simplify folds them away before reverse-mode is applied.
    #
    # This shape is common in user code: a reverse-mode kernel reads fields through runtime index-range guards
    # around dynamic-loop bodies that carry floats across iterations. Without the save/restore this produces NaN
    # on Metal and zero-valued gradients on CPU.
    n_iter = 4
    n_active = 3
    n_max = n_active + 2  # outer loop iterates past n_active; body guarded by `i < n_active`.

    x = qd.field(qd.f32, shape=n_max, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)
    n_arr = qd.field(qd.i32, shape=())

    @qd.kernel
    def compute():
        for i in range(n_max):
            if i < n_arr[None]:
                v = x[i]
                acc = 0.0
                for _ in range(n_iter):
                    acc = acc + v
                    v = v * 0.95 + 0.01
                y[None] += acc

    for i in range(n_max):
        x[i] = 1.0 + 0.1 * i
    n_arr[None] = n_active
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n_max):
        x.grad[i] = 0.0
    compute.grad()

    expected = sum(0.95**k for k in range(n_iter))
    for i in range(n_active):
        assert x.grad[i] == pytest.approx(expected, rel=1e-6)
    for i in range(n_active, n_max):
        assert x.grad[i] == 0.0


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_if_cond_snapshot_through_dynamic_for():
    # Pins MakeAdjoint::visit(IfStmt) cond-snapshot behaviour. Reverse-mode AD through a runtime `if` whose
    # cond is a stack-backed alloca load, nested inside a dynamic-range for-loop, must evaluate the reverse
    # if-cond at the forward-time value - not by re-running `stack_load_top` at reverse-time. Without the
    # snapshot, BackupSSA's cross-block clone of `if_stmt->cond` re-reads the cond's backing adstack at
    # reverse time, where the top has advanced due to the short-circuit push emitted inside the forward if
    # body; the reverse branch flips, the accumulation never runs, and gradients silently come out all-zero.
    #
    # Internal details: `cfg_optimization=False` is load-bearing - with it enabled, store-to-load forwarding
    # collapses the tautological `if (stack_load_top after push_true) { ... }` short-circuit wrapper before
    # MakeAdjoint sees it, and the multi-push-per-iter pattern that drives the bug vanishes. The outer
    # `qd.ndrange` (not a plain Python `for`) is required: the plain-`for` variant keeps the outer index as
    # a direct loop index rather than a cast alloca, and the enclosing-loop cast is what forces the inner
    # cond alloca to be stack-promoted. `qd.cast(i_b, qd.i32)` is also load-bearing for the same reason -
    # without it the cond alloca stays unpromoted and the bug does not surface. The inner-loop bound `n[None]`
    # pulled from a field, not a Python literal, forces the inner for to be compiled as a runtime-dynamic
    # range rather than statically unrolled; the buggy stack clone cannot arise on the fully-unrolled path.
    # The min shape is 2 iterations (`i == 0` writes a vector from `x`; else reads a constant `c[i]`); the
    # expected grad `[1, 1, 1, 0]` makes a flipped reverse branch visible immediately because flipping drops
    # the `x[0..2]` accumulation entirely and the whole grad comes out `[0, 0, 0, 0]`.
    vec3 = qd.types.vector(3, qd.f32)

    outputs = qd.field(dtype=vec3, shape=(2, 1), needs_grad=True)
    constants = qd.field(dtype=vec3, shape=(2,))
    n_iter = qd.field(dtype=qd.i32, shape=())
    inputs = qd.field(dtype=qd.f32, shape=(4, 1), needs_grad=True)

    @qd.kernel
    def my_kernel():
        for i_batch in qd.ndrange(outputs.shape[1]):
            i_batch = qd.cast(i_batch, qd.i32)
            for i_inner in range(n_iter[None]):
                if i_inner == 0:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[0, i_batch], inputs[1, i_batch], inputs[2, i_batch]], dt=qd.f32
                    )
                else:
                    outputs[i_inner, i_batch] = constants[i_inner]

    outputs.grad.from_numpy(np.ones((2, 1, 3), dtype=np.float32))
    n_iter[None] = 2

    my_kernel.grad()

    grad = inputs.grad.to_numpy().squeeze()
    assert grad[0] == 1.0
    assert grad[1] == 1.0
    assert grad[2] == 1.0
    assert grad[3] == 0.0


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False, ad_stack_size=32)
def test_adstack_if_cond_snapshot_adaptive_sizing():
    # Reverse-mode AD through an `if/elif/elif/else` chain inside a dynamic for-loop must compile
    # and produce the correct per-input gradient. The companion test above pins the silently-zero
    # gradient bug on a single `if/else`; this one pins a compile-time crash that only surfaces
    # once the chain has several arms with distinct stack-backed conds.
    #
    # Internal details: every arm of the chain lowers to its own IfStmt whose cond is an
    # AdStackLoadTopStmt, so MakeAdjoint emits one snapshot adstack per arm. The crash is not
    # about gradient values - it is a codegen abort ("Adaptive autodiff stack's size should have
    # been determined") that fires when the adaptive-sizing pass leaves any one of those snapshot
    # stacks with max_size still zero. Four arms is the smallest shape where that pass's
    # Bellman-Ford walk reliably fails to size at least one snapshot stack; a single `if/else` is
    # always sized successfully. `ad_stack_size=32` is load-bearing - the default `ad_stack_size=0`
    # (adaptive) puts the cond stack itself through the same sizing pass, which incidentally sizes
    # the snapshot stacks too; only when the cond stack is stamped with a fixed size and skipped by
    # the pass does the snapshot-stack-only walk expose the miscount. Every caveat from the companion
    # test about `cfg_optimization=False`, the `qd.ndrange`/`qd.cast` pair, and the runtime-dynamic
    # inner-range bound still applies - without them the snapshot adstack is never created at all
    # and the crash cannot arise.
    vec3 = qd.types.vector(3, qd.f32)

    outputs = qd.field(dtype=vec3, shape=(4, 1), needs_grad=True)
    constants = qd.field(dtype=vec3, shape=(4,))
    n_iter = qd.field(dtype=qd.i32, shape=())
    inputs = qd.field(dtype=qd.f32, shape=(4, 1), needs_grad=True)

    @qd.kernel
    def my_kernel():
        for i_batch in qd.ndrange(outputs.shape[1]):
            i_batch = qd.cast(i_batch, qd.i32)
            for i_inner in range(n_iter[None]):
                if i_inner == 0:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[0, i_batch], inputs[1, i_batch], inputs[2, i_batch]], dt=qd.f32
                    )
                elif i_inner == 1:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[1, i_batch], inputs[2, i_batch], inputs[3, i_batch]], dt=qd.f32
                    )
                elif i_inner == 2:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[0, i_batch], inputs[2, i_batch], inputs[3, i_batch]], dt=qd.f32
                    )
                else:
                    outputs[i_inner, i_batch] = constants[i_inner]

    outputs.grad.from_numpy(np.ones((4, 1, 3), dtype=np.float32))
    n_iter[None] = 4

    my_kernel.grad()

    grad = inputs.grad.to_numpy().squeeze()
    assert grad[0] == 2.0
    assert grad[1] == 2.0
    assert grad[2] == 3.0
    assert grad[3] == 2.0


@test_utils.test(require=qd.extension.adstack)
def test_adstack_sibling_for_loops_reverse_order():
    # Reverse-mode AD through two sibling dynamic for-loops in the same container, where the second loop reads a global
    # that the first loop wrote, must execute the second loop's reverse before the first loop's reverse. Otherwise the
    # first-loop reverse reads an uninitialised (zero) adjoint of the intermediate global, clears it, and the gradient
    # the second-loop reverse later populates propagates nowhere. Left unfixed, `inputs.grad` comes out all-zeros
    # despite a well-defined non-zero analytic derivative.
    #
    # Internal details: MakeAdjoint runs per-IB, and for this shape both sibling fors' bodies are their own IBs
    # (innermost loops with global ops). The reverse-mode transform therefore never visits the container block that
    # holds them, so nothing flips their order. ReverseOuterLoops flips each loop's `reversed` iteration direction but
    # historically left sibling order alone; the fix adds a pairwise swap of sibling for-loops inside every non-IB
    # container block the pass walks through. Non-loop statements (range-bound loads, alloca, etc.) stay at their
    # original positions so SSA operands still dominate both swapped fors. The outer `for _ in range(1)` dummy is the
    # smallest shape that places the two siblings inside a non-IB container (the frontend rejects a bare sequence of
    # top-level for-loops as "mixed usage of for-loops and statements without looping"); `n[None]` from a field forces
    # the inner ranges to be dynamic so the bug manifests (static-unrolled ranges go through a different path that
    # already works).
    size = 3
    n = qd.field(qd.i32, shape=())

    inputs = qd.field(qd.f32, shape=size, needs_grad=True)
    weights = qd.field(qd.f32, shape=size)
    scratch = qd.field(qd.f32, shape=size, needs_grad=True)
    outputs = qd.field(qd.f32, shape=size, needs_grad=True)

    @qd.kernel
    def my_kernel():
        for _ in range(1):
            for i in range(n[None]):
                scratch[i] = inputs[i]
            for i in range(n[None]):
                outputs[i] = scratch[i] * weights[i]

    n[None] = size
    for i in range(size):
        inputs[i] = float(i + 1)
        weights[i] = float(i + 1) * 0.5

    my_kernel()
    outputs.grad.from_numpy(np.ones(size, dtype=np.float32))
    my_kernel.grad()

    grad = inputs.grad.to_numpy()
    for i in range(size):
        assert grad[i] == float(i + 1) * 0.5


@pytest.mark.parametrize("n", [3, 5])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_for_bound_is_enclosing_loop_index(n):
    # Reverse-mode AD must handle a triangular-nested loop, where an inner for-loop's upper bound is itself an
    # enclosing loop's index (a per-iteration value, not a loop invariant). The kernel mirrors a classic
    # lower-triangular sweep like an in-place Cholesky update. `w` additionally accumulates a linear function of
    # the outer counter alongside the inner sum, so both a per-iteration value and its reverse-mode gradient flow
    # are exercised. `x` entries are distinct (0.1, 0.2, ...) so the inner for's contribution to each `x.grad[k]`
    # differs per iteration; a uniform `x` would collapse several contributions into the same number and would
    # let a reverse pass over the wrong iteration range still match the expected sum.
    #
    # Internal details: two pieces of the autodiff pipeline are load-bearing together.
    # (1) AdStackAllocaJudger::visit(RangeForStmt) recognises allocas whose LocalLoad feeds a RangeForStmt begin or
    #     end and promotes them to an adstack, so each forward iteration pushes the current bound and each reverse
    #     iteration pops the matching one. This is the only promotion path for a pure loop-counter alloca (LOAD-only
    #     into the inner bound, no LOAD-then-STORE cycle), so the `local_loaded_` short-circuit in visit(LocalStoreStmt)
    #     cannot cover it. The comparison has to resolve the LocalLoad chain (compare `ll->src` to the backup, not
    #     the operand itself or the cursor) because `begin`/`end` are always value-producing stmts, not the alloca,
    #     and the mutable cursor only names the first matching load's instance.
    # (2) BackupSSA::visit(RangeForStmt) spills cross-block operands on the for-stmt itself (not just its body), so
    #     the reverse clone's `end` operand - pointing at the forward-scope AdStackLoadTop - is rematerialised via
    #     op->clone() inside the reverse scope (the existing AdStackLoadTopStmt branch in generic_visit).
    # Without either piece, this kernel fails: IR-verify reports `RangeForStmt cannot have operand LocalLoadStmt`
    # without (2); LLVM codegen hits "Instruction does not dominate all uses" without (2) after (1); or the reverse
    # inner-loop iteration count is the last forward value without (1), silently corrupting gradients for the
    # earliest inner indices (those visited most often across outer iterations) - bigger `n` exposes more affected
    # indices, which is why the parametrize sweep catches a regression that a single small `n` can alias past.
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)
    w = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            for j in range(n):
                if i < n and j < i + 1:
                    s = 0.0
                    for k in range(j):
                        s = s + x[k] * x[k]
                    w[None] += qd.cast(j, qd.f32) * x[i]
                    y[None] += qd.sqrt(qd.math.clamp(x[i] + 1.0 - s, 0.01, 1e9))

    x_vals = [0.1 * (k + 1) for k in range(n)]
    for k in range(n):
        x[k] = x_vals[k]
    y[None] = 0.0
    w[None] = 0.0
    compute()
    y.grad[None] = 1.0
    w.grad[None] = 1.0
    for k in range(n):
        x.grad[k] = 0.0
    compute.grad()

    expected = [0.0] * n
    for i in range(n):
        for j in range(i + 1):
            s = sum(x_vals[k] * x_vals[k] for k in range(j))
            arg = x_vals[i] + 1.0 - s
            d_arg = 1.0 / (2.0 * arg**0.5)
            expected[i] += d_arg
            for k in range(j):
                expected[k] += d_arg * (-2.0 * x_vals[k])
            # w_contribution = cast(j) * x[i]: d/dx[i] += j
            expected[i] += float(j)
    for k in range(n):
        assert x.grad[k] == test_utils.approx(expected[k], rel=1e-5)


def test_adstack_vector_subscript_selfop_no_warnings(tmp_path):
    # Exercises reverse-mode differentiation of a common Vector pattern: a small Vector is built with a literal
    # initializer, one slot is overwritten by static subscript, and the whole Vector is then used in an in-place
    # op whose right-hand side reads the same Vector (e.g. a self-normalization `q *= q.norm_sqr()`). The test
    # guards that the backward compile completes without emitting any "Loading variable N before anything is
    # stored to it" UD-chain warnings, which is the signature of a reverse-grad kernel that would read
    # uninitialized adjoint slots at runtime.
    #
    # Internal details: the pattern lives on the experimental adstack path (`ad_stack_experimental_enabled=True`),
    # where `ReplaceLocalVarWithStacks` lowers every stack-backed static subscript store into a full-tensor push.
    # Devs modifying `ReplaceLocalVarWithStacks::visit(LocalStoreStmt)` or the reach-in analysis in
    # `ir/control_flow_graph.cpp` must preserve the invariant that every non-target slot in that rebuilt tensor
    # has a reaching definition the store-to-load-forwarding walker can see - otherwise the warning fires here.
    # The warning is emitted by the C++ logger during `kernel.grad()` compilation, so the check runs in a
    # subprocess to capture stderr reliably regardless of log-sink state in the parent test session.
    child_script = textwrap.dedent(
        """
        import quadrants as qd

        qd.init(arch=qd.cpu, ad_stack_experimental_enabled=True, ad_stack_size=32)


        @qd.func
        def f(x):
            q = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=qd.f32)
            q[1] = x
            q *= q.norm_sqr()
            return q


        x = qd.field(qd.f32, shape=(), needs_grad=True)
        y = qd.field(qd.f32, shape=(), needs_grad=True)


        @qd.kernel
        def k():
            q = f(x[None])
            y[None] = q[0] + q[1] + q[2] + q[3]


        x[None] = 1.5
        k()
        y.grad[None] = 1.0
        k.grad()
        """
    )
    script_path = tmp_path / "vector_subscript_selfop.py"
    script_path.write_text(child_script)
    env_no_cache = {"QD_OFFLINE_CACHE": "0"}
    env = {**os.environ, **env_no_cache}
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, check=True, env=env)
    stderr = result.stderr.decode()
    assert "Loading variable" not in stderr, (
        "reverse-mode AD emitted 'Loading variable N before anything is stored to it' warnings for a Vector "
        "subscript-assign + self-referencing in-place op pattern; stderr was:\n" + stderr
    )


@test_utils.test(require=qd.extension.adstack, ad_stack_size=4096)
def test_adstack_ndrange_over_ndarray_shape_does_not_oversize_heap():
    # Regression test: a grad kernel whose range is derived at launch time from an ndarray shape (e.g.
    # `qd.ndrange(arr.shape[0], arr.shape[1])`) used to inherit `advisory_total_num_threads =
    # kMaxNumThreadsGridStrideLoop = 131072` from the SPIR-V codegen fallback, and the runtime sized the
    # per-dispatch adstack heap as `131072 * per_thread_stride * sizeof(float)`. For this kernel's ten
    # loop-carried f32 variables at `ad_stack_size=4096`, that is `131072 * 10 * 2 * 4096 * 4 bytes = 40
    # GiB`. Apple Silicon's `MTLDevice.maxBufferLength` is ~75% of unified memory (e.g. ~28 GiB on an M4 Max
    # with 48 GiB unified, smaller on lower-end configs), so the allocation failed. Before the RHI layer
    # checked for nil, that failure was silently wrapped as `RhiResult::success` with a nil MTLBuffer; every
    # downstream `setBuffer:atIndex:2` bound nil, writes dropped and reads returned 0, and the backward
    # produced NaN gradients without any error. With the fix, the codegen records the shape-lookup product
    # backing the runtime-resolved `end_stmt` into `RangeForAttributes::end_shape_product`, the runtime
    # `launch_kernel` reads each shape from the `LaunchContextBuilder` args buffer and tightens
    # `advisory_total_num_threads` to `actual_iter_count = rows * cols = 6`, so only ~240 KB of adstack heap
    # is allocated and the gradient is correct.
    #
    # Internal details: `ad_stack_size=4096` + ten loop-carried f32 variables is tuned so that the pre-fix
    # 131072-thread allocation request crosses the smallest plausible Apple Silicon `maxBufferLength` - the
    # test would otherwise silently pass on hardware with large unified memory. The original oversize symptom
    # only surfaced on the SPIR-V heap-backed adstack path whose per-dispatch sizing depends on the advisory
    # thread count; the LLVM path sizes the adstack slab once per runtime against `num_cpu_threads` and cannot
    # exhibit the same nil-buffer regression. The test still runs on every backend so the finite-difference
    # cross-check catches a future regression in the grad computation regardless of which path it lives in.
    rows, cols = 2, 3

    @qd.kernel
    def compute(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(arr.shape[0], arr.shape[1]):
            v0 = arr[i, j]
            v1 = arr[i, j] + 0.1
            v2 = arr[i, j] + 0.2
            v3 = arr[i, j] + 0.3
            v4 = arr[i, j] + 0.4
            v5 = arr[i, j] + 0.5
            v6 = arr[i, j] + 0.6
            v7 = arr[i, j] + 0.7
            v8 = arr[i, j] + 0.8
            v9 = arr[i, j] + 0.9
            for _ in range(3):
                v0 = v0 * 1.01 + 0.01
                v1 = v1 * 1.02 + 0.02
                v2 = v2 * 1.03 + 0.03
                v3 = v3 * 1.04 + 0.04
                v4 = v4 * 1.05 + 0.05
                v5 = v5 * 1.06 + 0.06
                v6 = v6 * 1.07 + 0.07
                v7 = v7 * 1.08 + 0.08
                v8 = v8 * 1.09 + 0.09
                v9 = v9 * 1.10 + 0.10
            out[0] += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9

    arr_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    arr = qd.ndarray(qd.f32, shape=(rows, cols), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    arr.from_numpy(arr_np)
    arr.grad.from_numpy(np.zeros_like(arr_np))
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))

    compute(arr, out)
    compute.grad(arr, out)
    qd.sync()

    got_grad = arr.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"ndrange-over-shape grad returned NaN: {got_grad}"

    # Analytic oracle. The kernel is affine in `arr[i, j]` (each `v_k` is `v_k * c_k + d_k` for three
    # iterations, so `d(v_k_final) / d(arr[i, j]) = c_k^3`), and `out[0]` sums all ten recurrences, so the
    # closed-form gradient per cell is `sum_k c_k^3`. Independent of the backward emission so a
    # wrong-but-non-NaN gradient (the failure mode when the adstack heap was bound to Metal's nil-fallback
    # and reads came back as zero) still trips the assertion; tolerance bounded by f32 accumulation roundoff
    # only, not finite-difference cancellation.
    coeffs = np.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10], dtype=np.float64)
    expected_per_cell = float((coeffs**3).sum())
    # `rtol=1e-4` rather than tight-to-backward-roundoff because on AMD Vulkan (RADV) the adjoint accumulation
    # through ten loop-carried recurrences drifts by a few parts in 1e5 relative to the analytic value; the
    # tighter bound catches the drift without a corresponding correctness signal. The regression this test
    # guards against (nil device buffer -> zero read -> NaN adjoint) trips any tolerance at all.
    np.testing.assert_allclose(got_grad, np.full_like(arr_np, expected_per_cell), rtol=1e-4, atol=0)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_bounded_inner_loop_sized_by_structural_prepass():
    # Pins that reverse-mode AD on SPIR-V backends through a statically bounded inner `range(N)` sizes the
    # adstack from the product of enclosing `RangeForStmt` trip counts via the structural pre-pass, not
    # from any compile-time fallback. There is no `default_ad_stack_size` anymore: any alloca the pre-pass
    # cannot bound is a hard compile error, so the only way this test passes is through the structural
    # walk correctly folding the constant trip counts.
    #
    # Internal details: `irpass::determine_ad_stack_size` runs a structural pre-pass that walks each adaptive
    # `AdStackAllocaStmt`'s push sites and computes `max_size` from the product of enclosing `RangeForStmt`
    # trip counts when every enclosing range has a constant integer begin/end (folded through `BinaryOpStmt`).
    # Runs on every backend: on SPIR-V this is the only bound-derivation path; on LLVM the inner-range bounds
    # are rewritten through `LoopIndexStmt` in a shape the structural analyzer does not fold, and the kernel
    # routes through the symbolic-tree runtime-evaluator path instead - the gradient assertion catches a
    # regression in either path.
    x = qd.field(qd.f32, shape=(1,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(5):
                v = qd.sin(v) + 0.1
            y[None] += v

    n_iter = 5

    x[0] = 0.3
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    v = 0.3
    dv_dx = 1.0
    for _ in range(n_iter):
        dv_dx = math.cos(v) * dv_dx
        v = math.sin(v) + 0.1
    # `pytest.approx` rather than `test_utils.approx` so the tight f32 tolerance isn't floored to the
    # backend `get_rel_eps()` - a wrong gradient from an unresolved adstack would drift by orders of
    # magnitude, not parts per million, but a tight bound still catches subtler regressions.
    assert x.grad[0] == pytest.approx(dv_dx, rel=1e-6)


@test_utils.test(require=qd.extension.adstack, arch=[qd.cpu, qd.cuda, qd.metal, qd.vulkan], default_ip=qd.i64)
def test_adstack_bounded_inner_loop_pre_pass_handles_i64_bounds():
    # Pins that the structural pre-pass in `irpass::determine_ad_stack_size` accepts integer
    # `ConstStmt` loop bounds of any signed or unsigned width (not just i32). A kernel compiled
    # with `default_ip=qd.i64` materialises `RangeForStmt` begin/end as i64 `ConstStmt`s; the
    # pre-pass's interval evaluator (`try_eval_int_range`) must read `ConstStmt` leaves through
    # `val_int()` / `val_uint()` rather than hard-coding `val_int32()`, which would trip a dtype
    # assert inside `TypedConstant` and abort compilation before the Bellman-Ford fallback could
    # take over.
    #
    # Internal details: a simple `for _ in range(3)` reverse-mode kernel is enough to exercise the const-leaf
    # read path; if the evaluator trips the dtype assert the test fails at `compute.grad()` with an assertion
    # inside quadrants rather than at the numerical comparison. Runs on every backend: LLVM reads i64 const
    # leaves through the same `val_int()` / `val_uint()` helpers the SPIR-V pre-pass uses, so a dtype-assert
    # regression would surface on either path.
    x = qd.field(qd.f32, shape=(1,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(3):
                v = qd.sin(v) + 0.1
            y[None] += v

    x[0] = 0.2
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    v = 0.2
    dv_dx = 1.0
    for _ in range(3):
        dv_dx = math.cos(v) * dv_dx
        v = math.sin(v) + 0.1
    # `default_ip=qd.i64` only forces i64 loop bounds; floating-point arithmetic stays at
    # `default_fp` (f32). `pytest.approx` bypasses the backend `get_rel_eps()` floor so the tight
    # bound actually bites.
    assert x.grad[0] == pytest.approx(dv_dx, rel=1e-6)


@pytest.mark.parametrize("trip_mode", ["element", "shape"])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_nested_tripcount_gradient(trip_mode):
    # Pins the reverse-mode gradient through three nested range-fors whose inner bounds toggle between the
    # two shapes the structural pre-pass must recognise as trip-count sources: an ndarray element read
    # (`range(n[i])`, lowering to a `MaxOverRange`-wrapped `ExternalTensorRead`) and a shape-only reference
    # (`range(n.shape[0])`, lowering to a bare `ExternalTensorShape`). Both forms drive a nested adstack with
    # `x[j] * x[j]` pushes on every innermost iteration; a trip-count bound that underestimates the true
    # push count trips an overflow at `qd.sync()` rather than a silent numerical drift.
    #
    # Internal details: `qd.static(IS_ELEMENT)` picks the range at compile time so each parametrisation emits
    # a different `SizeExpr` tree without any runtime branch in the kernel. The `element` leg is the shape
    # the `ExternalTensorRead`-as-trip-body grammar gap is narrowed around; the `shape` leg is the
    # always-resolvable baseline and pins that shape-only nesting keeps working when the element form is
    # rejected or widened.
    IS_ELEMENT = trip_mode == "element"
    N_X = 32
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(n: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_e in range(n.shape[0]):
            for i_l in range(n[i_e]) if qd.static(IS_ELEMENT) else range(n.shape[0]):
                accum = 0.0
                for j in range(n[i_l]) if qd.static(IS_ELEMENT) else range(n.shape[0]):
                    accum = accum + x[j] * x[j]
                loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1
    n_np = np.array([2, 3, 4, 1], dtype=np.int32)
    compute(n_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(n_np)
    qd.sync()

    n_shape = int(n_np.shape[0])
    expected_loss = 0.0
    expected_grad = [0.0] * N_X
    for i_e in range(n_shape):
        middle_end = int(n_np[i_e]) if IS_ELEMENT else n_shape
        for i_l in range(middle_end):
            inner_end = int(n_np[i_l]) if IS_ELEMENT else n_shape
            for j in range(inner_end):
                expected_loss += 0.1 * 0.1
                expected_grad[j] += 2.0 * 0.1

    assert loss[None] == pytest.approx(expected_loss, rel=1e-5)
    for k in range(N_X):
        if expected_grad[k] == 0.0:
            assert x.grad[k] == pytest.approx(0.0, abs=1e-7)
        else:
            assert x.grad[k] == pytest.approx(expected_grad[k], rel=1e-5)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Silent-wrong-gradient when a reverse-mode kernel mutates an ndarray element that is then used as a "
        "trip count inside the same kernel, and the caller resets the host-side ndarray buffer between the "
        "forward and backward calls. The per-launch upload feeds the sizer stale zero values, the adstack is "
        "sized for that snapshot, and the reverse pass under-pops - `x.grad` ends up zero rather than "
        "overflowing. A walker-level static guard on `kernel writes ndarray N, kernel also reads N as a "
        "trip count` would have to run pre-autodiff (autodiff+DIE elide the offending store from the reverse "
        "IR whenever the write is not load-bearing for the gradient), false-positives on every legitimate "
        "read-after-write within a kernel, and still misses the cross-kernel case (kernel A writes, kernel B "
        "reads as trip count at reverse time). Documented as a known limitation in "
        "`docs/source/user_guide/autodiff.md`; the test is kept as a regression pin for a future runtime "
        "re-sizing or IR-integrity fix."
    ),
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_sizer_trip_count_ndarray_mutated_after_launch_read():
    # Pins the silent-wrong-gradient pattern where the sizer's dispatch-entry ndarray snapshot does not
    # match what the kernel actually runs against. The kernel writes `n[i] = 10` and immediately uses
    # `n[i_e]` as the inner-loop trip count; the caller resets `n_np` on the host between `compute()` and
    # `compute.grad()` so the backward dispatch uploads zeros. The sizer picks `max_size = 1`, the reverse
    # pass under-pops, and `x.grad` reads as zero instead of the analytical `0.8` at `x[i] = 0.1`.
    #
    # Note: this is a different class of bug than a runtime overflow that fires when the walker's captured
    # tree mathematically under-bounds the true push count - that one surfaces as a hard `RuntimeError` even
    # when the sizer reads the right ndarray values. The two share a common theme (sizer's bound disagrees
    # with runtime push count) but different root causes and remediations.
    N_X = 16
    N = 4

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(n: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(n.shape[0]):
            n[i] = 10
        for i_e in range(n.shape[0]):
            accum = 0.0
            for j in range(n[i_e]):
                accum = accum + x[j] * x[j]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1
    n_np = np.zeros(N, dtype=np.int32)
    compute(n_np)
    n_np[:] = 0
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(n_np)
    qd.sync()

    assert loss[None] == pytest.approx(4 * 10 * 0.1 * 0.1, rel=1e-5)
    for k in range(10):
        assert x.grad[k] == pytest.approx(4 * 2.0 * 0.1, rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_field_load_bounded_loop_evaluated_per_launch():
    # Pins the host-evaluated SizeExpr path end-to-end: a reverse-mode adstack whose inner-loop bound is a scalar
    # i32 field load must size the per-thread heap slice from the live field value at each launch. The
    # structural pre-pass captures a `SizeExpr::FieldLoad` leaf; the launcher evaluates it via
    # `SNodeRwAccessorsBank` before each dispatch and resizes the heap accordingly. The kernel is run with
    # `n_iter_fld[None]` set to 1, 20 and then 50 in sequence: each launch picks up the current field value,
    # resizes the adstack heap, and runs to completion with the analytical gradient `0.95 ** n_iter`.
    #
    # Internal details: the symbolic bound tree is flattened into the serialisable `SerializedSizeExpr` form and
    # stored inside the per-backend per-alloca task attributes, so this test exercises the same path whether the
    # kernel is freshly compiled or restored from the offline cache. On LLVM the bound is published into
    # `LLVMRuntime::adstack_{per_thread_stride,offsets,max_sizes}` by `publish_adstack_metadata` before each
    # dispatch; on SPIR-V it is uploaded into the `AdStackMetadata` StorageBuffer that the shader reads at every
    # push / load-top site.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    n_iter_fld = qd.field(qd.i32, shape=())
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter_fld[None]):
                v = v * 0.95 + 0.01
            y[None] += v

    for n_iter in (1, 20, 50):
        x[0] = 0.1
        n_iter_fld[None] = n_iter
        y[None] = 0.0
        compute()
        y.grad[None] = 1.0
        x.grad[0] = 0.0
        compute.grad()
        qd.sync()
        expected = 0.95**n_iter
        assert x.grad[0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("ndarray_kind", ["numpy", "qd_ndarray"])
@test_utils.test(require=qd.extension.adstack, arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.metal])
def test_adstack_inner_range_bounded_by_ndarray_read_at_outer_index(ndarray_kind):
    # Pins the `ExternalTensorRead`-over-`LoopIndex` `MaxOverRange` wrap in the `SizeExpr` pre-pass: a
    # reverse-mode adstack whose inner range `range(a[i])` is bounded by a scalar ndarray read at the enclosing
    # outer loop index. The pre-pass must build `MaxOverRange(var, 0, outer_end, ExternalTensorRead(a, [var]))`
    # for the alloca's multiplier; the launch-time evaluator enumerates the outer range, reads `a[var]` at each
    # iteration, and takes the max to size the per-thread adstack heap exactly.
    #
    # Internal details: runs on every backend. On CPU the launch-time SizeExpr is evaluated host-side via
    # `evaluate_adstack_size_expr`, with ndarray element reads going through the real host pointer
    # `set_host_accessible_ndarray_ptrs` mirrored into `array_ptrs`. On CUDA / AMDGPU the host encodes the
    # tree into device-side bytecode (`encode_adstack_size_expr_device_bytecode`) and calls
    # `runtime_eval_adstack_size_expr` to run the interpreter on the device. On Metal / Vulkan the bytecode
    # is emitted as a SPIR-V compute shader launched from `GfxRuntime::launch_kernel`. All three paths are
    # the only way to resolve an `ExternalTensorRead` against a GPU-private ndarray without round-tripping
    # the whole allocation to host. Asserts the analytical gradient `0.95 ** a[i]` per outer iteration so a
    # regression in the wrap or in either the host or device evaluator shows up as a value mismatch rather
    # than an overflow crash. Parametrised over the ndarray argument kind because `numpy`/torch inputs lower
    # through
    # `set_arg_external_array_with_shape` (writes the raw host pointer straight into `array_ptrs`) while
    # `qd.ndarray` inputs lower through `set_arg_ndarray_impl` + `set_ndarray_ptrs` (stashes a
    # `DeviceAllocation *` first, then the launcher resolves it). The CPU launcher mirrors the resolved
    # pointer back into `array_ptrs`; the CUDA / AMDGPU launchers don't need to because the device interpreter
    # reads the ndarray data pointer straight out of `ctx->arg_buffer` at the offset the encoder precomputed
    # from `args_type`, sidestepping the host-side `array_ptrs` map entirely.
    N = 4
    arr_data = np.array([2, 3, 1, 2], dtype=np.int32)

    x = qd.field(qd.f32, shape=(N,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in x:
            v = x[i]
            n = a[i]
            for _ in range(n):
                v = v * 0.95 + 0.01
            y[None] += v

    for i in range(N):
        x[i] = 0.1

    if ndarray_kind == "numpy":
        a = arr_data
    else:
        a = qd.ndarray(qd.i32, shape=(N,))
        a.from_numpy(arr_data)

    compute(a)
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad(a)
    qd.sync()

    for i in range(N):
        expected = 0.95 ** int(arr_data[i])
        assert x.grad[i] == pytest.approx(expected, rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_range_bounded_by_multidim_ndarray_read():
    # Pins multi-axis stride handling in the `ExternalTensorRead` evaluator. The sizer routes a
    # reverse-mode inner trip count `range(a[i, j])` through `SizeExpr::ExternalTensorRead(a, [var_i, var_j])`
    # for a 2-D ndarray `a`; the host evaluator, the CUDA / AMDGPU device interpreter in the LLVM runtime,
    # and the SPIR-V sizer compute shader must all fold the indices into a C-order linear offset
    # `i * shape[1] + j`, not the naive stride-1 sum `i + j`. The worst-case shape below isolates a single
    # non-zero entry at `a[2, 2] = 100` so the stride-1 path (sum over `(i, j)` with `i + j < rows + cols -
    # 1`) visits only the leading diagonal and the first row/column, all of which are zero; the buggy
    # sizer then picks `max = 0`, clamps to 1, and the `a[2, 2] = 100` cell pushes 100 times into an
    # adstack sized for 1. The fixed evaluator picks `max = 100` and the kernel runs to completion with
    # the analytical gradient. Any backend whose sizer still uses the stride-1 sum raises
    # `QuadrantsAssertionError: Adstack overflow` at the next `qd.sync()`.
    #
    # Internal details: CPU uses the host evaluator (`evaluate_adstack_size_expr` in
    # `adstack_size_expr_eval.cpp`) which reads shapes off `LaunchContextBuilder` via the same
    # `SHAPE_POS_IN_NDARRAY` path that `ExternalTensorShape` leaves use. CUDA / AMDGPU encode the node
    # bytecode and the device interpreter in `runtime/llvm/runtime_module/runtime.cpp`'s
    # `runtime_eval_adstack_size_expr` sums indices; Metal / Vulkan drive the SPIR-V sizer shader in
    # `codegen/spirv/adstack_sizer_shader.cpp` whose `compute_linear_index` does the same accumulation.
    # All three need per-axis stride support. The kernel uses nested `for i: for j:` rather than
    # `qd.ndrange(shape[0], shape[1])` so the pre-pass sees two distinct `LoopIndexStmt`s (one per axis)
    # as the `ExternalPtrStmt` index operands; the 2-index ndrange lowering flattens `(i, j)` through
    # `div`/`mod` arithmetic that `determine_ad_stack_size.cpp::build_value_expr` does not fold, so the
    # pre-pass would fall back to `default_ad_stack_size` and the kernel would pass trivially on every
    # backend without exercising the multi-axis evaluator.
    rows, cols = 3, 5
    arr_np = np.zeros((rows, cols), dtype=np.int32)
    arr_np[2, 2] = 100  # sole non-zero cell; stride-1 sum never visits (i=2, j=2) -> max evaluated to 0
    # Sanity: stride-1 sum over `(i, j)` with `i + j < rows + cols - 1` stays within the zero band.
    n_max_true = int(arr_np.max())
    n_max_stride_one = 0
    for i in range(rows):
        for j in range(cols):
            if i + j < rows + cols - 1:
                n_max_stride_one = max(n_max_stride_one, int(arr_np.flatten()[min(i + j, rows * cols - 1)]))
    assert n_max_true == 100 and n_max_stride_one == 0

    N_X = 8
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=2)):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                v = x[0]
                for _ in range(a[i, j]):
                    v = v * 0.95 + 0.01
                y[None] += v

    for i in range(N_X):
        x[i] = 0.1

    compute(arr_np)
    y.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_np)
    qd.sync()

    # `y = sum_{i, j} v_final(a[i, j])` where `v_final(n) = 0.95 ** n * x[0] + const_terms(n)`. Gradient wrt
    # x[0] is `sum_{i, j} 0.95 ** a[i, j]`; with `a[2, 2] = 100` and all other cells zero the dominant
    # contribution is `14 + 0.95 ** 100 ~ 14.006`.
    expected = float(np.sum(0.95 ** arr_np.astype(np.float64)))
    assert x.grad[0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("outer_bound", ["const", "dynamic"])
@test_utils.test(require=qd.extension.adstack, arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.metal])
def test_adstack_ext_tensor_read_indexed_by_stashed_outer_loop_var(outer_bound):
    # Pins the `ExternalPtrStmt` indexed by `AdStackLoadTopStmt` grammar gap. The kernel walks a
    # parent/child hierarchical-array layout: an outer parallel-for whose body casts its loop variable
    # (`i_l = qd.cast(i_l_, qd.i32)`), branches on `ndarray[i_l] != -1`, and drives a nested range-for
    # from `ndarray[i_l] - ndarray[i_l]`. Under `ad_stack_experimental_enabled=True` the autodiff
    # pipeline stashes the cast loop index onto a dedicated adstack and reloads it via `stack_load_top`
    # so the reverse pass can reconstruct it; every downstream `ndarray[i_l]` lowers to
    # `ExternalPtrStmt(arr, [AdStackLoadTopStmt])`. The pre-pass upper-bounds the loaded value by
    # recognising the stash pattern (single loop-index push plus const-zero initialiser) and folding
    # through to the backing `LoopIndexStmt`.
    #
    # Internal details: runs on every backend - LLVM evaluates the stash-backed SizeExpr through
    # `publish_adstack_metadata`, SPIR-V through `GfxRuntime::launch_kernel`'s AdStackMetadata upload. Parametrised
    # over `outer_bound` because const and dynamic outer range-for bounds lower very differently - a constant
    # collapses into the offload's `const_end` at offload time (no prep task), a dynamic bound lowers to a prep
    # serial task that writes the value into the kernel's global-temporary buffer for the main range-for task to
    # read back. The grammar covers both paths via `resolve_global_tmp_value`, so the
    # `ExternalPtrStmt`-with-stashed-index pattern works whether the outermost parallel-for is sized at launch
    # time (e.g. `arr.shape[0]`) or hard-coded.
    N_ENT = 1
    link_start_np = np.array([0], dtype=np.int32)
    link_end_np = np.array([2], dtype=np.int32)
    joint_start_np = np.array([0, 1], dtype=np.int32)
    joint_end_np = np.array([1, 3], dtype=np.int32)
    parent_idx_np = np.array([-1, 0], dtype=np.int32)

    N_X = 4
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    IS_CONST = outer_bound == "const"

    @qd.kernel
    def compute(
        link_start: qd.types.ndarray(dtype=qd.i32, ndim=1),
        link_end: qd.types.ndarray(dtype=qd.i32, ndim=1),
        joint_start: qd.types.ndarray(dtype=qd.i32, ndim=1),
        joint_end: qd.types.ndarray(dtype=qd.i32, ndim=1),
        parent_idx: qd.types.ndarray(dtype=qd.i32, ndim=1),
    ):
        for i_e in range(N_ENT) if qd.static(IS_CONST) else range(link_start.shape[0]):
            for i_l_ in range(link_start[i_e], link_end[i_e]):
                i_l = qd.cast(i_l_, qd.i32)
                accum = x[i_l] * 0.5
                if parent_idx[i_l] != -1:
                    accum = accum + x[parent_idx[i_l]] * 0.3
                n_joints = joint_end[i_l] - joint_start[i_l]
                for i_j_ in range(n_joints):
                    i_j = i_j_ + joint_start[i_l]
                    accum = accum + x[i_j] * x[i_j]
                loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1

    compute(link_start_np, link_end_np, joint_start_np, joint_end_np, parent_idx_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(link_start_np, link_end_np, joint_start_np, joint_end_np, parent_idx_np)
    qd.sync()

    # Analytical gradients at x[i] = 0.1:
    #   x[0]: 0.5 (i_l=0 self) + 0.3 (i_l=1 parent) + 2*x[0] (i_j=0, i_l=0) = 1.0
    #   x[1]: 0.5 (i_l=1 self) + 2*x[1] (i_j=1, i_l=1)                      = 0.7
    #   x[2]:                                   2*x[2] (i_j=2, i_l=1)      = 0.2
    #   x[3]: 0
    assert x.grad[0] == pytest.approx(1.0, rel=1e-6)
    assert x.grad[1] == pytest.approx(0.7, rel=1e-6)
    assert x.grad[2] == pytest.approx(0.2, rel=1e-6)
    assert x.grad[3] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_field_ptr_indexed_by_stashed_outer_loop_var():
    # Pins the `GlobalPtrStmt`-index stash-chase extension in the `SizeExpr` pre-pass. The kernel reads
    # two scalar quadrants fields `link_start[i_outer]` / `link_end[i_outer]` as the bounds of an inner
    # range-for, where `i_outer` is an outer parallel-for index that `ad_stack_experimental_enabled=True`
    # stashes onto a dedicated adstack for the reverse pass. Every downstream `link_start[i_outer]` then
    # lowers to `GlobalPtrStmt(<field>, [AdStackLoadTopStmt])`. Before the fix, the pre-pass's
    # `GlobalPtrStmt` branch rejected any non-const index and the reverse-mode adstack bound would hard-
    # error as "unresolved after Bellman-Ford + structural pre-pass"; the fix walks the index through the
    # same stash chase the `ExternalPtrStmt` branch uses and falls back to the snode's
    # `shape_along_axis(axis)` as a safe upper bound when the stash has no single loop-index push.
    #
    # Internal details: runs on every backend - LLVM evaluates the stash-backed `SizeExpr` through
    # `publish_adstack_metadata`, SPIR-V through `GfxRuntime::launch_kernel`'s `AdStackMetadata` upload.
    # The inner `range(link_start[i_outer_c], link_end[i_outer_c])` fans into four differentiable-body
    # iterations per outer index; each touches a distinct `x[i_inner]` so the analytical gradient is the
    # same constant per slot and a bound-too-small regression surfaces as either the old "unresolved"
    # error or an adstack-overflow at `qd.sync()`.
    N_OUTER = 4
    link_start = qd.field(qd.i32, shape=(N_OUTER,))
    link_end = qd.field(qd.i32, shape=(N_OUTER,))

    N_X = 6
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i_outer in range(arr.shape[0]):
            i_oc = qd.cast(i_outer, qd.i32)
            for i_inner in range(link_start[i_oc], link_end[i_oc]):
                v = x[i_inner] * 0.5
                for _ in range(2):
                    v = v * 0.95 + 0.01
                loss[None] += v

    link_start[0] = 0
    link_start[1] = 1
    link_start[2] = 3
    link_start[3] = 4
    link_end[0] = 1
    link_end[1] = 3
    link_end[2] = 4
    link_end[3] = 6
    for i in range(N_X):
        x[i] = 0.1

    arr = np.zeros(N_OUTER, dtype=np.float32)
    compute(arr)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr)
    qd.sync()

    # Analytical gradient at each `x[i_inner]`: each slot is read once, each read contributes
    # `d(loss)/d(x) = 0.5 * 0.95 * 0.95 = 0.45125`.
    for i in range(N_X):
        assert x.grad[i] == pytest.approx(0.45125, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_triangular_ndrange_self_referential_push_idempotency():
    # Pins the phase-2 idempotency-at-zero probe ordering in `build_value_expr` for a reverse-mode push whose
    # value expression is self-referential by construction. The kernel couples a 2D `qd.ndrange` outer parallel
    # scan with a triangular inner `for j in range(i_outer, ...)` and a nested `range(begin_fld[j], end_fld[j])`
    # whose bounds come from scalar fields indexed by the stashed outer index. On the experimental adstack path
    # that shape lowers reverse-mode pushes of the form `sub(load_top($S), load_top($S))` where both load_top
    # reads target the very stack the push feeds - a zero-net push that must be treated as idempotent at zero
    # rather than rejected as a stash data-flow cycle. The structural pre-pass must try the idempotency probe
    # (which substitutes `load_top(self) -> 0`) BEFORE its generic visited-set cycle guard fires; without that
    # ordering the cycle guard aborts the walk first, the probe never runs, and the grad kernel fails to
    # compile with "stash data-flow cycle ... idempotency-at-zero probe could not discharge it".
    #
    # Internal details: names are deliberately domain-neutral. `outer_size` + `batch_probe.shape[1]` feed the
    # 2D `qd.ndrange` (the batch-dim probe is a grad-requiring field whose only in-kernel use is the shape
    # read, matching the minimal pattern that triggers the cycle). `group_begin` / `group_end` gate the middle
    # range, `sub_begin` / `sub_end` gate the innermost range, `src_offset` / `dst_offset` offset the scalar
    # scatter. The scalar reads from `src_buf[src_offset + k, i_b]` packed into a `qd.Vector` then written into
    # `dst_buf[dst_offset + j, i_b]` via `qd.static(range(3))` is load-bearing: without a differentiable
    # Vector-packed reverse body the `sub(load_top($S), load_top($S))` push shape does not arise. The trailing
    # `dst_vec_buf[i_l, i_b] = src_vec_buf[i_l, n_sub, i_b]` Vector copy outside the innermost range but inside
    # the triangular loop is also load-bearing - removing it collapses the enclosing-loop structure enough that
    # the pre-pass resolves the adstack by other means. `cfg_optimization=False` + `ad_stack_experimental_enabled=True`
    # are the minimum flags that surface the cycle: with CFG optimization on the store-to-load forwarder collapses
    # the self-referential push before the sizing pass sees it, and the non-experimental path uses a different
    # sizing strategy that sidesteps the probe altogether.
    outer_size = qd.field(qd.i32, shape=(1,))
    group_begin = qd.field(qd.i32, shape=(1,))
    group_end = qd.field(qd.i32, shape=(1,))
    group_base = qd.Vector.field(3, qd.f32, shape=(1,))
    sub_begin = qd.field(qd.i32, shape=(1,))
    sub_end = qd.field(qd.i32, shape=(1,))
    src_offset = qd.field(qd.i32, shape=(1,))
    dst_offset = qd.field(qd.i32, shape=(1,))
    dst_buf = qd.field(qd.f32, shape=(6, 1), needs_grad=True)
    src_buf = qd.field(qd.f32, shape=(7, 1), needs_grad=True)
    batch_probe = qd.Vector.field(3, qd.f32, shape=(1, 1), needs_grad=True)
    dst_vec_buf = qd.Vector.field(4, qd.f32, shape=(1, 1), needs_grad=True)
    src_vec_buf = qd.Vector.field(4, qd.f32, shape=(1, 2, 1), needs_grad=True)

    @qd.kernel
    def compute(
        batch_probe: qd.template(),
        dst_vec_buf: qd.template(),
        src_vec_buf: qd.template(),
        group_base: qd.template(),
        sub_begin: qd.template(),
        sub_end: qd.template(),
        src_offset: qd.template(),
        dst_offset: qd.template(),
        dst_buf: qd.template(),
        outer_size: qd.template(),
        group_begin: qd.template(),
        group_end: qd.template(),
        src_buf: qd.template(),
    ):
        for i_e, i_b in qd.ndrange(outer_size.shape[0], batch_probe.shape[1]):
            for j_e in range(i_e, outer_size.shape[0]):
                for i_l in range(group_begin[j_e], group_end[j_e]):
                    base = group_base[i_l]
                    n_sub = sub_end[i_l] - sub_begin[i_l]
                    for k_ in range(n_sub):
                        k = k_ + sub_begin[i_l]
                        s = src_offset[k]
                        a = src_buf[s + 4, i_b]
                        b = src_buf[s + 5, i_b]
                        c = src_buf[s + 6, i_b]
                        packed = qd.Vector([a, b, c], dt=qd.f32)
                        d = dst_offset[k]
                        for j in qd.static(range(3)):
                            dst_buf[d + j, i_b] = base[j]
                            dst_buf[d + 3 + j, i_b] = packed[j]
                    dst_vec_buf[i_l, i_b] = src_vec_buf[i_l, n_sub, i_b]

    outer_size[0] = 1
    group_begin[0] = 0
    group_end[0] = 1
    sub_begin[0] = 0
    sub_end[0] = 1
    src_offset[0] = 0
    dst_offset[0] = 0

    # Pre-fix the grad compile raises RuntimeError("stash data-flow cycle ..."); post-fix it must compile
    # cleanly and run to completion. The assertion is the absence of that RuntimeError - no gradient value is
    # checked because the minimal-shape fields have a single element and the bug is purely a compile-time
    # cycle-detection regression.
    compute.grad(
        batch_probe,
        dst_vec_buf,
        src_vec_buf,
        group_base,
        sub_begin,
        sub_end,
        src_offset,
        dst_offset,
        dst_buf,
        outer_size,
        group_begin,
        group_end,
        src_buf,
    )
    qd.sync()


@pytest.mark.parametrize("inner_loop_shape", ["begin_end", "sub_then_range"])
@test_utils.test(require=qd.extension.adstack, arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.metal])
def test_adstack_structural_pre_pass_fuses_sub_of_max_over_range_with_matching_shape_ends(inner_loop_shape):
    # Covers the two user-facing surface forms of a reverse-mode kernel whose inner range-for trip count is
    # the difference between two reads of parallel ndarrays indexed by the SAME outer loop. Both lower to a
    # Sub-of-two-MaxOverRange where the `end` operands are structurally equal (both come from the single
    # enclosing outer loop's end, not from each read's own ndarray shape), so the walker's strict-equality
    # fusion path already fires without the `ExternalTensorShape` same-axis extension. The test pins that
    # strict path continues to work on both spellings and that the resulting adstack bound matches the actual
    # reverse-pass push count.
    #
    # Internal details: runs on every backend now that the runtime-evaluator ships on both the LLVM and SPIR-V
    # paths. Two surface spellings share a single test body via `qd.static` because both produce the same
    # `expr_sub` call at walker time, just via different `build_value_expr` recursion paths:
    #   - `begin_end`: `for i_j_ in range(start[i_o], end[i_o])` - `compute_bounded_adstack_size` multiplies
    #     `end_upper - begin_lower`, `resolve_loop_begin_lower_bound` drops non-const begins to `Const(0)`,
    #     so the fused bound is the `end[i_o]` MaxOverRange alone (no two-operand Sub is built for this
    #     spelling); the test still passes because the body's push count is bounded by `end[i_o]`, which the
    #     walker tracks soundly via the single MaxOverRange.
    #   - `sub_then_range`: user materialises `n_inner = end[i_o] - start[i_o]` and passes `range(n_inner)`;
    #     this makes the `Sub` explicit in the range-for's `end` stmt, `build_value_expr` recurses into both
    #     operands, each wraps with the SAME outer-loop end, and strict fusion collapses the pair.
    # `qd.cast(e - s, qd.f32)` is a multiplicative factor inside the body so `full_simplify` does not inline
    # the subtraction back into the range-for bounds and collapse the two patterns.
    IS_BEGIN_END = inner_loop_shape == "begin_end"
    N_X = 8

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(start_arr: qd.types.ndarray(dtype=qd.i32, ndim=1), end_arr: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_o in range(start_arr.shape[0]):
            s = start_arr[i_o]
            e = end_arr[i_o]
            accum = 0.0
            for i_j_ in range(s, e) if qd.static(IS_BEGIN_END) else range(e - s):
                i_j = i_j_ if qd.static(IS_BEGIN_END) else (i_j_ + s)
                accum = accum + x[i_j] * x[i_j] * qd.cast(e - s, qd.f32)
            loss[None] += accum

    # `start = [0, 3]`, `end = [3, 4]`: per-slot trips are (3, 1). `e - s` per slot is (3, 1) so the inner
    # body adds `trip * x[i_j]^2` across 4 inner iterations.
    for i in range(N_X):
        x[i] = 0.1
    start_np = np.array([0, 3], dtype=np.int32)
    end_np = np.array([3, 4], dtype=np.int32)

    compute(start_np, end_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(start_np, end_np)
    qd.sync()

    # loss = sum over i_j in [0, 3) of `3 * x[i_j]^2` + sum over i_j in [3, 4) of `1 * x[i_j]^2`
    #      = 3 * 3 * 0.01 + 1 * 0.01 = 0.10.
    assert loss[None] == pytest.approx(0.10, rel=1e-5)
    # d(loss)/dx[j] = 2 * trip * x[j]; j in 0..2 has trip=3, j=3 has trip=1, j>=4 never visited.
    for j in range(3):
        assert x.grad[j] == pytest.approx(2 * 3 * 0.1, rel=1e-5)
    assert x.grad[3] == pytest.approx(2 * 1 * 0.1, rel=1e-5)
    for j in range(4, N_X):
        assert x.grad[j] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack, arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.metal])
def test_adstack_structural_pre_pass_fuses_sub_of_max_over_range_with_mismatched_shape_ends():
    # Pins the `expr_sub` fusion for the Sub-of-two-MaxOverRange shape that the walker builds when an inner
    # range-for's trip count is computed as the difference between two ndarray reads whose indices come from
    # two DIFFERENT enclosing range-fors. Each read wraps into its own `MaxOverRange(outer_i, 0, shape(arr_i),
    # ExtRead(arr_i, [outer_i]))`; the `end` operands are then `ExternalTensorShape` nodes pointing at
    # distinct `arg_id`s so `expr_equal` rejects them and the pre-fusion walker falls back to
    # `max_i arr_a[i] - max_j arr_b[j]`, which under-counts `max_i (arr_a[i] - arr_b[i])` whenever the two
    # per-index maxima land at different slots. With `arr_a = [1, 5]`, `arr_b = [4, 0]` the unfused bound
    # collapses to `5 - 4 = 1` per outer pair and the full trip multiplier undershoots the actual push count
    # of 7 (the (1,1) pair alone pushes 5), so the reverse pass overflows the heap and raises at `qd.sync()`.
    # The fusion emits the tight `MaxOverRange(v, 0, shape(arr_a), Sub(arr_a[v], arr_b[v]))` which correctly
    # evaluates to 5, and the adstack gets sized to fit.
    #
    # Internal details: runs on every backend now that the runtime-evaluator ships on both the LLVM and SPIR-V
    # paths. The trivial `range(1)` wrapper keeps the kernel AST inside a top-level for-loop, which the
    # autodiff front-end requires (`reverse_segments` rejects mixed
    # statement-plus-for kernel bodies).
    N_X = 16

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr_a: qd.types.ndarray(dtype=qd.i32, ndim=1), arr_b: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for _dummy in range(1):
            accum = 0.0
            for i_a in range(arr_a.shape[0]):
                for i_b in range(arr_b.shape[0]):
                    n = arr_a[i_a] - arr_b[i_b]
                    if n > 0:
                        for k in range(n):
                            accum = accum + x[k] * x[k]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1
    arr_a_np = np.array([1, 5], dtype=np.int32)
    arr_b_np = np.array([4, 0], dtype=np.int32)

    compute(arr_a_np, arr_b_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_a_np, arr_b_np)
    qd.sync()

    # For (i_a, i_b) in {(0,0), (0,1), (1,0), (1,1)} the inner trip n = max(0, arr_a[i_a] - arr_b[i_b]) is
    # {0, 1, 1, 5}. Total push count is 7 (the (1,1) pair contributes 5); the pre-fusion adstack bound of 4
    # overflows. loss = (0 + 1 + 1 + 5) * x[0..4]^2 = 0.07 with every x[i] = 0.1.
    # x.grad[k] = 2 * (count of inner iterations that visit index k) * 0.1. k = 0 is visited by every
    # non-empty pair (3 visits), k in [1, 4] is visited only by the (1, 1) pair (1 visit each), k >= 5 is
    # never visited.
    assert loss[None] == pytest.approx(0.07, rel=1e-5)
    assert x.grad[0] == pytest.approx(0.6, rel=1e-5)
    for k in range(1, 5):
        assert x.grad[k] == pytest.approx(0.2, rel=1e-5)
    for k in range(5, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack, arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.metal], cfg_optimization=False)
def test_adstack_spirv_metadata_per_task_buffer():
    # SPIR-V launcher used to share a single grow-on-demand `AdStackMetadata` device buffer across every
    # task in a kernel. Per-task `(stride_float, stride_int, offset_i, max_size_i, ...)` tables were
    # host-memcpy'd into that buffer inside the cmdlist record loop, and the `bindings` descriptor for each
    # task's dispatch captured the same buffer handle. Record is host-synchronous but execute is deferred,
    # so by submit time the buffer holds only the LAST task's metadata and every dispatch in the cmdlist
    # reads those bytes. Earlier tasks then see shorter sibling stacks' `max_size` where their own should
    # be - e.g. a stack whose sizer wrote `max_size=9` observes a runtime `max_size=3`, its first guarded
    # push trips the `count < max_size` check at `count=3`, the overflow flag flips, and `qd.sync()` raises
    # even though the kernel's actual per-thread push count fits the per-stack bound the sizer computed.
    #
    # Internal details: `cfg_optimization=False` is load-bearing - with it enabled, the CFG pass sinks / merges
    # the bind-and-dispatch pair in a way that masks the cross-task buffer reuse on this kernel shape; with it
    # disabled the raw record-then-execute race surfaces. The pinned regression is SPIR-V-specific (the LLVM
    # path publishes metadata host-side via `publish_adstack_metadata` directly into each launch's own
    # `AdStackSizingInfo` with no cross-task aliasing), but the test runs on every backend so a future
    # regression in either path that produces wrong values rather than an overflow is still caught. The kernel
    # shape (two sibling `qd.ndrange` offloads, the second one carrying a triangular i<=j<k nested loop that
    # stashes a multiplicative reduction onto its own adstack) is the minimum that exhibits the bug: you need
    # at least two tasks in the same kernel so the second task's record overwrites the first task's metadata
    # before submit. The post-fix runtime allocates a fresh metadata buffer per task record and retires it into
    # `ctx_buffers_` so it stays alive until the sync window closes.
    tri_mat = qd.field(dtype=qd.f32, shape=(2, 7, 7, 1))
    src_mat = qd.field(dtype=qd.types.matrix(3, 3, qd.f32), shape=(1, 1), needs_grad=True)
    dst_mat = qd.field(dtype=qd.types.matrix(3, 3, qd.f32), shape=(1, 1), needs_grad=True)
    state = qd.field(dtype=qd.types.vector(3, qd.f32), shape=(1, 1), needs_grad=True)
    group_offset = qd.field(dtype=qd.i32, shape=(1,))
    group_size = qd.field(dtype=qd.i32, shape=(1,))
    group_size[0] = 7

    @qd.kernel
    def kernel_two_offloads_with_tri_reduce():
        for i_0, i_b in qd.ndrange(state.shape[0], state.shape[1]):
            dst_mat[i_0, i_b] = src_mat[i_0, i_b]

        for i_g, i_b in qd.ndrange(state.shape[0], state.shape[1]):
            base = group_offset[i_g]
            for p_i0 in range(group_size[i_g]):
                for p_j0 in range(p_i0 + 1):
                    i_pr = base + p_i0
                    j_pr = base + p_j0
                    acc = qd.f32(0.0)
                    for p_k0 in range(p_j0):
                        k_pr = base + p_k0
                        acc = acc + (tri_mat[1, i_pr, k_pr, i_b] * tri_mat[1, j_pr, k_pr, i_b])
                    tri_mat[1, i_pr, j_pr, i_b] = acc

    # Pre-fix: raises `Adstack overflow (offending stack_id=0)` at `qd.sync()` because the first offload's
    # metadata buffer was overwritten by the second offload's host memcpy before the cmdlist ran, so the
    # first offload's f32 stack 0 saw `max_size=3` (the second offload's int stack 0 value) instead of its
    # own sizer-computed 9. Post-fix: finishes cleanly because each task gets its own metadata buffer.
    kernel_two_offloads_with_tri_reduce.grad()
    qd.sync()
