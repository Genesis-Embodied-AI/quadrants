import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from contextlib import nullcontext

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang.exception import QuadrantsAssertionError
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
    # Prevents drift between the Python unary op registry and the C++ `unary_collections` set declared in
    # `quadrants/transforms/auto_diff/auto_diff_common.h`. Every unary op whose `MakeAdjoint` branch (defined in
    # `quadrants/transforms/auto_diff/make_adjoint.cpp`) accumulates onto `stmt->operand` must be either in
    # `unary_collections` (nonlinear: needs per-iteration operand spilling on the adstack inside dynamic loops) or
    # in the local `_KNOWN_LINEAR_UNARY_OPS` allow-list (reverse formula uses a compile-time constant coefficient,
    # so the single-slot spill path is correct for it). Forgetting to classify a new diffable unary op falls back
    # to the single-slot spill and produces silently wrong gradients in dynamic loops.
    #
    # Four invariants are checked, all of them symmetric:
    #   (1) Every diffable-math op detected in MakeAdjoint is in `cpp_nonlinear` OR in `_KNOWN_LINEAR_UNARY_OPS`.
    #   (2) Every op in `cpp_nonlinear` has a matching diffable-math branch in MakeAdjoint.
    #   (3) Every op in `_KNOWN_LINEAR_UNARY_OPS` has a matching diffable-math branch in MakeAdjoint.
    #   (4) `cpp_nonlinear` and `_KNOWN_LINEAR_UNARY_OPS` are disjoint (an op cannot be both nonlinear and linear).
    auto_diff_dir = pathlib.Path(__file__).resolve().parents[2] / "quadrants" / "transforms" / "auto_diff"
    common_src = (auto_diff_dir / "auto_diff_common.h").read_text()
    make_adjoint_src = (auto_diff_dir / "make_adjoint.cpp").read_text()

    cc_match = re.search(r"unary_collections\s*\{([^}]+)\}", common_src)
    assert cc_match is not None, "unary_collections not located in auto_diff_common.h"
    cpp_nonlinear = set(re.findall(r"UnaryOpType::(\w+)", cc_match.group(1)))

    make_adjoint_start = make_adjoint_src.find("class MakeAdjoint")
    assert make_adjoint_start != -1, "class MakeAdjoint not located in make_adjoint.cpp"
    adj_start = make_adjoint_src.find("void visit(UnaryOpStmt *stmt) override", make_adjoint_start)
    assert adj_start != -1, "MakeAdjoint::visit(UnaryOpStmt*) not located in make_adjoint.cpp"
    adj_end = make_adjoint_src.find("void visit(", adj_start + 10)
    assert adj_end != -1, "next visitor method after MakeAdjoint::visit(UnaryOpStmt*) not found"
    adj_block = make_adjoint_src[adj_start:adj_end]
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
        f"Add each one to `unary_collections` in quadrants/transforms/auto_diff/auto_diff_common.h (if "
        f"nonlinear) or to `_KNOWN_LINEAR_UNARY_OPS` in this file (if linear)."
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
    reason=(
        "Reverse-mode NaN/Inf poisoning semantics is TBD (f64 variant). Same divergence as the f32 case: PyTorch "
        "propagates NaN in the backward graph; Quadrants runs `1 / operand` verbatim and returns a finite number."
    ),
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


@test_utils.test(require=qd.extension.adstack)
def test_eliminate_recomputable_pushes_preserves_zero_body_store():
    # Cross-checks `dloss/dc` against the analytic value `2 * cos(c) * cos(sin(c))` for an adstack-mode kernel where
    # `tmp` receives one recomputable body push (`tmp = qd.sin(c[None])`) plus one conditional zero body store
    # (`if reset[i] != 0: tmp = 0.0`). The reset mask `[0, 1, 0, 1]` zeros `tmp` for two of four iterations so only
    # the other two contribute to the gradient.
    #
    # Internal details: the adstack promotion of `tmp` comes from `AdStackAllocaJudger::visit(UnaryOpStmt)` (`tmp`
    # feeds the non-linear `qd.sin(tmp)` accumulator); the load+store rule does not fire because the offload-level
    # for-loop sits outside the IB and `dynamic_for_depth_` stays 0 throughout the judger walk. The body push value
    # `sin(GlobalLoad(c))` is a recomputable chain by `RecomputableChainAnalyzer`, which makes the stack a candidate
    # for `EliminateRecomputableAdStackPushes`. The eligibility gate must count the conditional `tmp = 0.0` as a
    # body push: two body pushes for one stack disqualify the stack and the original IR survives. A weaker gate that
    # classifies the conditional zero as the prologue init - e.g. detecting init by literal-zero value rather than
    # by position relative to the alloca - drops the user's zero store, rewires every `load_top` to `sin(c)`, and
    # produces `c.grad` exactly 2x the analytic value because the gradient flows through every iteration regardless
    # of the reset mask.
    n = 4
    c = qd.field(qd.f32, shape=(), needs_grad=True)
    reset = qd.field(qd.i32, shape=n)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            # Recomputable body push: sin of a globally-loaded scalar. The chain `sin(GlobalLoad(c))` has interior
            # side-effect-free ops only and no LoopIndex / LocalLoad leaves, so RecomputableChainAnalyzer returns
            # true.
            tmp = qd.sin(c[None])
            if reset[i] != 0:
                # Real `tmp = 0.0` body store. Lowers to AdStackPushStmt with value zero. A weaker eligibility gate
                # misclassifies this push as an init prologue push and silently erases it.
                tmp = 0.0
            # Use tmp as a non-linear unary op operand so AdStackAllocaJudger marks the alloca stack-needed via its
            # visit(UnaryOpStmt) rule, which is what makes this an AdStack in the first place (the load+store rule
            # does not fire here because the offload-level for-loop is outside the IB and `dynamic_for_depth_` stays
            # 0).
            loss[None] += qd.sin(tmp)

    c[None] = 0.5
    reset[0] = 0
    reset[1] = 1
    reset[2] = 0
    reset[3] = 1
    loss[None] = 0.0
    c.grad[None] = 0.0

    compute()
    loss.grad[None] = 1.0
    compute.grad()

    # Forward semantics: tmp_i = sin(c) when reset[i]==0, 0 when reset[i]!=0. Two of four iterations are not reset,
    # so loss == 2 * sin(sin(c)). Gradient: dloss/dc contributes cos(c)*cos(sin(c)) per non-reset iteration, so
    # dloss/dc == 2 * cos(c) * cos(sin(c)).
    n_no_reset = 2
    expected_loss = n_no_reset * math.sin(math.sin(0.5))
    expected_grad = n_no_reset * math.cos(0.5) * math.cos(math.sin(0.5))
    assert loss[None] == pytest.approx(expected_loss, rel=1e-5)
    assert c.grad[None] == pytest.approx(expected_grad, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False, ad_stack_experimental_enabled=True)
def test_backup_ssa_load_top_with_subsequent_pushes():
    # Pins `BackupSSA::is_load_top_stable` against the reverse-mode aliasing bug where cloning an
    # `AdStackLoadTopStmt` at the reverse cursor reads the stack's live top - which differs from the original
    # forward-time top whenever the same stack received another push between the original load_top and the end of
    # the IB. The minimal IR shape that exhibits this:
    #
    #   - A multi-axis `qd.ndrange(...)` whose flat loop index is decomposed into per-axis indices via repeated
    #     `floordiv` / `sub` over an adstack-backed value (each step pushes its result, then a subsequent
    #     `load_top` reads that pushed value to feed the next decomposition step).
    #   - A guarded division `out = in / mass` inside the kernel: `MakeAdjoint` emits two reverse formulas
    #     (`adj(in) += adj(out) / mass`, `adj(mass) += -adj(out) * in / mass^2`) that re-evaluate `mass` and `in`
    #     at the reverse cursor; that re-evaluation walks the per-axis index DAG, which BackupSSA was happy to
    #     reconstruct from `floordiv(load_top(stack), const)` clones - the load_top clones read the stack's
    #     final top, which is the LAST push (the inner axis) instead of the intermediate flat index.
    #
    # With the stable-load-top guard, BackupSSA detects the unsafe load_top, falls back to the per-IB alloca
    # spill (`load(op)`), and the reverse re-evaluation reads the per-iteration forward-time index, producing
    # the analytic gradients. Without the guard, the reverse pass reads `g_mass[j/2, j]` and `g_in[j/2, j]`
    # instead of `g_mass[i, j]` / `g_in[i, j]`, so cells whose forward gate was true at `(i=1, j=1)` end up
    # dividing `adj(out)` by `g_mass[0, 1]` (= 0) and the gradient comes out as `inf`/`nan`.
    #
    # Internal details: `cfg_optimization=False` keeps the compound `if true && (g_mass[i, j] > 0)` form alive
    # (the `&&` lowering inserts the multi-push into the gate stack that the chain analyzer must clone through);
    # `ad_stack_experimental_enabled=True` is the configuration where the cloned-load_top path actually fires.
    g_mass = qd.field(qd.f32, shape=(2, 2), needs_grad=True)
    g_in = qd.field(qd.f32, shape=(2, 2), needs_grad=True)
    g_out = qd.field(qd.f32, shape=(2, 2), needs_grad=True)

    @qd.kernel
    def gated_div():
        for i, j in qd.ndrange(2, 2):
            if g_mass[i, j] > 0.0:
                g_out[i, j] = g_in[i, j] / g_mass[i, j]

    g_mass[1, 1] = 2.0
    g_in[1, 1] = 4.0
    g_out.grad[1, 1] = 1.0
    gated_div.grad()

    # Forward: g_out[1, 1] = g_in[1, 1] / g_mass[1, 1] = 2.0; gate is false at every other cell so all other
    # gradients are zero. With dy = 1.0 on the active cell:
    #   d(g_in[1, 1])   = 1 / g_mass[1, 1]                = 0.5
    #   d(g_mass[1, 1]) = -g_in[1, 1] / g_mass[1, 1]^2    = -1.0
    assert g_in.grad[1, 1] == pytest.approx(0.5, rel=1e-6)
    assert g_mass.grad[1, 1] == pytest.approx(-1.0, rel=1e-6)
    # Inactive cells must not have received any gradient. A regression where `BackupSSA` re-clones the
    # load_top at the reverse cursor reads `g_mass[j/2, j] = g_mass[0, 1] = 0`, so 1/g_mass[0, 1] = inf
    # would land on `g_in.grad[0, 1]` via the wrong index. Probing every other cell catches both the inf
    # spill and any silent off-by-one accumulation.
    for i in range(2):
        for j in range(2):
            if (i, j) == (1, 1):
                continue
            assert g_in.grad[i, j] == pytest.approx(0.0, abs=1e-6)
            assert g_mass.grad[i, j] == pytest.approx(0.0, abs=1e-6)


@test_utils.test(
    require=qd.extension.adstack,
    cfg_optimization=False,
    force_scalarize_matrix=True,
    ad_stack_experimental_enabled=True,
)
def test_eliminate_recomputable_pushes_rejects_mutated_snode_chain_leaf():
    # Pins the `mutated_snodes` guard inside `RecomputableChainAnalyzer::is_recomputable` against the chain-clone
    # post-write read miscompile: a forward `GlobalLoadStmt` chain leaf reading a SNode the same kernel mutates
    # cannot be re-cloned at the reverse cursor - the cloned load re-issues the read after the forward writes have
    # updated the SNode, producing wrong gradients (`nan` / `inf`).
    #
    # Internal details:
    #   - Kernel shape: three top-level for-loops over the same `field` SNode.
    #     1. Atomic-add into `field[base, 0]` keyed by `base = floor(data[0]).cast(i32)`.
    #     2. Gated divide-by-self over the whole `field`, gate predicate `field[I, 0][0] > 0.0`.
    #     3. Consumer that reads `field[base, 0]` and accumulates into `data[1]` (the gradient endpoint).
    #   - `mutated_snodes(IB) = {field}`.
    #   - Without the guard: the analyzer admits the chain `GlobalLoad(GlobalPtr(field, base))` as recomputable;
    #     ERAP drops the adstack carrying that chain's value; `BackupSSA::generic_visit` re-clones the chain at
    #     the reverse cursor; the cloned `GlobalLoadStmt` reads `field` POST stage 2 (the divide-by-self has set
    #     every gated cell to the all-ones vector), so the adjoint flowing through `data[0]` blows up.
    #   - With the guard: the chain is rejected, ERAP keeps the original push/pop, reverse pops the iter-k value
    #     verbatim, and `data.grad[0]` matches the analytic all-ones vector.
    #   - `force_scalarize_matrix=True` is structural: the matrix-typed `field` value path is what ERAP latches
    #     onto; with scalarization off the chain's leaf shape changes and the bug no longer fires.
    #   - `cfg_optimization=False` keeps the gate's compound `&&` lowering alive so the reverse clone path
    #     actually triggers; with cfg-opt the cond folds and the clone never happens.
    #   - The trailing `1` axis on `field` matches `MakeAdjoint::visit(RangeForStmt)`'s reverse iteration;
    #     collapsing it folds the loop into a shape that misses ERAP's eligibility entirely.
    vec3 = qd.types.vector(3, qd.f32)
    data = qd.field(dtype=vec3, shape=(2,), needs_grad=True)
    field = qd.field(dtype=vec3, shape=(2, 2, 2, 1), needs_grad=True)

    data[0] = qd.Vector([1.0, 1.0, 1.0])

    @qd.kernel
    def k(data: qd.template(), field: qd.template()):
        for _ in qd.ndrange(1):
            base = qd.floor(data[0]).cast(qd.i32)
            field[base, 0] += data[0]
        for ii, jj, kk, i_b in qd.ndrange(2, 2, 2, 1):
            I = (ii, jj, kk)
            if field[I, i_b][0] > 0.0:
                field[I, i_b] = field[I, i_b] / field[I, i_b]
        for _ in qd.ndrange(1):
            base = qd.floor(data[0]).cast(qd.i32)
            data[1] = data[0] + field[base, 0]

    field[1, 1, 1, 0] = qd.Vector([1.0, 1.0, 1.0])
    data.grad[1] = qd.Vector([1.0, 1.0, 1.0])
    k.grad(data, field)

    expected = [1.0, 1.0, 1.0]
    for d in range(3):
        assert math.isfinite(data.grad[0][d]), f"non-finite at axis {d}: {data.grad[0][d]}"
        assert data.grad[0][d] == pytest.approx(expected[d], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("wrap_inner_in_func", [False, True])
@test_utils.test(ad_stack_experimental_enabled=False)
def test_adstack_basic_gradient_negative(wrap_inner_in_func, n_iter):
    # Negative counterpart of `test_adstack_basic_gradient`: with the adstack disabled the backward compiler
    # cannot reverse a dynamic `range(n_iter)`, so `compute.grad()` raises `QuadrantsCompilationError("Cannot use
    # non static range in Backwards mode")` deterministically for every `n_iter`. Inlined rather than reusing
    # `_run_basic_gradient` because the shall-not-pass path never reaches the gradient assertion, so a shared
    # helper would carry a dead `rel_tol` argument down this branch.
    #
    # Internal details: the `wrap_inner_in_func` axis covers both shapes that should produce the same diagnostic.
    # When False, the dynamic-range loop sits directly inside the outer struct-for, so `ctx.loop_depth > 0` at the
    # range-for visit and the AST transformer raises. When True, the same dynamic-range loop body lives inside a
    # nested `@qd.func` and is invoked from the outer struct-for; the func body must inherit the caller's
    # `loop_depth` so the nested-range check still fires - if the func compile path resets `loop_depth` back to 0
    # the kernel compiles silently and the backward pass produces a wrong gradient with no diagnostic.
    n = 4
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.func
    def inner(v_in: qd.f32) -> qd.f32:
        v = v_in
        for _ in range(n_iter):
            v = v * 0.95 + 0.01
        return v

    if wrap_inner_in_func:

        @qd.kernel
        def compute():
            for i in x:
                y[None] += inner(x[i])

    else:

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


_REVERSE_GLOBAL_CROSS_ITER_PATTERNS = [
    # `(has_outer_for, in_func, load_kind)` toggles for `_run_reverse_global_cross_iter`. `load_kind` selects which
    # `out[...]` index expression the inner range loads from: `cross_iter` is the canonical `out[i-1]` shape with a
    # genuine cross-iteration RAW; `constant_outside_iter_range` reads from a slot of `out` that the loop never
    # writes to, so the load and store live on disjoint iteration slices and the diagnostic must not fire.
    pytest.param(False, False, "cross_iter", id="kernel_top_inline_cross_iter"),
    pytest.param(False, True, "cross_iter", id="kernel_top_in_func_cross_iter"),
    pytest.param(True, False, "cross_iter", id="outer_for_inline_cross_iter"),
    pytest.param(True, True, "cross_iter", id="outer_for_in_func_cross_iter"),
    pytest.param(False, False, "constant_outside_iter_range", id="kernel_top_inline_constant_outside"),
    pytest.param(False, True, "constant_outside_iter_range", id="kernel_top_in_func_constant_outside"),
    pytest.param(True, False, "constant_outside_iter_range", id="outer_for_inline_constant_outside"),
    pytest.param(True, True, "constant_outside_iter_range", id="outer_for_in_func_constant_outside"),
]


def _run_reverse_global_cross_iter(has_outer_for, in_func, load_kind, expect_raise):
    # Single factory covering the reverse-mode shapes that read and write the same `needs_grad` global field
    # `out` from inside an inner non-static range, parametrised on three orthogonal axes:
    #   - `has_outer_for`: when True the inner range is nested under a `range(1)`, so it is AST-nested at
    #     `loop_depth == 1`; when False the inner range becomes the offload-level loop after inlining. The
    #     toggle is materialised via the `range(1) if qd.static(...) else qd.static(range(1))` inline-if pattern
    #     (already used by `_run_unary_loop_carried` in this file) so a single kernel template covers both.
    #   - `in_func`: when True the inner range body lives inside a `@qd.func` invoked from the kernel; when
    #     False the same body is written directly inside the kernel.
    #   - `load_kind`: `cross_iter` puts the canonical `out[i-1]` cross-iteration RAW shape in the body;
    #     `constant_outside_iter_range` replaces the load with `out[n]` (a slot strictly outside the iteration
    #     range `[0, n)`), so the load and store live on disjoint iteration slices and there is no real
    #     cross-iter dependency. Pins the offload-level guard's `references_loop_index` gate.
    n = 4
    out_shape = 2 * n if load_kind == "constant_outside_iter_range" else n
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=out_shape, needs_grad=True)

    @qd.func
    def func_body_cross_iter():
        for i in range(n):
            if i == 0:
                out[i] = x[None]
            else:
                out[i] = out[i - 1] + x[None]

    @qd.func
    def func_body_constant_outside():
        for i in range(n):
            out[i] = out[n] + x[None]

    @qd.kernel
    def compute():
        for _ in range(1) if qd.static(has_outer_for) else qd.static(range(1)):
            if qd.static(in_func):
                if qd.static(load_kind == "cross_iter"):
                    func_body_cross_iter()
                else:
                    func_body_constant_outside()
            else:
                if qd.static(load_kind == "cross_iter"):
                    for i in range(n):
                        if i == 0:
                            out[i] = x[None]
                        else:
                            out[i] = out[i - 1] + x[None]
                else:
                    for i in range(n):
                        out[i] = out[n] + x[None]

    x[None] = 1.0
    if load_kind == "constant_outside_iter_range":
        out[n] = 0.5
    compute()
    for i in range(out_shape):
        out.grad[i] = 0.0
    if load_kind == "cross_iter":
        out.grad[n - 1] = 1.0
        expected_x_grad = float(n)
    else:
        for i in range(n):
            out.grad[i] = 1.0
        expected_x_grad = float(n)
    x.grad[None] = 0.0
    if expect_raise:
        with pytest.raises(
            (qd.QuadrantsCompilationError, RuntimeError),
            match=r"non static range|Cross-iteration read-after-write",
        ):
            compute.grad()
    else:
        compute.grad()
        assert float(x.grad[None]) == pytest.approx(expected_x_grad, rel=1e-6)


@pytest.mark.parametrize("has_outer_for, in_func, load_kind", _REVERSE_GLOBAL_CROSS_ITER_PATTERNS)
@test_utils.test(require=qd.extension.adstack)
def test_reverse_global_cross_iter_with_adstack(has_outer_for, in_func, load_kind):
    # Reverse-mode AD on a kernel that reads and writes the same `needs_grad` global field from inside a non-static
    # range must either produce the analytical gradient or raise a clear diagnostic. With adstack on, only the
    # kernel-top shapes with a genuine cross-iteration RAW raise; the outer-for shapes and the
    # constant-outside-iter-range shapes produce the analytical gradient.
    #
    # Internal details: `AdStackAllocaJudger` only promotes cross-iter dependencies on local `AllocaStmt`s, never
    # through global field stores; cross-iter through a global needs_grad field at the offload level is therefore
    # not handled in either adstack setting and must raise. The offload-level guard in `auto_diff.cpp` pairs same-
    # SNode `(GlobalStore, GlobalLoad)` and raises on structurally-different index `Stmt*`s, but only when both
    # indices transitively reference a `LoopIndexStmt` - which suppresses the false positive on the
    # constant-outside-iter-range shape where the load reads a slot the loop never writes to. The outer-for shapes
    # route the cross-iter chain through an AST-nested range whose backward the AD pass handles correctly via the
    # IB analysis when adstack is on.
    expect_raise = (not has_outer_for) and load_kind == "cross_iter"
    _run_reverse_global_cross_iter(has_outer_for, in_func, load_kind, expect_raise)


@pytest.mark.parametrize("has_outer_for, in_func, load_kind", _REVERSE_GLOBAL_CROSS_ITER_PATTERNS)
@test_utils.test()
def test_reverse_global_cross_iter_without_adstack(has_outer_for, in_func, load_kind):
    # Same eight shapes as the with-adstack companion. With adstack off, every outer-for shape raises via the
    # AST-level `loop_depth > 0` check (purely on nesting, regardless of body content), and the kernel-top
    # cross-iter shapes raise via the offload-level guard in `auto_diff.cpp`. The kernel-top
    # constant-outside-iter-range shape is the only combination that compiles cleanly: nesting is zero so the
    # AST guard does not fire, and the load index is iteration-independent so the offload-level guard's
    # `references_loop_index` gate also does not fire.
    #
    # Internal details: the caller-loop-depth propagation in `_func_base.py` is what makes the @qd.func variant
    # of the outer-for shape behave identically to the inline variant - without it, the func context would
    # start at `loop_depth = 0` and the AST guard would never fire for the func-routed nested shape.
    expect_raise = has_outer_for or load_kind == "cross_iter"
    _run_reverse_global_cross_iter(has_outer_for, in_func, load_kind, expect_raise)


_REVERSE_SIBLING_COMPONENT_PATTERNS = [
    # `(has_outer_for, in_func)` toggles for `_run_reverse_sibling_component_correct_gradient`. Same axes as the
    # cross-iter matrix, minus the `load_kind` dimension (the body shape here is fixed: same loop index on
    # axis 0 of `out`, differing constants on axis 1).
    pytest.param(False, False, id="kernel_top_inline"),
    pytest.param(False, True, id="kernel_top_in_func"),
    pytest.param(True, False, id="outer_for_inline"),
    pytest.param(True, True, id="outer_for_in_func"),
]


def _run_reverse_sibling_component_correct_gradient(has_outer_for, in_func):
    # Drives a kernel that writes `out[i, 0]` and reads `out[i, 1]` on a 2-axis `needs_grad` field. The store
    # and load both reference `LoopIndexStmt(i)` on axis 0 (same `Stmt*`) and have differing `ConstStmt`s on
    # axis 1. The offload-level guard's per-axis rule treats the differing constant axis as iter-independent
    # sibling access, so the kernel must compile and produce the analytical gradient.
    n = 4
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=(n, 2), needs_grad=True)

    @qd.func
    def func_body():
        for i in range(n):
            out[i, 1] = x[None]
            out[i, 0] = out[i, 1] * 2.0

    @qd.kernel
    def compute():
        for _ in range(1) if qd.static(has_outer_for) else qd.static(range(1)):
            if qd.static(in_func):
                func_body()
            else:
                for i in range(n):
                    out[i, 1] = x[None]
                    out[i, 0] = out[i, 1] * 2.0

    x[None] = 1.0
    compute()
    for i in range(n):
        out.grad[i, 0] = 1.0
        out.grad[i, 1] = 0.0
    x.grad[None] = 0.0
    compute.grad()
    # `out[i, 0] = (out[i, 1] = x) * 2.0 = 2x` per iter; seed `out.grad[i, 0] = 1` for every i.
    # `d(out[i, 0]) / d(x) = 2` per iter -> `x.grad = 2 * n`.
    assert float(x.grad[None]) == pytest.approx(2.0 * n, rel=1e-6)


@pytest.mark.parametrize("has_outer_for, in_func", _REVERSE_SIBLING_COMPONENT_PATTERNS)
@test_utils.test(require=qd.extension.adstack)
def test_reverse_sibling_component_correct_gradient_with_adstack(has_outer_for, in_func):
    # Sibling-component access on a multi-axis `needs_grad` field must compile and produce the analytical
    # gradient. The store and load share the same `LoopIndexStmt` on axis 0 and differ only on a constant
    # axis, which is sibling access in the same iteration, not cross-iter RAW.
    #
    # Internal details: the offload-level guard's per-axis `indices_have_no_cross_iter_dependency` rule
    # treats axes where neither index references `LoopIndexStmt` as iter-independent and ignores their
    # difference, so `out[i, 0]` vs `out[i, 1]` is not flagged.
    _run_reverse_sibling_component_correct_gradient(has_outer_for, in_func)


def _run_reverse_kernel_top_indep_iters_correct_gradient():
    # Independent-iteration kernel-top range-for (no cross-iteration RAW). Both adstack settings must
    # produce the analytical gradient `2 * n` for the per-iteration store `out[i] = x[None] * 2`.
    n = 4
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            out[i] = x[None] * 2.0

    x[None] = 1.0
    compute()
    for i in range(n):
        out.grad[i] = 1.0
    x.grad[None] = 0.0
    compute.grad()
    assert float(x.grad[None]) == pytest.approx(2.0 * n, rel=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_reverse_kernel_top_indep_iters_with_adstack():
    # Kernel-top non-static range whose body has no cross-iter RAW on a `needs_grad` field must
    # produce the analytical gradient with adstack on.
    #
    # Internal details: the offload-level guard pairs only same-SNode `(store, load)`, so a body
    # that only stores `out[i] = x[None] * 2` (no load on `out`) compiles cleanly. Adstack itself has
    # nothing to do here either - each iteration is independent.
    _run_reverse_kernel_top_indep_iters_correct_gradient()


@test_utils.test()
def test_reverse_kernel_top_indep_iters_without_adstack():
    # Same independent-iteration shape as the with-adstack companion, must produce the analytical
    # gradient with adstack off.
    #
    # Internal details: the AST-level `loop_depth > 0` check does not fire (range-for is at
    # `loop_depth == 0`) and there is no cross-iter chain to drop, so the gradient comes out correct
    # without the adstack.
    _run_reverse_kernel_top_indep_iters_correct_gradient()


def _run_reverse_kernel_top_in_place_accum_correct_gradient():
    # In-place accumulation `out[i] = out[i] + x[None]` at the kernel-top range-for. Sanity check that
    # the offload-level guard's pointer-equality on index `Stmt*`s allows this shape - store and
    # load on `out` share the same `LoopIndexStmt` so `indices_pointer_match` returns true and the
    # guard does not fire.
    n = 4
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            out[i] = out[i] + x[None]

    x[None] = 1.0
    for i in range(n):
        out[i] = 0.0
    compute()
    for i in range(n):
        out.grad[i] = 1.0
    x.grad[None] = 0.0
    compute.grad()
    # `out[i] = out[i] + x` reduces to `out[i] += x`; with `out[i]` initialised to 0, the per-iteration
    # adjoint accumulates 1 into `x.grad` for each of n elements when `out.grad[i] = 1`.
    assert float(x.grad[None]) == pytest.approx(float(n), rel=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_reverse_kernel_top_in_place_accum_with_adstack():
    # In-place accumulation `out[i] = out[i] + x[None]` at the offload-level loop must compile and
    # produce the analytical gradient: the offload-level guard intentionally exempts this shape.
    #
    # Internal details: the store dest `out[i]` and the load src `out[i]` reference the same
    # `LoopIndexStmt`, so `OffloadLevelGlobalCrossIterRAWChecker::indices_pointer_match` returns true
    # and the guard does not fire. The per-iteration adjoint already handles in-place accumulation
    # correctly: `x.grad` accumulates 1 per element.
    _run_reverse_kernel_top_in_place_accum_correct_gradient()


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
    # sum_{k=0..4} (k+1) = 15. Five exactly-representable f32 accumulations so the result is exact up to a handful of
    # ULPs.
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


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, debug=True)
def test_adstack_overflow_raises():
    # Runs a backward pass with a for-loop longer than the adstack can hold, and asserts the overflow surfaces as a
    # regular Python exception on the next `qd.sync()` - not a silent wrong gradient and not a process crash. This
    # is what users see when their differentiable kernel is too deep for the current `ad_stack_size`, and the error
    # message should tell them how to raise the capacity.
    #
    # Internal detail: both LLVM and SPIR-V defer the error to the next `qd.sync()` (same pattern as CUDA async
    # errors) so we do not pay a sync-per-launch. LLVM polls `runtime->adstack_overflow_flag` from
    # `LlvmProgramImpl::synchronize()` via `check_adstack_overflow()`; SPIR-V's gfx runtime raises via `QD_ERROR`
    # on sync. The bounds-check codepath in both backends is gated on `debug`; release builds elide it on the
    # premise that `determine_ad_stack_size` produces a tight upper bound. This test deliberately misconfigures
    # the capacity below the kernel's actual peak push count, which the sizer cannot foresee, so the runtime
    # check has to be live for the deferred raise to fire - `debug=True` keeps it live.
    compute, _, _ = _overflowing_compute()
    # On LLVM the runtime raises QuadrantsAssertionError (subclass of AssertionError) from
    # check_adstack_overflow; on SPIR-V the gfx runtime raises RuntimeError via QD_ERROR. We accept either,
    # matching only the message prefix.
    with pytest.raises(QuadrantsAssertionError, match=r"[Aa]dstack overflow"):
        compute.grad()
        qd.sync()


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, debug=True)
def test_adstack_overflow_flag_resets_after_catch():
    # Once `check_adstack_overflow()` raises, the runtime must clear its overflow flag so a subsequent `qd.sync()`
    # (with no new overflowing grad launch in between) returns normally. Without the reset the user would see a
    # stale overflow exception every time they sync after the first one, which makes diagnosis and recovery
    # impossible. `debug=True` keeps the per-push bounds check live.
    compute, _, _ = _overflowing_compute()
    with pytest.raises(QuadrantsAssertionError, match=r"[Aa]dstack overflow"):
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
    # `offline_cache=False` is load-bearing: a cached compile from one run could mask a regression that flipped the
    # codegen back to the function-scope path; the test must force a fresh compile every run so the `QD_ERROR_IF` on a
    # regressed tree actually fires and terminates the process.
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


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, debug=True)
def test_adstack_overflow_multithreaded():
    # Multi-element field so several threads execute the overflowing grad body in parallel. Asserts the overflow
    # still surfaces as a single Python exception rather than deadlocking, crashing, or racing on the flag. Every
    # thread writes the same flag value (non-zero), so a race on the write is benign; this test pins that the
    # read side is also safe (one raise per sync regardless of how many threads flipped the bit). `debug=True`
    # keeps the per-push bounds check live (release-build codegen elides it - see `test_adstack_overflow_raises`
    # for the rationale).
    compute, _, _ = _overflowing_compute(n_elements=16)
    with pytest.raises(QuadrantsAssertionError, match=r"[Aa]dstack overflow"):
        compute.grad()
        qd.sync()


@pytest.mark.parametrize("force_sync", [False, True])
def test_adstack_overflow_caught_then_clean_teardown(tmp_path, force_sync):
    # This test runs the kernel in a child process (not via `@test_utils.test`, which iterates arches), so it
    # cannot rely on the decorator's `require=qd.extension.adstack` skip. Guard manually: skip if the CPU backend
    # was not built with the adstack extension, matching what the sibling overflow tests get from the decorator.
    if not is_extension_supported(qd.cpu, qd.extension.adstack):
        pytest.skip("adstack extension not available on cpu")

    # Pins the per-launch-raise + clean-teardown contract for `Program::finalize()`. The per-launch
    # `check_adstack_overflow_and_assert()` poll wired into `Program::launch_kernel` surfaces an overflow at
    # the very next kernel-launch entry on synchronous backends (CPU). On async backends (CUDA / AMDGPU /
    # Metal / Vulkan) the kernel may still be in flight when `launch_kernel` returns, so the post-launch poll
    # reads a not-yet-set flag - the overflow is then surfaced at the next `qd.sync()` via the post-drain
    # check in `LlvmProgramImpl::synchronize_and_assert` (or the host-mapped readback in
    # `GfxRuntime::synchronize`). The `force_sync` parametrisation toggles whether the user issues an
    # explicit `qd.sync()`. Either way the teardown contract holds: the two teardown `synchronize()` calls
    # inside `Program::finalize()` re-enter the LLVM `check_adstack_overflow_and_assert` path, and
    # `LlvmProgramImpl::pre_finalize()` must have set `finalizing_ = true` early enough that the per-launch
    # poll AND the `synchronize_and_assert` poll BOTH short-circuit during the destructor. If either path
    # re-raised, the destructor would `std::terminate()` instead of returning a clean exit code (-6 / SIGABRT
    # on macOS, std::terminate's _Exit on linux). The subprocess asserts the child returns with exit code 0.
    #
    # Internal details: the subprocess is launched from a temp file because `python -c "<kernel>"` breaks
    # Quadrants' kernel source-inspect (`getsourcelines` cannot find the source of an inlined `-c` string).
    # `debug=True` is required because release-build LLVM codegen elides the per-push bounds check on the
    # premise the sizer's bound is tight, so a manually-misconfigured `ad_stack_size=32` kernel only flips
    # the runtime overflow flag in debug mode. Without the flag set there is no flag for the teardown
    # guard to swallow, and the bug this test pins cannot trigger.
    child_script = textwrap.dedent(
        f"""
        from contextlib import nullcontext

        import pytest
        import quadrants as qd
        from quadrants.lang.exception import QuadrantsAssertionError

        qd.init(arch=qd.cpu, ad_stack_experimental_enabled=True, ad_stack_size=32, debug=True)

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

        # CPU is the only arch the child runs on for now; synchronous-backend semantics apply.
        # The per-launch poll wired into `Program::launch_kernel` surfaces the overflow at `compute.grad()`.
        # On a future async-arch parametrisation, flip `is_sync_backend` and the qd.sync() branch becomes
        # the raising path while compute.grad() drops to nullcontext.
        is_sync_backend = True
        force_sync = {force_sync}

        def raises_overflow():
            return pytest.raises(QuadrantsAssertionError, match=r"[Aa]dstack overflow")

        with raises_overflow() if is_sync_backend else nullcontext():
            compute.grad()
        if force_sync:
            with raises_overflow() if not is_sync_backend else nullcontext():
                qd.sync()
        # Process exits without `qd.sync()` (when force_sync=False). Teardown's two `synchronize()` calls
        # plus their per-launch polls must short-circuit on `finalizing_`; otherwise the destructor
        # double-raises and the process exits non-zero / SIGABRTs.
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


@pytest.mark.needs_torch
@test_utils.test(require=qd.extension.adstack, debug=True)
def test_adstack_overflow_diagnostic_and_auto_recovery():
    import torch

    # Cross-backend regression for the always-on overflow detection + diagnostic + auto-recovery
    # contract shipped on this branch. The kernel's inner trip count is bounded by `n[0]`, an `int32`
    # ndarray. The per-task adstack metadata cache invalidation tracks `Ndarray.write` /
    # `Ndarray.fill` via `Program::ndarray_data_gen_` - mutations that route through Quadrants APIs
    # invalidate cleanly. DLPack zero-copy mutations (`.to_torch(copy=False)`) bypass that tracking,
    # so the cache holds a stale `max_size`; the next reverse launch overflows.
    #
    # The contract pinned here:
    #   1. The first launch with `n[0] = 2` populates the cache with `max_size = 2`.
    #   2. A DLPack-backed torch view writes `n[0] = 64`. Quadrants's gen counter is NOT bumped.
    #   3. The next reverse launch reads cached `max_size = 2`, pushes 64, overflows. The host poll
    #      raises with the enriched diagnostic naming kernel + offload-task index. The raise site
    #      ALSO bulk-invalidates the adstack-sizer caches on its way out.
    #   4. The user catches the exception. They do NOT need to manually adjust `ad_stack_size`. On the NEXT reverse
    #      launch, the sizer reruns from scratch (cache invalidated), reads the mutated `n[0] = 64`, sizes capacity
    #      to 64, and the kernel runs to completion with the correct gradient.
    #
    # This auto-recovery contract is what lets the user's training-loop code recover from a
    # transient cache-staleness window without per-iteration retries: the offending data has
    # already been mutated in place; once the cache reflects it, every subsequent run just works.
    n = qd.ndarray(qd.i32, shape=(1,))
    x = qd.field(qd.f32, shape=(1,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(n_arr: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in x:
            v = x[i]
            for _ in range(n_arr[0]):
                y[None] += qd.sin(v)
                v = v + 1.0

    # Step 1: small `n[0]`, kernel runs cleanly, cache populated with `max_size = 2`.
    n[0] = 2
    x[0] = 0.1
    y[None] = 0.0
    compute(n)
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad(n)
    qd.sync()

    # Step 1.5: Python-side mutation through `Ndarray.write` (`n[0] = 8`). The setter routes through Quadrants's
    # tracking and bumps `ndarray_data_gen_` for the bound DeviceAllocation, so the per-task adstack metadata
    # cache invalidates cleanly: the next launch reruns the sizer with `n[0] = 8`, sizes capacity to 8, and the
    # kernel runs to completion without raising. This pins the clean-invalidation contract on every backend
    # (no DLPack involvement, no overflow, no recovery exception).
    n[0] = 8
    y[None] = 0.0
    compute(n)
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad(n)
    qd.sync()
    expected_after_clean_grow = sum(math.cos(0.1 + k) for k in range(8))
    assert x.grad[0] == pytest.approx(expected_after_clean_grow, rel=1e-4)

    # Reset state for the DLPack-bypass scenario: bring `n[0]` back down to 2 through the Quadrants-tracked
    # setter so the next cached `max_size` is the small value the bypass mutation will outgrow.
    n[0] = 2
    y[None] = 0.0
    compute(n)
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad(n)
    qd.sync()

    # The DLPack-bypass scenario below requires `to_torch(copy=False)` which is unsupported on
    # Vulkan because Quadrants and torch do not currently share a command queue there
    # (`_can_zerocopy_field` returns false on i32 ndarrays and the export raises
    # `Zero-copy not available for arch=vulkan, dtype=i32`). Steps 1 + 4 above already verified
    # the cleanly-running path; bail out before the bypass-mutation portion on Vulkan.
    if qd.lang.impl.get_runtime().prog.config().arch == qd.vulkan:
        return

    # Step 2: DLPack-bypass mutation. `Ndarray.write` would have bumped `ndarray_data_gen_` and
    # invalidated the cache cleanly; `to_torch(copy=False)` shares storage with no Quadrants hook,
    # so the cache sees no change. On Metal `to_torch(copy=False)` returns an `mps:0` tensor and
    # writes through it dispatch asynchronously through Metal Performance Shaders; an explicit
    # `torch.mps.synchronize()` is required to flush those writes to the shared buffer the
    # Quadrants device kernel reads from. Without it the next Quadrants launch sees the stale
    # `n[0] = 2` and the overflow detection misses entirely. CPU / CUDA / AMDGPU paths do not
    # need the equivalent on this code path because their `to_torch` returns a tensor on a
    # device where writes are coherent without an additional sync.
    n_view = n.to_torch(copy=False)
    n_view[0] = 64
    if qd.lang.impl.get_runtime().prog.config().arch == qd.metal:
        torch.mps.synchronize()
    qd.sync()

    # Step 3: next reverse launch may overflow. On backends with a stale-cache shortcut (LLVM-GPU
    # `try_llvm_per_task_ad_stack_cache_hit`, SPIR-V `try_per_task_ad_stack_cache_hit`) the cached
    # `max_size = 2` is reused because `ndarray_data_gen` has not been bumped, the kernel pushes 64,
    # and the host poll raises with the enriched diagnostic naming kernel + offload-task index. On
    # LLVM-CPU the host-eval branch always re-evaluates the size expression per launch via
    # `try_size_expr_cache_hit`, which observes the live ndarray read and self-invalidates on
    # mismatch - that path never raises here, so the second compute.grad call returns cleanly. The
    # `pytest.raises if shortcut else nullcontext` pattern handles both paths uniformly without arch-narrowing the
    # test.
    y[None] = 0.0
    compute(n)
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    backend_uses_per_task_cache_shortcut = qd.lang.impl.get_runtime().prog.config().arch != qd.cpu
    raises_overflow = pytest.raises(QuadrantsAssertionError, match=r"[Aa]dstack overflow")
    with raises_overflow if backend_uses_per_task_cache_shortcut else nullcontext() as exc_info:
        compute.grad(n)
        qd.sync()
    if exc_info is not None:
        msg = str(exc_info.value)
        # When the inner range is bounded by an ndarray read, the user sees the actual mutated size in the
        # error (e.g. allocated=[1], required=[64]) and a recovery flow pointing at the tensor mutation
        # performed outside Quadrants's tracking. The generic "this might also be a Quadrants bug"
        # alternative only appears when the diagnostic cannot pin the cause down.
        assert "DLPack" in msg, f"missing DLPack-bypass cause hint in: {msg}"
        assert "Restart" in msg, f"missing recovery flow in: {msg}"
        assert "Offending task" in msg, f"missing identity block in: {msg}"
        assert "compute" in msg, f"missing kernel name in: {msg}"
        assert "Synchronous sizer rerun: required max_size = [" in msg, f"missing sync-sizer-rerun line in: {msg}"

    # Step 4: auto-recovery. If the previous launch overflowed, the raise site bulk-invalidated the
    # adstack-sizer caches when the synchronous sizer rerun confirmed a stale-cache cause. The next
    # reverse launch reruns the sizer from scratch, reads `n[0] = 64`, sizes capacity to 64, and
    # the kernel runs cleanly with no second overflow. Either way (stale-cache backend that
    # recovered or auto-invalidating CPU backend that never overflowed), the closed-form gradient
    # below is the contract for every backend.
    y[None] = 0.0
    compute(n)
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad(n)
    qd.sync()
    # Closed-form gradient sanity: y = sum_{k=0..n-1} sin(x + k), so dy/dx = sum cos(x + k).
    expected = sum(math.cos(0.1 + k) for k in range(64))
    assert x.grad[0] == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("n_iter", [30, 100])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_near_capacity(n_iter):
    # Pins that a field-load-bounded reverse-mode loop sizes its adstack from the live field value at each launch.
    # Parametrized on both sides of the previous K+2=32 overflow boundary: `n_iter=30` would have required 32 slots,
    # `n_iter=100` would have required 102 slots. Both cases now run to completion with the analytical gradient since
    # the structural pre-pass captures the symbolic trip count and the host launcher evaluates it per dispatch.
    #
    # Internal details: the trip count is loaded from a runtime field (`n_iter_fld`) rather than a Python int constant
    # so the `irpass::determine_ad_stack_size` structural pre-pass captures a `SizeExpr::FieldLoad` (not a `Const`). The
    # host evaluator in `LlvmRuntimeExecutor::publish_adstack_metadata` reads `n_iter_fld` via `SNodeRwAccessorsBank`,
    # recomputes the per-launch stride / offsets / max-sizes, and writes them into the runtime metadata buffers that the
    # LLVM codegen for `AdStack*` reads via `LLVMRuntime_get_adstack_*`. Restricted to LLVM-CPU here: SPIR-V still bakes
    # `max_size` as a codegen-time immediate (future work will move SPIR-V onto the same per-launch metadata path, at
    # which point the arch restriction can drop). Companion to `test_adstack_overflow_raises` which still exercises the
    # explicit `ad_stack_size=32` knob (that path forces every adstack to exactly 32 slots and intentionally overflows).
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
    # holds them, so nothing flips their order. ReverseOuterLoops flips each loop's `reversed` iteration direction and
    # also pairwise-swaps sibling for-loops inside every non-IB container block the pass walks through. Non-loop
    # statements (range-bound loads, alloca, etc.) stay at their original positions so SSA operands still dominate both
    # swapped fors. The outer `for _ in range(1)` dummy is the smallest shape that places the two siblings inside a
    # non-IB container (the frontend rejects a bare sequence of top-level for-loops as "mixed usage of for-loops and
    # statements without looping"); `n[None]` from a field forces the inner ranges to be dynamic so the bug manifests
    # (static-unrolled ranges go through a different path that already works).
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
    # Asserts that a grad kernel whose range is derived at launch time from an ndarray shape (e.g.
    # `qd.ndrange(arr.shape[0], arr.shape[1])`) sizes the per-dispatch adstack heap from the actual launch-time iter
    # count rather than from the SPIR-V codegen's grid-stride advisory cap (`kMaxNumThreadsGridStrideLoop = 131072`).
    # Sizing from the cap on a small workload would request `131072 * per_thread_stride * sizeof(float)` (e.g. ~40 GiB
    # at 10 f32 vars and `ad_stack_size=4096`), exceeding Apple Silicon's `MTLDevice.maxBufferLength` (~28 GiB on a 48
    # GiB-unified M4 Max), and the Metal RHI's nil-buffer fallback would silently bind nil at `setBuffer:atIndex:2` so
    # writes drop, reads return 0, and the backward NaNs. The codegen records the shape-lookup product backing the
    # runtime-resolved `end_stmt` into `RangeForAttributes::end_shape_product`; the runtime `launch_kernel` reads each
    # shape from the `LaunchContextBuilder` args buffer and tightens `advisory_total_num_threads` to `actual_iter_count
    # = rows * cols = 6`, so only ~240 KB of adstack heap is allocated.
    #
    # Internal details: `ad_stack_size=4096` + ten loop-carried f32 variables is tuned so that the cap-fallback
    # 131072-thread allocation request crosses the smallest plausible Apple Silicon `maxBufferLength` - the test would
    # otherwise silently pass on hardware with large unified memory. The oversize symptom only surfaces on the SPIR-V
    # heap-backed adstack path whose per-dispatch sizing depends on the advisory thread count; the LLVM path sizes the
    # adstack slab once per runtime against `num_cpu_threads` and cannot exhibit the same nil-buffer regression. The
    # test still runs on every backend so the finite-difference cross-check catches a regression in the grad computation
    # regardless of which path it lives in.
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

    # Analytic oracle. The kernel is affine in `arr[i, j]` (each `v_k` is `v_k * c_k + d_k` for three iterations, so
    # `d(v_k_final) / d(arr[i, j]) = c_k^3`), and `out[0]` sums all ten recurrences, so the closed-form gradient per
    # cell is `sum_k c_k^3`. Independent of the backward emission so a wrong-but-non-NaN gradient (the failure mode when
    # the adstack heap was bound to Metal's nil-fallback and reads came back as zero) still trips the assertion;
    # tolerance bounded by f32 accumulation roundoff only, not finite-difference cancellation.
    coeffs = np.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10], dtype=np.float64)
    expected_per_cell = float((coeffs**3).sum())
    # `rtol=1e-4` rather than tight-to-backward-roundoff because on AMD Vulkan (RADV) the adjoint accumulation
    # through ten loop-carried recurrences drifts by a few parts in 1e5 relative to the analytic value; the
    # tighter bound catches the drift without a corresponding correctness signal. The regression this test
    # guards against (nil device buffer -> zero read -> NaN adjoint) trips any tolerance at all.
    np.testing.assert_allclose(got_grad, np.full_like(arr_np, expected_per_cell), rtol=1e-4, atol=0)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_bounded_inner_loop_sized_by_structural_prepass():
    # Pins that reverse-mode AD on SPIR-V backends through a statically bounded inner `range(N)` sizes the adstack from
    # the product of enclosing `RangeForStmt` trip counts via the structural pre-pass, not from any compile-time
    # fallback. There is no `default_ad_stack_size` anymore: any alloca the pre-pass cannot bound is a hard compile
    # error, so the only way this test passes is through the structural walk correctly folding the constant trip counts.
    #
    # Internal details: `irpass::determine_ad_stack_size` runs a structural pre-pass that walks each adaptive
    # `AdStackAllocaStmt`'s push sites and computes `max_size` from the product of enclosing `RangeForStmt` trip counts
    # when every enclosing range has a constant integer begin/end (folded through `BinaryOpStmt`). Runs on every
    # backend: on SPIR-V this is the only bound-derivation path; on LLVM the inner-range bounds are rewritten through
    # `LoopIndexStmt` in a shape the structural analyzer does not fold, and the kernel routes through the symbolic-tree
    # runtime-evaluator path instead - the gradient assertion catches a regression in either path.
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


@test_utils.test(require=qd.extension.adstack, default_ip=qd.i64)
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


@pytest.mark.xfail(
    reason=(
        "Cross-kernel sibling of `test_adstack_sizer_trip_count_ndarray_mutated_after_launch_read`. When a "
        "reverse-mode kernel uses `a[i_e]` as a loop trip count on a `qd.ndarray` and a separate kernel "
        "mutates `a` on device between the forward and `.grad()` calls, the backward sizer re-dispatches "
        "and reads the post-mutation value, so the reverse pass walks more inner iterations than the "
        "forward pushed and accumulates gradient at indices the forward never visited. Documented as a "
        "known limitation in `docs/source/user_guide/autodiff.md`."
    ),
    strict=True,
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_sizer_trip_count_qd_ndarray_mutated_by_separate_kernel():
    # Pins the cross-kernel silent-wrong-gradient pattern: a reverse-mode kernel reads `a[i_e]` as an inner-loop trip
    # count on a device-resident `qd.ndarray`, and a separate kernel mutates `a` between the forward and `.grad()`
    # calls. The reverse pass walks with the post-mutation trip count, so `x.grad` ends up non-zero at indices the
    # forward never visited.
    #
    # Internal details: the forward sizer reads `a = 5` and the main kernel pushes 5 entries per outer iter; the sibling
    # kernel then writes `a = 10` on device; the backward sizer re-dispatches, reads `a = 10`, and the reverse pass
    # walks 10 inner iterations. The result is `x.grad[5..9] = 0.8` at `x[k] = 0.1` instead of the analytical `0.0`.
    # `qd.ndarray` (rather than numpy) is required so the sibling kernel's device write persists across launches; the
    # same construction with a numpy ndarray may not reproduce because per-launch h2d uploads can erase the sibling
    # kernel's device write.
    N_X = 16
    N = 4

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def init_to_5(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            a[i] = 5

    @qd.kernel
    def overwrite_to_10(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            a[i] = 10

    @qd.kernel
    def use_bound(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_e in range(a.shape[0]):
            for j in range(a[i_e]):
                loss[None] += x[j] * x[j]

    for i in range(N_X):
        x[i] = 0.1
    a = qd.ndarray(qd.i32, shape=(N,))
    init_to_5(a)
    use_bound(a)
    overwrite_to_10(a)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    use_bound.grad(a)
    qd.sync()

    assert loss[None] == pytest.approx(4 * 5 * 0.1 * 0.1, rel=1e-5)
    for k in range(N_X):
        if k < 5:
            assert x.grad[k] == pytest.approx(2 * 4 * 0.1, rel=1e-5)
        else:
            assert x.grad[k] == pytest.approx(0.0, abs=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_field_load_bounded_loop_evaluated_per_launch():
    # Pins the host-evaluated SizeExpr path end-to-end: a reverse-mode adstack whose inner-loop bound is a scalar i32
    # field load must size the per-thread heap slice from the live field value at each launch. The structural pre-pass
    # captures a `SizeExpr::FieldLoad` leaf; the launcher evaluates it via `SNodeRwAccessorsBank` before each dispatch
    # and resizes the heap accordingly. The kernel is run with `n_iter_fld[None]` set to 1, 20 and then 50 in sequence:
    # each launch picks up the current field value, resizes the adstack heap, and runs to completion with the analytical
    # gradient `0.95 ** n_iter`.
    #
    # Internal details: the symbolic bound tree is flattened into the serialisable `SerializedSizeExpr` form and stored
    # inside the per-backend per-alloca task attributes, so this test exercises the same path whether the kernel is
    # freshly compiled or restored from the offline cache. On LLVM the bound is published into
    # `LLVMRuntime::adstack_{per_thread_stride,offsets,max_sizes}` by `publish_adstack_metadata` before each dispatch;
    # on SPIR-V it is uploaded into the `AdStackMetadata` StorageBuffer that the shader reads at every push / load-top
    # site.
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
@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_range_bounded_by_ndarray_read_at_outer_index(ndarray_kind):
    # Pins the `ExternalTensorRead`-over-`LoopIndex` `MaxOverRange` wrap in the `SizeExpr` pre-pass: a reverse-mode
    # adstack whose inner range `range(a[i])` is bounded by a scalar ndarray read at the enclosing outer loop index. The
    # pre-pass must build `MaxOverRange(var, 0, outer_end, ExternalTensorRead(a, [var]))` for the alloca's multiplier;
    # the launch-time evaluator enumerates the outer range, reads `a[var]` at each iteration, and takes the max to size
    # the per-thread adstack heap exactly.
    #
    # Internal details: runs on every backend. On CPU the launch-time SizeExpr is evaluated host-side via
    # `evaluate_adstack_size_expr`, with ndarray element reads going through the real host pointer
    # `set_host_accessible_ndarray_ptrs` mirrored into `array_ptrs`. On CUDA / AMDGPU the host encodes the tree into
    # device-side bytecode (`encode_adstack_size_expr_device_bytecode`) and calls `runtime_eval_adstack_size_expr` to
    # run the interpreter on the device. On Metal / Vulkan the bytecode is emitted as a SPIR-V compute shader launched
    # from `GfxRuntime::launch_kernel`. All three paths are the only way to resolve an `ExternalTensorRead` against a
    # GPU-private ndarray without round-tripping the whole allocation to host. Asserts the analytical gradient `0.95 **
    # a[i]` per outer iteration so a regression in the wrap or in either the host or device evaluator shows up as a
    # value mismatch rather than an overflow crash. Parametrised over the ndarray argument kind because `numpy`/torch
    # inputs lower through `set_arg_external_array_with_shape` (writes the raw host pointer straight into `array_ptrs`)
    # while `qd.ndarray` inputs lower through `set_arg_ndarray_impl` + `set_ndarray_ptrs` (stashes a `DeviceAllocation
    # *` first, then the launcher resolves it). The CPU launcher mirrors the resolved pointer back into `array_ptrs`;
    # the CUDA / AMDGPU launchers don't need to because the device interpreter reads the ndarray data pointer straight
    # out of `ctx->arg_buffer` at the offset the encoder precomputed from `args_type`, sidestepping the host-side
    # `array_ptrs` map entirely.
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


@pytest.mark.parametrize(
    "trip_count_source",
    ["qd_ndarray", "field"],
    ids=["qd_ndarray", "field"],
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_metadata_cache_invalidates_on_host_mutation(trip_count_source):
    # Pins per-task adstack metadata cache invalidation against host-side mutation of the structure that supplies
    # the inner trip count. A reverse-mode kernel whose inner range is `range(n[i])` populates the cache on the
    # first launch with `max_size = n[i]` evaluated against the current contents. A subsequent host-side mutation
    # via `Ndarray::write` (qd.ndarray case) or via the `SNodeRwAccessorsBank` writer kernel (field case) must
    # bump the matching generation counter so the next launch evicts the entry and re-runs the sizer.
    #
    # Internal details: the cache key on every backend is `(AdStackSizingInfo *, snode_write_gen[snode_ids],
    # ndarray_data_gen[devalloc])`. The qd.ndarray path goes through `ndarray_data_gen` keyed by the
    # `DeviceAllocation` holder address; the field path goes through `snode_write_gen` keyed by `SNode::id`.
    # Without the bump on either side the second launch sees the same key, returns the stale `max_size` for the
    # previous contents, and the reverse pass walks the wrong number of inner iters per outer iteration. On Metal
    # the symptom is heap reads from out-of-bounds slots that produce garbage gradients (e.g. `x.grad[k]` in the
    # dozens instead of the analytical `0.8`); on CPU the host-eval path replays observed reads and recovers
    # without the explicit gen bump, so the test asserts gradient values rather than an overflow trap to catch the
    # bug on every backend uniformly.
    N = 4
    N_X = 16

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    if trip_count_source == "qd_ndarray":
        n_obj = qd.ndarray(qd.i32, shape=(N,))

        @qd.kernel
        def compute(n: qd.types.ndarray(dtype=qd.i32, ndim=1)):
            for i_e in range(n.shape[0]):
                accum = 0.0
                for j in range(n[i_e]):
                    accum = accum + x[j] * x[j]
                loss[None] += accum

        def set_n(val):
            for i in range(N):
                n_obj[i] = val

        def call_compute():
            compute(n_obj)

        def call_compute_grad():
            compute.grad(n_obj)

    else:
        n_obj = qd.field(qd.i32, shape=(N,))

        @qd.kernel
        def compute():
            for i_e in range(N):
                accum = 0.0
                for j in range(n_obj[i_e]):
                    accum = accum + x[j] * x[j]
                loss[None] += accum

        def set_n(val):
            for i in range(N):
                n_obj[i] = val

        def call_compute():
            compute()

        def call_compute_grad():
            compute.grad()

    set_n(8)
    for i in range(N_X):
        x[i] = 0.1
    loss[None] = 0.0
    call_compute()
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    call_compute_grad()
    qd.sync()
    assert loss[None] == pytest.approx(N * 8 * 0.01, rel=1e-5)
    for k in range(8):
        assert x.grad[k] == pytest.approx(N * 2 * 0.1, rel=1e-5)
    for k in range(8, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)

    set_n(16)
    for i in range(N_X):
        x[i] = 0.1
    loss[None] = 0.0
    call_compute()
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    call_compute_grad()
    qd.sync()
    assert loss[None] == pytest.approx(N * 16 * 0.01, rel=1e-5)
    for k in range(N_X):
        assert x.grad[k] == pytest.approx(N * 2 * 0.1, rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_range_bounded_by_multidim_ndarray_read():
    # Pins multi-axis stride handling in the `ExternalTensorRead` evaluator. The sizer routes a reverse-mode inner trip
    # count `range(a[i, j])` through `SizeExpr::ExternalTensorRead(a, [var_i, var_j])` for a 2-D ndarray `a`; the host
    # evaluator, the CUDA / AMDGPU device interpreter in the LLVM runtime, and the SPIR-V sizer compute shader must all
    # fold the indices into a C-order linear offset `i * shape[1] + j`, not the naive stride-1 sum `i + j`. The
    # worst-case shape below isolates a single non-zero entry at `a[2, 2] = 100` so the stride-1 path (sum over `(i, j)`
    # with `i + j < rows + cols - 1`) visits only the leading diagonal and the first row/column, all of which are zero;
    # the buggy sizer then picks `max = 0`, clamps to 1, and the `a[2, 2] = 100` cell pushes 100 times into an adstack
    # sized for 1. The fixed evaluator picks `max = 100` and the kernel runs to completion with the analytical gradient.
    # Any backend whose sizer still uses the stride-1 sum raises `QuadrantsAssertionError: Adstack overflow` at the next
    # `qd.sync()`.
    #
    # Internal details: CPU uses the host evaluator (`evaluate_adstack_size_expr` in `adstack_size_expr_eval.cpp`) which
    # reads shapes off `LaunchContextBuilder` via the same `SHAPE_POS_IN_NDARRAY` path that `ExternalTensorShape` leaves
    # use. CUDA / AMDGPU encode the node bytecode and the device interpreter in
    # `runtime/llvm/runtime_module/runtime.cpp`'s `runtime_eval_adstack_size_expr` sums indices; Metal / Vulkan drive
    # the SPIR-V sizer shader in `codegen/spirv/adstack_sizer_shader.cpp` whose `compute_linear_index` does the same
    # accumulation. All three need per-axis stride support. The kernel uses nested `for i: for j:` rather than
    # `qd.ndrange(shape[0], shape[1])` so the pre-pass sees two distinct `LoopIndexStmt`s (one per axis) as the
    # `ExternalPtrStmt` index operands; the 2-index ndrange lowering flattens `(i, j)` through `div`/`mod` arithmetic
    # that `determine_ad_stack_size.cpp::build_value_expr` does not fold, so the pre-pass would fall back to
    # `default_ad_stack_size` and the kernel would pass trivially on every backend without exercising the multi-axis
    # evaluator.
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
@test_utils.test(require=qd.extension.adstack)
def test_adstack_ext_tensor_read_indexed_by_stashed_outer_loop_var(outer_bound):
    # Pins the `ExternalPtrStmt` indexed by `AdStackLoadTopStmt` grammar gap. The kernel walks a parent/child
    # hierarchical-array layout: an outer parallel-for whose body casts its loop variable (`i_l = qd.cast(i_l_,
    # qd.i32)`), branches on `ndarray[i_l] != -1`, and drives a nested range-for from `ndarray[i_l] - ndarray[i_l]`.
    # Under `ad_stack_experimental_enabled=True` the autodiff pipeline stashes the cast loop index onto a dedicated
    # adstack and reloads it via `stack_load_top` so the reverse pass can reconstruct it; every downstream
    # `ndarray[i_l]` lowers to `ExternalPtrStmt(arr, [AdStackLoadTopStmt])`. The pre-pass upper-bounds the loaded value
    # by recognising the stash pattern (single loop-index push plus const-zero initialiser) and folding through to the
    # backing `LoopIndexStmt`.
    #
    # Internal details: runs on every backend - LLVM evaluates the stash-backed SizeExpr through
    # `publish_adstack_metadata`, SPIR-V through `GfxRuntime::launch_kernel`'s AdStackMetadata upload. Parametrised over
    # `outer_bound` because const and dynamic outer range-for bounds lower very differently - a constant collapses into
    # the offload's `const_end` at offload time (no prep task), a dynamic bound lowers to a prep serial task that writes
    # the value into the kernel's global-temporary buffer for the main range-for task to read back. The grammar covers
    # both paths via `resolve_global_tmp_value`, so the `ExternalPtrStmt`-with-stashed-index pattern works whether the
    # outermost parallel-for is sized at launch time (e.g. `arr.shape[0]`) or hard-coded.
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
    # Pins the `GlobalPtrStmt`-index stash-chase extension in the `SizeExpr` pre-pass. The kernel reads two scalar
    # quadrants fields `link_start[i_outer]` / `link_end[i_outer]` as the bounds of an inner range-for, where `i_outer`
    # is an outer parallel-for index that `ad_stack_experimental_enabled=True` stashes onto a dedicated adstack for the
    # reverse pass. Every downstream `link_start[i_outer]` then lowers to `GlobalPtrStmt(<field>,
    # [AdStackLoadTopStmt])`. The pre-pass's `GlobalPtrStmt` branch must walk the index through the same stash chase the
    # `ExternalPtrStmt` branch uses and fall back to the snode's `shape_along_axis(axis)` as a safe upper bound when the
    # stash has no single loop-index push, otherwise the reverse-mode adstack bound hard-errors as "unresolved after
    # Bellman-Ford + structural pre-pass".
    #
    # Internal details: runs on every backend - LLVM evaluates the stash-backed `SizeExpr` through
    # `publish_adstack_metadata`, SPIR-V through `GfxRuntime::launch_kernel`'s `AdStackMetadata` upload. The inner
    # `range(link_start[i_outer_c], link_end[i_outer_c])` fans into four differentiable-body iterations per outer index;
    # each touches a distinct `x[i_inner]` so the analytical gradient is the same constant per slot and a
    # bound-too-small regression surfaces as either the old "unresolved" error or an adstack-overflow at `qd.sync()`.
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
    # Pins the phase-2 idempotency-at-zero probe ordering in `build_value_expr` for a reverse-mode push whose value
    # expression is self-referential by construction. The kernel couples a 2D `qd.ndrange` outer parallel scan with a
    # triangular inner `for j in range(i_outer, ...)` and a nested `range(begin_fld[j], end_fld[j])` whose bounds come
    # from scalar fields indexed by the stashed outer index. On the experimental adstack path that shape lowers
    # reverse-mode pushes of the form `sub(load_top($S), load_top($S))` where both load_top reads target the very stack
    # the push feeds - a zero-net push that must be treated as idempotent at zero rather than rejected as a stash
    # data-flow cycle. The structural pre-pass must try the idempotency probe (which substitutes `load_top(self) -> 0`)
    # BEFORE its generic visited-set cycle guard fires; without that ordering the cycle guard aborts the walk first, the
    # probe never runs, and the grad kernel fails to compile with "stash data-flow cycle ... idempotency-at-zero probe
    # could not discharge it".
    #
    # Internal details: names are deliberately domain-neutral. `outer_size` + `batch_probe.shape[1]` feed the 2D
    # `qd.ndrange` (the batch-dim probe is a grad-requiring field whose only in-kernel use is the shape read, matching
    # the minimal pattern that triggers the cycle). `group_begin` / `group_end` gate the middle range, `sub_begin` /
    # `sub_end` gate the innermost range, `src_offset` / `dst_offset` offset the scalar scatter. The scalar reads from
    # `src_buf[src_offset + k, i_b]` packed into a `qd.Vector` then written into `dst_buf[dst_offset + j, i_b]` via
    # `qd.static(range(3))` is load-bearing: without a differentiable Vector-packed reverse body the `sub(load_top($S),
    # load_top($S))` push shape does not arise. The trailing `dst_vec_buf[i_l, i_b] = src_vec_buf[i_l, n_sub, i_b]`
    # Vector copy outside the innermost range but inside the triangular loop is also load-bearing - removing it
    # collapses the enclosing-loop structure enough that the pre-pass resolves the adstack by other means.
    # `cfg_optimization=False` + `ad_stack_experimental_enabled=True` are the minimum flags that surface the cycle: with
    # CFG optimization on the store-to-load forwarder collapses the self-referential push before the sizing pass sees
    # it, and the non-experimental path uses a different sizing strategy that sidesteps the probe altogether.
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

    # The grad compile must complete without raising RuntimeError("stash data-flow cycle ..."). The assertion is the
    # absence of that RuntimeError - no gradient value is checked because the minimal-shape fields have a single element
    # and the regression is purely compile-time cycle detection.
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
@test_utils.test(require=qd.extension.adstack)
def test_adstack_structural_pre_pass_fuses_sub_of_max_over_range_with_matching_shape_ends(inner_loop_shape):
    # Covers the two user-facing surface forms of a reverse-mode kernel whose inner range-for trip count is the
    # difference between two reads of parallel ndarrays indexed by the SAME outer loop. Both lower to a
    # Sub-of-two-MaxOverRange where the `end` operands are structurally equal (both come from the single enclosing outer
    # loop's end, not from each read's own ndarray shape), so the walker's strict-equality fusion path already fires
    # without the `ExternalTensorShape` same-axis extension. The test pins that strict path continues to work on both
    # spellings and that the resulting adstack bound matches the actual reverse-pass push count.
    #
    # Internal details: runs on every backend now that the runtime-evaluator ships on both the LLVM and SPIR-V paths.
    # Two surface spellings share a single test body via `qd.static` because both produce the same `expr_sub` call at
    # walker time, just via different `build_value_expr` recursion paths:
    #   - `begin_end`: `for i_j_ in range(start[i_o], end[i_o])` - `compute_bounded_adstack_size` multiplies `end_upper
    #     - begin_lower`, `resolve_loop_begin_lower_bound` drops non-const begins to `Const(0)`, so the fused bound is
    #     the `end[i_o]` MaxOverRange alone (no two-operand Sub is built for this spelling); the test still passes
    #     because the body's push count is bounded by `end[i_o]`, which the walker tracks soundly via the single
    #     MaxOverRange.
    #   - `sub_then_range`: user materialises `n_inner = end[i_o] - start[i_o]` and passes `range(n_inner)`; this makes
    #     the `Sub` explicit in the range-for's `end` stmt, `build_value_expr` recurses into both operands, each wraps
    #     with the SAME outer-loop end, and strict fusion collapses the pair.
    # `qd.cast(e - s, qd.f32)` is a multiplicative factor inside the body so `full_simplify` does not inline the
    # subtraction back into the range-for bounds and collapse the two patterns.
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


@test_utils.test(require=qd.extension.adstack)
def test_adstack_structural_pre_pass_fuses_sub_of_max_over_range_with_mismatched_shape_ends():
    # Pins the `expr_sub` fusion for the Sub-of-two-MaxOverRange shape that the walker builds when an inner range-for's
    # trip count is computed as the difference between two ndarray reads whose indices come from two DIFFERENT enclosing
    # range-fors. Each read wraps into its own `MaxOverRange(outer_i, 0, shape(arr_i), ExtRead(arr_i, [outer_i]))`; the
    # `end` operands are then `ExternalTensorShape` nodes pointing at distinct `arg_id`s so `expr_equal` rejects them
    # and the pre-fusion walker falls back to `max_i arr_a[i] - max_j arr_b[j]`, which under-counts `max_i (arr_a[i] -
    # arr_b[i])` whenever the two per-index maxima land at different slots. With `arr_a = [1, 5]`, `arr_b = [4, 0]` the
    # unfused bound collapses to `5 - 4 = 1` per outer pair and the full trip multiplier undershoots the actual push
    # count of 7 (the (1,1) pair alone pushes 5), so the reverse pass overflows the heap and raises at `qd.sync()`. The
    # fusion emits the tight `MaxOverRange(v, 0, shape(arr_a), Sub(arr_a[v], arr_b[v]))` which correctly evaluates to 5,
    # and the adstack gets sized to fit.
    #
    # Internal details: runs on every backend now that the runtime-evaluator ships on both the LLVM and SPIR-V paths.
    # The trivial `range(1)` wrapper keeps the kernel AST inside a top-level for-loop, which the autodiff front-end
    # requires (`reverse_segments` rejects mixed statement-plus-for kernel bodies).
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

    # For (i_a, i_b) in {(0,0), (0,1), (1,0), (1,1)} the inner trip n = max(0, arr_a[i_a] - arr_b[i_b]) is {0, 1, 1, 5}.
    # Total push count is 7 (the (1,1) pair contributes 5); the pre-fusion adstack bound of 4 overflows. loss = (0 + 1 +
    # 1 + 5) * x[0..4]^2 = 0.07 with every x[i] = 0.1. x.grad[k] = 2 * (count of inner iterations that visit index k) *
    # 0.1. k = 0 is visited by every non-empty pair (3 visits), k in [1, 4] is visited only by the (1, 1) pair (1 visit
    # each), k >= 5 is never visited.
    assert loss[None] == pytest.approx(0.07, rel=1e-5)
    assert x.grad[0] == pytest.approx(0.6, rel=1e-5)
    for k in range(1, 5):
        assert x.grad[k] == pytest.approx(0.2, rel=1e-5)
    for k in range(5, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    "arr_a_values, arr_b_values",
    [
        ([5, 5, 5, 5], [4, 0]),
        ([0, 0, 0, 7], [1, 1]),
    ],
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_fuses_sub_of_max_over_range_with_mismatched_lengths_is_safe(arr_a_values, arr_b_values):
    # Pins that a reverse-mode kernel whose inner trip count is `arr_a[i_a] - arr_b[i_b]` over two independent outer
    # loops of shape `arr_a.shape[0]` and `arr_b.shape[0]` computes the correct gradient when the two ndarrays along the
    # fused axis have different lengths - both when the longer ndarray's peak fits inside the shorter ndarray's shape
    # and when it sits past it.
    #
    # Internal details: the `expr_sub` MaxOverRange fusion in `determine_ad_stack_size.cpp` must produce a bound that
    # simultaneously keeps the fused `arr_a[v] - arr_b[v]` body in-bounds for the shorter ndarray and covers `max_ia
    # arr_a[ia] - max_ib arr_b[ib]` from the unfused form. A too-tight fused end OOB-reads the shorter ndarray at launch
    # (`cudaErrorIllegalAddress` on CUDA); a too-permissive clamp silently drops the longer ndarray's peak-past-shape
    # pushes and overflows the adstack at `qd.sync()`. The two parametrisations exercise the same invariant at both
    # boundary conditions - touch the cross-ndarray fusion path with care for both.
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
    arr_a_np = np.array(arr_a_values, dtype=np.int32)
    arr_b_np = np.array(arr_b_values, dtype=np.int32)

    compute(arr_a_np, arr_b_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_a_np, arr_b_np)
    qd.sync()

    expected_pushes = 0
    expected_visits = [0] * N_X
    for ia in range(len(arr_a_values)):
        for ib in range(len(arr_b_values)):
            n = max(0, int(arr_a_values[ia]) - int(arr_b_values[ib]))
            expected_pushes += n
            for k in range(min(n, N_X)):
                expected_visits[k] += 1
    assert loss[None] == pytest.approx(expected_pushes * 0.01, rel=1e-5)
    for k in range(N_X):
        if expected_visits[k] == 0:
            assert x.grad[k] == pytest.approx(0.0, abs=1e-7)
        else:
            assert x.grad[k] == pytest.approx(2 * expected_visits[k] * 0.1, rel=1e-5)


@pytest.mark.parametrize(
    "x_unused_val",
    [0.1, 100.0],
    ids=["uniform_x", "amplified_unused_x"],
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_sub_of_max_over_range_fusion_does_not_mix_fieldload_and_extread(x_unused_val):
    # Pins that a reverse-mode kernel whose inner trip count is `fld[i] - arr[i]` - a scalar field minus an ndarray
    # element, both indexed by the outer loop variable - compiles and produces the correct gradient on every supported
    # backend.
    #
    # Internal details: the walker in `determine_ad_stack_size.cpp` wraps each operand of the inner `Sub` in its own
    # `MaxOverRange(i, 0, shape, leaf)` (`FieldLoad` on the field side, `ExternalTensorRead` on the ndarray side);
    # `expr_sub`'s `end_eq` branch then sees structurally-equal `ExternalTensorShape(arr, 0)` ends and would fuse them
    # into a single `MaxOverRange(v, 0, shape, Sub(FieldLoad(fld, [v]), ExternalTensorRead(arr, [v])))`. The LLVM
    # encoder's closed-subtree lift only folds `FieldLoad` leaves with no free bound vars, so the mixed body - whose
    # `FieldLoad` carries the free `v` - falls through to `encode_subtree`'s `FieldLoad` branch and hard-errors on the
    # LLVM path (CUDA / AMDGPU have no on-device SNode access). The fusion must therefore decline whenever its
    # synthesised body would pair a bound-var-indexed `FieldLoad` with an `ExternalTensorRead`; the unfused
    # `Sub(MaxOverRange(i, 0, shape, FieldLoad), MaxOverRange(i, 0, shape, ExternalTensorRead))` keeps each operand
    # closed and host-foldable on both encoders.
    #
    # The `x_unused_val` parametrization sets `x[8..15]` to `x_unused_val` while keeping `x[0..7]` at `0.1`. Only
    # `x[0..7]` is reached by the kernel under correct sizing, so `x_unused_val` does not affect the expected loss /
    # gradient at all and the assertions are identical across parametrizations. The `amplified_unused_x` variant
    # (`x_unused_val=100.0`) exists so that any regression that mis-routes a stack push / pop to a slot outside the
    # intended index range surfaces as a multi-order-of-magnitude gradient delta (e.g. a single spurious visit to `x[8]`
    # produces `x.grad[8]=200.0` instead of the `0.2` an `x_unused_val=0.1` setup would produce), so the failure cannot
    # be misread as a tolerance issue. The `uniform_x` (`x_unused_val=0.1`) parametrization keeps the baseline loss /
    # gradient magnitudes that the rest of the kernel was originally tuned against.
    N = 4
    N_X = 16

    fld = qd.field(qd.i32, shape=(N,))
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for _dummy in range(1):
            accum = 0.0
            for i in range(arr.shape[0]):
                n = fld[i] - arr[i]
                if n > 0:
                    for k in range(n):
                        accum = accum + x[k] * x[k]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1 if i < 8 else x_unused_val
    for i in range(N):
        fld[i] = 10
    arr_np = np.array([2, 2, 2, 2], dtype=np.int32)

    compute(arr_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_np)
    qd.sync()

    # Each of the 4 outer iterations runs `fld[i] - arr[i] = 10 - 2 = 8` inner iters. Total pushes = 32.
    # x[0..7] is visited 4 times (once per outer iter); x[8..] is never visited, so the loss and gradients
    # are independent of `x_unused_val`.
    assert loss[None] == pytest.approx(4 * 8 * 0.01, rel=1e-5)
    for k in range(8):
        assert x.grad[k] == pytest.approx(4 * 2 * 0.1, rel=1e-5)
    for k in range(8, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_spirv_metadata_per_task_buffer():
    # The SPIR-V launcher must allocate a fresh `AdStackMetadata` device buffer per task inside the cmdlist record loop,
    # not share a single grow-on-demand buffer across every task in a kernel. With a shared buffer, per-task
    # `(stride_float, stride_int, offset_i, max_size_i, ...)` tables host-memcpy'd into it would be overwritten by later
    # tasks' metadata before the deferred dispatch executes (record is host-synchronous, execute is deferred), so by
    # submit time the buffer holds only the LAST task's metadata and every dispatch in the cmdlist reads those bytes.
    # Earlier tasks then see shorter sibling stacks' `max_size` where their own should be - e.g. a stack whose sizer
    # wrote `max_size=9` observes a runtime `max_size=3`, its first guarded push trips the `count < max_size` check at
    # `count=3`, the overflow flag flips, and `qd.sync()` raises even though the kernel's actual per-thread push count
    # fits the per-stack bound the sizer computed.
    #
    # Internal details: `cfg_optimization=False` is load-bearing - with it enabled, the CFG pass sinks / merges the
    # bind-and-dispatch pair in a way that masks the cross-task buffer reuse on this kernel shape; with it disabled the
    # raw record-then-execute race surfaces. The pinned regression is SPIR-V-specific (the LLVM path publishes metadata
    # host-side via `publish_adstack_metadata` directly into each launch's own `AdStackSizingInfo` with no cross-task
    # aliasing), but the test runs on every backend so a future regression in either path that produces wrong values
    # rather than an overflow is still caught. The kernel shape (two sibling `qd.ndrange` offloads, the second one
    # carrying a triangular i<=j<k nested loop that stashes a multiplicative reduction onto its own adstack) is the
    # minimum that exhibits the bug: you need at least two tasks in the same kernel so the second task's record
    # overwrites the first task's metadata before submit. The post-fix runtime allocates a fresh metadata buffer per
    # task record and retires it into `ctx_buffers_` so it stays alive until the sync window closes.
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

    # The grad call must finish cleanly: a regression that shares one metadata buffer across tasks would have the first
    # offload's metadata overwritten by the second offload's host memcpy before the cmdlist ran, so the first offload's
    # f32 stack 0 would see `max_size=3` (the second offload's int stack 0 value) instead of its own sizer-computed 9,
    # and `qd.sync()` would raise `Adstack overflow (offending stack_id=0)`.
    kernel_two_offloads_with_tri_reduce.grad()
    qd.sync()


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [4, 32])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_linear_only_accumulator_with_nonlinear_operand(n_iter):
    # Cross-check `d/dx sum_i sum_j sin(x[i] + j*step)` against PyTorch autograd for the kernel shape
    # that mixes a loop-carried accumulator (`acc = acc + ...`) with a non-linear operand
    # (`sin(a)` where `a = x[i] + j*step`) inside an unrolled inner loop.
    #
    # Internal details: the operand `a` feeds `sin`, which is in `NonLinearOps::unary_collections`, so
    # `AdStackAllocaJudger::visit(UnaryOpStmt)` promotes its alloca to an adstack and the reverse pass
    # reads `a` back via `AdStackLoadTopStmt`. The accumulator `acc` only feeds linear `add`, never
    # reaches the non-linear / index / control-flow consumer visitors, but its adjoint chain still has
    # to weave through the adstack push / pop sites the operand introduces. `n_iter=32` keeps the
    # unrolled body wide enough that any miscalculation in the slot offset for the operand's stack -
    # e.g. an off-by-one in the inline `count - 1` saturation, a misaligned slot stride, or a missed
    # primal+adjoint zero-init on push - surfaces here as a wrong gradient long before it would surface
    # as an overflow at runtime.
    import torch

    n = 4
    step = 0.07
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in qd.static(range(n_iter)):
                a = x[i] + qd.cast(j, qd.f32) * step
                acc = acc + qd.sin(a)
            y[None] += acc

    for i in range(n):
        x[i] = 0.18 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.18 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch.float32)
        for j in range(n_iter):
            a_t = x_t[i] + float(j) * step
            acc_t = acc_t + torch.sin(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    assert y[None] == pytest.approx(y_t.item(), rel=1e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=1e-6)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_inner", [4, 32])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_repeated_load_top_of_outer_value_in_unrolled_inner_loop(n_inner):
    # Cross-check `d/dx sum_i sum_j cos(sin(x[i])) * (1 + j*0.01)` against PyTorch autograd for the
    # kernel shape where an outer-loop value (`a = sin(x[i])`) is consumed by a non-linear unary op
    # `cos(a)` repeatedly inside an unrolled inner loop, producing N consecutive `AdStackLoadTopStmt`
    # reads of the same stack in the reverse pass with no intervening Push / Pop / AccAdjoint.
    #
    # Internal details: each inner-loop iteration's reverse-pass adjoint formula reads `a`'s primal
    # (for `d/da cos(a) = -sin(a)`) and adjoint via the inline slot-pointer math. With `n_inner=32`
    # straight-line LoadTops on the same stack inside one block, the inline codegen emits 32
    # independent count-load + slot-GEP + value-load sequences. A regression that miscalculates the
    # saturating `count - 1` index under repeated reads, races the count alloca's mem2reg promotion
    # across the LoadTops, or short-circuits the slot offset math, surfaces here as a wrong gradient.
    import torch

    n = 3
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            a = qd.sin(x[i])
            acc = 0.0
            for j in qd.static(range(n_inner)):
                acc = acc + qd.cos(a) * (1.0 + qd.cast(j, qd.f32) * 0.01)
            y[None] += acc

    for i in range(n):
        x[i] = 0.21 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.21 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        a_t = torch.sin(x_t[i])
        acc_t = torch.zeros((), dtype=torch.float32)
        for j in range(n_inner):
            acc_t = acc_t + torch.cos(a_t) * (1.0 + float(j) * 0.01)
        y_t = y_t + acc_t
    y_t.backward()

    # Tolerance is `rel=5e-6` rather than the f32 floor `1e-6` because the kernel is unusually deep on the
    # numeric side: each contribution is `cos(sin(x[i])) * (1 + j*0.01)` and the test sums up to 32 of them per
    # outer iteration, so the per-sum drift accumulates to a few ULPs at the order of magnitude of the final
    # value across every backend that auto-promotes to FMA or drops guard bits (CUDA's NVPTX and Vulkan's
    # SPIR-V both observed). The test's purpose is to pin slot-pointer / count-recurrence correctness under
    # repeated LoadTop, not bit-exact float behavior; 5e-6 is the smallest tolerance that still passes on every
    # backend while staying inside the f32 noise band.
    assert y[None] == pytest.approx(y_t.item(), rel=5e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=5e-6)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [4, 16])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], require=qd.extension.adstack)
def test_adstack_f16_unrolled_pushes_alignment(n_iter):
    # Cross-check `d/dx sum_i sum_j sin(x[i] + j*step)` against PyTorch autograd for an `f16` field whose
    # operand `a = x[i] + j*step` adstack-promotes through `sin`. Each adstack slot is `2 * element_size`
    # = 4 bytes wide, so slot N starts at byte offset `8 + 4*N` from the per-thread slab base. Slot 0 is
    # 8-aligned but every odd slot index (1, 3, 5, ...) lands on a 4-byte-aligned address that is NOT
    # 8-aligned.
    #
    # Internal details: `AdStackPushStmt`'s inline IR emits a `llvm.memset` zeroing the primal+adjoint
    # slot pair before storing the pushed value. The destination alignment passed to `CreateMemSet` must
    # reflect the actual slot pointer alignment - `min(8, 2 * element_size)` = 4 bytes for f16. An
    # over-stated 8-byte alignment lets NVPTX / AMDGCN lowering pick a wider store (`st.b64`) than the
    # 4-aligned slot pointer can satisfy, which traps as a misaligned-address fault on stricter GPU
    # backends or silently corrupts adjoint state on tolerant ones - either way producing wrong
    # gradients. `n_iter=16` ensures multiple non-zero slot offsets get exercised, including odd
    # slot indices where the alignment claim matters most. Tolerance `rel=4e-3` is the f16 precision
    # floor for sums of ~16 `sin` values. SPIR-V backends (Metal / MoltenVK / Vulkan) are excluded
    # from the arch list because their MSL / Vulkan shader compilation fails on the reverse-mode
    # `y[None] += acc` adstack shape with `qd.f16` fields - the host-side type-check warns
    # `Atomic add may lose precision: f16 <- f32` and the underlying compute pipeline build then
    # rejects the shader, so the alignment regression this test pins on the LLVM CUDA / AMDGPU
    # backends cannot run on the SPIR-V backends regardless of slot alignment correctness.
    import torch

    n = 4
    step = 0.05
    x = qd.field(qd.f16, shape=n, needs_grad=True)
    y = qd.field(qd.f16, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in qd.static(range(n_iter)):
                a = x[i] + qd.cast(j, qd.f16) * qd.f16(step)
                acc = acc + qd.sin(a)
            y[None] += acc

    for i in range(n):
        x[i] = 0.21 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.21 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch.float32)
        for j in range(n_iter):
            a_t = x_t[i] + float(j) * step
            acc_t = acc_t + torch.sin(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    assert float(y[None]) == pytest.approx(y_t.item(), rel=4e-3)
    for i in range(n):
        assert float(x.grad[i]) == pytest.approx(x_t.grad[i].item(), rel=4e-3)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [4, 16, 64])
@pytest.mark.parametrize("n_stacks", [1, 3])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_unrolled_many_pushes_across_multiple_stacks(n_iter, n_stacks):
    # Cross-check `d/dx sum_i sum_s sum_j sin(x[i] + j*step + s*offset)` against PyTorch autograd for the
    # kernel shape where `n_stacks` parallel adstacks each receive `n_iter` straight-line pushes inside an
    # unrolled inner loop, fanning out to a separate non-linear `sin` per stack per iteration.
    #
    # Internal details: this is the regime the LLVM release-build inline AdStack codegen targets - each `s`
    # promotes its operand `a = x[i] + j*step + s*offset` onto its own per-stack `alloca i64` count, and
    # the unrolled inner loop body emits `n_iter` consecutive `count++` increments per stack. After mem2reg
    # lifts the alloca to SSA and GVN folds the increment chain to constants `0, 1, ..., n_iter - 1`, the
    # only memory ops left in the unrolled body are the slot stores. `n_iter=64` is well above the adstack
    # capacity floor of 32, so any regression that miscalculates the slot offset under multiple sibling
    # stacks (e.g. by hoisting the count alloca shared across stacks instead of per-stack), that
    # short-circuits / silently drops pushes via a wrong saturating subtract on count, or that races a
    # cross-stack increment, surfaces here as a wrong gradient long before it would surface as an overflow
    # at runtime. The cross-check tolerance `rel=1e-6` is the f32 floor; a few-ULP drift per `sin` over up
    # to `n_stacks * n_iter <= 192` summed terms still fits within it.
    import torch

    n = 4
    step = 0.07
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for s in qd.static(range(n_stacks)):
                s_offset = qd.cast(s, qd.f32) * 0.13
                for j in qd.static(range(n_iter)):
                    a = x[i] + qd.cast(j, qd.f32) * step + s_offset
                    acc += qd.sin(a)
            y[None] += acc

    for i in range(n):
        x[i] = 0.21 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.21 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch.float32)
        for s in range(n_stacks):
            s_offset_t = float(s) * 0.13
            for j in range(n_iter):
                a_t = x_t[i] + float(j) * step + s_offset_t
                acc_t = acc_t + torch.sin(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    assert y[None] == pytest.approx(y_t.item(), rel=1e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=1e-6)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_inner", [4, 32])
@test_utils.test(require=qd.extension.adstack, ad_stack_size=64)
def test_adstack_min_loop_carried_serial_range_for(n_inner):
    # Cross-check `d/dx sum_i acc_n` where `acc` is initialized to 1.0 and updated per iteration of a serial
    # `for j in range(n_inner)` body via `acc = qd.min(acc * 0.5 + 0.05, x[i] + j*0.05)`. The min winner flips
    # between the lhs and rhs across iterations (rhs wins on iter 0 because acc starts at 1.0 vs x[i]=~0.3,
    # then lhs wins on later iterations as acc shrinks). The reverse pass must use the per-iteration forward
    # lhs/rhs to route the gradient correctly.
    #
    # Internal details: pins the snap-stack fix in `MakeAdjoint::visit(BinaryOpStmt)`'s min/max branch. The
    # forward cmp `lhs < rhs` (min) / `rhs < lhs` (max) is computed at forward time and pushed onto a
    # dedicated 1-push-per-bin-execution adstack, then read back in reverse with a matching pop. Without the
    # snap-stack, BackupSSA spills `bin->lhs` / `bin->rhs` to single overwrite-each-iteration allocas and
    # every reverse iteration reads the last forward iteration's values, so the cmp flips on iterations where
    # the actual winner changed and the gradient routes through the wrong branch (visible as `x.grad=0`
    # instead of the analytical `0.125` for `x[i]=0.31`, `n_inner=4`).
    import torch

    n = 4
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 1.0
            for j in range(n_inner):
                acc = qd.min(acc * 0.5 + 0.05, x[i] + qd.cast(j, qd.f32) * 0.05)
            y[None] += acc

    for i in range(n):
        x[i] = 0.31 + i * 0.07
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.31 + i * 0.07 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.tensor(1.0, dtype=torch.float32)
        for j in range(n_inner):
            acc_t = torch.minimum(acc_t * 0.5 + 0.05, x_t[i] + float(j) * 0.05)
        y_t = y_t + acc_t
    y_t.backward()

    assert y[None] == pytest.approx(y_t.item(), rel=1e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=1e-4)


@pytest.mark.parametrize("gated_fraction", [0.0, 0.05, 0.5, 1.0])
@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_ndarray_gate_grad_correct(gated_fraction):
    # Asserts gradient correctness for reverse-mode kernels of shape `for i in range(n): if selector[i] > eps:
    # <adstack-using gradient work>` where `selector` is an ndarray argument. Parametrised over the gate-pass fraction
    # (0%, 5%, 50%, 100%) so the savings path (sparse), the half-claim row mapping, the dispatch-equivalent fallback
    # (full), and the empty-reducer-count edge case are all exercised against an analytic gradient oracle; a
    # wrong-but-non-NaN gradient (the failure mode when row-claim and heap-sizing disagree) trips the assertion.
    #
    # Internal details: the codegen pattern matcher captures the gating predicate as a `StaticBoundExpr` carrying the
    # ndarray's `arg_id` and the comparison `> eps`; the runtime walks the gating ndarray (host-side on CPU,
    # single-thread reducer kernel on CUDA / AMDGPU, compute-shader reducer on SPIR-V), counts threads with `selector[i]
    # > eps`, and sizes the float adstack heap to that count. The lazy LCA-block atomic claim then maps each gated
    # thread to a unique row in `[0, count)`. `ad_stack_size=32` keeps per-stack max_size small so the worst-case heap
    # allocation is much larger than the gated subset actually consumes - amplifying the savings ratio so a regression
    # that breaks the reducer dispatch and silently falls back to worst-case sizing still produces a passing test, while
    # a regression that corrupts the row mapping fails on the gradient oracle. The kernel places the gate immediately
    # above the inner range-for so the LCA pre-pass places the float-LCA inside the gate, the precondition for the
    # bound_expr capture. `n=256` is deliberately larger than a typical CPU worker pool (~8 threads) so the CPU host
    # reducer must walk the full ndarray to count gate-passing iterations, not just the worker-pool prefix; a reducer
    # that walks `[0, num_cpu_threads)` undercounts in the sparse case and aliases every later iteration's claimed row
    # into a single slot. `gated_fraction=0.5` is the tightest catch for that class of bug because the count mismatch
    # then aliases ~128 iterations into a handful of rows, overwhelming the per-row stack's `max_size=32` headroom and
    # tripping the bounds-checked overflow on the debug build.
    n = 256
    n_iter = 8
    eps = 1e-9

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    np.random.seed(0)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = int(round(gated_fraction * n))
    selector_np = np.zeros(n, dtype=np.float32)
    if n_gated > 0:
        gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
        selector_np[gated_indices] = 1.0
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"static-bound-expr grad returned NaN: {got_grad}"

    # Analytic oracle. For gated i, the inner recurrence `v = v*c + d` over `n_iter` steps is linear in v with slope
    # `c^n_iter`, where `c = 1.05`. So `d(out[0])/d(x[i]) = c^n_iter` for gated i, 0 otherwise. `gated_fraction == 0` is
    # the per-task-reducer-count-zero edge case: every dispatched thread misses the gate, the reducer publishes capacity
    # = 0, the codegen-emitted clamp at the LCA-block claim site has to keep the row id at 0 (a naive `capacity - 1`
    # underflow to UINT32_MAX leaves the clamp inert and a divergent over-claim writes past the float-heap end).
    # Float-heap allocation is floored at one row precisely so the single-row fallback is always backed by real storage.
    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > eps, np.float32(expected_per_gated), np.float32(0.0))
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(
    require=[qd.extension.adstack, qd.extension.data64], ad_stack_size=32, ad_stack_sparse_threshold_bytes=0
)
def test_adstack_static_bound_expr_f64_gate_grad_correct():
    # Asserts gradient correctness for reverse-mode kernels with an f64-typed gating ndarray (`if selector_f64[i] >
    # 0.5`) above f32 adstack pushes. The reducer must dispatch through the f64 comparison arm; routing f64-captured
    # gates through the f32 arm misreads the source ndarray and produces wrong-but-non-NaN gradients on every gated
    # index where the bit pattern flips the bitcast comparison's outcome against the misdecoded threshold.
    #
    # Internal details: the captured `StaticAdStackBoundExpr` carries `field_dtype_is_float = True` AND
    # `field_dtype_is_double = True` plus the threshold in `literal_f64`. The SPIR-V reducer reads
    # `field_dtype_is_double` to select the 8-byte u64 PSB load (two 4-byte u32 loads at offsets 0 and 4 from `elem_idx
    # * 8`, reassembled into a u64 in registers because PSB requires Aligned 8 for a single 8-byte load), then
    # OpFOrd*-compares against the high+low threshold pair. `require=qd.extension.data64` skips on backends without f64
    # (e.g. Metal: Apple silicon does not advertise SPIR-V `Float64`, and the kernel codegen rejects the f64 ndarray at
    # the IR pre-pass). f32-push-only on the adstack heap because SPIR-V's adstack heap is a typed Array<f32> SSBO and
    # rejects f64 AdStackAllocaStmts; LLVM accepts both but the test stays on f32 push + f64 gate for backend parity.
    # Selector layout: non-gated cells at 0.25, gated cells at 1.0, threshold = 0.5. A misdecoded threshold of 0.0 would
    # spuriously include the 0.25 cells, doubling the gate-passing count - the per-i oracle fails on every non-gated
    # cell because the codegen clamps the over-claimed rows onto valid heap slots and the adjoint's reverse pop reads
    # back zeros (bootstrap-init slot) instead of the primal value.
    n = 256
    n_iter = 8
    threshold = 0.5

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f64, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > threshold:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    np.random.seed(0)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = np.full(n, 0.25, dtype=np.float64)
    gated_indices = np.sort(np.random.choice(n, size=n // 2, replace=False))
    selector_np[gated_indices] = 1.0

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"f64-gate static-bound-expr grad returned NaN: {got_grad}"

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > threshold, np.float32(expected_per_gated), np.float32(0.0))
    for i in range(n):
        assert got_grad[i] == pytest.approx(expected[i], rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("alloca_outside_gate", [False, True])
@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, debug=True, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_ndarray_gate_debug_build_grad_correct(alloca_outside_gate):
    # Asserts gradient correctness for reverse-mode kernels with a captured ndarray-backed gate under `debug=True`. The
    # debug build routes every adstack push / pop / load-top through the runtime helpers (`stack_push`,
    # `stack_top_primal`, ...) instead of the release build's inline emission, and those helpers read the count u64
    # prefix word from the heap row itself, so the lazy-row codegen has to keep the per-row count header consistent
    # across both alloca placements (inside vs above the gate). Parametrised over `alloca_outside_gate` to cover both
    # placements; either should produce gradients that match the analytic oracle.
    #
    # Internal details: each lazy float alloca needs its row's count header initialised to 0 BEFORE the first push and
    # AFTER the LCA-block atomic-rmw stores the per-thread claimed row id into `row_id_var`; emitting `stack_init` at
    # the alloca visit site (mirroring the eager path's `linear_thread_idx * stride + offset`) would dereference
    # `row_id_var` while it still holds its entry-block UINT32_MAX sentinel, writing the count u64 to `heap_float +
    # UINT32_MAX * stride_float + offset` (~64 GB past the heap base). The fix emits `stack_init` at the LCA block. The
    # `alloca_outside_gate` parametrisation covers both codegen shapes: `False` puts the `AdStackAllocaStmt` and the
    # autodiff bootstrap push in the if-true block (below the LCA) so the bootstrap push's `stack_push` runs after the
    # row claim and `row_id_var` is already valid; `True` puts them at the offload root (above the LCA) and requires the
    # bootstrap-skip guard at the push site to fire on the debug build, otherwise the runtime-helper `stack_push` runs
    # at the offload root with `row_id_var = UINT32_MAX` and writes the count u64 ~TB past the heap base, crashing the
    # worker with SIGSEGV / CUDA_ERROR_ILLEGAL_ADDRESS / hipErrorIllegalAddress at the first `compute.grad()`. Kernel
    # shape otherwise mirrors `test_adstack_static_bound_expr_ndarray_gate_grad_correct`; only delta is `debug=True`
    # flipping both the bounds-check codepath and the runtime-helper push / pop emission. `gated_fraction=0.5` places
    # ~half the LCA reaches on non-trivial rows in `[0, count)` so the row mapping must be correct (a regression that
    # always claims row 0 would still pass the 100% case) while keeping the test fast enough to run on every backend
    # without a parametrize sweep on the fraction axis.
    n = 256
    n_iter = 8
    eps = 1e-9
    gated_fraction = 0.5

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))

    if alloca_outside_gate:

        @qd.kernel
        def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
            for i in range(n):
                v = qd.cast(0.0, qd.f32)
                if selector[i] > eps:
                    v = x[i]
                    for _ in range(n_iter):
                        v = v * 1.05 + 0.05
                out[0] += v

    else:

        @qd.kernel
        def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
            for i in range(n):
                if selector[i] > eps:
                    v = x[i]
                    for _ in range(n_iter):
                        v = v * 1.05 + 0.05
                    out[0] += v

    np.random.seed(2)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = max(1, int(round(gated_fraction * n)))
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"debug-build static-bound-expr grad returned NaN: {got_grad}"

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > eps, np.float32(expected_per_gated), np.float32(0.0))
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("gated_fraction", [0.05, 0.5, 1.0])
@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_grad_correct(gated_fraction):
    # Asserts gradient correctness for reverse-mode kernels of shape `for i in selector: if selector[i] > eps:
    # <adstack-using gradient work>` where `selector` is a `qd.field(...)` placed under `qd.root.dense(...)` -the layout
    # most sparse-grid workloads use. SNode counterpart to `test_adstack_static_bound_expr_ndarray_gate_grad_correct`;
    # parametrised over the gate-pass fraction (5%, 50%, 100%) so a regression in the SNode root-buffer load path or the
    # byte-offset precomputation surfaces as a wrong gradient.
    #
    # Internal details: the codegen pattern matcher captures the gating predicate as a `StaticBoundExpr` carrying the
    # leaf snode id plus the precomputed `(byte_base_offset, byte_cell_stride, iter_count)` triple the runtime needs to
    # walk the field at dispatch time without re-emitting the SNode lookup chain. The runtime then dispatches the
    # bound-reducer compute shader against the bound root buffer, counts threads whose `selector[i] > eps`, and sizes
    # the float adstack heap to that count.
    n = 256
    n_iter = 8
    eps = 1e-9

    selector = qd.field(qd.f32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in selector:
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(1)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = max(1, int(round(gated_fraction * n)))
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    for i in range(n):
        x[i] = float(x_np[i])
        selector[i] = float(selector_np[i])
    out[None] = 0.0
    out.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute()
    compute.grad()
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > eps, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    assert not np.isnan(got_grad).any(), f"static-bound-expr snode grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_ndarray_gate_compound_index_grad_correct():
    # Pins gradient correctness when an ndarray-backed gating array is indexed by a compound expression
    # (`selector[i % K]` with K < n). The ndarray arm of `match_field_source` validates per-axis that every
    # `ExternalPtrStmt::indices[axis]` is a `LoopIndexStmt`; compound indices like `i % K` are
    # `BinaryOpStmt(mod, ...)` so the validation rejects the capture and the runtime falls back to
    # dispatched-threads worst-case sizing on every backend. The reverse-mode gradient comes out correct because
    # the float adstack heap is sized for the full thread count and there is no LCA-block claim aliasing.
    n = 256
    K = 64
    n_iter = 8
    eps = 1e-9

    selector = qd.ndarray(qd.f32, shape=(K,))
    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i % K] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    np.random.seed(3)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = (np.random.rand(K) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    gated_per_iter = selector_np[np.arange(n) % K] > eps
    expected = np.where(gated_per_iter, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"compound-index ndarray grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "gate_shape",
    [
        "compound_mod",
        "affine_div",
        "constant_index",
        "dynamic_load_index",
        "folding_two_axis_decomp",
    ],
)
@test_utils.test(
    arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal],
    require=qd.extension.adstack,
    ad_stack_size=32,
    ad_stack_sparse_threshold_bytes=0,
)
def test_adstack_static_bound_expr_snode_gate_non_bijective_index_grad_correct(gate_shape):
    # Pins gradient correctness on parallel-dispatched backends for `qd.field`-backed gates whose index expression
    # is not a per-iteration bijection with the SNode cell space. Five shapes exercised:
    #   * `compound_mod`:        `selector[i % K]` with K < n               (loop hits each cell n/K times)
    #   * `affine_div`:          `selector[i / 2]`                          (pairs of iterations alias onto one cell)
    #   * `constant_index`:      `selector[K // 2]`                         (every iteration hits the same cell)
    #   * `dynamic_load_index`:  `selector[idx_field[i]]`                   (axis is a runtime load, not loop-derivable)
    #   * `folding_two_axis_decomp`:
    #                            `selector[i % 8, (i // 8) % 8]` with `loop_iter > 64` and an oversized SNode
    #                                                                       (axes are value-distinct so the same_value
    #                                                                        dedup admits them and `loop_iter <=
    #                                                                        snode_iter_count` because the SNode is
    #                                                                        large, but the joint axis space is
    #                                                                        8 * 8 = 64 and folds n iterations onto
    #                                                                        that subspace; the joint-axis-product
    #                                                                        check refuses capture)
    # In all five the SNode arm of `match_field_source` must refuse capture so the runtime falls back to the
    # dispatched-threads worst-case heap. With cross-row aliasing on a wrongly-captured heap, the reverse pass
    # reads a different thread's gate bool from the boolean adstack - the wrong set of `i` values contributes
    # to the gradient and the per-`i` mismatch is detectable even with a linear inner recurrence (the body's
    # chain rule may not need the primal, but the gate's boolean adstack does). CPU is excluded from `arch=`
    # because its dispatch thread count is small enough that the alias rarely fires. The companion bijective
    # multi-axis decomposition `selector[i // K, i % K]` is covered by
    # `test_adstack_static_bound_expr_snode_gate_bijective_decomposed_index_grad_correct`.
    n = 256
    K = 64
    n_iter = 8
    eps = 1e-9

    affine_div_size = n  # snode size = n keeps `loop_iter <= snode_iter_count`, so the iter-count check passes
    fold_axis = 8  # `folding_two_axis_decomp` joint space is `fold_axis * fold_axis = 64 < n = 256`
    if gate_shape == "affine_div":
        selector_shape = (affine_div_size,)
    elif gate_shape == "folding_two_axis_decomp":
        selector_shape = (fold_axis, fold_axis)
    else:
        selector_shape = (K,)
    selector = qd.field(qd.f32, shape=selector_shape)
    idx_field = qd.field(qd.i32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            gate_idx = (
                (i % K,)
                if qd.static(gate_shape == "compound_mod")
                else (
                    (i // 2,)
                    if qd.static(gate_shape == "affine_div")
                    else (
                        (K // 2,)
                        if qd.static(gate_shape == "constant_index")
                        else (
                            (i % fold_axis, (i // fold_axis) % fold_axis)
                            if qd.static(gate_shape == "folding_two_axis_decomp")
                            else (idx_field[i],)
                        )
                    )
                )
            )
            if selector[gate_idx] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(2)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = (np.random.rand(*selector_shape) < 0.3).astype(np.float32)
    idx_field_np = (np.arange(n) % K).astype(np.int32)
    selector.from_numpy(selector_np)
    for i in range(n):
        x[i] = float(x_np[i])
        idx_field[i] = int(idx_field_np[i])
    out[None] = 0.0
    out.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute()
    compute.grad()
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    if gate_shape == "compound_mod":
        gated_per_iter = selector_np[np.arange(n) % K] > eps
    elif gate_shape == "affine_div":
        gated_per_iter = selector_np[np.arange(n) // 2] > eps
    elif gate_shape == "constant_index":
        gated_per_iter = np.full(n, bool(selector_np[K // 2] > eps))
    elif gate_shape == "folding_two_axis_decomp":
        idx = np.arange(n)
        gated_per_iter = selector_np[idx % fold_axis, (idx // fold_axis) % fold_axis] > eps
    else:
        gated_per_iter = selector_np[idx_field_np] > eps
    expected = np.where(gated_per_iter, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    assert not np.isnan(got_grad).any(), f"non-bijective-index snode grad returned NaN ({gate_shape}): {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_bijective_decomposed_index_grad_correct():
    # Bijective multi-axis gate decomposed from a single flat loop variable: `for i in range(n): if
    # selector[i // K, i % K] > eps:`. Each iteration visits a unique cell because `(i // K, i % K)` is the
    # canonical bijection from `[0, n_rows * K)` to `[0, n_rows) x [0, K)`. The two axes share their root
    # `LoopIndexStmt`, but `same_value`-based deduplication recognises `i // K` and `i % K` as having
    # different values, so the SNode arm captures and shrinks the float adstack heap to the gate-passing
    # iteration count. Companion to `test_adstack_static_bound_expr_snode_gate_non_bijective_index_grad_correct`,
    # which pins the rejected shapes on the same `for i in range(n)` loop.
    K = 8
    n_rows = 32
    n = n_rows * K
    n_iter = 8
    eps = 1e-9

    selector = qd.field(qd.f32, shape=(n_rows, K))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            if selector[i // K, i % K] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(11)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = (np.random.rand(n_rows, K) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    idx = np.arange(n)
    gated_per_iter = selector_np[idx // K, idx % K] > eps
    expected = np.where(gated_per_iter, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"bijective decomposed-index snode grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_bijective_linear_range_grad_correct():
    # Bijective single-axis gate: `for i in range(n): if field[i] > eps:`. The gate's index is a bare `LoopIndexStmt`,
    # every iteration visits a unique cell, the SNode arm accepts capture and the backward grad matches `1.05^n_iter`
    # on every gated cell.
    n = 256
    n_iter = 8
    eps = 1e-9
    field = qd.field(qd.f32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            if field[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    field_np = (np.random.rand(n) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    field.from_numpy(field_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.where(field_np > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"linear-range bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_bijective_multi_axis_structfor_grad_correct():
    # Bijective multi-axis StructFor gate: `for I, J, K in field3d: if field3d[I, J, K] > eps:`. Each axis is a distinct
    # bare `LoopIndexStmt` (one per StructFor axis), so the joint mapping is bijective and the SNode arm captures.
    n = 16
    n_iter = 8
    eps = 1e-9
    field3d = qd.field(qd.f32, shape=(n, n, n))
    x = qd.field(qd.f32, shape=(n, n, n), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for I, J, K in field3d:
            if field3d[I, J, K] > eps:
                v = x[I, J, K]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n**3).reshape(n, n, n)).astype(np.float32)
    field_np = (np.random.rand(n, n, n) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    field3d.from_numpy(field_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.where(field_np > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"multi-axis StructFor bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_bijective_multi_axis_ndrange_grad_correct():
    # Bijective multi-axis ndrange gate: `for ii, jj, kk in qd.ndrange(n, n, n): if grid[ii, jj, kk] > eps:`. Each
    # iteration visits a unique cell. After `lower_access` rewrites the linearised offset into a `floordiv` /
    # `mod` / `sub` arithmetic tree over a single `LoopIndexStmt`, the per-axis components hold structurally
    # different values, so `same_value`-based deduplication of the iterating axes admits the joint-bijective
    # decomposition uniformly across LLVM and SPIR-V backends.
    n = 16
    n_iter = 8
    eps = 1e-9
    grid = qd.field(qd.f32, shape=(n, n, n))
    x = qd.field(qd.f32, shape=(n, n, n), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for ii, jj, kk in qd.ndrange(n, n, n):
            if grid[ii, jj, kk] > eps:
                v = x[ii, jj, kk]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n**3).reshape(n, n, n)).astype(np.float32)
    grid_np = (np.random.rand(n, n, n) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    grid.from_numpy(grid_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.where(grid_np > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"multi-axis ndrange bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_bijective_slice_with_iter_grad_correct():
    # Bijective gate with one iterating axis plus a constant slice: `for i in range(n): if grid[i, 0] > eps:`. The
    # first axis varies with the loop (bijective), the second axis is a literal constant (slice). The SNode arm
    # accepts capture; the reducer over-counts by the slice factor (it walks all `cols` cells per row) but
    # over-allocation is safe.
    n = 256
    cols = 4
    n_iter = 8
    eps = 1e-9
    grid = qd.field(qd.f32, shape=(n, cols))
    x = qd.field(qd.f32, shape=(n, cols), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            if grid[i, 0] > eps:
                v = x[i, 0]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n * cols).reshape(n, cols)).astype(np.float32)
    grid_np = (np.random.rand(n, cols) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    grid.from_numpy(grid_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.zeros_like(x_np)
    expected[:, 0] = np.where(grid_np[:, 0] > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"slice-with-iter bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=32, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_multi_axis_ndrange_with_arg_index_grad_correct():
    # Pins SNode-arm bound-expr capture for the canonical sparse-grid kernel shape: `for ii, jj, kk, ib in
    # qd.ndrange(...): if grid[f, ii, jj, kk, ib] > eps:`. The leading axis `f` is a kernel `qd.i32` argument that
    # slices the SNode space without folding iterations together; the four iterating axes cover the loop's flat
    # index range exactly once. The capture must engage (the analysis sees `loop_iter <= snode_iter_count` and at
    # least one `LinearizeStmt::input` containing a `LoopIndexStmt`), and gradients must match the analytic
    # backward of the inner recurrence. Companion regression test for `mpm_grid_op` in Genesis MPM, where this
    # shape with `len(f) > 1` previously lost capture under an over-strict bare-`LoopIndexStmt` check.
    F = 2
    n = 4
    B = 1
    n_iter = 4
    eps = 1e-9

    grid = qd.field(qd.f32, shape=(F, n, n, n, B))
    x = qd.field(qd.f32, shape=(F, n, n, n, B), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(f: qd.i32) -> None:
        for ii, jj, kk, ib in qd.ndrange(n, n, n, B):
            if grid[f, ii, jj, kk, ib] > eps:
                v = x[f, ii, jj, kk, ib]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(3)
    x_np = (0.1 + 0.001 * np.arange(F * n * n * n * B).reshape(F, n, n, n, B)).astype(np.float32)
    grid_np = (np.random.rand(F, n, n, n, B) < 0.4).astype(np.float32)
    grid.from_numpy(grid_np)
    x.from_numpy(x_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))

    f_active = 0
    compute(f_active)
    compute.grad(f_active)
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.zeros_like(x_np)
    expected[f_active] = np.where(grid_np[f_active] > eps, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"multi-axis ndrange grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=0, debug=False, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_snode_gate_primal_dependent_grad_correct():
    # Asserts gradient correctness on the LLVM CPU host reducer for SNode-backed gates with a primal-dependent inner
    # recurrence. The CPU host reducer must walk the SNode field and publish the gate-passing count so the float adstack
    # heap can be sized to that count; without the walk, the heap falls back to `num_cpu_threads * stride_float` while
    # the codegen-emitted LCA-block atomic-rmw produces row ids `0..n_gated-1`, and the over-claimed rows OOB into
    # unmapped memory or alias adjacent buffers.
    #
    # Internal details: the SNode-backed `selector` field (placed under `qd.root.dense(...)`) makes the analysis pass
    # capture the gating predicate as a `StaticBoundExpr` carrying the SNode descriptor triple (`byte_base_offset`,
    # `byte_cell_stride`, `iter_count`). The host reducer in `publish_per_task_bound_count_cpu`
    # (`runtime/llvm/llvm_runtime_executor.cpp`) walks the SNode at `bound_count_length = snode_iter_count` and writes
    # the count into the per-task capacity slot. The inner recurrence `v = v * v + 0.05` is primal-dependent so any
    # cross-row aliasing would re-read a different thread's pushed primal and surface as a wrong gradient even when the
    # OOB write happens to land within the heap allocation's over-allocated tail. `ad_stack_size = 0` lets the sizer
    # pick the per-thread stride; with 8 cpu threads and `n_gated = 2048` the row counter advances well past the
    # eight-row fallback so the OOB write reliably escapes the page mapped by the heap allocation guard.
    n = 4096
    n_iter = 8
    eps = 1e-9

    selector = qd.field(qd.f32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in selector:
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * v + 0.05
                out[None] += v

    np.random.seed(1)
    x_np = (0.001 * np.ones(n)).astype(np.float32)
    n_gated = max(1, n // 2)
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros(n, dtype=np.float32))

    compute()
    compute.grad()
    qd.sync()

    expected = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if selector_np[i] <= eps:
            continue
        v = float(x_np[i])
        primals = [v]
        for _ in range(n_iter):
            v = v * v + 0.05
            primals.append(v)
        d = 1.0
        for k in range(n_iter):
            d = d * (2.0 * primals[n_iter - 1 - k])
        expected[i] = np.float32(d)

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any()
    assert not np.isinf(got_grad).any()
    for i in range(n):
        assert got_grad[i] == pytest.approx(expected[i], rel=1e-5, abs=1e-7)


@test_utils.test(
    require=[qd.extension.adstack, qd.extension.data64], ad_stack_size=0, debug=False, ad_stack_sparse_threshold_bytes=0
)
def test_adstack_static_bound_expr_snode_gate_multileaf_dense_grad_correct():
    # Asserts gradient correctness on the LLVM static-bound-expr SNode resolver for dense parents with multiple
    # mixed-size leaves. The resolver must read each leaf's byte offset in declaration order (matching the LLVM struct
    # compiler's layout); reading from a size-sorted source would walk the wrong leaf's bytes during the reducer
    # dispatch and over-count gate-passing cells.
    #
    # Internal details: the dense parent `qd.root.dense(qd.i, n).place(field_f64, field_f32)` has two leaves of sizes 8
    # and 4 bytes; the LLVM struct compiler lays them out in declaration order (f64 at offset 0, f32 at offset 8) while
    # a size-sorted layout would place the f32 leaf at offset 0 and the f64 leaf at offset 8. The captured gating
    # predicate `field_f32[i] > eps` rides through the LLVM static-bound-expr resolver: a size-sorted resolver makes the
    # runtime reducer walk the field at offset 0 (the f64 leaf's low-half bytes) every cell-stride bytes. With the f64
    # leaf seeded to `1.0` everywhere, a misread at the f64 leaf's offset comparison-passes for every cell (the bit
    # pattern of the f64 1.0 low half is non-zero and greater than the f32 eps when reinterpreted), the reducer reports
    # `n` gate-passing cells while the main kernel's actual gated pass count is `n_gated`, the float adstack heap is
    # mis-sized and the codegen-emitted clamp aliases legitimate gated iterations onto wrong rows. The non-linear
    # recurrence `v = v * v + 0.05` makes the per-iteration gradient primal-dependent so any cross-row aliasing surfaces
    # as a wrong gradient. The f32 selector layout puts non-gated cells at 0.0 and gated cells at 1.0 with `eps = 1e-9`.
    # `arch=[qd.cpu, qd.cuda, qd.amdgpu]` because this test targets the LLVM snode_resolver specifically; SPIR-V
    # backends use the SPIR-V struct compiler natively for both the reducer and the main kernel so they agree on the
    # size-sorted offsets and are unaffected.
    n = 256
    n_iter = 6
    eps = 1e-9

    field_f64 = qd.field(qd.f64)
    field_f32 = qd.field(qd.f32)
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)
    qd.root.dense(qd.i, n).place(field_f64, field_f32)

    @qd.kernel
    def compute() -> None:
        for i in field_f32:
            if field_f32[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * v + 0.05
                out[None] += v

    np.random.seed(1)
    # `x` varies with `i` so any cross-row aliasing under a mis-sized adstack heap surfaces as a gradient mismatch (the
    # reverse pop reads back a different thread's primal). A constant `x` would mask aliasing because every gated thread
    # pushes the same primal sequence and the pop comes back identical.
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = max(1, n // 2)
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    for i in range(n):
        x[i] = float(x_np[i])
        field_f32[i] = float(selector_np[i])
        field_f64[i] = 1.0
    out[None] = 0.0
    out.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute()
    compute.grad()
    qd.sync()

    expected = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if selector_np[i] <= eps:
            continue
        v = float(x_np[i])
        primals = [v]
        for _ in range(n_iter):
            v = v * v + 0.05
            primals.append(v)
        d = 1.0
        for k in range(n_iter):
            d = d * (2.0 * primals[n_iter - 1 - k])
        expected[i] = np.float32(d)

    got_grad = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    assert not np.isnan(got_grad).any()
    assert not np.isinf(got_grad).any()
    for i in range(n):
        assert got_grad[i] == pytest.approx(expected[i], rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("bound_shape", ["int_const", "scalar_field", "ndarray_shape", "ndarray_read", "two_arg_range"])
@test_utils.test(require=qd.extension.adstack, ad_stack_size=128)
def test_adstack_static_bound_expr_memory_savings_runs_clean(bound_shape):
    # Asserts gradient correctness across every loop-bound shape the autodiff sizer documents as supported
    # (`docs/source/user_guide/autodiff.md::Appendix A`) when the kernel uses a captured gating predicate above
    # adstack-using inner work. Each shape resolves to the same `n` iteration count at launch time so the analytic
    # oracle is identical across cases; a regression that drops shape-product / scalar-field / two-arg-range support
    # from `determine_ad_stack_size` or from the `analyze_adstack_static_bounds` pre-pass surfaces as a wrong gradient
    # on exactly the shape that broke.
    #
    # Internal details: the codegen pattern matcher must recognise `field[i] cmp literal` immediately above the
    # adstack-using inner work; the runtime then sizes the float adstack heap to the gate-passing iteration count
    # instead of `dispatched_threads * stride * sizeof(elem)`. The kernel body is a non-linear recurrence in `x[i]` (`v
    # = x[i] * x[i]; v = v * 1.05 + 0.05; ...`) so the analytic per-iteration gradient `2 * x[i] * 1.05^n_iter` varies
    # with `i`; a regression that under-sizes the float heap (reducer count diverging from main-pass claim count) clamps
    # multiple gated iterations into the same heap row, the row's stored primal comes from whichever iteration last
    # pushed it, and the reverse pass attributes that primal's chain-rule contribution to a different `i` than the one
    # that wrote it. The per-`i` analytic oracle catches that aliasing as a wrong gradient on the affected indices.
    n = 256
    n_iter = 16
    eps = 1e-9

    np.random.seed(0)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = np.zeros(n, dtype=np.float32)
    selector_np[: max(1, int(round(0.5 * n)))] = 1.0
    np.random.shuffle(selector_np)

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))
    bound_arr = qd.ndarray(qd.i32, shape=(n,))
    n_field = qd.field(qd.i32, shape=())
    start_arr = qd.ndarray(qd.i32, shape=(1,))
    stop_arr = qd.ndarray(qd.i32, shape=(1,))
    n_field[None] = n
    bound_arr.from_numpy(np.full(n, n, dtype=np.int32))
    start_arr.from_numpy(np.array([0], dtype=np.int32))
    stop_arr.from_numpy(np.array([n], dtype=np.int32))

    @qd.kernel
    def compute(
        x: qd.types.NDArray,
        selector: qd.types.NDArray,
        out: qd.types.NDArray,
        bound_arr: qd.types.NDArray,
        start_arr: qd.types.NDArray,
        stop_arr: qd.types.NDArray,
    ) -> None:
        # `qd.static(bound_shape == ...)` evaluates the comparison at kernel-compile time (`bound_shape` is a Python
        # closure constant), so the AST that reaches the codegen has only one of the five `range` forms surviving - no
        # helper has to materialise per parametrisation.
        for i in (
            range(n)
            if qd.static(bound_shape == "int_const")
            else (
                range(n_field[None])
                if qd.static(bound_shape == "scalar_field")
                else (
                    range(selector.shape[0])
                    if qd.static(bound_shape == "ndarray_shape")
                    else (
                        range(bound_arr[0])
                        if qd.static(bound_shape == "ndarray_read")
                        else range(start_arr[0], stop_arr[0])
                    )
                )
            )
        ):
            if selector[i] > eps:
                v = x[i] * x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out, bound_arr, start_arr, stop_arr)
    compute.grad(x, selector, out, bound_arr, start_arr, stop_arr)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"sparse-adstack-heap [{bound_shape}] grad returned NaN: {got_grad}"
    coeff = 1.05
    # `v = x[i] * x[i]` then `v = v * 1.05 + 0.05` repeated n_iter times. v_final = x[i]^2 * c^n + S where S is a
    # constant. d(v_final)/d(x[i]) = 2 * x[i] * c^n. Gated only.
    expected = np.where(selector_np > eps, np.float32(2.0 * x_np * coeff**n_iter), np.float32(0.0))
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=64, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_primal_dependent_inner_recurrence_grad_correct():
    # Asserts gradient correctness for reverse-mode kernels with a captured ndarray-backed gate above a primal-dependent
    # inner recurrence (`v = qd.sin(v) + 0.01`, whose chain rule `d(sin(v))/dv = cos(v)` depends on the stored primal).
    # Slot-aliasing companion to `test_adstack_static_bound_expr_memory_savings_runs_clean`: any regression that
    # under-sizes the float adstack heap aliases multiple gated iterations onto the same row, the reverse pass evaluates
    # `cos(slot)` against the wrong iteration's `v`, and the per-`i` gradient diverges from the analytic oracle by a
    # primal-dependent factor.
    #
    # Internal details: a regression that derives the reducer length from `array_runtime_sizes / sizeof(int32_t)` while
    # the launcher receives an element-count-unit value from `set_args_ndarray` undercounts by `sizeof(elem)`x for
    # `qd.ndarray` arguments and triggers exactly this aliasing. The `v = x[i]; for _: v = sin(v) + 0.01; out += v`
    # recurrence is strictly nonlinear so the per-`i` gradient is computed offline via numpy on the same recurrence. `n`
    # is chosen so that capacity-vs-claims under any under-sized reducer length aliases multiple gated iterations into
    # the last reachable row; the divergence between the codegen output and the numpy reference scales linearly with the
    # number of aliased iterations, so the assertion catches the regression on every backend that under-sizes.
    n = 512
    n_iter = 4
    eps = 1e-9

    np.random.seed(0)
    x_np = (0.05 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = np.ones(n, dtype=np.float32)

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = qd.sin(v) + 0.01
                out[0] += v

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    # numpy reference: chain rule for `v_k = sin(v_{k-1}) + 0.01` is `cos(v_{k-1})`. d(v_n)/d(x[i]) is the product of
    # `cos(v_k)` for k = 0..n_iter-1, where the v_k sequence is generated forward from x[i].
    v_np = x_np.copy()
    grad_np = np.ones(n, dtype=np.float64)
    for _ in range(n_iter):
        grad_np *= np.cos(v_np.astype(np.float64))
        v_np = np.sin(v_np) + np.float32(0.01)
    expected = grad_np.astype(np.float32)

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"primal-dependent inner-recurrence grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=2e-4, atol=2e-6)


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64, ad_stack_size=32)
def test_adstack_static_bound_expr_non_loop_var_index_falls_back_to_worst_case():
    # Asserts gradient correctness for reverse-mode kernels whose gating predicate uses a non-`LoopIndexStmt` index
    # expression (e.g. `selector[i % K]`, `selector[const]`, `selector[i + 1]`, `selector[other_field[i]]`). The
    # static-bound-expr capture must reject such gates so the heap-sizing path falls back to the dispatched-threads
    # worst case for that task, rather than walking `selector[0..length)` against a divergent claim-count basis and
    # aliasing iterations into the last reachable row.
    #
    # Internal details: the reducer walks the gating ndarray as `selector[0..length)` and counts gate-passing cells; the
    # main-kernel LCA-block atomic-rmw fires once per gated iteration of the actual index. A captured gate with a
    # non-loop-index index makes the two counts diverge, the codegen-emitted clamp aliases multiple gated iterations
    # into the last reachable row, and the result is silent gradient corruption on LLVM / hard overflow on SPIR-V. The
    # kernel below uses `selector[i % K]` so the same 4 selector cells are read `n / K = 16` times each but only
    # `n_gated = 4` of those reads pass the gate; without the rejection the reducer counts at most 4 gate-passing cells
    # in `selector[0..n)`, the float heap is sized for 4 rows while 16 gated LCA reaches happen on each row, and rows
    # 1..15 of every iteration's claim alias into row 0/1/2/3. `match_field_source`'s `LoopIndexStmt`-only check rejects
    # the gate capture for this task only (this `OffloadedStmt` / outer parallel-for); the rest of the kernel's tasks
    # still capture their gates if their index is the loop's own `LoopIndexStmt`. The rejected task falls back to the
    # worst-case `dispatched_threads * stride_float` heap sizing - safe (no aliasing), at the cost of the savings the
    # bound-reducer path would have given for that one task.
    n = 64
    K = 4
    n_iter = 8
    eps = 1e-12

    np.random.seed(0)
    # Spread `x` widely across the f64 representable range so per-`i` `cos(x[i])` differs by O(0.1) between adjacent
    # indices; under f64 precision the multi-thread CPU race produces a clearly observable drift in the per-`i`
    # chain-rule product when the gate-capture pretends `selector[i % K]` is loop-index-shaped.
    x_np = (0.5 + 0.05 * np.arange(n)).astype(np.float64)
    selector_np = np.zeros(n, dtype=np.float64)
    selector_np[:K] = 1.0  # first K cells gated; rest zero

    x = qd.ndarray(qd.f64, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f64, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f64, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i % K] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = qd.sin(v) + 0.01
                out[0] += v

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float64))
    out.grad.from_numpy(np.ones((1,), dtype=np.float64))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    # `v = sin(v) + c` has a primal-dependent chain rule `cos(v_{k-1})`. Each iteration's reverse pass multiplies
    # adjoints by `cos(stored_primal)`, so a slot read corrupted by a different iteration's push produces a
    # primal-dependent wrong factor. With selector[:K] = 1.0 every iteration is gated; numpy reference computes the
    # chain forward then products `cos(v_k)` for k = 0..n_iter-1.
    v_np = x_np.copy()
    grad_np = np.ones(n, dtype=np.float64)
    for _ in range(n_iter):
        grad_np *= np.cos(v_np)
        v_np = np.sin(v_np) + 0.01

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"non-loop-var-index grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, grad_np, rtol=1e-12, atol=1e-14)


@test_utils.test(
    arch=[qd.cuda, qd.amdgpu],
    require=[qd.extension.adstack, qd.extension.data64],
    default_fp=qd.f64,
    ad_stack_size=2048,
)
def test_adstack_gpu_dispatch_cap_uses_floor_division():
    # Asserts gradient correctness for CUDA / AMDGPU adstack-bearing kernels whose `block_dim` does not divide
    # `kAdStackMaxConcurrentThreads = 65536` evenly. The launcher must cap such kernels' grid using floor division so
    # the dispatched thread count stays within the float heap row count; ceiling division would over-dispatch the last
    # block and OOB-write past the heap end, manifesting as `cudaErrorIllegalAddress` (CUDA) / `hipErrorIllegalAddress`
    # (AMDGPU) at sync.
    #
    # Internal details: the launcher caps adstack-bearing tasks at `cap_blocks * block_dim` threads. With `block_dim =
    # 192` floor division gives `cap_blocks = floor(65536/192) = 341`, dispatched = `341 * 192 = 65472`; ceiling
    # division gives `342`, dispatched = `342 * 192 = 65664` - 128 threads past the heap row count.
    # `resolve_num_threads` floors at 65536 and the non-bound_expr float heap is sized at `n_threads * stride_float` for
    # `n_threads = 65536`, so any thread with `linear_thread_idx in [65536, 65664)` would index past the heap end. The
    # kernel has 65700 iterations so each dispatched thread reaches at least one `i` past 65536; with
    # `ad_stack_size=2048` the per-thread stride is ~16 KB at f64 so a misdispatch's OOB write lands in unmapped device
    # memory rather than aliasing into another adjacent buffer. arch=[qd.cuda, qd.amdgpu] only because Metal requires
    # `block_dim` to be a power of two. default_fp=qd.f64 because CUDA's libdevice `__nv_sinf` / `__nv_cosf` carry ~3
    # ULP error in f32 and a 6-deep sin/cos composition compounds to ~1.5e-5 relative drift against numpy's libm
    # reference, right at the rtol boundary; f64 transcendentals are ~1 ULP on both libdevice and rocm libm so the drift
    # drops to ~6e-15 relative and the tolerance can stay tight, and the f64 stride (8 B vs 4 B) doubles the per-thread
    # heap footprint, making OOB bugs strictly easier to detect.
    n = 65700
    block_dim = 192
    n_inner = 6

    x_np = (0.5 + 0.001 * np.arange(n)).astype(np.float64)

    x = qd.ndarray(qd.f64, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f64, shape=(1,), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.NDArray, out: qd.types.NDArray) -> None:
        qd.loop_config(block_dim=block_dim)
        for i in range(n):
            v = x[i]
            for _ in range(n_inner):
                v = qd.sin(v) + 0.01
            out[0] += v

    x.from_numpy(x_np)
    out.from_numpy(np.zeros((1,), dtype=np.float64))
    out.grad.from_numpy(np.ones((1,), dtype=np.float64))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, out)
    compute.grad(x, out)
    qd.sync()

    v_np = x_np.copy()
    grad_ref = np.ones(n, dtype=np.float64)
    for _ in range(n_inner):
        grad_ref *= np.cos(v_np)
        v_np = np.sin(v_np) + 0.01

    got_grad = x.grad.to_numpy()
    np.testing.assert_allclose(got_grad, grad_ref, rtol=1e-12, atol=1e-14)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=0, debug=False)
def test_adstack_static_bound_expr_device_sizer_per_kind_offsets_grad_correct():
    # Asserts gradient correctness on CUDA / AMDGPU for kernels that interleave float and int adstack allocas in source
    # order, when the SizeExpr contains an ExternalTensorRead leaf (so the device sizer runs instead of the host-eval
    # path). The device sizer must write per-kind running offsets into `adstack_offsets[stack_id]`, not the combined
    # prefix sum across all stacks; a combined prefix sum makes the codegen address each alloca's tape using a byte
    # offset that includes the other kind's strides, landing the tape inside an adjacent thread's slice and producing
    # wrong gradients on the cross-thread primal reload.
    #
    # Internal details: the codegen reads `adstack_offsets[stack_id]` as an offset within the per-kind slice (float
    # allocas: `heap_float + linear_tid * stride_float + offsets[i]`; int / u1 allocas: `heap_int + linear_tid *
    # stride_int + offsets[i]`). The kernel below interleaves two f32 allocas (`v0`, `v1`) and one i32 alloca (`j`) in
    # source order so the IR pre-scan in `init_offloaded_task_function` assigns stack ids 0 (float), 1 (int), 2 (float).
    # Under a combined prefix sum, `out_offsets[2] = step_v0 + step_j` - non-zero - which the codegen interprets as a
    # byte offset within `heap_float`'s slice for `v1`. With `stride_float = step_v0 + step_v1` and `step_v0 + step_j >
    # step_v0`, `v1`'s tape for thread `t` lands inside thread `(t+1)`'s float slice; thread `t`'s reverse pass then
    # reads `v1`'s saved primal that thread `(t+1)` wrote, which is x[(t+1)]'s tape. Restricted to LLVM CUDA / AMDGPU
    # because (a) CPU goes through `use_host_eval=true` and uses the host-eval branch of `publish_adstack_metadata`
    # whose per-kind write is correct, (b) Metal / Vulkan use the SPIR-V sizer compute shader
    # (`codegen/spirv/adstack_sizer_shader.cpp`) which already does per-kind offsets correctly. `ad_stack_size=0` lets
    # the SizeExpr's launch-time evaluator pick the per-launch bound; `debug=False` keeps the release-build inline push
    # / pop emit path so the tape addressing math goes through `get_ad_stack_base_llvm` rather than the runtime
    # helper-call path which would also exercise the bug but takes a different code path through `stack_init`.
    n_outer = 8
    a_np = np.array([2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32)

    x = qd.field(qd.f32, shape=(n_outer,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in x:
            v0 = x[i] * 1.0
            j = 0
            v1 = x[i] * 2.0
            n = a[i]
            for _ in range(n):
                v0 = v0 * 0.95 + 0.01
                j = j + 1
                v1 = v1 * 0.9 + 0.02
            y[None] += v0 + v1 + qd.cast(j, qd.f32) * 0.0

    for i in range(n_outer):
        x[i] = 0.1 + 0.05 * i

    compute(a_np)
    y.grad[None] = 1.0
    for i in range(n_outer):
        x.grad[i] = 0.0
    compute.grad(a_np)
    qd.sync()

    for i in range(n_outer):
        # d(v0_n + v1_n) / dx[i] = 1.0 * 0.95**a[i] + 2.0 * 0.9**a[i].
        expected = 1.0 * (0.95 ** int(a_np[i])) + 2.0 * (0.9 ** int(a_np[i]))
        assert x.grad[i] == pytest.approx(expected, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, ad_stack_size=0, ad_stack_sparse_threshold_bytes=0)
def test_adstack_static_bound_expr_resolve_length_walks_full_ndarray():
    # Asserts gradient correctness on Metal / Vulkan when an adstack-bearing kernel's gating ndarray is larger than the
    # SPIR-V grid-stride advisory cap (`kMaxNumThreadsGridStrideLoop = 131072`) and all gated cells live past the cap.
    # The launcher's reducer must walk the full flat element product of the gating ndarray (not just the first 131072
    # cells) so the float adstack heap is sized for every gated iteration; capping the walk at the advisory would size
    # the heap to zero rows on workloads whose gates only fire past index 131072 and silently corrupt gradients on every
    # gated index.
    #
    # Internal details: the kernel places all gated cells at indices [131072, 131072+n_gated_past_cap) and runs the
    # inner recurrence `v = v * 1.05 + 0.05` so the autodiff transform actually pushes loop-carried primals onto the
    # float adstack (a single `qd.sin(x[i])` would not - sin's adjoint reloads `x[i]` directly without consulting the
    # adstack). A reducer that walks only `selector[0..131072)` counts 0 gate-passing cells, the float heap is floored
    # at 1 row, and every gated iteration's `OpAtomicIAdd` on the row counter clamps back to row 0 via the
    # codegen-emitted `select(capacity == 0, 0, capacity - 1)` upper-bound; all n_gated_past_cap forward push streams
    # alias onto row 0 and the reverse pop reads back whichever iteration's primal landed last, producing one common
    # gradient value for every gated index instead of the per-i `1.05 ** n_iter` the analytic oracle expects.
    # arch=[qd.metal, qd.vulkan] because CPU and CUDA / AMDGPU launchers have their own `bound_count_length` derivation
    # paths whose advisory-cap shape is exercised by separate tests.
    n_gated_past_cap = 64  # enough to alias multiple iterations into a single row if the heap mis-sizes to one row
    advisory_cap = 131072  # SPIR-V kMaxNumThreadsGridStrideLoop
    n = advisory_cap + n_gated_past_cap
    n_iter = 4

    selector = qd.ndarray(qd.f32, shape=(n,))
    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > 1e-9:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    x_np = (0.001 * np.arange(n) + 0.1).astype(np.float32)
    selector_np = np.zeros(n, dtype=np.float32)
    selector_np[advisory_cap : advisory_cap + n_gated_past_cap] = 1.0  # all gated cells past the advisory cap
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got = x.grad.to_numpy()
    expected_per_gated = np.float32(1.05**n_iter)
    expected = np.where(selector_np > 1e-9, expected_per_gated, np.float32(0.0)).astype(np.float32)
    assert not np.isnan(got).any(), f"resolve_length grad returned NaN: {got[advisory_cap:advisory_cap + 8]}"
    for i in range(advisory_cap, advisory_cap + n_gated_past_cap):
        assert got[i] == pytest.approx(expected[i], rel=1e-5, abs=1e-7), (
            f"gated index {i} (past advisory_total_num_threads={advisory_cap}) gradient diverged: "
            f"got={got[i]} expected={expected[i]}"
        )


@pytest.mark.parametrize(
    "shape, body_kind",
    # `shape` selects whether the per-task sizer's `1<<24` host-eval cap fires; the smaller shape stays well below the
    # cap, the larger one crosses it. `body_kind` selects which body-leaf and combinator mix the recognizer must accept
    # and the encoder must lower correctly before the device walk. Each `(shape, body_kind)` combination is designed so
    # the body's max value over the captured ndarray is always `N_X`, keeping the asserted gradient identical across the
    # matrix.
    [
        (256, "extread"),
        ((1 << 24) + 1, "extread"),
        ((1 << 24) + 1, "shape_in_body"),
        ((1 << 24) + 1, "field_in_body"),
        ((1 << 24) + 1, "arith_combine"),
    ],
    ids=[
        "small_extread",
        "above_cap_extread",
        "above_cap_shape_in_body",
        "above_cap_field_in_body",
        "above_cap_arith_combine",
    ],
)
@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack, cfg_optimization=False)
def test_max_reducer_pins_stride_for_oversized_axis(shape, body_kind):
    # A reverse-mode kernel with a parallel-for over an arbitrarily large ndarray axis and an inner range-for bound to
    # a recognizer-accepted trip-count expression sizes its adstack at launch time and computes the right gradient,
    # without the per-task sizer's `1<<24` cap firing. GPU only: the max-reducer dispatch is GPU-specific - the host
    # evaluator handles equivalent shapes on CPU.
    #
    # Internal details: the kernel lowers to `MaxOverRange(0, a.shape[0], <body>)` in the per-stack `SizeExpr`.
    # `recognize_adstack_max_reducer_specs` captures the spec; the launcher dispatches the parallel max-reducer before
    # the per-task sizer walks the tree; `substitute_precomputed_max_over_range` rewrites the captured `MaxOverRange`
    # to `Const`. The above-cap variants place the only non-zero cell at `arr_np[-1] = N_X` so heap-stride correctness
    # depends on the dispatch walking every element of the axis rather than relying on a partial host-eval walk. The
    # `shape_in_body` / `field_in_body` variants additionally pin that closed leaves (`ExternalTensorShape`,
    # `FieldLoad`) host-fold to `kConst` at encode time and never reach the device interpreter; `arith_combine`
    # exercises every binary combinator (`Add`, `Sub`, `Mul`, `Max`) and `Const` leaf in a single body expression that
    # algebraically reduces to `a[i_e]`. The CPU codegen gate lives in
    # `codegen/llvm/codegen_llvm.cpp::finalize_offloaded_task_function`; the lifted host-eval cap lives in
    # `program/adstack/eval.cpp::evaluate_node`. On CPU `_get_max_reducer_dispatch_count` stays at 0 (no dispatch
    # fires), which is why this test pins it on GPU arches only.
    N_X = 4
    arr_np = np.zeros(shape, dtype=np.int32)
    arr_np[-1] = N_X
    # `qd.ndarray` rather than the numpy passthrough so the underlying device buffer is host-managed by Quadrants; numpy
    # passthrough (`kNone` H2D-blit) caps the device-side mirror at backend-specific limits on macOS Metal for arrays
    # above ~32 MB, which would prevent the dispatch from observing the cell at `arr_np[-1]` in the above-cap variant.
    arr = qd.ndarray(qd.i32, shape=(shape,))
    arr.from_numpy(arr_np)

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)
    # Closed `FieldLoad` leaf for the `field_in_body` variant. Set to zero so the body's max value remains `N_X`
    # regardless of the body kind, keeping the asserted gradient uniform across the parametrized matrix.
    gate = qd.field(qd.i32, shape=())
    gate[None] = 0

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_e in range(a.shape[0]):
            # `qd.static(...)` selects the body shape at kernel compile time so each parametrization compiles a
            # single-branch kernel; every form has algebraic max value `a[i_e]`. The `arith_combine` form exercises
            # `Add` / `Sub` / `Mul` / `Max` / `Const` together: outer `Max` of the two equal sub-expressions `a[i_e] +
            # 0` (`Add` + `Const`) and `a[i_e] * 1 - 0` (`Mul` + `Sub` + `Const`).
            n = (
                a[i_e]
                if qd.static(body_kind == "extread")
                else (
                    a[i_e] + (a.shape[0] - a.shape[0])
                    if qd.static(body_kind == "shape_in_body")
                    else (
                        max(a[i_e], gate[None])
                        if qd.static(body_kind == "field_in_body")
                        else max(a[i_e] + 0, a[i_e] * 1 - 0)
                    )
                )
            )
            accum = 0.0
            for j in range(n):
                accum = accum + x[j] * x[j]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()

    compute(arr)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr)
    qd.sync()

    # Only the last outer iteration walks the inner loop; every other iteration contributes nothing. The max-reducer
    # dispatch covers every element of `arr` so the heap stride lands at the actual maximum (= N_X), and
    # `compute.grad(arr)` plus `qd.sync()` runs to completion. The expected per-slot gradient is `2 * x[k]` since each
    # surviving inner iteration contributes `2 * x[k]` to the reverse pass.
    assert prog._get_max_reducer_dispatch_count() >= 1
    for k in range(N_X):
        assert x.grad[k] == pytest.approx(2 * 0.1, rel=1e-5)


@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack, cfg_optimization=False)
def test_max_reducer_dispatch_counts_advance_on_input_mutation():
    # Pins the dispatch + cache invalidation pipeline. The first launch must fire at least one max-reducer dispatch
    # (the kernel's `MaxOverRange(0, a.shape[0], a[var])` matches the recognizer grammar so the recognizer captures
    # the spec; the launcher dispatches once and bumps `Program.max_reducer_dispatch_count`). A subsequent host
    # mutation of the gating ndarray must bump `ndarray_data_gen` and force the next launch to re-dispatch, advancing
    # the counter beyond its post-first-launch value. Steady-state cache short-circuit on an unchanged ndarray is
    # backend-dependent (the CPU launcher's `set_host_accessible_ndarray_ptrs` path converts qd.ndarray reads to
    # `kNone` semantics and `bump_writes_for_kernel_llvm` then bumps the gen on every read; the SPIR-V launchers
    # preserve the qd.ndarray dev-alloc-type and only bump on writes), so this test asserts only the
    # mutation-triggers-redispatch contract that holds uniformly. GPU only: the max-reducer dispatch is GPU-specific -
    # the host evaluator handles equivalent shapes on CPU.
    N = 4

    x = qd.field(qd.f32, shape=(N,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            v = x[i]
            n = a[i]
            for _ in range(n):
                v = v * 0.95 + 0.01
            y[None] += v

    a = qd.ndarray(qd.i32, shape=(N,))
    a.from_numpy(np.array([2, 3, 1, 2], dtype=np.int32))
    for i in range(N):
        x[i] = 0.1

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()

    compute(a)
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad(a)
    qd.sync()
    after_first = prog._get_max_reducer_dispatch_count()
    assert after_first >= 1

    a.from_numpy(np.array([3, 3, 1, 2], dtype=np.int32))
    pre_mutation = prog._get_max_reducer_dispatch_count()
    compute(a)
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad(a)
    qd.sync()
    assert prog._get_max_reducer_dispatch_count() > pre_mutation


@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack)
def test_max_reducer_per_kernel_registry_id_isolation():
    # Two `qd.template()` instantiations of the same kernel hash to distinct registry ids so the per-spec max-reducer
    # cache keyed by `(registry_id, stack_id, mor_node_idx)` does not leak entries across them. GPU only: the max-
    # reducer dispatch is GPU-specific.
    #
    # Internal details: a single `@qd.kernel` definition takes a `qd.template()` flag and selects between two distinct
    # reverse-mode bodies that write to different loss fields. Each instantiation captures a `MaxOverRange` spec at the
    # same `(stack_id, mor_node_idx)` coordinates because the bodies are structurally identical, and they share the same
    # `arr` so a stale entry would pass the observation freshness check (same devalloc, same write gen). The dispatch-
    # count delta on the second instantiation pins that the SPIR-V launcher registers each task with the real kernel
    # name so the registry slot is unique per kernel handle.
    arr = qd.ndarray(qd.i32, shape=(2,))
    arr.from_numpy(np.array([0, 2], dtype=np.int32))

    x = qd.field(qd.f32, shape=(2,), needs_grad=True)
    loss_a = qd.field(qd.f32, shape=(), needs_grad=True)
    loss_b = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1), flag: qd.template()):
        for i in range(a.shape[0]):
            accum = 0.0
            for j in range(a[i]):
                accum = accum + x[j] * x[j]
            if qd.static(flag):
                loss_a[None] += accum
            else:
                loss_b[None] += accum

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()

    compute(arr, False)
    compute.grad(arr, False)
    qd.sync()
    after_false = prog._get_max_reducer_dispatch_count()

    compute(arr, True)
    compute.grad(arr, True)
    qd.sync()
    assert prog._get_max_reducer_dispatch_count() > after_false


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_max_reducer_grammar_fallback():
    # Pins the recognizer's grammar gate. A reverse-mode kernel whose inner trip count is a compile-time constant (no
    # `MaxOverRange` wrapper in the resulting `SizeExpr`) does not match the recognizer grammar and there is no spec for
    # `recognize_adstack_max_reducer_specs` to capture. The launcher's pre-publish dispatch finds an empty
    # `max_reducer_specs` list, fires no max-reducer dispatch, and the per-task sizer's existing host / device evaluator
    # handles the constant trip count via its `Const` leaf path. The dispatch counter must stay at zero and the
    # analytical gradient must still match. Pins the "any kernel outside the captured grammar runs unchanged" contract
    # so future grammar broadening cannot silently drop the fallback path.
    N = 4
    K = 3

    x = qd.field(qd.f32, shape=(N,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(N):
            v = x[i]
            for _ in range(K):
                v = v * 0.95 + 0.01
            y[None] += v

    for i in range(N):
        x[i] = 0.1

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()

    compute()
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad()
    qd.sync()

    assert prog._get_max_reducer_dispatch_count() == 0
    expected = 0.95**K
    for i in range(N):
        assert x.grad[i] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "body_kind",
    [
        "field_bv",
        "field_bv_plus_arr_bv",
        "arr_bv_plus_field_bv",
        "max_field_bv_arr_bv",
        "max_field_bv_const",
        "field_bv_arith_combine",
        "field_bv_indexed_by_field_load",
        "arr_bv_indexed_by_field_load",
    ],
)
@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack)
def test_max_reducer_field_load_bound_var_dispatch(body_kind):
    # A reverse-mode kernel whose inner range-for trip count reads a `qd.field` indexed by the outer chain variable
    # captures via the parallel max-reducer dispatch and produces the analytical gradient. The body-shape
    # parametrization exercises every supported composition: bound-var FieldLoad on its own, mixed with bound-var ETR
    # via `Add` / `Max`, combined with `Const` / arithmetic, and the nested-load worst-case form (`field[field[i]]` /
    # `arr[field[i]]`). GPU only: the max-reducer dispatch is GPU-specific - the host evaluator handles equivalent
    # shapes on CPU.
    #
    # Internal details: each variant lowers to `MaxOverRange(0, M, body)` where `body` is bound-var-indexed
    # `FieldLoad(field_a, [bound_var])` or a recognizer-accepted composition that includes one. The relaxed
    # `max_reducer_body_is_recognizable::FieldLoad` arm accepts the leaf, the encoder emits a `kFieldLoad` device node
    # whose base pointer is pre-resolved on host (PSB on SPIR-V, `runtime->roots[id] + place_byte_offset` on LLVM),
    # and the dispatch reads `field_a[i]` for every `i` and keeps the max. The two `_indexed_by_field_load` variants
    # exercise the conservative-wrapper path: `SerializedSizeExprNode::indices` carries one int32 per axis (no
    # subtree refs), so the trip-count builder substitutes `MaxOverRange(var, 0, leaf_snode.shape, body=Load(snode,
    # [var]))` that iterates the leaf snode's full axis - the recognizer captures it via the same bound-var route and
    # the dispatched max equals `max_k field_a[k]` (resp. `max_k arr[k]`). Across all variants the body's max value
    # over the indexed range is `N_X`, keeping the asserted gradient identical.
    N_X = 4
    M = 8
    # Field-a holds the bound-var-indexed counter values: peak value `N_X` lands at the last cell, so a per-element walk
    # is necessary to observe the heap-stride correctness; a partial walk that stops at the first non-zero cell would
    # under-bound the heap stride.
    field_a = qd.field(qd.i32, shape=(M,))
    field_a_init = np.zeros(M, dtype=np.int32)
    field_a_init[-1] = N_X
    for i in range(M):
        field_a[i] = int(field_a_init[i])
    # Field-b is the inner-index source for the `_indexed_by_field_load` variants. Setting every cell to the index of
    # field_a's peak (M-1) routes every outer iteration to the cell holding `N_X`; the dispatch's worst-case wrapper
    # walks field_a's full axis regardless, so the max reduction still observes `N_X` and the gradient stays uniform.
    field_b = qd.field(qd.i32, shape=(M,))
    for i in range(M):
        field_b[i] = M - 1
    arr = qd.ndarray(qd.i32, shape=(M,))
    arr.from_numpy(field_a_init)

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_e in range(M):
            # Each variant is an algebraic identity over the value at `field_a[i_e]` (or `field_a[field_b[i_e]]` for the
            # nested-load forms): max value over the captured axis is `N_X` so the asserted gradient stays uniform.
            n = (
                field_a[i_e]
                if qd.static(body_kind == "field_bv")
                else (
                    field_a[i_e] + (a[i_e] - a[i_e])
                    if qd.static(body_kind == "field_bv_plus_arr_bv")
                    else (
                        a[i_e] + (field_a[i_e] - a[i_e])
                        if qd.static(body_kind == "arr_bv_plus_field_bv")
                        else (
                            max(field_a[i_e], a[i_e])
                            if qd.static(body_kind == "max_field_bv_arr_bv")
                            else (
                                max(field_a[i_e], 0)
                                if qd.static(body_kind == "max_field_bv_const")
                                else (
                                    max(field_a[i_e] + 0, field_a[i_e] * 1 - 0)
                                    if qd.static(body_kind == "field_bv_arith_combine")
                                    else (
                                        field_a[field_b[i_e]]
                                        if qd.static(body_kind == "field_bv_indexed_by_field_load")
                                        else a[field_b[i_e]]
                                    )
                                )
                            )
                        )
                    )
                )
            )
            accum = 0.0
            for j in range(n):
                accum = accum + x[j] * x[j]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()

    compute(arr)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr)
    qd.sync()

    # Only one outer iteration walks the inner loop with a non-zero count (the cell at position `M-1` for the direct
    # variants, or every iteration via field_b -> field_a[M-1] for the nested variants); each surviving inner
    # iteration contributes `2 * x[k]` to `x.grad[k]`. The recognizer captures every variant via the bound-var
    # FieldLoad / ETR path so the dispatch counter must advance.
    assert prog._get_max_reducer_dispatch_count() >= 1
    if body_kind in ("field_bv_indexed_by_field_load", "arr_bv_indexed_by_field_load"):
        # Nested-load worst-case: every outer iteration routes to the peak cell so the reverse pass accumulates `M`
        # times.
        expected = 2 * 0.1 * M
    else:
        expected = 2 * 0.1
    for k in range(N_X):
        assert x.grad[k] == pytest.approx(expected, rel=1e-5)


@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack)
def test_max_reducer_field_load_bound_var_cache_invalidates_on_snode_mutation():
    # A reverse-mode kernel whose inner trip count reads a `qd.field` indexed by the outer chain variable redispatches
    # the max-reducer when the gating field is mutated between launches. GPU only: the max-reducer dispatch is
    # GPU-specific - the host evaluator handles equivalent shapes on CPU.
    #
    # Internal details: the encoder emits a `kFieldLoad` device node and pushes a `FieldLoadObs` carrying the snode id
    # and the live `snode_write_gen` snapshot. On the second launch's `try_max_reducer_cache_hit`,
    # `replay_one_observation`'s `FieldLoadObs` arm fast-skips on a matching gen and otherwise falls through to the
    # invalidate path (`obs.indices == {}` means the gen counter is the sole staleness signal for max-reducer body
    # observations). Mutating `field_a[M-1]` from Python bumps `snode_write_gen` so the second launch's replay
    # invalidates the entry and the dispatch counter advances beyond `after_first`.
    M = 8
    N_X = 4

    field_a = qd.field(qd.i32, shape=(M,))
    for i in range(M):
        field_a[i] = 0
    field_a[M - 1] = 2

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i_e in range(M):
            n = field_a[i_e]
            accum = 0.0
            for j in range(n):
                accum = accum + x[j] * x[j]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()

    compute()
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad()
    qd.sync()
    after_first = prog._get_max_reducer_dispatch_count()
    assert after_first >= 1

    # Bump field_a's peak value to force a different max; the snode write must bump `snode_write_gen` and the next
    # launch's cache replay must invalidate, advancing the dispatch counter.
    field_a[M - 1] = 4
    pre_mutation = prog._get_max_reducer_dispatch_count()
    compute()
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad()
    qd.sync()
    assert prog._get_max_reducer_dispatch_count() > pre_mutation


@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack, cfg_optimization=False)
def test_above_cap_out_of_grammar_kernel_raises():
    # A reverse-mode kernel whose inner `range(...)` trip count is bound to an out-of-grammar `MaxOverRange` body and
    # whose iteration count exceeds the `1<<24` adstack-sizer cap surfaces a `QuadrantsAssertionError` at `qd.sync()`.
    # GPU only: on CPU the host-eval cap is lifted to UINT32_MAX, so a shape of `(1<<24)+1` resolves without raising.
    #
    # Internal details: the recognizer's body grammar accepts only `Const / ExternalTensorRead / Add / Sub / Mul / Max
    # / ExternalTensorShape / FieldLoad(literal-or-bound-var indices)`, and `max_reducer_body_is_recognizable` further
    # restricts `ExternalTensorRead` leaves to dtypes whose value range cannot collide with the cache-revalidation
    # sentinel (`INT64_MIN`) - `i8 / i16 / i32 / u8 / u16 / u32` only. An `i64` ndarray read passes the host evaluator
    # (`evaluate_node`'s `ExternalTensorRead` arm reads any integer dtype) but fails the recognizer's dtype check, so
    # the whole spec is dropped and the per-task sizer walks the outer `MaxOverRange` itself. With `a.shape[0] >
    # 1<<24` the cap fires on the host evaluator (`QD_ERROR_IF` in `adstack_size_expr_eval.cpp::evaluate_node`, raised
    # as `RuntimeError` on the CPU host fast path) and on the SPIR-V on-device sizer (the trailing overflow-flag slot
    # of the metadata buffer, raised as `QuadrantsAssertionError` from the host post-readback in
    # `publish_adstack_metadata_spirv`). The CUDA and AMDGPU LLVM-GPU sizer short-circuits the walk and returns 0 from
    # `device_eval_node`'s `kMaxOverRange` arm so the single-thread on-device dispatch stays within the driver's TDR
    # window; the cap-hit then surfaces indirectly via the existing `stack_push` overflow infrastructure on a
    # subsequent main-kernel launch, and the resulting diagnostic message attribution depends on the kernel layout.
    # That indirect path is covered by `test_adstack_overflow_diagnostic_and_auto_recovery`.
    N_X = 4
    shape = (1 << 24) + 1
    # All-zero gating ndarray keeps the forward kernel's actual inner-loop work at zero on every thread; the cap-hit is
    # purely a property of the symbolic `MaxOverRange` iteration count, so we do not need any cell to be non-zero for
    # the per-task sizer's walk to overflow the guard.
    a_data = np.zeros(shape, dtype=np.int64)
    a = qd.ndarray(qd.i64, shape=(shape,))
    a.from_numpy(a_data)

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i64, ndim=1)):
        for i_e in range(a.shape[0]):
            # `a` is an `i64` ndarray, so the inner `MaxOverRange`'s `end` is an `ExternalTensorRead` with leaf dtype
            # `i64`. `max_reducer_body_is_recognizable` rejects `i64 / u64` leaves (the cache-revalidation sentinel
            # `INT64_MIN` is a legal value of an `i64` cell, so a mutated cache entry could false-hit on revalidation).
            # The whole spec is dropped and the per-task sizer walks the outer `MaxOverRange(0, shape[0], ...)` itself,
            # hits the `1<<24` cap, and raises on every backend.
            for j_e in range(a[i_e]):
                n = a[j_e]
                accum = 0.0
                for k in range(n):
                    accum = accum + x[k] * x[k]
                loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1

    # The host evaluator on CPU raises `RuntimeError` directly from `prog.launch_kernel` (the `QD_ERROR_IF` path
    # surfaces as `RuntimeError` to Python); the device sizers raise `QuadrantsAssertionError` from `qd.sync()` once
    # the overflow flag is polled. The match-set covers both backends uniformly.
    with pytest.raises((QuadrantsAssertionError, RuntimeError)):
        compute(a)
        loss.grad[None] = 1.0
        for i in range(N_X):
            x.grad[i] = 0.0
        compute.grad(a)
        qd.sync()


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_offset_array_difference_loop_trip_grad_correct():
    # A reverse-mode kernel whose inner loop trip count is the difference of two adjacent reads of an offset array
    # (`starts[i + 1] - starts[i]`, the ragged-segment length pattern) must size the loop-carried adstack for the
    # widest segment, and the analytical gradient must match the closed form. The structural pre-pass wraps each of
    # the two reads in a whole-shape `MaxOverRange` fallback because the recovered loop index is opaque; the two
    # wrappers carry alpha-equal `ExternalTensorShape` ends, so before the fix `expr_sub` fused them into
    # `MaxOverRange(starts[v] - starts[v]) = 0`, collapsing the loop's push multiplier to zero. The stack was then
    # sized for the root pushes only and overflowed once a segment had more than one element (loud
    # `QuadrantsAssertionError` on SPIR-V, silently replicated per-lane gradients on a `__debug__`-disabled build).
    # The outer parallel `ndrange` forces the reverse pass to spill and recover the loop index (via a stash), so
    # both `starts` reads index through an opaque expression and take the whole-shape `MaxOverRange` fallback -
    # the shape that fuses to zero before the fix. A compile-time outer `range` would keep the index concrete and
    # not exercise the fusion.
    starts_np = np.array([0, 2, 3, 6], dtype=np.int32)  # segment lengths 2, 1, 3
    n_seg = starts_np.size - 1
    total = int(starts_np[-1])
    n_env = 2
    x_np = np.linspace(0.2, 0.9, total * n_env).astype(np.float32).reshape(total, n_env)

    starts = qd.ndarray(qd.i32, shape=(n_seg + 1,))
    starts.from_numpy(starts_np)
    x = qd.ndarray(qd.f32, shape=(total, n_env), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(n_seg, n_env), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.ndarray(), out: qd.types.ndarray(), starts: qd.types.ndarray()):
        for i, i_env in qd.ndrange(n_seg, n_env):
            seg_start = starts[i]
            seg_end = starts[i + 1]
            acc = qd.f32(0.0)
            for j in range(seg_end - seg_start):
                acc = acc + x[seg_start + j, i_env] * x[seg_start + j, i_env]
            out[i, i_env] = acc

    x.from_numpy(x_np)
    compute(x, out, starts)
    out.grad.from_numpy(np.ones((n_seg, n_env), dtype=np.float32))
    x.grad.fill(0.0)
    compute.grad(x, out, starts)

    # out[i, e] = sum_{j in segment i} x[j, e]^2, so d(out)/d(x[j, e]) = 2 * x[j, e] for the owning segment.
    grad = x.grad.to_numpy()
    for j in range(total):
        for e in range(n_env):
            assert grad[j, e] == pytest.approx(2.0 * x_np[j, e], rel=1e-5)
