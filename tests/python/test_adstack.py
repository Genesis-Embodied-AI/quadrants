import math
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

    assert loss.dual[None] == test_utils.approx(expected, rel=1e-4)


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
    arch=qd.metal,
    require=qd.extension.adstack,
    ad_stack_size=65536,
)
def test_adstack_shader_compile_failure_raises():
    # Asks the compiler to build a Metal shader whose per-thread private-memory footprint is too large for Apple's
    # shader translator to accept. The test asserts the kernel fails to build with a regular Python `RuntimeError`
    # saying the pipeline couldn't be created, instead of silently launching a null pipeline (which would either
    # crash the process or corrupt subsequent kernels).
    #
    # Internal detail: the oversized `ad_stack_size` combined with several independent loop-carried variables
    # forces enough Function-scope private memory per thread that Apple's MSL translator rejects the pipeline
    # with `XPC_ERROR_CONNECTION_INTERRUPTED` at create time. A single loop-carried variable is not enough - the
    # Metal compiler is willing to spill a single oversized private array to device memory on its own and the
    # pipeline still builds; four independent adstacks at the same capacity defeat the spill heuristic. The test
    # is restricted to Metal because Vulkan drivers vary widely on what per-thread Function-scope footprint they
    # will accept, so calibrating a single threshold that every CI Vulkan driver rejects is brittle.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            a = x[i]
            b = x[i]
            c = x[i]
            d = x[i]
            for _ in range(10):
                a = qd.sin(a)
                b = qd.sin(b)
                c = qd.sin(c)
                d = qd.sin(d)
            y[None] += a + b + c + d

    x[0] = 0.1
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    with pytest.raises(RuntimeError, match=r"[Ff]ailed to create pipeline"):
        compute.grad()


def _overflowing_compute(n_elements=1, n_iter=64):
    # Shared kernel for the overflow tests. Builds `compute`, loads inputs, seeds the output gradient, and returns
    # `(compute, x, y)` so each test can drive the grad launch and read back assertions itself. `n_iter=64` + 2
    # adstack preamble pushes = 66 pushes, comfortably above `default_ad_stack_size=32`; `n_elements` controls how
    # many threads run the overflowing loop in parallel.
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
    assert x.grad[0] == test_utils.approx(expected, rel=1e-3)


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


@test_utils.test(require=qd.extension.adstack)
def test_adstack_near_capacity():
    # Runs a backward pass with a for-loop sized to just barely fit inside the adstack (one iteration away from
    # overflow) and asserts the gradient comes out correctly. Companion to `test_adstack_overflow_raises` - this is
    # the "and it still works at the boundary" side.
    #
    # Internal detail: the transform emits two adstack pushes before the loop body (one for the initial adjoint
    # slot, one for the primal's starting value), so a loop of K iterations produces K+2 pushes. With
    # `default_ad_stack_size=32`, that bounds K at 30.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(30):
                y[None] += qd.sin(v)
                v = v + 1.0

    x[0] = 0.1
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    expected = sum(math.cos(0.1 + k) for k in range(30))
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


def test_adstack_codegen_budget_guard_runs_in_child_process(tmp_path):
    # Per-task codegen guard: the sum of `AdStackAllocaStmt::size_in_bytes()` in a single LLVM task must not cross
    # the ~256 KB CPU worker-thread stack budget. Beyond that the frame silently clobbers adjacent stack memory and
    # the reverse pass returns zero / garbage gradients. The guard runs inside the LLVM compilation worker thread
    # pool; the underlying `QD_ERROR_IF` throws across a thread boundary that does not propagate the exception
    # back to Python, so it surfaces as a loud `std::terminate` / SIGABRT rather than a catchable Python
    # exception. The test runs the overflowing kernel in a child process and asserts the child aborts with a
    # non-zero exit code and the guard message reaches stderr; that is enough to prove the guard fires and does
    # not let silent stack-frame clobbering through.
    if not is_extension_supported(qd.cpu, qd.extension.adstack):
        pytest.skip("adstack extension not available on cpu")
    if not is_extension_supported(qd.cpu, qd.extension.data64):
        pytest.skip("f64 extension not available on cpu")

    _run_budget_guard_child(tmp_path)


def _run_budget_guard_child(tmp_path):
    child_script = textwrap.dedent(
        """
        import quadrants as qd

        qd.init(arch=qd.cpu, ad_stack_experimental_enabled=True, ad_stack_size=4096, default_fp=qd.f64)

        n = 4
        x = qd.field(qd.f64, shape=n, needs_grad=True)
        y = qd.field(qd.f64, shape=(), needs_grad=True)
        n_iter = qd.field(qd.i32, shape=())

        @qd.kernel
        def compute():
            for i in x:
                v1 = x[i]
                v2 = x[i]
                v3 = x[i]
                v4 = x[i]
                v5 = x[i]
                for _ in range(n_iter[None]):
                    v1 = qd.sin(v1)
                    v2 = qd.sin(v2)
                    v3 = qd.sin(v3)
                    v4 = qd.sin(v4)
                    v5 = qd.sin(v5)
                y[None] += v1 + v2 + v3 + v4 + v5

        for i in range(n):
            x[i] = 0.1 + 0.1 * i
        n_iter[None] = 3
        y[None] = 0.0
        compute()
        y.grad[None] = 1.0
        for i in range(n):
            x.grad[i] = 0.0
        compute.grad()
        """
    )
    script_path = tmp_path / "budget_guard_child.py"
    script_path.write_text(child_script)
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, check=False)
    assert (
        result.returncode != 0
    ), "child exited with returncode 0 but the budget guard was expected to terminate the process"
    combined = (result.stdout + result.stderr).decode()
    assert "autodiff-stack budget exceeded" in combined, (
        f"expected guard message in child output; got:\nstdout:\n{result.stdout.decode()}\n"
        f"stderr:\n{result.stderr.decode()}"
    )


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
        assert x.grad[i] == test_utils.approx(expected, rel=1e-4)
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
        assert x.grad[k] == test_utils.approx(expected[k], rel=1e-4)


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
    import os

    env = {**os.environ, **env_no_cache}
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, check=True, env=env)
    stderr = result.stderr.decode()
    assert "Loading variable" not in stderr, (
        "reverse-mode AD emitted 'Loading variable N before anything is stored to it' warnings for a Vector "
        "subscript-assign + self-referencing in-place op pattern; stderr was:\n" + stderr
    )
