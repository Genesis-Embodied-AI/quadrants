import math
import pathlib
import re

import pytest

import quadrants as qd

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
    ("sin", 0.3, -0.4),
    ("cos", 0.3, -0.4),
    ("abs", 0.3, -0.4),
    ("tanh", 0.3, -0.4),
    ("exp", 0.3, -0.4),
    # Ops restricted to positive/subunit operands use a smaller step and zero offset to
    # stay inside their domain across every `x_val` and `n_iter` combination.
    # `tan` joins this positive-domain group because its singularity at pi/2 ~= 1.57 lies outside
    # the positive-path operand's reach for every `x_val` and `n_iter` combination.
    ("tan", 0.05, 0.0),
    ("log", 0.05, 0.0),
    ("sqrt", 0.05, 0.0),
    ("rsqrt", 0.05, 0.0),
    ("asin", 0.05, 0.0),
    ("acos", 0.05, 0.0),
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
@pytest.mark.parametrize("op_name,step,offset", _UNARY_OPS_PARAMS)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_unary_loop_carried(op_name, step, offset, x_val, n_iter):
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
    _run_unary_loop_carried(qd.f32, op_name, step, offset, x_val, n_iter, rel_tol=1e-4)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("x_val", [0.001, 0.15, 0.26, 0.399])
@pytest.mark.parametrize("op_name,step,offset", _UNARY_OPS_PARAMS)
@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_unary_loop_carried_f64(op_name, step, offset, x_val, n_iter):
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


def _run_basic_gradient(qd_dtype, n_iter, rel_tol, approx=test_utils.approx):
    # Builds the kernel, runs forward + backward, and asserts a correct gradient. `approx` defaults to
    # `test_utils.approx` which is correct for f32 (its backend-specific floor kicks in); f64 callers must pass
    # `pytest.approx` to honor a tight `rel_tol=1e-14` that `test_utils.approx` would otherwise floor to 1e-6.
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
    for i in range(n):
        assert x.grad[i] == approx(expected, rel=rel_tol)


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
    # f64 counterpart of `test_adstack_basic_gradient`. Uses `pytest.approx` so the tight `rel_tol=1e-14` is
    # honored; `test_utils.approx` would floor it to `get_rel_eps()` (typically 1e-6) and silently pass an
    # f32-precision regression.
    _run_basic_gradient(qd.f64, n_iter=n_iter, rel_tol=1e-14, approx=pytest.approx)


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


def _run_sum_linear(qd_dtype, use_static_loop, use_varying_coeff, n_iter, rel_tol, approx=test_utils.approx):
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
    for i in range(n):
        assert x.grad[i] == approx(expected, rel=rel_tol)


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
    # f64 counterpart uses `pytest.approx` so the tight rel_tol is not floored by `test_utils.approx`.
    _run_sum_linear(qd.f64, use_static_loop, use_varying_coeff, n_iter, rel_tol=1e-14, approx=pytest.approx)
