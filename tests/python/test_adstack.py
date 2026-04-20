import math

import pytest

import quadrants as qd

from tests import test_utils

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
    # Pins MakeDual (forward mode) for the ops whose reverse formulas were just patched (tan/tanh/exp) plus the
    # two audit-adjacent operand-based ops (log/sqrt). Forward mode is argued safe-by-construction because it
    # runs in primal order and does not rely on BackupSSA spilling, but that invariant is not itself tested by
    # the reverse-mode regression above; this test pins it. Note the asymmetry with MakeAdjoint::tanh / exp:
    # this PR's reverse-mode recompute of `tanh(stmt->operand)` / `exp(stmt->operand)` has a detailed comment
    # explaining why reusing the forward `stmt` would be wrong under BackupSSA; the forward-mode formulas
    # reuse the same `stmt` value without a safety comment because MakeDual runs in primal order, so `stmt`
    # is the current-iteration value. This test pins that forward-order invariant.
    ["tan", "tanh", "exp", "log", "sqrt"],
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


def _run_nan_propagation(qd_dtype, op_name, x_val):
    qd_op = getattr(qd, op_name)

    x = qd.field(qd_dtype, shape=(), needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

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
    _run_nan_propagation(qd.f32, op_name, x_val)


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
    _run_nan_propagation(qd.f64, op_name, x_val)
