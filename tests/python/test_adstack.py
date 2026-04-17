import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.needs_torch
@pytest.mark.parametrize(
    "qd_op,torch_op",
    [
        (qd.abs, "abs"),
        (qd.sin, "sin"),
        (qd.cos, "cos"),
        (qd.tan, "tan"),
        (qd.asin, "asin"),
        (qd.acos, "acos"),
        (qd.log, "log"),
        (qd.sqrt, "sqrt"),
        (qd.tanh, "tanh"),
        (qd.exp, "exp"),
        (qd.rsqrt, "rsqrt"),
    ],
)
@test_utils.test(require=qd.extension.adstack, ad_stack_experimental_enabled=True)
def test_adstack_unary_loop_carried(qd_op, torch_op):
    # Baseline regression test for reverse-mode AD through dynamic loops whose only loop-variant
    # operand feeding a non-linear unary op is read via `LocalLoad` (no subsequent `LocalStore` back
    # into the same alloca in the inner body). Under that pattern, `AdStackAllocaJudger` only
    # promotes the operand alloca to an `AdStackAllocaStmt` if the unary op is listed in
    # `NonLinearOps::unary_collections`. Without promotion, `BackupSSA` spills the operand into a
    # single plain alloca that every forward iteration overwrites, so the reversed loop reads the
    # last forward value for every backward step and produces silently wrong gradients.
    #
    # The inner body redefines `a` from scratch as `x[i] + j * 0.1` every iteration - no
    # `LocalStore(a)` after `LocalLoad(a)`, and the `cast + add` chain is outside
    # `binary_collections`, so promotion hinges solely on `unary_collections` membership. Dropping
    # any of the parametrized ops out of `unary_collections` makes this test fail. Follow-up split
    # PRs extend the parametrize list as they add new ops (tan) or fix ops that were already in
    # `unary_collections` but had buggy reverse formulas (tanh/exp recompute) or were missing
    # entirely (rsqrt).
    import torch

    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in range(3):
                a = x[i] + qd.cast(j, qd.f32) * 0.1
                acc += qd_op(a)
            y[None] += acc

    x_val = 0.5
    x[0] = x_val
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    x_t = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    acc_t = torch.zeros((), dtype=torch.float32)
    for j in range(3):
        a_t = x_t + float(j) * 0.1
        acc_t = acc_t + getattr(torch, torch_op)(a_t)
    acc_t.backward()

    assert x.grad[0] == test_utils.approx(x_t.grad.item(), rel=1e-3)
