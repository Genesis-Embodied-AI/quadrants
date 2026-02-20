import quadrants as qd
from quadrants.types.enums import SNodeGradType

from tests import test_utils


@test_utils.test()
def test_snode_grad_type():
    x = qd.field(float, shape=(), needs_grad=True, needs_dual=True)
    assert x.snode.ptr.get_snode_grad_type() == SNodeGradType.PRIMAL
    assert x.grad.snode.ptr.get_snode_grad_type() == SNodeGradType.ADJOINT
    assert x.dual.snode.ptr.get_snode_grad_type() == SNodeGradType.DUAL


@test_utils.test()
def test_snode_grad_type_lazy():
    x = qd.field(float, shape=())
    qd.root.lazy_grad()
    qd.root.lazy_dual()
    assert x.snode.ptr.get_snode_grad_type() == SNodeGradType.PRIMAL
    assert x.grad.snode.ptr.get_snode_grad_type() == SNodeGradType.ADJOINT
    assert x.dual.snode.ptr.get_snode_grad_type() == SNodeGradType.DUAL


@test_utils.test()
def test_snode_clear_gradient():
    x = qd.field(float, shape=(), needs_grad=True, needs_dual=True)
    y = qd.field(float, shape=(), needs_grad=True, needs_dual=True)

    x[None] = 1.0

    @qd.kernel
    def compute():
        y[None] += x[None] ** 2

    with qd.ad.Tape(loss=y):
        compute()

    with qd.ad.FwdMode(loss=y, param=x):
        compute()

    assert x.grad[None] == 2.0

    with qd.ad.Tape(loss=y):
        compute()

    assert y.dual[None] == 2.0


# TODO: Add test for `adjoint_checkbit` after #5801 merged.
