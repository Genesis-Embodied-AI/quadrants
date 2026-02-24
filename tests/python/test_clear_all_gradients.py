import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@test_utils.test(exclude=[qd.vulkan])
def test_clear_all_gradients():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    z = qd.field(qd.f32)
    w = qd.field(qd.f32)

    n = 128

    qd.root.place(x)
    qd.root.dense(qd.i, n).place(y)
    qd.root.dense(qd.i, n).dense(qd.j, n).place(z, w)
    qd.root.lazy_grad()

    x.grad[None] = 3
    for i in range(n):
        y.grad[i] = 3
        for j in range(n):
            z.grad[i, j] = 5
            w.grad[i, j] = 6

    qd.ad.clear_all_gradients()
    assert impl.get_runtime().get_num_compiled_functions() == 3

    assert x.grad[None] == 0
    for i in range(n):
        assert y.grad[i] == 0
        for j in range(n):
            assert z.grad[i, j] == 0
            assert w.grad[i, j] == 0

    qd.ad.clear_all_gradients()
    # No more kernel compilation
    assert impl.get_runtime().get_num_compiled_functions() == 3
