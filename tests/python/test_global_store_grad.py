"""
import quadrants as qd

qd.lang.impl.current_cfg().print_ir = True


def test_global_store_branching():
    # qd.reset()

    N = 16
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    qd.root.dense(qd.i, N).place(x)
    qd.root.dense(qd.i, N).place(y)
    qd.root.lazy_grad()

    @qd.kernel
    def oldeven():
        for i in range(N):
            if i % 2 == 0:
                x[i] = y[i]

    for i in range(N):
        x.grad[i] = 1

    oldeven()
    oldeven.grad()

    for i in range(N):
        assert y.grad[i] == (i % 2 == 0)
"""
