import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_matrix_non_constant_index():
    m = qd.Matrix.field(2, 2, qd.f32, 5, needs_grad=True)
    n = qd.Matrix.field(2, 2, qd.f32, 5, needs_grad=True)
    loss = qd.field(qd.f32, (), needs_grad=True)

    n.fill(0)

    @qd.kernel
    def func1():
        for i in range(5):
            for j, k in qd.ndrange(2, 2):
                m[i][j, k] = (j + 1) * (k + 1) * n[i][j, k]
                loss[None] += m[i][j, k]

    loss.grad[None] = 1.0
    func1.grad()

    for i in range(5):
        for j in range(2):
            for k in range(2):
                assert n.grad[i][j, k] == (j + 1) * (k + 1)
