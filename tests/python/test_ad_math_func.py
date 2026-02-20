import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.adstack)
def test_polar_decompose_2D():
    # `polar_decompose3d` in current Quadrants version (v1.1) does not support autodiff,
    # becasue it mixed usage of for-loops and statements without looping.
    dim = 2
    F_1 = qd.Matrix.field(dim, dim, dtype=qd.f32, shape=(), needs_grad=True)
    F = qd.Matrix.field(dim, dim, dtype=qd.f32, shape=(), needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def polar_decompose_2D():
        r, s = qd.polar_decompose(F[None])
        F_1[None] += r

    with qd.ad.Tape(loss=loss):
        polar_decompose_2D()
