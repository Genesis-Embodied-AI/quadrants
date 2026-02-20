import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_offload_order():
    n = 128
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=n, needs_grad=True)
    z = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def forward():
        for i in x:
            y[i] = x[i]

        # for i in x:
        #     z[None] += y[i]

    with qd.ad.Tape(z):
        forward()

    # for i in range(n):
    #     assert x.grad[i] == 1
