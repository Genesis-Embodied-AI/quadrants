import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.data64)
def test_global_buffer_misalignment():
    @qd.kernel
    def test(x: qd.f32):
        a = x
        b = qd.cast(0.12, qd.f64)
        for i in range(8):
            b += a

    for i in range(8):
        test(0.1)
