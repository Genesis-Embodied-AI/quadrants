import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ftz_f32():
    a = qd.field(dtype=qd.f32, shape=2)

    @qd.kernel
    def foo():
        a[0] = 1e-45
        a[1] = 1e-10 * 1e-35

    foo()
    assert a[0] == 0
    assert a[1] == 0


@test_utils.test(require=qd.extension.data64)
def test_ftz_f64():
    a = qd.field(dtype=qd.f64, shape=2)

    @qd.kernel
    def foo():
        a[0] = 1e-323
        x = 1e-300
        y = 1e-23
        a[1] = x * y

    foo()
    assert a[0] == 0
    assert a[1] == 0
