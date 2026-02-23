import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_arg_load():
    x = qd.field(qd.i32)
    y = qd.field(qd.f32)

    qd.root.place(x, y)

    @qd.kernel
    def set_i32(v: qd.i32):
        x[None] = v

    @qd.kernel
    def set_f32(v: qd.f32):
        y[None] = v

    set_i32(123)
    assert x[None] == 123

    set_i32(456)
    assert x[None] == 456

    set_f32(0.125)
    assert y[None] == 0.125

    set_f32(1.5)
    assert y[None] == 1.5


@test_utils.test(require=qd.extension.data64)
def test_arg_load_f64():
    x = qd.field(qd.i32)
    y = qd.field(qd.f32)

    qd.root.place(x, y)

    @qd.kernel
    def set_f64(v: qd.f64):
        y[None] = qd.cast(v, qd.f32)

    @qd.kernel
    def set_i64(v: qd.i64):
        y[None] = v

    set_i64(789)
    assert y[None] == 789

    set_f64(2.5)
    assert y[None] == 2.5


@test_utils.test()
def test_ndarray():
    N = 128
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, N).place(x)

    @qd.kernel
    def set_f32(v: qd.types.ndarray()):
        for i in range(N):
            x[i] = v[i] + i

    import numpy as np

    v = np.ones((N,), dtype=np.float32) * 10
    set_f32(v)
    for i in range(N):
        assert x[i] == 10 + i
