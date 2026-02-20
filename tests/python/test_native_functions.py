import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test(exclude=[qd.amdgpu])
def test_abs():
    x = qd.field(qd.f32)

    N = 16

    qd.root.dense(qd.i, N).place(x)

    @qd.kernel
    def func():
        for i in range(N):
            x[i] = abs(-i)
            print(x[i])
            qd.static_print(x[i])

    func()

    for i in range(N):
        assert x[i] == i


@test_utils.test()
def test_int():
    x = qd.field(qd.f32)

    N = 16

    qd.root.dense(qd.i, N).place(x)

    @qd.kernel
    def func():
        for i in range(N):
            x[i] = int(x[i])
            x[i] = float(int(x[i]) // 2)

    for i in range(N):
        x[i] = i + 0.4

    func()

    for i in range(N):
        assert x[i] == i // 2


@test_utils.test()
def test_minmax():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    z = qd.field(qd.f32)
    minimum = qd.field(qd.f32)
    maximum = qd.field(qd.f32)

    N = 16

    qd.root.dense(qd.i, N).place(x, y, z, minimum, maximum)

    @qd.kernel
    def func():
        for i in range(N):
            minimum[i] = min(x[i], y[i], z[i])
            maximum[i] = max(x[i], y[i], z[i])

    for i in range(N):
        x[i] = i
        y[i] = N - i
        z[i] = i - 2 if i % 2 else i + 2

    func()

    assert np.allclose(
        minimum.to_numpy(),
        np.minimum(np.minimum(x.to_numpy(), y.to_numpy()), z.to_numpy()),
    )
    assert np.allclose(
        maximum.to_numpy(),
        np.maximum(np.maximum(x.to_numpy(), y.to_numpy()), z.to_numpy()),
    )
