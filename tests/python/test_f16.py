import math

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.util import has_pytorch

from tests import test_utils

archs_support_f16 = [qd.cpu, qd.cuda, qd.vulkan]


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_snode_read_write():
    dtype = qd.f16
    x = qd.field(dtype, shape=())
    x[None] = 0.3
    print(x[None])
    assert x[None] == test_utils.approx(0.3, rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_float16():
    dtype = qd.float16
    x = qd.field(dtype, shape=())
    x[None] = 0.3
    print(x[None])
    assert x[None] == test_utils.approx(0.3, rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_to_numpy():
    n = 16
    x = qd.field(qd.f16, shape=n)

    @qd.kernel
    def init():
        for i in x:
            x[i] = i * 2

    init()
    y = x.to_numpy()
    for i in range(n):
        assert y[i] == 2 * i


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_from_numpy():
    n = 16
    y = qd.field(dtype=qd.f16, shape=n)
    x = np.arange(n, dtype=np.half)
    y.from_numpy(x)

    @qd.kernel
    def init():
        for i in y:
            y[i] = 3 * i

    init()
    z = y.to_numpy()
    for i in range(n):
        assert z[i] == i * 3


@pytest.mark.sm70
@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_f16)
def test_to_torch():
    n = 16
    x = qd.field(qd.f16, shape=n)

    @qd.kernel
    def init():
        for i in x:
            x[i] = i * 2

    init()
    y = x.to_torch()
    print(y)
    for i in range(n):
        assert y[i] == 2 * i


@pytest.mark.sm70
@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_f16)
def test_from_torch():
    import torch

    n = 16
    y = qd.field(dtype=qd.f16, shape=n)
    # torch doesn't have rand implementation for float16 so we need to create float first and then convert
    x = torch.arange(0, n).to(torch.float16)
    y.from_torch(x)

    @qd.kernel
    def init():
        for i in y:
            y[i] = 3 * i

    init()
    z = y.to_torch()
    for i in range(n):
        assert z[i] == i * 3


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_binary_op():
    dtype = qd.f16
    x = qd.field(dtype, shape=())
    y = qd.field(dtype, shape=())
    z = qd.field(dtype, shape=())

    @qd.kernel
    def add():
        x[None] = y[None] + z[None]
        x[None] = x[None] * z[None]

    y[None] = 0.2
    z[None] = 0.72
    add()
    u = x.to_numpy()
    assert u[None] == test_utils.approx(0.6624, rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_rand_promote():
    dtype = qd.f16
    x = qd.field(dtype, shape=(4, 4))

    @qd.kernel
    def init():
        for i, j in x:
            x[i, j] = qd.random(dtype=dtype)
            print(x[i, j])

    init()


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_unary_op():
    dtype = qd.f16
    x = qd.field(dtype, shape=())
    y = qd.field(dtype, shape=())

    @qd.kernel
    def foo():
        x[None] = -y[None]
        x[None] = qd.floor(x[None])
        y[None] = qd.ceil(y[None])

    y[None] = -1.4
    foo()
    assert x[None] == test_utils.approx(1, rel=1e-3)
    assert y[None] == test_utils.approx(-1, rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_extra_unary_promote():
    dtype = qd.f16
    x = qd.field(dtype, shape=())
    y = qd.field(dtype, shape=())

    @qd.kernel
    def foo():
        x[None] = abs(y[None])

    y[None] = -0.3
    foo()
    assert x[None] == test_utils.approx(0.3, rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16, exclude=qd.vulkan)
def test_binary_extra_promote():
    x = qd.field(dtype=qd.f16, shape=())
    y = qd.field(dtype=qd.f16, shape=())
    z = qd.field(dtype=qd.f16, shape=())

    @qd.kernel
    def foo():
        y[None] = x[None] ** 2
        z[None] = qd.atan2(y[None], 0.3)

    x[None] = 0.1
    foo()
    assert z[None] == test_utils.approx(math.atan2(0.1**2, 0.3), rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_arg_f16():
    dtype = qd.f16
    x = qd.field(dtype, shape=())
    y = qd.field(dtype, shape=())

    @qd.kernel
    def foo(a: qd.f16, b: qd.f32, c: qd.f16):
        x[None] = y[None] + a + b + c

    y[None] = -0.3
    foo(0.3, 0.4, 0.5)
    assert x[None] == test_utils.approx(0.9, rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_fractal_f16():
    n = 320
    pixels = qd.field(dtype=qd.f16, shape=(n * 2, n))

    @qd.func
    def complex_sqr(z):
        return qd.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2], dt=qd.f16)

    @qd.kernel
    def paint(t: float):
        for i, j in pixels:  # Parallelized over all pixels
            c = qd.Vector([-0.8, qd.cos(t) * 0.2], dt=qd.f16)
            z = qd.Vector([i / n - 1, j / n - 0.5], dt=qd.f16) * 2
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i, j] = 1 - iterations * 0.02

    paint(0.03)


# TODO(): Vulkan support
@pytest.mark.sm70
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_atomic_add_f16():
    f = qd.field(dtype=qd.f16, shape=(2))

    @qd.kernel
    def foo():
        # Parallel sum
        for i in range(1000):
            f[0] += 1.12

        # Serial sum
        for _ in range(1):
            for i in range(1000):
                f[1] = f[1] + 1.12

    foo()
    assert f[0] == test_utils.approx(f[1], rel=1e-3)


# TODO(): Vulkan support
@pytest.mark.sm70
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_atomic_max_f16():
    f = qd.field(dtype=qd.f16, shape=(2))

    @qd.kernel
    def foo():
        # Parallel max
        for i in range(1000):
            qd.atomic_max(f[0], 1.12 * i)

        # Serial max
        for _ in range(1):
            for i in range(1000):
                f[1] = qd.max(1.12 * i, f[1])

    foo()
    assert f[0] == test_utils.approx(f[1], rel=1e-3)


# TODO(): Vulkan support
@pytest.mark.sm70
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_atomic_min_f16():
    f = qd.field(dtype=qd.f16, shape=(2))

    @qd.kernel
    def foo():
        # Parallel min
        for i in range(1000):
            qd.atomic_min(f[0], -3.13 * i)

        # Serial min
        for _ in range(1):
            for i in range(1000):
                f[1] = qd.min(-3.13 * i, f[1])

    foo()
    assert f[0] == test_utils.approx(f[1], rel=1e-3)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16)
def test_cast_f32_to_f16():
    @qd.kernel
    def func() -> qd.f16:
        a = qd.cast(23.0, qd.f32)
        b = qd.cast(4.0, qd.f32)
        return qd.cast(a * b, qd.f16)

    assert func() == pytest.approx(23.0 * 4.0, 1e-4)


@pytest.mark.sm70
@test_utils.test(arch=archs_support_f16, require=qd.extension.data64)
def test_cast_f64_to_f16():
    @qd.kernel
    def func() -> qd.f16:
        a = qd.cast(23.0, qd.f64)
        b = qd.cast(4.0, qd.f64)
        return qd.cast(a * b, qd.f16)

    assert func() == pytest.approx(23.0 * 4.0, 1e-4)


@pytest.mark.sm70
@test_utils.test(arch=[qd.cuda], half2_vectorization=True)
def test_half2_vectorize():
    half2 = qd.types.vector(n=2, dtype=qd.f16)

    table = half2.field(shape=(40), needs_grad=True)
    embeddings = half2.field(shape=(40, 16), needs_grad=True)
    B = 1

    @qd.kernel
    def test(B: qd.i32):
        for i, level in qd.ndrange(B, 16):
            w = 4.0
            local_feature = qd.Vector([qd.f16(0.0), qd.f16(0.0)])
            for index in qd.static(range(64)):
                local_feature += w * table[index]

            embeddings[i, level] = local_feature

    test(B)

    for i in range(10):
        test.grad(B)

    qd.sync()

    for i in range(40):
        for j in range(16):
            embeddings.grad[i, j] = half2(1.0)

    for i in range(1000):
        test.grad(B)
    qd.sync()

    assert (table.grad.to_numpy() == 64).all()
