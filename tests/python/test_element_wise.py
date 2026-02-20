import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _c_mod(a, b):
    return a - b * int(float(a) / b)


@pytest.mark.parametrize("lhs_is_mat,rhs_is_mat", [(True, True), (True, False), (False, True)])
@test_utils.test(fast_math=False, exclude=[qd.vulkan])
def test_binary_f(lhs_is_mat, rhs_is_mat):
    x = qd.Matrix.field(3, 2, qd.f32, 16)
    if lhs_is_mat:
        y = qd.Matrix.field(3, 2, qd.f32, ())
    else:
        y = qd.field(qd.f32, ())
    if rhs_is_mat:
        z = qd.Matrix.field(3, 2, qd.f32, ())
    else:
        z = qd.field(qd.f32, ())

    if lhs_is_mat:
        y.from_numpy(np.array([[0, 2], [9, 3.1], [7, 4]], np.float32))
    else:
        y[None] = 6.1
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.float32))
    else:
        z[None] = 5

    @qd.kernel
    def func():
        x[0] = y[None] + z[None]
        x[1] = y[None] - z[None]
        x[2] = y[None] * z[None]
        x[3] = y[None] / z[None]
        x[4] = y[None] // z[None]
        x[5] = y[None] % z[None]
        x[6] = y[None] ** z[None]
        x[7] = y[None] == z[None]
        x[8] = y[None] != z[None]
        x[9] = y[None] > z[None]
        x[10] = y[None] >= z[None]
        x[11] = y[None] < z[None]
        x[12] = y[None] <= z[None]
        x[13] = qd.atan2(y[None], z[None])
        x[14] = qd.min(y[None], z[None])
        x[15] = qd.max(y[None], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert test_utils.allclose(x[0], y + z)
    assert test_utils.allclose(x[1], y - z)
    assert test_utils.allclose(x[2], y * z)
    assert test_utils.allclose(x[3], y / z)
    assert test_utils.allclose(x[4], y // z)
    assert test_utils.allclose(x[5], y % z)
    assert test_utils.allclose(x[6], y**z)
    assert test_utils.allclose(x[7].astype(bool), y == z)
    assert test_utils.allclose(x[8].astype(bool), y != z)
    assert test_utils.allclose(x[9].astype(bool), y > z)
    assert test_utils.allclose(x[10].astype(bool), y >= z)
    assert test_utils.allclose(x[11].astype(bool), y < z)
    assert test_utils.allclose(x[12].astype(bool), y <= z)
    assert test_utils.allclose(x[13], np.arctan2(y, z))
    assert test_utils.allclose(x[14], np.minimum(y, z))
    assert test_utils.allclose(x[15], np.maximum(y, z))


@pytest.mark.parametrize("is_mat", [(True, True), (True, False), (False, True)])
@test_utils.test()
def test_binary_i(is_mat):
    lhs_is_mat, rhs_is_mat = is_mat

    x = qd.Matrix.field(3, 2, qd.i32, 20)
    if lhs_is_mat:
        y = qd.Matrix.field(3, 2, qd.i32, ())
    else:
        y = qd.field(qd.i32, ())
    if rhs_is_mat:
        z = qd.Matrix.field(3, 2, qd.i32, ())
    else:
        z = qd.field(qd.i32, ())

    if lhs_is_mat:
        y.from_numpy(np.array([[0, 2], [9, 3], [7, 4]], np.int32))
    else:
        y[None] = 6
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.int32))
    else:
        z[None] = 5

    @qd.kernel
    def func():
        x[0] = y[None] + z[None]
        x[1] = y[None] - z[None]
        x[2] = y[None] * z[None]
        x[3] = y[None] // z[None]
        x[4] = qd.raw_div(y[None], z[None])
        x[5] = y[None] % z[None]
        x[6] = qd.raw_mod(y[None], z[None])
        x[7] = y[None] ** z[None]
        x[8] = y[None] == z[None]
        x[9] = y[None] != z[None]
        x[10] = y[None] > z[None]
        x[11] = y[None] >= z[None]
        x[12] = y[None] < z[None]
        x[13] = y[None] <= z[None]
        x[14] = y[None] & z[None]
        x[15] = y[None] ^ z[None]
        x[16] = y[None] | z[None]
        x[17] = qd.min(y[None], z[None])
        x[18] = qd.max(y[None], z[None])
        x[19] = y[None] << z[None]

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert test_utils.allclose(x[0], y + z)
    assert test_utils.allclose(x[1], y - z)
    assert test_utils.allclose(x[2], y * z)
    assert test_utils.allclose(x[3], y // z)
    assert test_utils.allclose(x[4], y // z)
    assert test_utils.allclose(x[5], y % z)
    assert test_utils.allclose(x[6], y % z)
    assert test_utils.allclose(x[7], y**z, rel=1e-5)
    assert test_utils.allclose(x[8].astype(bool), y == z)
    assert test_utils.allclose(x[9].astype(bool), y != z)
    assert test_utils.allclose(x[10].astype(bool), y > z)
    assert test_utils.allclose(x[11].astype(bool), y >= z)
    assert test_utils.allclose(x[12].astype(bool), y < z)
    assert test_utils.allclose(x[13].astype(bool), y <= z)
    assert test_utils.allclose(x[14], y & z)
    assert test_utils.allclose(x[15], y ^ z)
    assert test_utils.allclose(x[16], y | z)
    assert test_utils.allclose(x[17], np.minimum(y, z))
    assert test_utils.allclose(x[18], np.maximum(y, z))
    assert test_utils.allclose(x[19], y << z)


@pytest.mark.parametrize("rhs_is_mat", [True, False])
@test_utils.test(fast_math=False)
def test_writeback_binary_f(rhs_is_mat):
    x = qd.Matrix.field(3, 2, qd.f32, 9)
    y = qd.Matrix.field(3, 2, qd.f32, ())
    if rhs_is_mat:
        z = qd.Matrix.field(3, 2, qd.f32, ())
    else:
        z = qd.field(qd.f32, ())

    y.from_numpy(np.array([[0, 2], [9, 3.1], [7, 4]], np.float32))
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.float32))
    else:
        z[None] = 5

    @qd.kernel
    def func():
        for i in x:
            x[i] = y[None]
        if qd.static(rhs_is_mat):
            x[0] = z[None]
        else:
            x[0].fill(z[None])
        x[1] += z[None]
        x[2] -= z[None]
        x[3] *= z[None]
        x[4] /= z[None]
        x[5] //= z[None]
        x[6] %= z[None]
        qd.atomic_min(x[7], z[None])
        qd.atomic_max(x[8], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert test_utils.allclose(x[1], y + z)
    assert test_utils.allclose(x[2], y - z)
    assert test_utils.allclose(x[3], y * z)
    assert test_utils.allclose(x[4], y / z)
    assert test_utils.allclose(x[5], y // z)
    assert test_utils.allclose(x[6], y % z)
    assert test_utils.allclose(x[7], np.minimum(y, z))
    assert test_utils.allclose(x[8], np.maximum(y, z))


@pytest.mark.parametrize("rhs_is_mat", [(True, True), (True, False)])
@test_utils.test()
def test_writeback_binary_i(rhs_is_mat):
    x = qd.Matrix.field(3, 2, qd.i32, 12)
    y = qd.Matrix.field(3, 2, qd.i32, ())
    if rhs_is_mat:
        z = qd.Matrix.field(3, 2, qd.i32, ())
    else:
        z = qd.field(qd.i32, ())

    y.from_numpy(np.array([[0, 2], [9, 3], [7, 4]], np.int32))
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.int32))
    else:
        z[None] = 5

    @qd.kernel
    def func():
        for i in x:
            x[i] = y[None]
        x[0] = z[None]
        x[1] += z[None]
        x[2] -= z[None]
        x[3] *= z[None]
        x[4] //= z[None]
        x[5] %= z[None]
        x[6] &= z[None]
        x[7] |= z[None]
        x[8] ^= z[None]
        qd.atomic_min(x[10], z[None])
        qd.atomic_max(x[11], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert test_utils.allclose(x[1], y + z)
    assert test_utils.allclose(x[2], y - z)
    assert test_utils.allclose(x[3], y * z)
    assert test_utils.allclose(x[4], y // z)
    assert test_utils.allclose(x[5], y % z)
    assert test_utils.allclose(x[6], y & z)
    assert test_utils.allclose(x[7], y | z)
    assert test_utils.allclose(x[8], y ^ z)
    assert test_utils.allclose(x[10], np.minimum(y, z))
    assert test_utils.allclose(x[11], np.maximum(y, z))


@test_utils.test()
def test_unary():
    xi = qd.Matrix.field(3, 2, qd.i32, 4)
    yi = qd.Matrix.field(3, 2, qd.i32, ())
    xf = qd.Matrix.field(3, 2, qd.f32, 15)
    yf = qd.Matrix.field(3, 2, qd.f32, ())

    yi.from_numpy(np.array([[3, 2], [9, 0], [7, 4]], np.int32))
    yf.from_numpy(np.array([[0.3, 0.2], [0.9, 0.1], [0.7, 0.4]], np.float32))

    @qd.kernel
    def func():
        xi[0] = -yi[None]
        xi[1] = ~yi[None]
        xi[2] = not yi[None]
        xi[3] = abs(yi[None])
        xf[0] = -yf[None]
        xf[1] = abs(yf[None])
        xf[2] = qd.sqrt(yf[None])
        xf[3] = qd.sin(yf[None])
        xf[4] = qd.cos(yf[None])
        xf[5] = qd.tan(yf[None])
        xf[6] = qd.asin(yf[None])
        xf[7] = qd.acos(yf[None])
        xf[8] = qd.tanh(yf[None])
        xf[9] = qd.floor(yf[None])
        xf[10] = qd.ceil(yf[None])
        xf[11] = qd.exp(yf[None])
        xf[12] = qd.log(yf[None])
        xf[13] = qd.rsqrt(yf[None])
        xf[14] = qd.round(yf[None])

    func()
    xi = xi.to_numpy()
    yi = yi.to_numpy()
    xf = xf.to_numpy()
    yf = yf.to_numpy()
    assert test_utils.allclose(xi[0], -yi)
    assert test_utils.allclose(xi[1], ~yi)
    assert test_utils.allclose(xi[3], np.abs(yi))
    assert test_utils.allclose(xf[0], -yf)
    assert test_utils.allclose(xf[1], np.abs(yf))
    assert test_utils.allclose(xf[2], np.sqrt(yf), rel=1e-5)
    assert test_utils.allclose(xf[3], np.sin(yf), rel=1e-4)
    assert test_utils.allclose(xf[4], np.cos(yf), rel=1e-4)
    assert test_utils.allclose(xf[5], np.tan(yf), rel=1e-4)
    # vulkan need 1e-3
    assert test_utils.allclose(xf[6], np.arcsin(yf), rel=1e-3)
    assert test_utils.allclose(xf[7], np.arccos(yf), rel=1e-3)
    assert test_utils.allclose(xf[8], np.tanh(yf), rel=1e-4)
    assert test_utils.allclose(xf[9], np.floor(yf), rel=1e-5)
    assert test_utils.allclose(xf[10], np.ceil(yf), rel=1e-5)
    assert test_utils.allclose(xf[11], np.exp(yf), rel=1e-5)
    assert test_utils.allclose(xf[12], np.log(yf), rel=1e-5)
    assert test_utils.allclose(xf[13], 1 / np.sqrt(yf), rel=1e-5)
    assert test_utils.allclose(xf[14], np.round(yf), rel=1e-5)


@pytest.mark.parametrize(
    "is_mat",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
    ],
)
@test_utils.test()
def test_ternary_i(is_mat):
    cond_is_mat, lhs_is_mat, rhs_is_mat = is_mat
    x = qd.Matrix.field(3, 2, qd.i32, 1)
    if cond_is_mat:
        y = qd.Matrix.field(3, 2, qd.i32, ())
    else:
        y = qd.field(qd.i32, ())
    if lhs_is_mat:
        z = qd.Matrix.field(3, 2, qd.i32, ())
    else:
        z = qd.field(qd.i32, ())
    if rhs_is_mat:
        w = qd.Matrix.field(3, 2, qd.i32, ())
    else:
        w = qd.field(qd.i32, ())

    if cond_is_mat:
        y.from_numpy(np.array([[0, 2], [9, 0], [7, 4]], np.int32))
    else:
        y[None] = 0
    if lhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.int32))
    else:
        z[None] = 5
    if rhs_is_mat:
        w.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.int32))
    else:
        w[None] = 4

    @qd.kernel
    def func():
        x[0] = qd.select(y[None], z[None], w[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    w = w.to_numpy()
    assert test_utils.allclose(x[0], np.int32(np.bool_(y)) * z + np.int32(1 - np.bool_(y)) * w)
