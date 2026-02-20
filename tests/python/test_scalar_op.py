import operator as ops

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

binary_func_table = [
    (ops.add,) * 2,
    (ops.sub,) * 2,
    (ops.mul,) * 2,
    (ops.truediv,) * 2,
    (ops.floordiv,) * 2,
    (ops.mod,) * 2,
    (ops.pow,) * 2,
    (ops.and_,) * 2,
    (ops.or_,) * 2,
    (ops.xor,) * 2,
    (ops.eq,) * 2,
    (ops.ne,) * 2,
    (ops.lt,) * 2,
    (ops.le,) * 2,
    (ops.gt,) * 2,
    (ops.ge,) * 2,
    (qd.max, np.maximum),
    (qd.min, np.minimum),
    (qd.atan2, np.arctan2),
]

unary_func_table = [
    (ops.neg,) * 2,
    (ops.invert,) * 2,
    (qd.lang.ops.logical_not, np.logical_not),
    (qd.lang.ops.abs, np.abs),
    (qd.exp, np.exp),
    (qd.log, np.log),
    (qd.sin, np.sin),
    (qd.cos, np.cos),
    (qd.tan, np.tan),
    (qd.asin, np.arcsin),
    (qd.acos, np.arccos),
    (qd.tanh, np.tanh),
    (qd.round, np.round),
    (qd.floor, np.floor),
    (qd.ceil, np.ceil),
]


@pytest.mark.parametrize("ti_func,np_func", binary_func_table)
def test_python_scope_vector_binary(ti_func, np_func):
    qd.init()
    x = qd.Vector([2, 3])
    y = qd.Vector([5, 4])

    result = ti_func(x, y).to_numpy()
    if ti_func in [ops.eq, ops.ne, ops.lt, ops.le, ops.gt, ops.ge]:
        result = result.astype(bool)
    expected = np_func(x.to_numpy(), y.to_numpy())
    assert test_utils.allclose(result, expected)


@pytest.mark.parametrize("ti_func,np_func", unary_func_table)
def test_python_scope_vector_unary(ti_func, np_func):
    qd.init()
    x = qd.Vector([2, 3] if ti_func in [ops.invert, qd.lang.ops.logical_not] else [0.2, 0.3])

    result = ti_func(x).to_numpy()
    if ti_func in [qd.lang.ops.logical_not]:
        result = result.astype(bool)
    expected = np_func(x.to_numpy())
    assert test_utils.allclose(result, expected)


def test_python_scope_matmul():
    qd.init()
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    x = qd.Matrix(a)
    y = qd.Matrix(b)

    result = (x @ y).to_numpy()
    expected = a @ b
    assert test_utils.allclose(result, expected)


def test_python_scope_linalg():
    qd.init()
    a = np.array([3, 4, -2])
    b = np.array([-5, 0, 6])
    x = qd.Vector(a)
    y = qd.Vector(b)

    assert test_utils.allclose(x.dot(y), np.dot(a, b))
    assert test_utils.allclose(x.norm(), np.sqrt(np.dot(a, a)))
    assert test_utils.allclose(x.normalized().to_numpy(), a / np.sqrt(np.dot(a, a)))
    assert x.any() == 1  # To match that of Quadrants IR, we return -1 for True
    assert y.all() == 0


@test_utils.test(arch=[qd.x64, qd.cuda, qd.metal])
def test_16_min_max():
    @qd.kernel
    def min_u16(a: qd.u16, b: qd.u16) -> qd.u16:
        return qd.min(a, b)

    @qd.kernel
    def min_i16(a: qd.i16, b: qd.i16) -> qd.i16:
        return qd.min(a, b)

    @qd.kernel
    def max_u16(a: qd.u16, b: qd.u16) -> qd.u16:
        return qd.max(a, b)

    @qd.kernel
    def max_i16(a: qd.i16, b: qd.i16) -> qd.i16:
        return qd.max(a, b)

    a, b = 4, 2
    assert min_u16(a, b) == min(a, b)
    assert min_i16(a, b) == min(a, b)
    assert max_u16(a, b) == max(a, b)
    assert max_i16(a, b) == max(a, b)


@test_utils.test()
def test_32_min_max():
    @qd.kernel
    def min_u32(a: qd.u32, b: qd.u32) -> qd.u32:
        return qd.min(a, b)

    @qd.kernel
    def min_i32(a: qd.i32, b: qd.i32) -> qd.i32:
        return qd.min(a, b)

    @qd.kernel
    def max_u32(a: qd.u32, b: qd.u32) -> qd.u32:
        return qd.max(a, b)

    @qd.kernel
    def max_i32(a: qd.i32, b: qd.i32) -> qd.i32:
        return qd.max(a, b)

    a, b = 4, 2
    assert min_u32(a, b) == min(a, b)
    assert min_i32(a, b) == min(a, b)
    assert max_u32(a, b) == max(a, b)
    assert max_i32(a, b) == max(a, b)


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_64_min_max():
    @qd.kernel
    def min_u64(a: qd.u64, b: qd.u64) -> qd.u64:
        return qd.min(a, b)

    @qd.kernel
    def min_i64(a: qd.i64, b: qd.i64) -> qd.i64:
        return qd.min(a, b)

    @qd.kernel
    def max_u64(a: qd.u64, b: qd.u64) -> qd.u64:
        return qd.max(a, b)

    @qd.kernel
    def max_i64(a: qd.i64, b: qd.i64) -> qd.i64:
        return qd.max(a, b)

    a, b = 4, 2
    assert min_u64(a, b) == min(a, b)
    assert min_i64(a, b) == min(a, b)
    assert max_u64(a, b) == max(a, b)
    assert max_i64(a, b) == max(a, b)


@test_utils.test()
def test_min_max_vector_starred():
    @qd.kernel
    def min_starred() -> qd.i32:
        a = qd.Vector([1, 2, 3])
        b = qd.Vector([4, 5, 6])
        return qd.min(*a, *b)

    @qd.kernel
    def max_starred() -> qd.i32:
        a = qd.Vector([1, 2, 3])
        b = qd.Vector([4, 5, 6])
        return qd.max(*a, *b)

    assert min_starred() == 1
    assert max_starred() == 6
