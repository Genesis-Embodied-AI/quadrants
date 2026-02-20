import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.util import has_pytorch

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_unary_op():
    @qd.kernel
    def floor():
        a = 1
        b = qd.floor(a)

    with pytest.raises(qd.QuadrantsTypeError, match="'floor' takes real inputs only"):
        floor()


@test_utils.test(arch=qd.cpu)
def test_binary_op():
    @qd.kernel
    def bitwise_float():
        a = 1
        b = 3.1
        c = a & b

    with pytest.raises(qd.QuadrantsTypeError, match=r"unsupported operand type\(s\) for '&'"):
        bitwise_float()


@test_utils.test(arch=qd.cpu, print_full_traceback=False)
def test_ternary_op():
    @qd.kernel
    def select():
        a = qd.math.vec2(1.0, 1.0)
        b = 3
        c = qd.math.vec3(1.0, 1.0, 2.0)
        d = a if b else c

    with pytest.raises(qd.QuadrantsCompilationError, match="Cannot broadcast tensor to tensor"):
        select()


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=[qd.cpu], print_full_traceback=False)
def test_subscript():
    a = qd.ndarray(qd.i32, shape=(10, 10))

    @qd.kernel
    def ndarray(x: qd.types.ndarray()):
        b = x[3, 1.1]

    with pytest.raises(qd.QuadrantsTypeError, match="indices must be integers"):
        ndarray(a)


@test_utils.test()
def test_0d_ndarray():
    @qd.kernel
    def foo() -> qd.i32:
        a = np.array(3, dtype=np.int32)
        return a

    assert foo() == 3


@test_utils.test()
def test_non_0d_ndarray():
    @qd.kernel
    def foo():
        a = np.array([1])

    with pytest.raises(
        qd.QuadrantsTypeError,
        match="Only 0-dimensional numpy array can be used to initialize a scalar expression",
    ):
        foo()


@test_utils.test(arch=qd.cpu)
def test_assign():
    f = qd.Vector.field(4, dtype=qd.i32, shape=())

    @qd.kernel
    def floor():
        f[None] = qd.Vector([1, 2, 3])

    with pytest.raises(
        qd.QuadrantsTypeError,
        match=r"cannot assign '\[Tensor \(3\) i32\]' to '\[Tensor \(4\) i32\]'",
    ):
        floor()
