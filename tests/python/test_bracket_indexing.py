import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_bracket_indexing_field():
    a = qd.field(qd.i32, ())

    @qd.kernel
    def k1():
        a[()] += 1

    k1()
    assert a[()] == 1


@test_utils.test()
def test_bracket_indexing_ndarray():
    a = qd.ndarray(qd.i32, ())

    @qd.kernel
    def k1(a: qd.types.NDArray[qd.i32, 0]):
        a[()] += 1

    k1(a)
    assert a[()] == 1


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray], ids=["field", "ndarray"])
@pytest.mark.parametrize("scope", ["python", "kernel"])
@pytest.mark.parametrize("read_index", [None, ()], ids=["none", "empty_tuple"])
@pytest.mark.parametrize("write_index", [None, ()], ids=["none", "empty_tuple"])
@test_utils.test()
def test_0dim_read_write(tensor_type, scope, read_index, write_index):
    # Cover both scalar-index spellings, including mixed reads and writes.
    TensorType = qd.types.NDArray if tensor_type == qd.ndarray else qd.Template

    def update(x: TensorType):
        a = x[read_index] + 1
        x[write_index] = 2 * a

    if scope == "kernel":
        update = qd.kernel(update)

    x = tensor_type(dtype=qd.f32, shape=())
    x[write_index] = 1.0
    assert x[read_index] == 1.0
    update(x)
    assert x[read_index] == 4.0

    y = tensor_type(dtype=qd.math.vec2, shape=())
    y[write_index] = [1.0, 2.0]
    np.testing.assert_array_equal(y[read_index].to_numpy(), [1.0, 2.0])
    update(y)
    np.testing.assert_array_equal(y[read_index].to_numpy(), [4.0, 6.0])
