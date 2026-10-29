import numpy as np
import pytest

import quadrants as qd
from quadrants._lib.core.quadrants_python import DataTypeCxx

from tests.helpers import test_utils


@pytest.mark.parametrize("container_type,operation", [("field", "fill"), ("field", "allocation"), ("ndarray", "fill")])
@pytest.mark.parametrize("element_class,element_shape", [("scalar", ()), ("vector", (2,)), ("matrix", (2, 2))])
@pytest.mark.parametrize(
    "dtype,fill_value",
    [
        (qd.i64, 2**40),
        (qd.i64, -(2**40)),
        (qd.u64, 2**63 + 5),
        (qd.f64, 1.0000000000000002),
        (qd.f64, 1e300),
        (qd.f64, 1e-300),
    ],
)
@test_utils.test(require=qd.extension.data64, default_ip=qd.i32, default_fp=qd.f32)
def test_fill_destination_dtype(
    container_type: str,
    operation: str,
    element_class: str,
    element_shape: tuple[int, ...],
    dtype: DataTypeCxx,
    fill_value: int | float,
) -> None:
    # Immediate and allocation-time fills must preserve the destination's range and precision despite i32/f32 defaults.
    components = np.full(element_shape, fill_value).tolist()
    kwargs = {"fill": components} if operation == "allocation" else {}
    if element_class == "scalar":
        factory = qd.field if container_type == "field" else qd.ndarray
        array = factory(dtype=dtype, shape=(), **kwargs)
    elif element_class == "vector":
        factory = qd.Vector.field if container_type == "field" else qd.Vector.ndarray
        array = factory(n=2, dtype=dtype, shape=(), **kwargs)
    elif element_class == "matrix":
        factory = qd.Matrix.field if container_type == "field" else qd.Matrix.ndarray
        array = factory(n=2, m=2, dtype=dtype, shape=(), **kwargs)
    if operation == "fill":
        array.fill(components)
    np.testing.assert_array_equal(array.to_numpy(), np.full(element_shape, fill_value))


@pytest.mark.parametrize("dtype,fill_value", [(qd.i64, 2**63), (qd.u64, -1)])
@pytest.mark.parametrize("operation", ["fill", "allocation"])
@test_utils.test(require=qd.extension.data64, default_ip=qd.i32)
def test_fill_integer_out_of_range(dtype: DataTypeCxx, fill_value: int, operation: str) -> None:
    # The destination integer type must still reject constants outside its representable range.
    kwargs = {"fill": fill_value} if operation == "allocation" else {}
    array = qd.field(dtype=dtype, shape=(), **kwargs)
    with pytest.raises(qd.QuadrantsTypeError, match="exceeded the range"):
        if operation == "fill":
            array.fill(fill_value)
        else:
            qd.sync()
