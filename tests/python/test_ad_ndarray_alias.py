import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize(
    "dtype,square",
    [
        pytest.param(qd.f32, False, id="scalar-identity"),
        pytest.param(qd.types.vector(3, qd.f32), False, id="vector-identity"),
        pytest.param(qd.types.matrix(2, 3, qd.f32), False, id="matrix-identity"),
        pytest.param(qd.f32, True, id="scalar-scaled-square"),
        pytest.param(qd.types.vector(3, qd.f32), True, id="vector-scaled-square"),
        pytest.param(qd.types.matrix(2, 3, qd.f32), True, id="matrix-scaled-square"),
    ],
)
@test_utils.test(offline_cache=False)
def test_ndarray_grad_load_before_clear(dtype, square: bool) -> None:
    x = qd.tensor(dtype, (2, 4, 5), backend=qd.Backend.NDARRAY, needs_grad=True)
    y = qd.tensor(dtype, x.shape, backend=qd.Backend.NDARRAY, needs_grad=True)

    @qd.kernel
    def compute(x: qd.Tensor, y: qd.Tensor) -> None:
        for I in qd.grouped(x):
            if qd.static(square):
                y[I] = 3 * x[I] * x[I]
            else:
                y[I] = x[I]

    x.fill(3)
    x.grad.fill(2)
    compute(x, y)
    y.grad.fill(1)
    # The backward kernel must retain the upstream gradient before clearing its buffer.
    compute.grad(x, y)
    actual = x.grad.to_numpy()
    np.testing.assert_array_equal(actual, np.full_like(actual, 20 if square else 3))
