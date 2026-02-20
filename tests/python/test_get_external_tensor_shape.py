import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.util import has_pytorch

from tests import test_utils

if has_pytorch():
    import torch


@pytest.mark.parametrize("size", [[1], [1, 2, 3, 4]])
@test_utils.test()
def test_get_external_tensor_shape_access_numpy(size):
    @qd.kernel
    def func(x: qd.types.ndarray(), index: qd.template()) -> qd.i32:
        return x.shape[index]

    x_hat = np.ones(size, dtype=np.int32)
    for idx, y_ref in enumerate(size):
        y_hat = func(x_hat, idx)
        assert y_ref == y_hat, "Size of axis {} should equal {} and not {}.".format(idx, y_ref, y_hat)


@pytest.mark.parametrize("size", [[1, 1], [2, 2]])
@test_utils.test()
def test_get_external_tensor_shape_sum_numpy(size):
    @qd.kernel
    def func(x: qd.types.ndarray()) -> qd.i32:
        y = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y += x[i, j]
        return y

    x_hat = np.ones(size, dtype=np.int32)
    x_ref = x_hat
    y_hat = func(x_hat)
    y_ref = x_ref.sum()
    assert y_ref == y_hat, "Output should equal {} and not {}.".format(y_ref, y_hat)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@pytest.mark.parametrize("size", [[1, 2, 3, 4]])
@test_utils.test()
def test_get_external_tensor_shape_access_torch(size):
    @qd.kernel
    def func(x: qd.types.ndarray(), index: qd.template()) -> qd.i32:
        return x.shape[index]

    x_hat = torch.ones(size, dtype=torch.int32, device="cpu")
    for idx, y_ref in enumerate(size):
        y_hat = func(x_hat, idx)
        assert y_ref == y_hat, "Size of axis {} should equal {} and not {}.".format(idx, y_ref, y_hat)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@pytest.mark.parametrize("size", [[1, 2, 3, 4]])
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_get_external_tensor_shape_access_ndarray(size):
    @qd.kernel
    def func(x: qd.types.ndarray(), index: qd.template()) -> qd.i32:
        return x.shape[index]

    x_hat = qd.ndarray(qd.i32, shape=size)
    for idx, y_ref in enumerate(size):
        y_hat = func(x_hat, idx)
        assert y_ref == y_hat, "Size of axis {} should equal {} and not {}.".format(idx, y_ref, y_hat)
