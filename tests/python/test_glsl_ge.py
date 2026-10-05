import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.f32])
@test_utils.test(arch=[qd.vulkan, qd.metal], offline_cache=False, external_optimization_level=0)
def test_ge_shader_and_fallbacks(dtype):
    if dtype == qd.i32:
        values = np.array([-(2**31), -17, -1, 0, 1, 17, 2**31 - 1], dtype=np.int32)
    elif dtype == qd.u32:
        values = np.array([0, 1, 17, 2**31, 2**32 - 1], dtype=np.uint32)
    else:
        values = np.array([-np.inf, -1.5, -0.0, 0.0, 1.5, np.inf, np.nan], dtype=np.float32)
    data = qd.ndarray(dtype=dtype, shape=len(values))
    data.from_numpy(values)
    out = qd.ndarray(dtype=qd.i32, shape=(len(values), len(values), 2))

    @qd.kernel
    def k_compare(data: qd.types.ndarray(), out: qd.types.ndarray()):
        for i, j in qd.ndrange(data.shape[0], data.shape[0]):
            out[i, j, 0] = data[i] >= data[j]
            out[i, j, 1] = data[j] >= data[i]

    k_compare(data, out)
    expected = values[:, None] >= values[None, :]
    np.testing.assert_array_equal(out.to_numpy(), np.stack([expected, expected.T], axis=-1))
