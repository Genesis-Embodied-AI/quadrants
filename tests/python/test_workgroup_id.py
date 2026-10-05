import numpy as np

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@test_utils.test(arch=[qd.vulkan, qd.metal], offline_cache=False)
def test_workgroup_id_shader_helper():
    out = qd.ndarray(dtype=qd.i32, shape=(200, 2))

    @qd.kernel
    def k_block_indices(out: qd.types.ndarray(dtype=qd.i32, ndim=2)):
        qd.loop_config(block_dim=32)
        for i in range(200):
            out[i, 0] = impl.call_internal("workgroupId")
            out[i, 1] = impl.call_internal("workgroupId")

    k_block_indices(out)
    expected = np.repeat((np.arange(200, dtype=np.int32) // 32)[:, None], 2, axis=1)
    np.testing.assert_array_equal(out.to_numpy(), expected)
