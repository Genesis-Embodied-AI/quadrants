import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cuda)
def test_global_thread_idx():
    n = 128
    x = qd.field(qd.i32, shape=n)

    @qd.kernel
    def func():
        for i in range(n):
            tid = qd.global_thread_idx()
            x[tid] = tid

    func()
    assert np.arange(n).sum() == x.to_numpy().sum()
