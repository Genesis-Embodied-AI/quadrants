"""Regression: the adstack sizer overflow word must start at zero.

The shader stores that word only when a ``MaxOverRange`` walk exceeds its cap. Device allocations are not
zero-filled. With MoltenVK placement heaps off, a later launch can reuse a block that still holds a nonzero
word, and the host reports that stale word as a huge range. This process sets ``MVK_CONFIG_USE_MTLHEAP=0``
before Quadrants loads MoltenVK, then repeats the second reverse launch that hits the recycled word.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import quadrants as qd

from tests import test_utils

_CHILD = textwrap.dedent(
    """\
    import os

    os.environ["MVK_CONFIG_USE_MTLHEAP"] = "0"

    import numpy as np
    import quadrants as qd
    from quadrants.lang import impl

    qd.init(arch=qd.vulkan, enable_fallback=False, cfg_optimization=False, print_full_traceback=True)

    N = 4
    x = qd.field(qd.f32, shape=(N,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            v = x[i]
            n = a[i]
            for _ in range(n):
                v = v * 0.95 + 0.01
            y[None] += v

    a = qd.ndarray(qd.i32, shape=(N,))
    a.from_numpy(np.array([2, 3, 1, 2], dtype=np.int32))
    for i in range(N):
        x[i] = 0.1

    prog = impl.get_runtime().prog
    prog._reset_max_reducer_dispatch_count()
    compute(a)
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad(a)
    qd.sync()
    after_first = prog._get_max_reducer_dispatch_count()
    assert after_first >= 1, after_first

    a.from_numpy(np.array([3, 3, 1, 2], dtype=np.int32))
    pre_mutation = prog._get_max_reducer_dispatch_count()
    compute(a)
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad(a)
    qd.sync()
    assert prog._get_max_reducer_dispatch_count() > pre_mutation
    """
)


def test_sizer_overflow_word_is_cleared_when_heaps_are_off():
    if sys.platform != "darwin":
        pytest.skip("MoltenVK placement-heap recycling")
    if qd.vulkan not in test_utils.expected_archs():
        pytest.skip("vulkan is not selected")
    env = os.environ.copy()
    env["MVK_CONFIG_USE_MTLHEAP"] = "0"
    proc = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
