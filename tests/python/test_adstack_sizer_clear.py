"""Regression: clear deliberately poisoned adstack metadata before the sizer runs.

The shader writes the overflow word only when a ``MaxOverRange`` walk exceeds its cap. Poisoning the metadata
buffer with nonzero words makes a missing clear fail without relying on allocator reuse or a particular platform.
"""

import os
import subprocess
import sys

import numpy as np

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

TEST_RAN = "test ran"
RET_SUCCESS = 42


def _sizer_overflow_child(args: list[str]) -> None:
    # Match the harness: requiring the adstack extension turns this flag on.
    qd.init(
        arch=qd.vulkan,
        enable_fallback=False,
        cfg_optimization=False,
        ad_stack_experimental_enabled=True,
        print_full_traceback=True,
    )

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

    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@test_utils.test(arch=qd.vulkan, require=qd.extension.adstack)
def test_sizer_overflow_word_is_cleared_after_poisoning():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["QD_TEST_ADSTACK_SIZER_POISON"] = "1"
    proc = subprocess.run(
        [sys.executable, __file__, _sizer_overflow_child.__name__], env=env, capture_output=True, text=True
    )
    assert proc.returncode == RET_SUCCESS, proc.stdout + "\n" + proc.stderr
    assert TEST_RAN in proc.stdout, proc.stdout + "\n" + proc.stderr
    assert "[adstack_sizer] poisoned metadata buffers" in proc.stderr, proc.stdout + "\n" + proc.stderr


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
