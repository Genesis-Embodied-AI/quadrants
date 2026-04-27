import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.skip("FIXME: This test is causing OOM error on the CI, silently crashing jobs...")
@test_utils.test(arch=qd.gpu)
def test_huge_allocation_fail_at_allocate_time():
    """Ensure huge allocation fails at allocate time and not at memset to 0"""
    # No match= filter: the exact error message varies across backends
    # (LLVM pool, CUDA malloc_async, Metal, Vulkan). We only care that
    # OOM raises a RuntimeError rather than crashing or silently succeeding.
    with pytest.raises(RuntimeError):
        allocations = []
        while True:
            allocations.append(qd.ndarray(qd.u8, 1024 * 1024 * 1024))
