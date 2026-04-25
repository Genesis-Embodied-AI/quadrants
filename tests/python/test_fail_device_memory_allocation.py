import pytest

import quadrants as qd

from tests import test_utils


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


@test_utils.test(arch=qd.gpu)
def test_zero_size_ndarray_does_not_crash():
    """Zero-size ndarrays should allocate successfully, not raise OOM."""
    x = qd.ndarray(qd.f32, (0,))
    y = qd.ndarray(qd.f32, (0, 4))
    z = qd.ndarray(qd.u8, (0,))
