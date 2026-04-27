import sys

import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="FIXME: macOS unified memory swaps to disk instead of OOMing, causing this test to hang",
)
@test_utils.test(arch=qd.gpu)
def test_huge_allocation_fail_at_allocate_time():
    """Ensure huge allocation fails at allocate time and not at memset to 0"""
    # No match= filter: the exact error message varies across backends (LLVM pool, CUDA malloc_async, Vulkan).
    # We only care that OOM raises a RuntimeError rather than crashing or silently succeeding.
    # Metal is excluded: its unified memory model means GPU allocations consume system RAM, and the unbounded loop
    # can OOM-kill the CI runner before Metal reports an allocation failure.
    # Vulkan on macOS uses MoltenVK (Metal under the hood) with the same unified-memory OOM risk.
    if sys.platform == "darwin" and qd.lang.impl.current_cfg().arch == qd.vulkan:
        pytest.skip("Vulkan on macOS uses MoltenVK with unified memory; same OOM-kill risk as Metal")
    with pytest.raises(RuntimeError):
        allocations = []
        while True:
            allocations.append(qd.ndarray(qd.u8, 1024 * 1024 * 1024))
