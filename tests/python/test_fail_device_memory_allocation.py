import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.gpu])
def test_huge_allocation_fail_at_allocate_time():
    """Ensure huge allocation fails at allocate time and not at memset to 0"""
    with pytest.raises(
        RuntimeError,
        match="Failed to allocate memory",
    ):
        N = 0x7F_FF_FF_FF
        x0 = qd.ndarray(qd.u8, (N))
