"""Tests for ``qd.algorithms.*`` device-wide primitives.

Covers:

- ``quadrants._scratch`` — the shared ``Field(u32)`` scratch buffer that backs
  every device algorithm.
- (Forthcoming as each algo lands) ``qd.algorithms.device_reduce_*``,
  ``qd.algorithms.device_exclusive_scan_*``, ``select`` / ``compact``,
  ``device radix sort``, and ``reduce-by-key``.

Each test runs across the full ``arch=qd.gpu`` parametrization so the kernels
are exercised on CUDA, AMDGPU, Vulkan, and Metal (where the host supports
each).
"""

import quadrants as qd
from quadrants import _scratch

from tests import test_utils


@test_utils.test(arch=qd.gpu)
def test_scratch_allocates_with_expected_capacity():
    """First call returns a Field(u32) sized to the configured byte budget."""
    s = _scratch.get_scratch_u32()
    assert s.dtype == qd.u32
    assert s.shape == (_scratch.DEFAULT_SCRATCH_BYTES // 4,)
    assert _scratch.scratch_capacity_u32() == _scratch.DEFAULT_SCRATCH_BYTES // 4


@test_utils.test(arch=qd.gpu)
def test_scratch_is_shared_across_calls():
    """The same Field instance is returned on repeated calls within a runtime."""
    s1 = _scratch.get_scratch_u32()
    s2 = _scratch.get_scratch_u32()
    assert s1 is s2


@test_utils.test(arch=qd.gpu)
def test_scratch_round_trips_bit_cast_f32():
    """Smoke: write f32 values into the u32 scratch via qd.bit_cast and read
    them back. Verifies the bit_cast pattern used by every algorithm."""
    s = _scratch.get_scratch_u32()
    N = 64

    @qd.kernel
    def write():
        for i in range(N):
            v = qd.f32(i) * qd.f32(0.5) - qd.f32(7.25)
            s[i] = qd.bit_cast(v, qd.u32)

    out = qd.field(qd.f32, shape=N)

    @qd.kernel
    def read():
        for i in range(N):
            out[i] = qd.bit_cast(s[i], qd.f32)

    write()
    read()
    for i in range(N):
        expected = i * 0.5 - 7.25
        assert out[i] == expected, f"slot {i}: got {out[i]}, expected {expected}"
