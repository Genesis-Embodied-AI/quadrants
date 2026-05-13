"""Tests for ``qd.algorithms.*`` device-wide primitives.

Covers:

- ``quadrants._scratch`` — the shared ``Field(u32)`` scratch buffer that backs
  every device algorithm.
- ``qd.algorithms.device_reduce_{add,min,max}`` — two-or-more-pass tree
  reduction with shared scratch + ``bit_cast``.
- (Forthcoming as each algo lands) ``qd.algorithms.device_exclusive_scan_*``,
  ``select`` / ``compact``, ``device radix sort``, and ``reduce-by-key``.

Each test runs across the full ``arch=qd.gpu`` parametrization so the kernels
are exercised on CUDA, AMDGPU, Vulkan, and Metal (where the host supports
each).
"""

import math

import numpy as np
import pytest

import quadrants as qd
from quadrants import _scratch
from quadrants.lang.util import to_numpy_type

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


# Sizes spanning the trivial single-element path (1), within-one-block
# (< BLOCK_DIM), block boundaries (BLOCK_DIM, BLOCK_DIM + 1), within-two-passes
# (BLOCK_DIM ** 2 and slightly above), and a "large-ish" three-pass-trigger
# size that exercises the recursion loop without making the test slow.
# (Quadrants doesn't support 0-shape tensors, so N=0 isn't exercised — the
# driver has a defensive N==0 branch but no caller can actually trigger it.)
_REDUCE_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]


def _np_dtype_for(qd_dtype):
    return to_numpy_type(qd_dtype)


def _fill_field(f, vals):
    """Host-side initialization of a Field from a numpy array. Uses
    ``from_numpy`` so layout permutations are handled correctly."""
    f.from_numpy(np.asarray(vals, dtype=_np_dtype_for(f.dtype)))


def _alloc_input_out(dtype, N):
    """Allocate a 1-D input field of size ``N`` and a 1-element output field of
    the same dtype. Field-backed because that's the cheaper allocation for
    Quadrants and exercises the polymorphic ``qd.Tensor`` annotation in the
    kernels."""
    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=1)
    return inp, out


@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.f32])
@test_utils.test(arch=qd.gpu)
def test_device_reduce_add(dtype, N):
    """device_reduce_add matches numpy.sum across the full size sweep + dtype set."""
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    if dtype == qd.f32:
        # Keep values bounded so f32 sums don't drift far from numpy's f64 ref.
        host = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)
    elif dtype == qd.u32:
        host = rng.integers(0, 1000, size=N, dtype=np.uint32)
    else:
        host = rng.integers(-1000, 1000, size=N, dtype=np.int32)
    _fill_field(inp, host)

    qd.algorithms.device_reduce_add(inp, out=out)

    got = out.to_numpy()[0]
    if dtype == qd.f32:
        expected = float(np.sum(host.astype(np.float64)))
        # f32 tree-reduce drift: tolerate ~N * eps_f32 in the worst case;
        # with bounded random inputs at N=200k, rtol=1e-4 is plenty.
        assert math.isclose(
            got, expected, rel_tol=1e-4, abs_tol=1e-4
        ), f"f32 reduce_add(N={N}): got {got}, expected {expected}"
    else:
        expected = int(np.sum(host.astype(np.int64)))
        assert int(got) == expected, f"{dtype} reduce_add(N={N}): got {got}, expected {expected}"


_MIN_IDENTITY = {qd.i32: 2**31 - 1, qd.u32: 2**32 - 1, qd.f32: float("inf")}
_MAX_IDENTITY = {qd.i32: -(2**31), qd.u32: 0, qd.f32: float("-inf")}


@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.f32])
@test_utils.test(arch=qd.gpu)
def test_device_reduce_min(dtype, N):
    """device_reduce_min(identity=type-positive-extreme) matches numpy.min."""
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    if dtype == qd.f32:
        host = rng.uniform(-10.0, 10.0, size=N).astype(np.float32)
    elif dtype == qd.u32:
        host = rng.integers(0, 10000, size=N, dtype=np.uint32)
    else:
        host = rng.integers(-10000, 10000, size=N, dtype=np.int32)
    _fill_field(inp, host)

    qd.algorithms.device_reduce_min(inp, _MIN_IDENTITY[dtype], out=out)
    got = out.to_numpy()[0]
    expected = host.min()

    if dtype == qd.f32:
        assert got == pytest.approx(expected, abs=1e-6)
    else:
        assert int(got) == int(expected), f"{dtype} reduce_min(N={N}): got {got}, expected {expected}"


@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.f32])
@test_utils.test(arch=qd.gpu)
def test_device_reduce_max(dtype, N):
    """device_reduce_max(identity=type-negative-extreme) matches numpy.max."""
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    if dtype == qd.f32:
        host = rng.uniform(-10.0, 10.0, size=N).astype(np.float32)
    elif dtype == qd.u32:
        host = rng.integers(0, 10000, size=N, dtype=np.uint32)
    else:
        host = rng.integers(-10000, 10000, size=N, dtype=np.int32)
    _fill_field(inp, host)

    qd.algorithms.device_reduce_max(inp, _MAX_IDENTITY[dtype], out=out)
    got = out.to_numpy()[0]
    expected = host.max()

    if dtype == qd.f32:
        assert got == pytest.approx(expected, abs=1e-6)
    else:
        assert int(got) == int(expected), f"{dtype} reduce_max(N={N}): got {got}, expected {expected}"


@test_utils.test(arch=qd.gpu)
def test_device_reduce_rejects_missing_identity_for_min():
    """Calling device_reduce_min without identity should raise — there's no
    portable type-extreme derivable from a value alone."""
    inp = qd.field(qd.i32, shape=4)
    out = qd.field(qd.i32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_reduce_min(inp, out=out)  # type: ignore[call-arg]


@test_utils.test(arch=qd.gpu)
def test_device_reduce_rejects_dtype_mismatch():
    """input and out must have the same dtype."""
    inp = qd.field(qd.i32, shape=4)
    out = qd.field(qd.f32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_reduce_add(inp, out=out)


@test_utils.test(arch=qd.gpu)
def test_device_reduce_rejects_unsupported_dtype():
    """First-land dtype set is {i32, u32, f32}; i64 / f64 are follow-up."""
    inp = qd.field(qd.i64, shape=4)
    out = qd.field(qd.i64, shape=1)
    with pytest.raises(NotImplementedError):
        qd.algorithms.device_reduce_add(inp, out=out)
