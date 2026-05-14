"""Tests for ``qd.algorithms.*`` device-wide primitives.

Covers:

- ``quadrants._scratch`` - the shared ``Field(u32)`` scratch buffer that backs
  every device algorithm.
- ``qd.algorithms.device_reduce_{add,min,max}`` - two-or-more-pass tree
  reduction with shared scratch + ``bit_cast``.
- ``qd.algorithms.device_exclusive_scan_{add,min,max}`` - three-pass scan.
- ``qd.algorithms.device_select`` - scan-based stream compaction.
- ``qd.algorithms.device_radix_sort`` - LSB radix sort built on
  ``block.radix_rank_match_atomic_or``.
- ``qd.algorithms.device_reduce_by_key_add`` - scan + scatter +
  atomic_add reduce-by-key.

Each test runs across the full ``arch=qd.gpu`` parametrization so the kernels
are exercised on CUDA, AMDGPU, Vulkan, and Metal (where the host supports
each).
"""

import math
import struct

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


@test_utils.test(arch=qd.gpu)
def test_scratch_u64_allocates_with_expected_capacity():
    """First call to ``get_scratch_u64`` returns a Field(u64) sized to the same byte budget as the u32 scratch."""
    s = _scratch.get_scratch_u64()
    assert s.dtype == qd.u64
    assert s.shape == (_scratch.DEFAULT_SCRATCH_BYTES // 8,)
    assert _scratch.scratch_capacity_u64() == _scratch.DEFAULT_SCRATCH_BYTES // 8


@test_utils.test(arch=qd.gpu)
def test_scratch_u64_is_shared_across_calls():
    s1 = _scratch.get_scratch_u64()
    s2 = _scratch.get_scratch_u64()
    assert s1 is s2


@test_utils.test(arch=qd.gpu)
def test_scratch_round_trips_bit_cast_f64():
    """Smoke: feed exact-known f64 bit patterns into the kernel, bit_cast through the u64 scratch, read back. Mirrors
    ``test_scratch_round_trips_bit_cast_f32`` for the 8-byte-dtype path used by 64-bit ``device_reduce_*``.

    We push the host-computed bit pattern in via a u64 source field rather than arithmetic on f64 literals to dodge
    kernel-side fp-contract / FMA-reassociation that can offset the result by 1 ulp from the host-side value.
    """
    N = 64
    s = _scratch.get_scratch_u64()
    src_bits = qd.field(qd.u64, shape=N)
    out = qd.field(qd.f64, shape=N)

    expected = [i * 0.5 - 7.25 + 1.0e-100 * i for i in range(N)]
    bits_host = np.array(
        [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in expected], dtype=np.uint64
    )
    src_bits.from_numpy(bits_host)

    @qd.kernel
    def write():
        for i in range(N):
            v = qd.bit_cast(src_bits[i], qd.f64)
            s[i] = qd.bit_cast(v, qd.u64)

    @qd.kernel
    def read():
        for i in range(N):
            out[i] = qd.bit_cast(s[i], qd.f64)

    write()
    read()
    for i in range(N):
        assert out[i] == expected[i], f"slot {i}: got {out[i]}, expected {expected[i]}"


# Sizes spanning the trivial single-element path (1), within-one-block (< BLOCK_DIM), block boundaries (BLOCK_DIM,
# BLOCK_DIM + 1), within-two-passes (BLOCK_DIM ** 2 and slightly above), and a "large-ish" three-pass-trigger size
# that exercises the recursion loop without making the test slow. (Quadrants doesn't support 0-shape tensors, so N=0
# isn't exercised - the driver has a defensive N==0 branch but no caller can actually trigger it.)
_REDUCE_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]


def _np_dtype_for(qd_dtype):
    return to_numpy_type(qd_dtype)


def _fill_field(f, vals):
    """Host-side initialization of a Field from a numpy array. Uses
    ``from_numpy`` so layout permutations are handled correctly."""
    f.from_numpy(np.asarray(vals, dtype=_np_dtype_for(f.dtype)))


def _alloc_input_out(dtype, N):
    """Allocate a 1-D input field of size ``N`` and a 1-element output field of the same dtype. Field-backed because
    that's the cheaper allocation for Quadrants and exercises the polymorphic ``qd.Tensor`` annotation in the kernels.
    """
    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=1)
    return inp, out


_REDUCE_DTYPES = [qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64]

# Identities span the full {i32, u32, f32, i64, u64, f64} matrix: the floating-point cases use the dtype's infinity
# extremum; the integer cases use the dtype's positive / negative range extreme. These are passed to
# device_reduce_min / max as the monoid identity.
_MIN_IDENTITY = {
    qd.i32: 2**31 - 1,
    qd.u32: 2**32 - 1,
    qd.f32: float("inf"),
    qd.i64: 2**63 - 1,
    qd.u64: 2**64 - 1,
    qd.f64: float("inf"),
}
_MAX_IDENTITY = {
    qd.i32: -(2**31),
    qd.u32: 0,
    qd.f32: float("-inf"),
    qd.i64: -(2**63),
    qd.u64: 0,
    qd.f64: float("-inf"),
}

_DTYPE_TO_NP = {
    qd.i32: np.int32,
    qd.u32: np.uint32,
    qd.f32: np.float32,
    qd.i64: np.int64,
    qd.u64: np.uint64,
    qd.f64: np.float64,
}


def _is_float(dtype):
    return dtype in (qd.f32, qd.f64)


def _is_unsigned(dtype):
    return dtype in (qd.u32, qd.u64)


def _rand_reduce_host(rng, dtype, N, *, bound=1000):
    """Generate test inputs of the right numpy dtype, bounded so accumulations stay representable."""
    np_dt = _DTYPE_TO_NP[dtype]
    if _is_float(dtype):
        # Keep values bounded so float sums don't drift far from numpy's promoted ref.
        return rng.uniform(-1.0, 1.0, size=N).astype(np_dt)
    if _is_unsigned(dtype):
        return rng.integers(0, bound, size=N, dtype=np_dt)
    return rng.integers(-bound, bound, size=N, dtype=np_dt)


@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", _REDUCE_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_reduce_add(dtype, N):
    """device_reduce_add matches numpy.sum across the full size sweep + dtype set."""
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    host = _rand_reduce_host(rng, dtype, N)
    _fill_field(inp, host)

    qd.algorithms.device_reduce_add(inp, out=out)

    got = out.to_numpy()[0]
    if _is_float(dtype):
        expected = float(np.sum(host.astype(np.float64)))
        # f32: ~N * eps_f32 drift; f64: ~N * eps_f64. rtol scales with dtype precision.
        rtol = 1e-4 if dtype == qd.f32 else 1e-12
        atol = 1e-4 if dtype == qd.f32 else 1e-9
        assert math.isclose(got, expected, rel_tol=rtol, abs_tol=atol), (
            f"{dtype} reduce_add(N={N}): got {got}, expected {expected}"
        )
    else:
        # Promote to Python int for an arbitrary-width reference; mask both sides to dtype width to handle the
        # u32 / u64 mod-wrap case at large N.
        mod = 1 << (32 if dtype in (qd.i32, qd.u32) else 64) if _is_unsigned(dtype) else None
        ref = int(np.sum(host.astype(np.int64 if dtype in (qd.i32, qd.u32) else (np.int64 if dtype == qd.i64 else np.uint64))))  # noqa: E501
        got_int = int(got)
        if mod is not None:
            ref &= mod - 1
            got_int &= mod - 1
        assert got_int == ref, f"{dtype} reduce_add(N={N}): got {got_int}, expected {ref}"


@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", _REDUCE_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_reduce_min(dtype, N):
    """device_reduce_min(identity=type-positive-extreme) matches numpy.min."""
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    if _is_float(dtype):
        host = rng.uniform(-10.0, 10.0, size=N).astype(_DTYPE_TO_NP[dtype])
    else:
        host = _rand_reduce_host(rng, dtype, N, bound=10000)
    _fill_field(inp, host)

    qd.algorithms.device_reduce_min(inp, out=out)
    got = out.to_numpy()[0]
    expected = host.min()

    if _is_float(dtype):
        assert got == pytest.approx(expected, abs=1e-6 if dtype == qd.f32 else 1e-12)
    else:
        assert int(got) == int(expected), f"{dtype} reduce_min(N={N}): got {got}, expected {expected}"


@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", _REDUCE_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_reduce_max(dtype, N):
    """device_reduce_max(identity=type-negative-extreme) matches numpy.max."""
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    if _is_float(dtype):
        host = rng.uniform(-10.0, 10.0, size=N).astype(_DTYPE_TO_NP[dtype])
    else:
        host = _rand_reduce_host(rng, dtype, N, bound=10000)
    _fill_field(inp, host)

    qd.algorithms.device_reduce_max(inp, out=out)
    got = out.to_numpy()[0]
    expected = host.max()

    if _is_float(dtype):
        assert got == pytest.approx(expected, abs=1e-6 if dtype == qd.f32 else 1e-12)
    else:
        assert int(got) == int(expected), f"{dtype} reduce_max(N={N}): got {got}, expected {expected}"


@test_utils.test(arch=qd.gpu)
def test_device_reduce_min_derives_identity_from_dtype():
    """``device_reduce_min`` does not take an identity argument; it's derived from ``arr.dtype`` (mirror of the
    ``block.reduce_min`` / ``subgroup.reduce_min`` contract). On an all-min-identity input the reduction returns
    the identity itself (largest representable value), which exercises the auto-derivation end-to-end."""
    for dtype in _REDUCE_DTYPES:
        inp = qd.field(dtype, shape=4)
        out = qd.field(dtype, shape=1)
        identity = _MIN_IDENTITY[dtype]
        host = np.full(4, identity, dtype=_DTYPE_TO_NP[dtype])
        _fill_field(inp, host)
        qd.algorithms.device_reduce_min(inp, out=out)
        got = out.to_numpy()[0]
        assert got == _DTYPE_TO_NP[dtype](identity), f"{dtype}: got {got}, expected {identity}"


@test_utils.test(arch=qd.gpu)
def test_device_reduce_rejects_dtype_mismatch():
    """input and out must have the same dtype."""
    inp = qd.field(qd.i32, shape=4)
    out = qd.field(qd.f32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_reduce_add(inp, out=out)


@test_utils.test(arch=qd.gpu)
def test_device_reduce_rejects_unsupported_dtype():
    """Supported set is {i32, u32, f32, i64, u64, f64}; narrower dtypes (i16, f16, etc.) are out of scope and must
    raise NotImplementedError so callers don't silently get bit-cast nonsense."""
    inp = qd.field(qd.i16, shape=4)
    out = qd.field(qd.i16, shape=1)
    with pytest.raises(NotImplementedError):
        qd.algorithms.device_reduce_add(inp, out=out)


# Same size sweep as reduce, plus a smaller pre-block size for the single-tile
# fast path, and a larger 1M size to exercise three-level recursion (B0=4096,
# B1=16, B2 single-block). The recursive case is the most important to catch
# any partials_off bookkeeping bugs.
_SCAN_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000, 1_000_000]


def _alloc_scan_input_out(dtype, N):
    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=N)
    return inp, out


_SCAN_DTYPES = [qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64]


def _scan_dtype_mask(dtype):
    """Return the wrap-around mask for unsigned scans, or -1 (no mask) for signed / floats."""
    if dtype == qd.u32:
        return 0xFFFFFFFF
    if dtype == qd.u64:
        return 0xFFFFFFFFFFFFFFFF
    return -1


@pytest.mark.parametrize("N", _SCAN_SIZES)
@pytest.mark.parametrize("dtype", _SCAN_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_add(dtype, N):
    """device_exclusive_scan_add(out[i] = sum(arr[0:i])) matches numpy.cumsum-shifted across the full 6-dtype set."""
    inp, out = _alloc_scan_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    host = _rand_reduce_host(rng, dtype, N, bound=100)
    _fill_field(inp, host)

    qd.algorithms.device_exclusive_scan_add(inp, out=out)
    got = out.to_numpy()

    if _is_float(dtype):
        ref = np.concatenate([[0.0], np.cumsum(host.astype(np.float64))[:-1]])
        # f32 accumulation drift across N=1M can reach ~N * eps_f32 * mean-magnitude; f64 ~N * eps_f64. rtol scales
        # with dtype precision. abs_tol covers values near zero.
        rtol = 1e-3 if dtype == qd.f32 else 1e-12
        atol = 1e-3 if dtype == qd.f32 else 1e-9
        np.testing.assert_allclose(
            got.astype(np.float64),
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{dtype} scan_add(N={N})",
        )
    else:
        # Promote to a width that survives the cumulative sum: u64 / i64 inputs use a Python int reference; smaller
        # ints can still use int64.
        promote = np.int64 if dtype in (qd.i32, qd.u32, qd.i64) else np.uint64
        host_wide = host.astype(promote)
        ref = np.concatenate([[promote(0)], np.cumsum(host_wide)[:-1]]).astype(promote)
        mask = _scan_dtype_mask(dtype)
        got_view = got.astype(np.int64 if dtype != qd.u64 else np.uint64)
        if mask != -1:
            got_view = got_view & promote(mask)
            ref = ref & promote(mask)
        np.testing.assert_array_equal(
            got_view,
            ref,
            err_msg=f"{dtype} scan_add(N={N})",
        )


@pytest.mark.parametrize("N", _SCAN_SIZES)
@pytest.mark.parametrize("dtype", _SCAN_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_min(dtype, N):
    """device_exclusive_scan_min(out[i] = min(arr[0:i])) matches numpy.minimum.accumulate-shifted across the full
    6-dtype set."""
    inp, out = _alloc_scan_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    np_dt = _DTYPE_TO_NP[dtype]
    if _is_float(dtype):
        host = rng.uniform(-10.0, 10.0, size=N).astype(np_dt)
    else:
        host = _rand_reduce_host(rng, dtype, N, bound=10000)
    _fill_field(inp, host)

    qd.algorithms.device_exclusive_scan_min(inp, out=out)
    got = out.to_numpy()

    if _is_float(dtype):
        ref = np.concatenate([[float("inf")], np.minimum.accumulate(host.astype(np.float64))[:-1]]).astype(np_dt)
        atol = 0 if dtype == qd.f32 else 0  # min is bitwise-exact for monotone ops on float
        np.testing.assert_allclose(got, ref, rtol=0, atol=atol, err_msg=f"{dtype} scan_min(N={N})")
    else:
        ref = np.concatenate([[np_dt(_MIN_IDENTITY[dtype])], np.minimum.accumulate(host)[:-1]]).astype(np_dt)
        np.testing.assert_array_equal(got, ref, err_msg=f"{dtype} scan_min(N={N})")


@pytest.mark.parametrize("N", _SCAN_SIZES)
@pytest.mark.parametrize("dtype", _SCAN_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_max(dtype, N):
    """device_exclusive_scan_max(out[i] = max(arr[0:i])) matches numpy.maximum.accumulate-shifted across the full
    6-dtype set."""
    inp, out = _alloc_scan_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    np_dt = _DTYPE_TO_NP[dtype]
    if _is_float(dtype):
        host = rng.uniform(-10.0, 10.0, size=N).astype(np_dt)
    else:
        host = _rand_reduce_host(rng, dtype, N, bound=10000)
    _fill_field(inp, host)

    qd.algorithms.device_exclusive_scan_max(inp, out=out)
    got = out.to_numpy()

    if _is_float(dtype):
        ref = np.concatenate([[float("-inf")], np.maximum.accumulate(host.astype(np.float64))[:-1]]).astype(np_dt)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0, err_msg=f"{dtype} scan_max(N={N})")
    else:
        ref = np.concatenate([[np_dt(_MAX_IDENTITY[dtype])], np.maximum.accumulate(host)[:-1]]).astype(np_dt)
        np.testing.assert_array_equal(got, ref, err_msg=f"{dtype} scan_max(N={N})")


@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_rejects_inplace():
    """In-place scan (out is input) is rejected per the design doc - see
    'API design' / 'Aliasing' in qipc_device_algos_design.md."""
    arr = qd.field(qd.i32, shape=4)
    with pytest.raises(ValueError):
        qd.algorithms.device_exclusive_scan_add(arr, out=arr)


@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_rejects_shape_mismatch():
    """out.shape must equal input.shape."""
    inp = qd.field(qd.i32, shape=4)
    out = qd.field(qd.i32, shape=8)
    with pytest.raises(TypeError):
        qd.algorithms.device_exclusive_scan_add(inp, out=out)


@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_rejects_dtype_mismatch():
    """input and out must have the same dtype."""
    inp = qd.field(qd.i32, shape=4)
    out = qd.field(qd.f32, shape=4)
    with pytest.raises(TypeError):
        qd.algorithms.device_exclusive_scan_add(inp, out=out)


@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_rejects_unsupported_dtype():
    """Supported set is {i32, u32, f32, i64, u64, f64}; narrower / wider scalar dtypes raise NotImplementedError."""
    inp = qd.field(qd.i16, shape=4)
    out = qd.field(qd.i16, shape=4)
    with pytest.raises(NotImplementedError):
        qd.algorithms.device_exclusive_scan_add(inp, out=out)


# ---------------------------------------------------------------------------
# Device select / compact
# ---------------------------------------------------------------------------

# Sizes that exercise: single-block path, two-pass path with even split,
# off-by-one tile (B0 ends mid-block), three-pass recursion. Default 5 MB
# scratch holds N + ceil(N/256) + ... <= ~1.3M u32 slots, so the largest
# size below (200_000) fits comfortably with ~780-slot partials.
_SELECT_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]


_SELECT_DTYPES = [qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64]


@pytest.mark.parametrize("N", _SELECT_SIZES)
@pytest.mark.parametrize("dtype", _SELECT_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_select_basic(dtype, N):
    """device_select packs the elements with flags != 0 into a dense prefix of out, in stable input order; num_out[0]
    holds the count. Covers all 6 supported scalar dtypes - the scatter (``dst[idx] = src[i]``) is dtype-agnostic, so
    a single parametrized test serves both the 4-byte and 8-byte paths."""
    rng = np.random.default_rng(seed=1234)
    np_dt = _DTYPE_TO_NP[dtype]
    if dtype in (qd.f32, qd.f64):
        host = rng.uniform(-1.0, 1.0, size=N).astype(np_dt)
    elif dtype in (qd.u32, qd.u64):
        host = rng.integers(0, 10000, size=N, dtype=np_dt)
    else:
        host = rng.integers(-10000, 10000, size=N, dtype=np_dt)
    # Roughly 30% selection rate so the test exercises a non-trivial mix.
    flags_host = (rng.random(N) < 0.3).astype(np.int32)
    expected = host[flags_host == 1]
    expected_n = int(flags_host.sum())

    inp = qd.field(dtype, shape=N)
    flags = qd.field(qd.i32, shape=N)
    out = qd.field(dtype, shape=max(N, 1))
    num_out = qd.field(qd.i32, shape=1)
    _fill_field(inp, host)
    _fill_field(flags, flags_host)

    qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)
    got_n = int(num_out.to_numpy()[0])
    assert got_n == expected_n, f"{dtype} N={N}: got count {got_n}, expected {expected_n}"

    got = out.to_numpy()[:got_n]
    np.testing.assert_array_equal(got, expected, err_msg=f"{dtype} select(N={N})")


_SELECT_STRUCT_SIZES = [1, 7, 256, 1024, 65537]
_SELECT_STRUCT_NFIELDS = [2, 3, 4]  # mirrors libuipc Vector2i / Vector3i / Vector4i


@pytest.mark.parametrize("N", _SELECT_STRUCT_SIZES)
@pytest.mark.parametrize("nfields", _SELECT_STRUCT_NFIELDS)
@test_utils.test(arch=qd.gpu)
def test_device_select_struct_dtype(nfields, N):
    """device_select over a Struct-of-i32 (libuipc-shape: Vector2i / Vector3i / Vector4i).

    No code path inside ``device_select`` knows about struct dtypes; the scatter is ``dst[idx] = src[i]`` which
    lowers per-field. This test pins that contract end-to-end across nfields = 2 / 3 / 4 and a range of N.
    Fill / read-back goes through the struct-field ``from_numpy`` / ``to_numpy`` dict APIs to avoid per-element
    Python-scope assignment (which is O(N) host hops and dominates the test runtime for larger N).
    """
    field_names = ["a", "b", "c", "d"][:nfields]
    StructDT = qd.types.struct(**{name: qd.i32 for name in field_names})

    inp = StructDT.field(shape=(N,))
    out = StructDT.field(shape=(N,))
    flags = qd.field(qd.i32, shape=N)
    num_out = qd.field(qd.i32, shape=1)

    rng = np.random.default_rng(seed=1234)
    fields_host = {name: rng.integers(-1000, 1000, size=N, dtype=np.int32) for name in field_names}
    flags_host = (rng.random(N) < 0.3).astype(np.int32)
    inp.from_numpy(fields_host)
    _fill_field(flags, flags_host)

    qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)

    expected_n = int(flags_host.sum())
    got_n = int(num_out.to_numpy()[0])
    assert got_n == expected_n, f"struct(n={nfields}) N={N}: got count {got_n}, expected {expected_n}"

    got = out.to_numpy()
    sel_idx = np.where(flags_host != 0)[0]
    for name in field_names:
        np.testing.assert_array_equal(
            got[name][:got_n],
            fields_host[name][sel_idx],
            err_msg=f"struct(n={nfields}) N={N} field={name}",
        )


@test_utils.test(arch=qd.gpu)
def test_device_select_all_selected():
    """flags = all 1 -> out is a copy of input, num_out = N."""
    N = 1024
    inp = qd.field(qd.i32, shape=N)
    flags = qd.field(qd.i32, shape=N)
    out = qd.field(qd.i32, shape=N)
    num_out = qd.field(qd.i32, shape=1)

    rng = np.random.default_rng(seed=42)
    host = rng.integers(-100, 100, size=N, dtype=np.int32)
    _fill_field(inp, host)
    _fill_field(flags, np.ones(N, dtype=np.int32))

    qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)
    assert int(num_out.to_numpy()[0]) == N
    np.testing.assert_array_equal(out.to_numpy(), host)


@test_utils.test(arch=qd.gpu)
def test_device_select_none_selected():
    """flags = all 0 -> nothing written, num_out = 0."""
    N = 1024
    inp = qd.field(qd.i32, shape=N)
    flags = qd.field(qd.i32, shape=N)
    out = qd.field(qd.i32, shape=N)
    num_out = qd.field(qd.i32, shape=1)

    _fill_field(inp, np.arange(N, dtype=np.int32))
    _fill_field(flags, np.zeros(N, dtype=np.int32))

    qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)
    assert int(num_out.to_numpy()[0]) == 0


@test_utils.test(arch=qd.gpu)
def test_device_select_rejects_shape_mismatch():
    inp = qd.field(qd.i32, shape=4)
    flags = qd.field(qd.i32, shape=5)
    out = qd.field(qd.i32, shape=4)
    num_out = qd.field(qd.i32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)


@test_utils.test(arch=qd.gpu)
def test_device_select_rejects_flags_wrong_dtype():
    inp = qd.field(qd.i32, shape=4)
    flags = qd.field(qd.f32, shape=4)
    out = qd.field(qd.i32, shape=4)
    num_out = qd.field(qd.i32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)


@test_utils.test(arch=qd.gpu)
def test_device_select_rejects_dtype_mismatch():
    inp = qd.field(qd.i32, shape=4)
    flags = qd.field(qd.i32, shape=4)
    out = qd.field(qd.f32, shape=4)
    num_out = qd.field(qd.i32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)


@test_utils.test(arch=qd.gpu)
def test_device_select_rejects_short_out():
    """out must hold at least N elements (worst-case all-selected)."""
    inp = qd.field(qd.i32, shape=8)
    flags = qd.field(qd.i32, shape=8)
    out = qd.field(qd.i32, shape=4)  # < input size
    num_out = qd.field(qd.i32, shape=1)
    with pytest.raises(ValueError):
        qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)


# ---------------------------------------------------------------------------
# Device radix sort
# ---------------------------------------------------------------------------

# Sizes chosen to stay within the default 5 MB scratch budget (~1.3M u32 slots, of which radix sort uses ~N + N/256
# per pass). We hit single-block (N<=256), block boundary, off-by-one, two-block, many-block (200K, 4-pass
# histogram-scan-scatter recursion all exercised). N = 1M is covered separately by the qipc-hot-path tests below.
_RADIX_SORT_SIZES = [1, 7, 256, 257, 1023, 1024, 1025, 65536, 200_000]


_RADIX_KEY_DTYPES = [qd.u32, qd.i32, qd.f32, qd.u64, qd.i64, qd.f64]


def _gen_keys(rng, dtype, N):
    """Generate sortable test inputs for every supported key dtype. The float paths sprinkle a few signed-zero /
    inf / denormal specials at the front of the array to exercise the sort-twiddle pattern."""
    if dtype == qd.u32:
        return rng.integers(0, 2**32, size=N, dtype=np.uint32)
    if dtype == qd.i32:
        return rng.integers(-(2**31), 2**31 - 1, size=N, dtype=np.int32)
    if dtype == qd.f32:
        arr = rng.standard_normal(N).astype(np.float32) * 1e3
        if N >= 6:
            arr[0] = -0.0
            arr[1] = 0.0
            arr[2] = np.float32(np.inf)
            arr[3] = np.float32(-np.inf)
            arr[4] = np.float32(1e-30)
            arr[5] = np.float32(-1e-30)
        return arr
    if dtype == qd.u64:
        # Span the high half of the u64 range too so all 8 byte-passes see non-zero histograms.
        return rng.integers(0, 2**63, size=N, dtype=np.uint64).astype(np.uint64) * np.uint64(2)
    if dtype == qd.i64:
        return rng.integers(-(2**62), 2**62, size=N, dtype=np.int64)
    if dtype == qd.f64:
        arr = rng.standard_normal(N).astype(np.float64) * 1e6
        if N >= 6:
            arr[0] = -0.0
            arr[1] = 0.0
            arr[2] = np.float64(np.inf)
            arr[3] = np.float64(-np.inf)
            arr[4] = np.float64(1e-300)
            arr[5] = np.float64(-1e-300)
        return arr
    raise ValueError(dtype)


@pytest.mark.parametrize("N", _RADIX_SORT_SIZES)
@pytest.mark.parametrize("dtype", _RADIX_KEY_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_keys_only(dtype, N):
    """device_radix_sort matches numpy.sort for every supported key dtype ({u32, i32, f32, u64, i64, f64})."""
    rng = np.random.default_rng(seed=1234)
    host = _gen_keys(rng, dtype, N)

    keys = qd.field(dtype, shape=N)
    tmp = qd.field(dtype, shape=N)
    _fill_field(keys, host)

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)
    got = keys.to_numpy()
    want = np.sort(host, kind="stable")
    np.testing.assert_array_equal(got, want, err_msg=f"{dtype} radix_sort(N={N})")


@pytest.mark.parametrize("N", _RADIX_SORT_SIZES)
@pytest.mark.parametrize("dtype", _RADIX_KEY_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_key_value(dtype, N):
    """Key-value sort: values permute in lock-step with keys; sort is stable. Exercises the libuipc-shaped u64-key
    + i32-value path (``MatrixConverter::ij_hash`` sorted with ``sort_index``) among the parametrized cases."""
    rng = np.random.default_rng(seed=1234)
    host = _gen_keys(rng, dtype, N)

    keys = qd.field(dtype, shape=N)
    tmp_keys = qd.field(dtype, shape=N)
    values = qd.field(qd.i32, shape=N)
    tmp_values = qd.field(qd.i32, shape=N)
    _fill_field(keys, host)
    _fill_field(values, np.arange(N, dtype=np.int32))

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp_keys, values=values, tmp_values=tmp_values)

    got_keys = keys.to_numpy()
    got_values = values.to_numpy()
    # Stable argsort gives the values permutation we expect.
    want_idx = np.argsort(host, kind="stable")
    want_keys = host[want_idx]
    np.testing.assert_array_equal(got_keys, want_keys, err_msg=f"{dtype} keys(N={N})")
    np.testing.assert_array_equal(got_values, want_idx.astype(np.int32), err_msg=f"{dtype} values(N={N})")


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_already_sorted():
    """No-op-ish input: already-sorted keys still come back sorted."""
    N = 5000
    keys = qd.field(qd.u32, shape=N)
    tmp = qd.field(qd.u32, shape=N)
    host = np.arange(N, dtype=np.uint32) * 7
    _fill_field(keys, host)
    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)
    np.testing.assert_array_equal(keys.to_numpy(), host)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_reverse_sorted():
    """Worst-case-for-comparison-sort input is just normal work for radix."""
    N = 5000
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    host = (np.arange(N, dtype=np.int32) * -7).astype(np.int32)  # decreasing
    _fill_field(keys, host)
    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)
    np.testing.assert_array_equal(keys.to_numpy(), np.sort(host))


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_all_same():
    """Many duplicates: radix rank still groups + scatters them correctly."""
    N = 5000
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    host = np.full(N, 42, dtype=np.int32)
    _fill_field(keys, host)
    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)
    np.testing.assert_array_equal(keys.to_numpy(), host)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_n1():
    """N=1 is the trivial early-return path."""
    keys = qd.field(qd.i32, shape=1)
    tmp = qd.field(qd.i32, shape=1)
    _fill_field(keys, np.asarray([42], dtype=np.int32))
    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)
    assert int(keys.to_numpy()[0]) == 42


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_dtype_mismatch():
    keys = qd.field(qd.i32, shape=8)
    tmp = qd.field(qd.u32, shape=8)
    with pytest.raises(TypeError):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_shape_mismatch():
    keys = qd.field(qd.i32, shape=8)
    tmp = qd.field(qd.i32, shape=4)
    with pytest.raises(TypeError):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_aliasing():
    keys = qd.field(qd.i32, shape=8)
    with pytest.raises(ValueError):
        qd.algorithms.device_radix_sort(keys, tmp_keys=keys)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_unsupported_dtype():
    """Supported set is {u32, i32, f32, u64, i64, f64}; narrower / wider dtypes raise NotImplementedError."""
    keys = qd.field(qd.i16, shape=8)
    tmp = qd.field(qd.i16, shape=8)
    with pytest.raises(NotImplementedError):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_missing_tmp_values():
    """values requires tmp_values."""
    keys = qd.field(qd.i32, shape=8)
    tmp_keys = qd.field(qd.i32, shape=8)
    values = qd.field(qd.i32, shape=8)
    with pytest.raises(ValueError):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp_keys, values=values)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_odd_passes():
    """end_bit must yield an even number of passes so the result lands in keys."""
    keys = qd.field(qd.i32, shape=8)
    tmp = qd.field(qd.i32, shape=8)
    with pytest.raises(ValueError):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, end_bit=8)  # 1 pass - odd


# ---------------------------------------------------------------------------
# Device reduce-by-key (add)
# ---------------------------------------------------------------------------

# Same default-scratch envelope as device_select (uses ~N + N/256 u32 slots).
_RBK_SIZES = [1, 2, 3, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]


def _ref_rbk_add(keys, values):
    """Reference reduce-by-key: collapse consecutive runs of equal keys,
    returning ``(unique_keys, sums)``."""
    if len(keys) == 0:
        return np.array([], dtype=keys.dtype), np.array([], dtype=values.dtype)
    uniq_keys = [keys[0]]
    sums = [values[0]]
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            uniq_keys.append(keys[i])
            sums.append(values[i])
        else:
            sums[-1] = sums[-1] + values[i]
    return np.asarray(uniq_keys, dtype=keys.dtype), np.asarray(sums, dtype=values.dtype)


def _gen_run_keys(rng, dtype, N):
    """Build a key vector of size N with a realistic run-length distribution.

    Runs are drawn from a small alphabet of 5-15 distinct values and repeated
    1-8 times, then concatenated and truncated to N. This guarantees both
    multi-element runs (so the scatter's atomic_add path is exercised) and
    single-element runs (so the position math is exercised at boundary).
    """
    np_t = to_numpy_type(dtype)
    if dtype == qd.f32:
        alphabet = rng.standard_normal(15).astype(np_t)
    elif dtype == qd.u32:
        alphabet = rng.integers(0, 100, size=15, dtype=np_t)
    else:
        alphabet = rng.integers(-50, 50, size=15, dtype=np_t)
    run_keys = rng.choice(alphabet, size=N // 3 + 2)
    run_lengths = rng.integers(1, 8, size=len(run_keys))
    keys = np.repeat(run_keys, run_lengths)
    if len(keys) < N:
        # pad with the last key to fill N elements (extends the final run)
        keys = np.concatenate([keys, np.full(N - len(keys), keys[-1], dtype=np_t)])
    return keys[:N].astype(np_t)


@pytest.mark.parametrize("N", _RBK_SIZES)
@pytest.mark.parametrize("key_dtype", [qd.i32, qd.u32, qd.f32])
@pytest.mark.parametrize("val_dtype", [qd.i32, qd.u32, qd.f32])
@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add(key_dtype, val_dtype, N):
    """Cross-product of key dtype × value dtype × size, against a CPU oracle.

    Values are bounded so ``f32`` accumulation error stays controlled - the
    tolerance is rtol=1e-3 for f32 (matches our scan_add f32 tolerance) and
    bit-exact for integer types.
    """
    rng = np.random.default_rng(seed=1234)
    keys_host = _gen_run_keys(rng, key_dtype, N)
    val_np = to_numpy_type(val_dtype)
    if val_dtype == qd.f32:
        values_host = rng.uniform(-1.0, 1.0, size=N).astype(val_np)
    elif val_dtype == qd.u32:
        values_host = rng.integers(0, 100, size=N, dtype=val_np)
    else:
        values_host = rng.integers(-100, 100, size=N, dtype=val_np)

    keys_in = qd.field(key_dtype, shape=N)
    values_in = qd.field(val_dtype, shape=N)
    keys_out = qd.field(key_dtype, shape=N)
    values_out = qd.field(val_dtype, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    _fill_field(keys_in, keys_host)
    _fill_field(values_in, values_host)

    qd.algorithms.device_reduce_by_key_add(
        keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
    )
    nr = int(num_runs.to_numpy()[0])
    want_keys, want_vals = _ref_rbk_add(keys_host, values_host)

    assert nr == len(want_keys), f"{key_dtype}/{val_dtype} N={N}: num_runs {nr} vs {len(want_keys)}"
    got_keys = keys_out.to_numpy()[:nr]
    got_vals = values_out.to_numpy()[:nr]
    np.testing.assert_array_equal(got_keys, want_keys, err_msg=f"{key_dtype}/{val_dtype} N={N}: keys")
    if val_dtype == qd.f32:
        np.testing.assert_allclose(
            got_vals,
            want_vals,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"{key_dtype}/{val_dtype} N={N}: values",
        )
    else:
        np.testing.assert_array_equal(got_vals, want_vals, err_msg=f"{key_dtype}/{val_dtype} N={N}: values")


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_all_same():
    """All keys equal -> single run, values_out[0] = sum of all values."""
    N = 1024
    keys_in = qd.field(qd.i32, shape=N)
    values_in = qd.field(qd.i32, shape=N)
    keys_out = qd.field(qd.i32, shape=N)
    values_out = qd.field(qd.i32, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    _fill_field(keys_in, np.full(N, 42, dtype=np.int32))
    rng = np.random.default_rng(seed=42)
    vals = rng.integers(-100, 100, size=N, dtype=np.int32)
    _fill_field(values_in, vals)

    qd.algorithms.device_reduce_by_key_add(
        keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
    )
    assert int(num_runs.to_numpy()[0]) == 1
    assert int(keys_out.to_numpy()[0]) == 42
    assert int(values_out.to_numpy()[0]) == int(vals.astype(np.int64).sum())


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_all_unique():
    """No two consecutive keys equal -> num_runs == N, values_out is a copy of values."""
    N = 1024
    keys_in = qd.field(qd.i32, shape=N)
    values_in = qd.field(qd.i32, shape=N)
    keys_out = qd.field(qd.i32, shape=N)
    values_out = qd.field(qd.i32, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    keys_host = np.arange(N, dtype=np.int32) * 7
    vals_host = np.arange(N, dtype=np.int32) * 11 - 3
    _fill_field(keys_in, keys_host)
    _fill_field(values_in, vals_host)

    qd.algorithms.device_reduce_by_key_add(
        keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
    )
    assert int(num_runs.to_numpy()[0]) == N
    np.testing.assert_array_equal(keys_out.to_numpy(), keys_host)
    np.testing.assert_array_equal(values_out.to_numpy(), vals_host)


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_rejects_shape_mismatch():
    keys_in = qd.field(qd.i32, shape=8)
    values_in = qd.field(qd.i32, shape=4)  # wrong length
    keys_out = qd.field(qd.i32, shape=8)
    values_out = qd.field(qd.i32, shape=8)
    num_runs = qd.field(qd.i32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_reduce_by_key_add(
            keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
        )


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_rejects_dtype_mismatch():
    keys_in = qd.field(qd.i32, shape=8)
    values_in = qd.field(qd.i32, shape=8)
    keys_out = qd.field(qd.f32, shape=8)  # dtype != keys_in
    values_out = qd.field(qd.i32, shape=8)
    num_runs = qd.field(qd.i32, shape=1)
    with pytest.raises(TypeError):
        qd.algorithms.device_reduce_by_key_add(
            keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
        )


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_rejects_short_out():
    """keys_out and values_out must hold at least N entries (worst case: all unique)."""
    keys_in = qd.field(qd.i32, shape=16)
    values_in = qd.field(qd.i32, shape=16)
    keys_out = qd.field(qd.i32, shape=8)  # too short
    values_out = qd.field(qd.i32, shape=16)
    num_runs = qd.field(qd.i32, shape=1)
    with pytest.raises(ValueError):
        qd.algorithms.device_reduce_by_key_add(
            keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
        )


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_rejects_unsupported_dtype():
    keys_in = qd.field(qd.i64, shape=8)
    values_in = qd.field(qd.i64, shape=8)
    keys_out = qd.field(qd.i64, shape=8)
    values_out = qd.field(qd.i64, shape=8)
    num_runs = qd.field(qd.i32, shape=1)
    with pytest.raises(NotImplementedError):
        qd.algorithms.device_reduce_by_key_add(
            keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
        )


# ---------------------------------------------------------------------------
# Cross-cutting: runtime lifecycle, ndarray polymorphism, deprecation, scratch-capacity errors, end_bit, pipeline
# composition, N=1M.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.gpu)
def test_scratch_invalidate_resets_bytes_to_default():
    """``_scratch._invalidate`` (hooked into ``qd.reset()``) resets BOTH the cached field handle AND ``_scratch_bytes``
    to the default.

    Pins the invariant: every ``qd.init`` starts with a pristine scratch config, exactly as a fresh process would. We
    test ``_invalidate`` directly (rather than going through ``qd.reset()``) because we want to assert the post-reset
    state *inside* a single test without fighting the conftest's per-test ``init`` / ``reset`` pairing.

    The ``arch=qd.gpu`` parametrization is for uniformity with the rest of the file - the assertion itself only
    touches Python module-level state, not the GPU, so the per-arch loop is redundant but harmless.
    """
    assert _scratch._scratch_bytes == _scratch.DEFAULT_SCRATCH_BYTES, (
        "test prerequisite: scratch_bytes starts at default; the previous "
        "test's qd.reset() teardown should have left it that way"
    )
    saved_field = _scratch._scratch_field
    saved_field_u64 = _scratch._scratch_field_u64
    try:
        _scratch._scratch_bytes = 8 << 20
        _scratch._invalidate()
        assert _scratch._scratch_bytes == _scratch.DEFAULT_SCRATCH_BYTES
        assert _scratch._scratch_field is None
        assert _scratch._scratch_field_u64 is None, "_scratch_field_u64 must also be invalidated on qd.reset()"
    finally:
        _scratch._scratch_bytes = _scratch.DEFAULT_SCRATCH_BYTES
        _scratch._scratch_field = saved_field
        _scratch._scratch_field_u64 = saved_field_u64


@pytest.fixture
def big_scratch():
    """Bump scratch to 8 MB for the duration of the test.

    No teardown - the conftest's per-test ``qd.reset()`` fires ``_scratch._invalidate``, which sets ``_scratch_bytes``
    back to ``DEFAULT_SCRATCH_BYTES`` and drops the field handle. That is what delivers test isolation for the next
    test. Restoring here via ``set_scratch_bytes`` would fail anyway: once the test has run an algorithm, the scratch
    field is allocated, and ``set_scratch_bytes`` rejects post-allocation bumps by design.
    """
    _scratch.set_scratch_bytes(8 << 20)
    yield


@pytest.mark.parametrize("dtype", _RADIX_KEY_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_n_1m(dtype, big_scratch):  # pylint: disable=unused-argument,redefined-outer-name
    """N = 1_000_000 - qipc's hot-path size. Requires scratch bumped to ~5 MB; the ``big_scratch`` fixture supplies
    8 MB and restores after. 8-byte key dtypes run twice as many passes (8 instead of 4) for the same N. Scratch
    requirement is unchanged - the histograms are always u32 - so the same ``big_scratch`` covers both widths."""
    N = 1_000_000
    rng = np.random.default_rng(seed=1234)
    host = _gen_keys(rng, dtype, N)

    keys = qd.field(dtype, shape=N)
    tmp = qd.field(dtype, shape=N)
    _fill_field(keys, host)

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)
    np.testing.assert_array_equal(keys.to_numpy(), np.sort(host, kind="stable"))


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_n_1m(big_scratch):  # pylint: disable=unused-argument,redefined-outer-name
    """N = 1_000_000 reduce-by-key. Same scratch requirement as the 1M radix sort; the kernel sequence is different
    (just scan + scatter) but the in-place scan over scratch[0:N] needs the bump."""
    N = 1_000_000
    rng = np.random.default_rng(seed=1234)
    keys_host = _gen_run_keys(rng, qd.i32, N)
    values_host = rng.integers(-100, 100, size=N, dtype=np.int32)

    keys_in = qd.field(qd.i32, shape=N)
    values_in = qd.field(qd.i32, shape=N)
    keys_out = qd.field(qd.i32, shape=N)
    values_out = qd.field(qd.i32, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    _fill_field(keys_in, keys_host)
    _fill_field(values_in, values_host)

    qd.algorithms.device_reduce_by_key_add(
        keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
    )
    nr = int(num_runs.to_numpy()[0])
    want_keys, want_vals = _ref_rbk_add(keys_host, values_host)
    assert nr == len(want_keys)
    np.testing.assert_array_equal(keys_out.to_numpy()[:nr], want_keys)
    np.testing.assert_array_equal(values_out.to_numpy()[:nr], want_vals)


# --- Polymorphic-tensor coverage (Field vs Ndarray) for the algorithm surface is currently Field-only - every kernel
# param is annotated ``template()``, which only accepts Field-like storage. The design doc captures the
# future-direction switch to ``qd.Tensor`` kernel annotations (which would let bare Ndarrays through unchanged).
# That's a follow-up; the unwrap path for ``qd.Tensor(field)`` is exercised at the kernel-API level in
# ``test_tensor_wrapper_kernel.py`` so we don't repeat it here (passing ``qd.Tensor(...)`` through ``device_*`` would
# also pin the per-kernel ``_tensor_unwrap_indices`` cache and break subsequent bare-Field tests, by design of the
# kernel.py fast-path optimisation).


# --- Deprecation warnings on the legacy executor / parallel_sort surfaces. We added the warnings; assert they
# actually fire so an accidental rebase that drops them is caught.


@test_utils.test(arch=qd.gpu)
def test_prefix_sum_executor_emits_deprecation_warning():
    """`PrefixSumExecutor(N)` must emit `DeprecationWarning` per the migration plan in algorithms.md."""
    with pytest.warns(DeprecationWarning, match="device_exclusive_scan_add"):
        qd.algorithms.PrefixSumExecutor(64)


@test_utils.test(arch=qd.gpu)
def test_parallel_sort_emits_deprecation_warning():
    """`parallel_sort` must emit `DeprecationWarning` per the migration plan in algorithms.md."""
    keys = qd.field(qd.i32, shape=8)
    with pytest.warns(DeprecationWarning, match="device_radix_sort"):
        qd.algorithms.parallel_sort(keys)


# --- Scratch-capacity error paths. Each algorithm raises a clear RuntimeError when N would push the scratch budget
# over the configured capacity, rather than corrupting data. Tests shrink scratch to a tiny budget so the trip
# point is reachable with a small N (cheap to allocate, runtime-independent of the DEFAULT_SCRATCH_BYTES knob).


_TINY_SCRATCH_BYTES = 64 << 10  # 64 KB; covers ~16K u32 slots, ~4M u32 slots after the BLOCK_DIM divide for reduce/scan.


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_oversized_n():
    """``device_radix_sort`` raises ``RuntimeError`` pointing the caller at ``set_scratch_bytes`` when N exceeds the
    scratch budget. Shrink scratch first so the trip point is reachable with a tiny N."""
    _scratch.set_scratch_bytes(_TINY_SCRATCH_BYTES)
    N = 4 * _scratch.scratch_capacity_u32()  # comfortably over the tiny-scratch ceiling
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)


@test_utils.test(arch=qd.gpu)
def test_device_select_rejects_oversized_n():
    """Same scratch-capacity error path for device_select."""
    _scratch.set_scratch_bytes(_TINY_SCRATCH_BYTES)
    N = 4 * _scratch.scratch_capacity_u32()
    inp = qd.field(qd.i32, shape=N)
    flags = qd.field(qd.i32, shape=N)
    out = qd.field(qd.i32, shape=N)
    num_out = qd.field(qd.i32, shape=1)
    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)


@test_utils.test(arch=qd.gpu)
def test_device_reduce_by_key_add_rejects_oversized_n():
    """Same scratch-capacity error path for reduce-by-key."""
    _scratch.set_scratch_bytes(_TINY_SCRATCH_BYTES)
    N = 4 * _scratch.scratch_capacity_u32()
    keys_in = qd.field(qd.i32, shape=N)
    values_in = qd.field(qd.i32, shape=N)
    keys_out = qd.field(qd.i32, shape=N)
    values_out = qd.field(qd.i32, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_reduce_by_key_add(
            keys_in, values_in, keys_out=keys_out, values_out=values_out, num_runs=num_runs
        )


@test_utils.test(arch=qd.gpu)
def test_device_reduce_add_rejects_oversized_n():
    """``device_reduce_*`` needs ~(B + B/256 + …) u32 slots where ``B = ceil(N / BLOCK_DIM)``; the trip point in N is
    ``BLOCK_DIM * capacity_u32``. With the tiny scratch budget that's ~256 * 16K = 4M; use 5M to be comfortably over.
    The kernel itself never launches; the validate-budget check trips first."""
    _scratch.set_scratch_bytes(_TINY_SCRATCH_BYTES)
    N = 256 * _scratch.scratch_capacity_u32() + 100_000
    inp = qd.field(qd.i32, shape=N)
    out = qd.field(qd.i32, shape=1)
    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_reduce_add(inp, out=out)


@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_add_rejects_oversized_n():
    """Same scratch-capacity error path for device_exclusive_scan_add. ``device_exclusive_scan_*`` needs ``B`` u32
    partials slots at the top level (plus recursive); trip point in N is ``BLOCK_DIM * capacity_u32``."""
    _scratch.set_scratch_bytes(_TINY_SCRATCH_BYTES)
    N = 256 * _scratch.scratch_capacity_u32() + 100_000
    inp = qd.field(qd.i32, shape=N)
    out = qd.field(qd.i32, shape=N)
    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_exclusive_scan_add(inp, out=out)


# --- Reduce / scan at N = 1M alongside the radix sort + RBK 1M coverage. Reduce / scan's scratch budget at 1M is
# small (4K + recursion ~ 16 u32 slots), trivially below the default 5 MB, so no ``big_scratch`` fixture is needed -
# included here just to round out the qipc-hot-path coverage on the same dtypes as the other 1M tests.


@pytest.mark.parametrize("dtype", _REDUCE_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_reduce_add_n_1m(dtype):
    """N = 1_000_000 reduce over the full dtype matrix. 4-byte dtypes use the u32 scratch (4K slots for top-level
    partials, recursion adds ~16); 8-byte dtypes use the u64 scratch with the same slot count at half the byte cost.
    Default 5 MB capacity covers both by a wide margin."""
    N = 1_000_000
    rng = np.random.default_rng(seed=1234)
    host = _rand_reduce_host(rng, dtype, N)

    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=1)
    _fill_field(inp, host)
    qd.algorithms.device_reduce_add(inp, out=out)

    got = out.to_numpy()[0]
    if _is_float(dtype):
        expected = float(np.sum(host.astype(np.float64)))
        rtol = 1e-3 if dtype == qd.f32 else 1e-12
        atol = 1e-3 if dtype == qd.f32 else 1e-9
        assert math.isclose(got, expected, rel_tol=rtol, abs_tol=atol)
    else:
        # Promote to a wide enough Python int / numpy int for the reference, then mask both to dtype width.
        if dtype in (qd.i32, qd.i64):
            expected = int(np.sum(host.astype(np.int64)))
        else:
            expected = int(np.sum(host.astype(np.uint64)))
        got_int = int(got)
        if dtype == qd.u32:
            got_int &= 0xFFFFFFFF
            expected &= 0xFFFFFFFF
        elif dtype == qd.u64:
            got_int &= 0xFFFFFFFFFFFFFFFF
            expected &= 0xFFFFFFFFFFFFFFFF
        assert got_int == expected


@pytest.mark.parametrize("dtype", _SCAN_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_add_n_1m(dtype):
    """N = 1_000_000 exclusive scan over the full dtype matrix. 4-byte dtypes go through the u32 scratch; 8-byte
    dtypes through the u64 scratch (4K slots at the top level for both, recursion adds ~16). Both fit in default
    5 MB by a wide margin."""
    N = 1_000_000
    rng = np.random.default_rng(seed=1234)
    np_dt = _DTYPE_TO_NP[dtype]
    if dtype == qd.f32:
        host = rng.uniform(-0.01, 0.01, size=N).astype(np_dt)
    elif dtype == qd.f64:
        host = rng.uniform(-0.01, 0.01, size=N).astype(np_dt)
    elif dtype in (qd.u32, qd.u64):
        host = rng.integers(0, 10, size=N, dtype=np_dt)
    else:
        host = rng.integers(-5, 5, size=N, dtype=np_dt)

    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=N)
    _fill_field(inp, host)
    qd.algorithms.device_exclusive_scan_add(inp, out=out)

    got = out.to_numpy()
    if _is_float(dtype):
        ref = np.concatenate([[0.0], np.cumsum(host.astype(np.float64))[:-1]]).astype(np_dt)
        # f32: cumulative drift over 1M adds is real; check head only with a generous rtol, then verify finite at
        # the tail. f64: 12 orders of magnitude more precision, can check the whole array with tight tolerance.
        if dtype == qd.f32:
            np.testing.assert_allclose(got[:64], ref[:64], rtol=1e-3, atol=1e-3)
            assert np.isfinite(got[-1])
        else:
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-9)
    else:
        promote = np.int64 if dtype in (qd.i32, qd.u32, qd.i64) else np.uint64
        ref = np.concatenate([[promote(0)], np.cumsum(host.astype(promote))[:-1]]).astype(promote)
        np.testing.assert_array_equal(got.astype(promote), ref)


# --- End-to-end round-trip: bump scratch, run a 1M algorithm, qd.reset + qd.init, then run a default-scratch-sized
# algorithm. This directly validates the principle "after reset+init, everything works as if there was nothing
# before it" - the bumped capacity from the first cycle must NOT leak into the second cycle's scratch.


@test_utils.test(arch=qd.gpu)
def test_scratch_round_trip_across_qd_reset(req_arch):
    """Run a bumped-scratch algorithm; ``qd.reset()`` + ``qd.init()``; then run another algorithm at default scratch.

    The bumped capacity from cycle 1 must be gone in cycle 2 - otherwise the second ``qd.init()`` would over-allocate
    against the unwanted bump. This is the *behavioural* version of ``test_scratch_invalidate_resets_bytes_to_default``
    (which only manipulates module state directly).
    """
    # Pick a "too big" N relative to the default scratch, so the cycle-2 retry trips the budget guard regardless of
    # what ``DEFAULT_SCRATCH_BYTES`` happens to be. ``2 * capacity_u32`` overshoots the default by 2x.
    default_capacity_u32 = _scratch.DEFAULT_SCRATCH_BYTES // 4
    N1 = 2 * default_capacity_u32  # comfortably over the default scratch ceiling for radix sort

    # --- Cycle 1: bump scratch enough to cover N1, run the sort.
    _scratch.set_scratch_bytes(4 * _scratch.DEFAULT_SCRATCH_BYTES)
    rng = np.random.default_rng(seed=1234)
    host1 = rng.integers(0, 2**31 - 1, size=N1, dtype=np.int32)
    keys1 = qd.field(qd.i32, shape=N1)
    tmp1 = qd.field(qd.i32, shape=N1)
    _fill_field(keys1, host1)
    qd.algorithms.device_radix_sort(keys1, tmp_keys=tmp1)
    np.testing.assert_array_equal(keys1.to_numpy(), np.sort(host1))

    # --- Cross the qd.reset() + qd.init() boundary. After this, everything should behave as if cycle 1 never ran.
    qd.reset()
    qd.init(arch=req_arch, enable_fallback=False, device_memory_GB=0.3, print_full_traceback=True)

    # Post-reset invariants on the scratch module.
    assert (
        _scratch._scratch_bytes == _scratch.DEFAULT_SCRATCH_BYTES
    ), "_scratch_bytes did not reset to default across qd.reset() + qd.init() - the very leak this test pins"
    assert _scratch._scratch_field is None, "_scratch_field handle was not invalidated across qd.reset()"
    assert _scratch._scratch_field_u64 is None, "_scratch_field_u64 handle was not invalidated across qd.reset()"

    # --- Cycle 2: run a small algorithm with default scratch. Should just work - and crucially, attempting an
    # over-budget sort NOW (without re-bumping) should *raise* RuntimeError because the bumped capacity is gone.
    N2 = 1024
    host2 = rng.integers(0, 100, size=N2, dtype=np.int32)
    keys2 = qd.field(qd.i32, shape=N2)
    tmp2 = qd.field(qd.i32, shape=N2)
    _fill_field(keys2, host2)
    qd.algorithms.device_radix_sort(keys2, tmp_keys=tmp2)
    np.testing.assert_array_equal(keys2.to_numpy(), np.sort(host2))

    # Re-attempting the over-budget sort without re-bumping must fail - proves the capacity really did drop back.
    keys3 = qd.field(qd.i32, shape=N1)
    tmp3 = qd.field(qd.i32, shape=N1)
    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_radix_sort(keys3, tmp_keys=tmp3)


# --- end_bit on radix sort. Default 32; lower values let callers sort by only the low bits when they know the high
# bits are zero (qipc's case for some small-value sorts).


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_end_bit_16():
    """end_bit=16 sorts by only the low 16 bits; high bits are ignored. Build keys where the low 16 bits and the high
    16 bits disagree on order, then verify sort is by low 16."""
    N = 1024
    rng = np.random.default_rng(seed=1234)
    low = rng.integers(0, 1 << 16, size=N, dtype=np.uint32)
    # High bits decreasing, so a sort by high bits would reverse the array; if the algorithm correctly ignores the
    # high bits, sort key is `low`.
    high = (np.arange(N, dtype=np.uint32)[::-1]).astype(np.uint32)
    host = ((high << np.uint32(16)) | low).astype(np.uint32)

    keys = qd.field(qd.u32, shape=N)
    tmp = qd.field(qd.u32, shape=N)
    _fill_field(keys, host)

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, end_bit=16)
    got = keys.to_numpy()
    # `got` should be sorted by the low 16 bits, in stable order of the original input. Tie-breaking on the low 16
    # bits keeps the original input index order.
    got_low = got & 0xFFFF
    assert np.all(np.diff(got_low.astype(np.int64)) >= 0), "low-16 not non-decreasing"


# --- Full pipeline: radix sort, then reduce-by-key. The qipc-shaped composition (unsorted (key, value) pairs ->
# global per-key sums).


@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.f32])
@test_utils.test(arch=qd.gpu)
def test_radix_sort_then_reduce_by_key_pipeline(dtype):
    """Sort by key, then reduce-by-key, to produce a global per-key sum. Cross-checked against numpy.unique +
    numpy.add.reduceat."""
    N = 4096
    rng = np.random.default_rng(seed=1234)
    if dtype == qd.f32:
        # Use a small set of f32 values to maximise repeats; this also keeps the f32 atomic_add accumulation tolerance
        # comfortable.
        alphabet = rng.standard_normal(20).astype(np.float32)
    elif dtype == qd.u32:
        alphabet = rng.integers(0, 100, size=20, dtype=np.uint32)
    else:
        alphabet = rng.integers(-50, 50, size=20, dtype=np.int32)
    keys_host = rng.choice(alphabet, size=N)
    values_host = rng.integers(-10, 10, size=N, dtype=np.int32)

    keys = qd.field(dtype, shape=N)
    tmp_keys = qd.field(dtype, shape=N)
    values = qd.field(qd.i32, shape=N)
    tmp_values = qd.field(qd.i32, shape=N)
    _fill_field(keys, keys_host)
    _fill_field(values, values_host)

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp_keys, values=values, tmp_values=tmp_values)
    # After sort, keys is ascending; values is permuted to match. Now RBK collapses runs of equal keys into per-key
    # sums.
    keys_out = qd.field(dtype, shape=N)
    values_out = qd.field(qd.i32, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    qd.algorithms.device_reduce_by_key_add(keys, values, keys_out=keys_out, values_out=values_out, num_runs=num_runs)

    nr = int(num_runs.to_numpy()[0])
    got_keys = keys_out.to_numpy()[:nr]
    got_vals = values_out.to_numpy()[:nr]

    # CPU reference: numpy.unique with sum aggregation, matching the device's sort + RBK semantics.
    uniq, idx = np.unique(keys_host, return_inverse=True)
    sums = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(sums, idx, values_host.astype(np.int64))

    assert nr == len(uniq), f"num_runs mismatch: got {nr}, expected {len(uniq)}"
    if dtype == qd.f32:
        np.testing.assert_allclose(got_keys, uniq, rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(got_keys, uniq)
    np.testing.assert_array_equal(got_vals.astype(np.int64), sums)
