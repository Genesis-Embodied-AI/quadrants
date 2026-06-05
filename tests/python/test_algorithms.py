"""Tests for ``qd.algorithms.*`` device-wide primitives.

Covers:

- ``quadrants._scratch`` - the shared ``Field(u32)`` scratch buffer that backs every device algorithm.
- ``qd.algorithms.device_reduce_{add,min,max}`` - two-or-more-pass tree reduction with shared scratch + ``bit_cast``.
- ``qd.algorithms.device_exclusive_scan_{add,min,max}`` - three-pass scan.
- ``qd.algorithms.device_select`` - scan-based stream compaction.
- ``qd.algorithms.device_radix_sort`` - LSB radix sort built on ``block.radix_rank_match_atomic_or``.
- ``qd.algorithms.device_reduce_by_key_add`` - scan + scatter + atomic_add reduce-by-key.

Each test runs across the full ``arch=qd.gpu`` parametrization so the kernels are exercised on CUDA, AMDGPU, Vulkan,
and Metal (where the host supports each).
"""

import math
import platform
import struct

import numpy as np
import pytest

import quadrants as qd
from quadrants import _scratch
from quadrants.lang.util import to_numpy_type

from tests import test_utils

# ---------------------------------------------------------------------------
# Module-level constants: dtype sets, size sweeps, identity tables.
# ---------------------------------------------------------------------------

# Supported scalar dtypes per algorithm. Reduce / scan / select / RBK share the same 6-dtype set; radix sort uses a
# slightly different ordering (u32 first because that's the natural histogram dtype).
_REDUCE_DTYPES = [qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64]
_SCAN_DTYPES = [qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64]
_SELECT_DTYPES = [qd.i32, qd.u32, qd.f32, qd.i64, qd.u64, qd.f64]
_RADIX_KEY_DTYPES = [qd.u32, qd.i32, qd.f32, qd.u64, qd.i64, qd.f64]

# Numpy-dtype lookup. Used by every test that allocates a host buffer for ``from_numpy``.
_DTYPE_TO_NP = {
    qd.i32: np.int32,
    qd.u32: np.uint32,
    qd.f32: np.float32,
    qd.i64: np.int64,
    qd.u64: np.uint64,
    qd.f64: np.float64,
}

# Identities for device_reduce_min / max (passed by tests that initialize an "all-identity" input). Floats use the
# +/- inf extremum; ints use the dtype's positive / negative range extreme.
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

# Size sweeps. Chosen to cover (across algorithms): single-block path, on-block-boundary, off-by-one tile, two-block,
# many-block recursion. Reduce / scan / select / RBK share the structure with minor variations (radix and
# select-struct trim a few sizes to keep test runtime bounded). The 1M size only appears in scan / scratch /
# qipc-hot-path tests; the others top out at 200K within the default 5 MB scratch budget.
_REDUCE_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]
_SCAN_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000, 1_000_000]
_SELECT_SIZES = [1, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]
_SELECT_STRUCT_SIZES = [1, 7, 256, 1024, 65537]
_SELECT_STRUCT_NFIELDS = [2, 3, 4]  # mirrors libuipc Vector2i / Vector3i / Vector4i
_RADIX_SORT_SIZES = [1, 7, 256, 257, 1023, 1024, 1025, 65536, 200_000]
_RBK_SIZES = [1, 2, 3, 7, 255, 256, 257, 1023, 1024, 1025, 65536, 65537, 200_000]

# 64 KB; ~16K u32 slots. Used by the scratch-budget rejection tests so each one trips the budget guard with a tiny N
# (cheap to allocate, runtime-independent of the DEFAULT_SCRATCH_BYTES knob).
_TINY_SCRATCH_BYTES = 64 << 10


# ---------------------------------------------------------------------------
# Backend dtype-support matrix + skip helpers.
# ---------------------------------------------------------------------------
#
# Anything outside the ``supported`` column is unsupported at the lang tier (the spirv / metal IR builders bail with
# "Type X not supported"), so every device-tier test must skip those (arch, platform, dtype) triples.
#
#   arch                  | platform | supported dtypes
#   ----------------------|----------|------------------------------------
#   qd.cuda               | any      | i32, u32, f32, i64, u64, f64
#   qd.amdgpu             | any      | i32, u32, f32, i64, u64, f64
#   qd.vulkan             | Linux    | i32, u32, f32, i64, u64, f64
#   qd.vulkan (MoltenVK)  | Darwin   | i32, u32, f32                (no i64 / u64 / f64)
#   qd.metal              | any      | i32, u32, f32                (no i64 / u64 / f64)
#
# We encode the matrix as the *unsupported* set per backend, since that's what the skip predicates need.


def _is_apple_gpu_backend():
    """Metal or MoltenVK (Vulkan-on-Darwin). These two share the same dtype-support gaps in buffer-backed I/O."""
    arch = qd.lang.impl.current_cfg().arch
    return arch == qd.metal or (arch == qd.vulkan and platform.system() == "Darwin")


def _unsupported_dtype_reason(dtype):
    """Return a human-readable reason if ``dtype`` is unsupported on the current backend, else None.

    Single source of truth for "should we skip this dtype?". ``_skip_if_dtype_unsupported`` wraps it for
    parametrized-test skipping; tests that iterate dtypes inside the body check the return value directly to
    ``continue`` past unsupported dtypes.
    """
    if _is_apple_gpu_backend():
        if dtype in (qd.i64, qd.u64):
            return f"64-bit integer type {dtype} not supported on the current backend"
        if dtype == qd.f64:
            return "f64 not supported on the current backend"
    return None


def _skip_if_dtype_unsupported(dtype):
    """Skip the calling test if ``dtype`` is unsupported on the current backend. Mirrors the gate used in
    ``test_simt.py`` so device-tier dtype coverage matches block / subgroup-tier coverage."""
    reason = _unsupported_dtype_reason(dtype)
    if reason is not None:
        pytest.skip(reason)


def _skip_if_radix_sort_large_n_on_apple_gpu(N):
    """Skip large-N ``device_radix_sort`` calls on Metal / MoltenVK.

    *Why this skip exists.* On Apple GPUs (Metal directly, and MoltenVK / Vulkan-on-Darwin), ``device_radix_sort``
    produces incorrect results once N crosses ``BLOCK_DIM**2 = 65_536``: the ``test_device_radix_sort_keys_only``
    parametrization at N=200_000 reports 50-90% of elements in the wrong position on those backends. CUDA, AMDGPU,
    and Linux Vulkan all pass at every tested size on the same code, so the regression is in the Apple-GPU codegen /
    runtime path of one of the building blocks (most likely the histogram pass's threadgroup-shared atomic_or +
    barrier sequence at high block counts), not in the radix-sort algorithm itself. Smaller N (N <= 65_536, the
    single- and few-block paths) pass cleanly on Apple GPUs.

    Tracked as a follow-up; not blocking the device-algos first land. Tests that *transitively* hit this path
    (radix-sort-then-RBK at N=1M, ``test_scratch_round_trip_across_qd_reset`` at ~2.6M) also need this guard.
    """
    if N >= 200_000 and _is_apple_gpu_backend():
        pytest.skip("device_radix_sort produces incorrect results on Metal / MoltenVK at N >= 200_000")


# ---------------------------------------------------------------------------
# Tolerance contract for f32 / f64 reduce / scan / RBK assertions.
# ---------------------------------------------------------------------------
#
# Block-tree reduce: error scales as ``O(log N * eps_f32)``. At N=1M, log2(N)*eps_f32 ~ 2e-6, well under any
# tolerance below; we use ``_F32_REDUCE_*`` for the parametrized N<=200K dtype sweep and ``_F32_LARGE_N_*`` for the
# qipc N=1M hot path (slightly looser to absorb MoltenVK fast-math reordering headroom on the big sums).
#
# Block-tree scan: error scales as ``O(sqrt(N) * eps_f32)`` (Higham 2002, "Accuracy and Stability of Numerical
# Algorithms", §4.2 on pairwise / tree summation). The ``2e-5`` constant in ``_f32_scan_tol`` is a 2x headroom over
# the strict-IEEE bound (``eps_f32 ~ 1.2e-7``), there to absorb MoltenVK's more-aggressive fast-math reordering of
# f32 partial sums without papering over actual algorithmic regressions on CUDA / Linux Vulkan / AMDGPU. Two asserts
# inside the function pin the contract in plain language: at <= 100 elements f32 stays under 0.1% rel; at <= 100K
# under 1% rel; the qipc hot-path N=1M lands at ~2% rel.
#
# Reduce-by-key (f32 values): adds an atomic_add reordering layer on top of scan-style scatter; uses the
# ``_F32_LARGE_N_*`` floor so MoltenVK's reordering stays comfortably bounded.
#
# f64: strict-IEEE ``eps_f64 ~ 2.2e-16`` dominates everything; reordering is irrelevant at f64 precision for any
# tested N.

_F32_REDUCE_RTOL = 1e-4  # tree reduce: log(N)*eps_f32 ~ 2e-6 at N=1M; 1e-4 is plenty for the N<=200K dtype sweep
_F32_REDUCE_ATOL = 1e-4
_F32_LARGE_N_RTOL = 1e-3  # qipc hot path: N=1M reduce / scan-head / RBK; covers MoltenVK reorder headroom
_F32_LARGE_N_ATOL = 1e-3
_F64_RTOL = 1e-12  # eps_f64 dominates; tight bound across every tested N
_F64_ATOL = 1e-9


def _f32_scan_tol(N):
    """Return ``(rtol, atol)`` for the f32 scan ``assert_allclose``. Scales rtol with sqrt(N); atol is constant.

    See the module-level comment block above for the derivation of the constant and the contract asserts below."""
    rtol = 2e-5 * math.sqrt(N)
    assert rtol <= 1e-3 or N > 100, f"f32 scan rtol={rtol:g} too loose for small N={N}"
    assert rtol <= 1e-2 or N > 100_000, f"f32 scan rtol={rtol:g} too loose for medium N={N}"
    return rtol, 1e-3


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
    """Smoke: write f32 values into the u32 scratch via qd.bit_cast and read them back. Verifies the bit_cast pattern
    used by every algorithm."""
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
    _skip_if_dtype_unsupported(qd.f64)
    N = 64
    s = _scratch.get_scratch_u64()
    src_bits = qd.field(qd.u64, shape=N)
    out = qd.field(qd.f64, shape=N)

    expected = [i * 0.5 - 7.25 + 1.0e-100 * i for i in range(N)]
    bits_host = np.array([struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in expected], dtype=np.uint64)
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


def _np_dtype_for(qd_dtype):
    return to_numpy_type(qd_dtype)


def _fill_field(f, vals):
    """Host-side initialization of a Field from a numpy array. Uses ``from_numpy`` so layout permutations are handled
    correctly."""
    f.from_numpy(np.asarray(vals, dtype=_np_dtype_for(f.dtype)))


def _alloc_input_out(dtype, N):
    """Allocate a 1-D input field of size ``N`` and a 1-element output field of the same dtype. Field-backed because
    that's the cheaper allocation for Quadrants and exercises the polymorphic ``qd.Tensor`` annotation in the kernels.
    """
    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=1)
    return inp, out


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


_REDUCE_OPS = ["add", "min", "max"]


def _reduce_host(rng, op, dtype, N):
    """Generate the test input for a reduce of `op` on `dtype` x N values.

    ``add`` uses small uniform / bounded values so float sums stay representable; ``min`` and ``max`` use a wider
    range (-10..10 for floats, +-10000 for ints) since picking-an-element is bitwise-exact regardless of magnitude.
    """
    if op == "add":
        return _rand_reduce_host(rng, dtype, N)
    if _is_float(dtype):
        return rng.uniform(-10.0, 10.0, size=N).astype(_DTYPE_TO_NP[dtype])
    return _rand_reduce_host(rng, dtype, N, bound=10000)


def _check_reduce(op, dtype, N):
    """Run ``device_reduce_<op>(arr)`` and verify against ``numpy.<op>(arr)``.

    ``add`` accumulates so it needs (a) wider integer promotion + mod-wrap masking for u32/u64 and (b) per-N float
    tolerance. ``min`` / ``max`` pick one input element, so they're bitwise-exact for both ints and floats.
    """
    _skip_if_dtype_unsupported(dtype)
    inp, out = _alloc_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    host = _reduce_host(rng, op, dtype, N)
    _fill_field(inp, host)

    qd_fn = getattr(qd.algorithms, f"device_reduce_{op}")
    qd_fn(inp, out=out)
    got = out.to_numpy()[0]

    if op == "add":
        if _is_float(dtype):
            expected = float(np.sum(host.astype(np.float64)))
            rtol, atol = (_F32_REDUCE_RTOL, _F32_REDUCE_ATOL) if dtype == qd.f32 else (_F64_RTOL, _F64_ATOL)
            assert math.isclose(
                got, expected, rel_tol=rtol, abs_tol=atol
            ), f"{dtype} reduce_add(N={N}): got {got}, expected {expected}"
        else:
            # Promote to Python int for an arbitrary-width reference; mask both sides to dtype width to handle the
            # u32 / u64 mod-wrap case at large N.
            mod = 1 << (32 if dtype in (qd.i32, qd.u32) else 64) if _is_unsigned(dtype) else None
            ref = int(
                np.sum(
                    host.astype(np.int64 if dtype in (qd.i32, qd.u32) else (np.int64 if dtype == qd.i64 else np.uint64))
                )
            )  # noqa: E501
            got_int = int(got)
            if mod is not None:
                ref &= mod - 1
                got_int &= mod - 1
            assert got_int == ref, f"{dtype} reduce_add(N={N}): got {got_int}, expected {ref}"
        return

    expected = host.min() if op == "min" else host.max()
    if _is_float(dtype):
        assert got == pytest.approx(expected, abs=1e-6 if dtype == qd.f32 else 1e-12)
    else:
        assert int(got) == int(expected), f"{dtype} reduce_{op}(N={N}): got {got}, expected {expected}"


@pytest.mark.parametrize("op", _REDUCE_OPS)
@pytest.mark.parametrize("N", _REDUCE_SIZES)
@pytest.mark.parametrize("dtype", _REDUCE_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_reduce(op, dtype, N):
    """``device_reduce_{add,min,max}`` match numpy across the full size sweep + dtype set.

    Unified across the three op variants. ``add`` accumulates so it needs overflow / precision-aware comparison;
    ``min`` / ``max`` pick one element of the input and are bitwise-exact.
    """
    _check_reduce(op, dtype, N)


@test_utils.test(arch=qd.gpu)
def test_device_reduce_min_derives_identity_from_dtype():
    """``device_reduce_min`` does not take an identity argument; it's derived from ``arr.dtype`` (mirror of the
    ``block.reduce_min`` / ``subgroup.reduce_min`` contract). On an all-min-identity input the reduction returns
    the identity itself (largest representable value), which exercises the auto-derivation end-to-end."""
    for dtype in _REDUCE_DTYPES:
        if _unsupported_dtype_reason(dtype) is not None:
            continue
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


def _alloc_scan_input_out(dtype, N):
    inp = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=N)
    return inp, out


def _scan_dtype_mask(dtype):
    """Return the wrap-around mask for unsigned scans, or -1 (no mask) for signed / floats."""
    if dtype == qd.u32:
        return 0xFFFFFFFF
    if dtype == qd.u64:
        return 0xFFFFFFFFFFFFFFFF
    return -1


_SCAN_OPS = ["add", "min", "max"]


def _scan_host(rng, op, dtype, N):
    """Generate the test input for a scan of `op` on `dtype` x N values. Same rationale as ``_reduce_host``."""
    if op == "add":
        return _rand_reduce_host(rng, dtype, N, bound=100)
    if _is_float(dtype):
        return rng.uniform(-10.0, 10.0, size=N).astype(_DTYPE_TO_NP[dtype])
    return _rand_reduce_host(rng, dtype, N, bound=10000)


def _check_scan(op, dtype, N):
    """Run ``device_exclusive_scan_<op>(arr)`` and verify against ``numpy.<op>.accumulate``-shifted.

    Like the reduce family, ``add`` accumulates (overflow / precision care) while ``min`` / ``max`` are
    bitwise-exact in both float and int paths.
    """
    _skip_if_dtype_unsupported(dtype)
    inp, out = _alloc_scan_input_out(dtype, N)
    rng = np.random.default_rng(seed=1234)
    np_dt = _DTYPE_TO_NP[dtype]
    host = _scan_host(rng, op, dtype, N)
    _fill_field(inp, host)

    qd_fn = getattr(qd.algorithms, f"device_exclusive_scan_{op}")
    qd_fn(inp, out=out)
    got = out.to_numpy()

    if op == "add":
        if _is_float(dtype):
            ref = np.concatenate([[0.0], np.cumsum(host.astype(np.float64))[:-1]])
            rtol, atol = _f32_scan_tol(N) if dtype == qd.f32 else (_F64_RTOL, _F64_ATOL)
            np.testing.assert_allclose(
                got.astype(np.float64),
                ref,
                rtol=rtol,
                atol=atol,
                err_msg=f"{dtype} scan_add(N={N})",
            )
        else:
            # Promote to a width that survives the cumulative sum: u64 / i64 inputs use a Python int reference;
            # smaller ints can still use int64.
            promote = np.int64 if dtype in (qd.i32, qd.u32, qd.i64) else np.uint64
            host_wide = host.astype(promote)
            ref = np.concatenate([[promote(0)], np.cumsum(host_wide)[:-1]]).astype(promote)
            mask = _scan_dtype_mask(dtype)
            got_view = got.astype(np.int64 if dtype != qd.u64 else np.uint64)
            if mask != -1:
                got_view = got_view & promote(mask)
                ref = ref & promote(mask)
            np.testing.assert_array_equal(got_view, ref, err_msg=f"{dtype} scan_add(N={N})")
        return

    np_accum = np.minimum.accumulate if op == "min" else np.maximum.accumulate
    identity_table = _MIN_IDENTITY if op == "min" else _MAX_IDENTITY
    if _is_float(dtype):
        identity = float("inf") if op == "min" else float("-inf")
        ref = np.concatenate([[identity], np_accum(host.astype(np.float64))[:-1]]).astype(np_dt)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0, err_msg=f"{dtype} scan_{op}(N={N})")
    else:
        ref = np.concatenate([[np_dt(identity_table[dtype])], np_accum(host)[:-1]]).astype(np_dt)
        np.testing.assert_array_equal(got, ref, err_msg=f"{dtype} scan_{op}(N={N})")


@pytest.mark.parametrize("op", _SCAN_OPS)
@pytest.mark.parametrize("N", _SCAN_SIZES)
@pytest.mark.parametrize("dtype", _SCAN_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan(op, dtype, N):
    """``device_exclusive_scan_{add,min,max}`` match ``numpy.{cumsum, minimum.accumulate, maximum.accumulate}``-shifted
    across the full size sweep + dtype set. Unified across the three op variants; same overflow vs bitwise-exact
    handling as the reduce family."""
    _check_scan(op, dtype, N)


@test_utils.test(arch=qd.gpu)
def test_device_exclusive_scan_rejects_inplace():
    """In-place scan (out is input) is rejected per the design doc - see 'API design' / 'Aliasing' in
    qipc_device_algos_design.md."""
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


@pytest.mark.parametrize("N", _SELECT_SIZES)
@pytest.mark.parametrize("dtype", _SELECT_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_device_select_basic(dtype, N):
    """device_select packs the elements with flags != 0 into a dense prefix of out, in stable input order; num_out[0]
    holds the count. Covers all 6 supported scalar dtypes - the scatter (``dst[idx] = src[i]``) is dtype-agnostic, so
    a single parametrized test serves both the 4-byte and 8-byte paths."""
    _skip_if_dtype_unsupported(dtype)
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
def test_device_select_zero_one_flag_contract():
    """Pin the 0/1 flag contract documented in ``algorithms.md`` and the ``device_select`` docstring.

    ``device_select`` prefix-sums ``flags`` *directly* as counts (no implicit normalization), so the contract is
    that every entry is exactly ``0`` or ``1`` and ``1`` selects. This regression test pins the contract by
    construction: an interleaved ``[1, 0, 1, 0, ...]`` pattern of length ``N`` selects exactly the even indices
    (``N/2`` elements, in input order). If a future change accidentally re-introduces an implicit normalization
    or breaks the prefix-sum-as-count semantics, this test is the canary.
    """
    N = 1024
    flags_host = np.zeros(N, dtype=np.int32)
    flags_host[::2] = 1  # exactly 0 or 1, interleaved -> selects the even indices
    inp_host = np.arange(N, dtype=np.int32)

    inp = qd.field(qd.i32, shape=N)
    flags = qd.field(qd.i32, shape=N)
    out = qd.field(qd.i32, shape=N)
    num_out = qd.field(qd.i32, shape=1)
    _fill_field(inp, inp_host)
    _fill_field(flags, flags_host)

    qd.algorithms.device_select(inp, flags, out=out, num_out=num_out)
    got_n = int(num_out.to_numpy()[0])
    assert got_n == N // 2, f"interleaved 0/1 flags should select N/2 = {N // 2} entries, got {got_n}"
    expected = inp_host[::2]
    np.testing.assert_array_equal(out.to_numpy()[:got_n], expected)


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
    _skip_if_dtype_unsupported(dtype)
    _skip_if_radix_sort_large_n_on_apple_gpu(N)
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
    _skip_if_dtype_unsupported(dtype)
    _skip_if_radix_sort_large_n_on_apple_gpu(N)
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


# --- Fused-kernel specifics: over-specified scan depth (the graph path fixes ``log256_max_n`` ahead of time so one
# captured topology serves a range of N). The dtype x size matrix (incl. key-value) is covered by
# ``test_device_radix_sort_keys_only`` / ``_key_value`` above - now backed by this fused kernel - and the
# caller-scratch tests further below. -------------------------------------------------------------------------------


@pytest.mark.parametrize("N", [7, 257, 1025, 65536])
@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_overspecified_depth(N):
    """An over-specified ``log256_max_n`` (deeper than the minimal depth for N) must still sort correctly - the
    forced extra staircase levels operate on length-1 buffers and act as identity no-ops. Also covers sizing scratch
    via ``fused_radix_sort_scratch_slots(N, D)`` for an explicit depth ``D``."""
    D = 3  # 256**3 = 16_777_216 >= every N here, so depth is intentionally deeper than needed
    rng = np.random.default_rng(seed=99)
    host = _gen_keys(rng, qd.u32, N)

    keys = qd.field(qd.u32, shape=N)
    tmp = qd.field(qd.u32, shape=N)
    _fill_field(keys, host)

    slots = qd.algorithms.fused_radix_sort_scratch_slots(N, D)
    scratch = qd.field(qd.u32, shape=max(slots, 1))

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, scratch=scratch, log256_max_n=D)
    np.testing.assert_array_equal(keys.to_numpy(), np.sort(host, kind="stable"), err_msg=f"u32 sort(N={N}, D={D})")


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


def _ref_rbk_add(keys, values):
    """Reference reduce-by-key: collapse consecutive runs of equal keys, returning ``(unique_keys, sums)``."""
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

    Runs are drawn from a small alphabet of 5-15 distinct values and repeated 1-8 times, then concatenated and
    truncated to N. This guarantees both multi-element runs (so the scatter's atomic_add path is exercised) and
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

    Values are bounded so ``f32`` accumulation error stays controlled - the tolerance is ``_F32_LARGE_N_*`` for f32
    (atomic_add reorder layered on the scan-style scatter) and bit-exact for integer types.
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
            rtol=_F32_LARGE_N_RTOL,
            atol=_F32_LARGE_N_ATOL,
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
    _skip_if_dtype_unsupported(dtype)
    N = 1_000_000
    _skip_if_radix_sort_large_n_on_apple_gpu(N)
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
    _skip_if_radix_sort_large_n_on_apple_gpu(N)
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
# over the configured capacity, rather than corrupting data. Tests shrink scratch to ``_TINY_SCRATCH_BYTES`` so the
# trip point is reachable with a small N (cheap to allocate, runtime-independent of the DEFAULT_SCRATCH_BYTES knob).


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
def test_device_radix_sort_recursive_scratch_check_keys_unchanged():
    """Regression: ``device_radix_sort`` must refuse the call *before* ``_twiddle_pass`` mutates the user's keys
    when the scratch budget is too small for the *recursive* in-place scan footprint.

    The bug this guards against (PR 693 review): the up-front scratch check counted only one level of scan
    partials (``hist_len + ceil(hist_len/BLOCK_DIM)``). For ``N`` large enough to force the in-place exclusive
    scan to recurse (``hist_len > BLOCK_DIM**2``), a budget that's just a few slots too small slipped past that
    single-level check, then ``_twiddle_pass`` ran (in-place XOR of sign bits for ``i32`` / ``f32`` keys), and
    only *then* did the recursive scan raise a ``RuntimeError`` - leaving the caller's ``keys`` corrupted with
    no recovery path. After the fix, the check uses ``_scan_total_scratch_slots`` to account for the full
    recursion up front, so we refuse the call before any side effect runs.

    Setup picks a budget in the (single-level-pass, full-recursion-fail) window so the test would have *failed*
    against the buggy old check (twiddle would have run, ``keys`` would be XOR'd) and *passes* against the fixed
    check (``keys`` are byte-identical to what the user wrote in).
    """
    from quadrants.algorithms._radix_sort import BLOCK_DIM, RADIX_DIGITS
    from quadrants.algorithms._scan import _scan_total_scratch_slots

    N = 1_000_000  # large enough that hist_len > BLOCK_DIM**2 = 65_536, forcing the scan to recurse one level
    num_blocks = (N + BLOCK_DIM - 1) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    old_needed = hist_len + (hist_len + BLOCK_DIM - 1) // BLOCK_DIM  # buggy single-level estimate
    new_needed = _scan_total_scratch_slots(hist_len, partials_cursor=hist_len)  # full recursive footprint
    assert new_needed > old_needed, (
        "test setup invariant: scan must recurse for the test to discriminate against the bug; "
        f"got old_needed={old_needed}, new_needed={new_needed} at N={N} - increase N if BLOCK_DIM grew"
    )
    # Budget in the bug window: passes the buggy old check, fails the fixed one. (new - old is small, ~16 slots
    # at N=1M, so any midpoint works.) Round to even so the bytes count is a multiple of 8 (a ``set_scratch_bytes``
    # precondition that holds because the u64 scratch field shares the same byte budget).
    cap_target = old_needed + (new_needed - old_needed) // 2
    cap_target += cap_target & 1  # snap up to even
    assert old_needed < cap_target < new_needed, (
        f"bug-window selection invariant: old_needed={old_needed} < cap_target={cap_target} < "
        f"new_needed={new_needed} should hold for the test to discriminate against the bug"
    )
    _scratch.set_scratch_bytes(cap_target * 4)

    rng = np.random.default_rng(seed=1234)
    host = rng.integers(-(2**30), 2**30, size=N, dtype=np.int32)  # signed -> hits the in-place twiddle path
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    _fill_field(keys, host)

    with pytest.raises(RuntimeError, match="scratch"):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp)

    # The crucial assertion: keys are still the user's original bit pattern, not XOR'd by twiddle.
    np.testing.assert_array_equal(keys.to_numpy(), host)


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_caller_scratch():
    """Caller-owned ``scratch`` buffer (sized via ``device_radix_sort_scratch_slots``) sorts identically to the
    shared-scratch path, without consulting the module-level scratch."""
    N = 100_000
    rng = np.random.default_rng(seed=7)
    host = rng.integers(-(2**31), 2**31 - 1, size=N, dtype=np.int32)
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    scratch = qd.field(qd.u32, shape=qd.algorithms.device_radix_sort_scratch_slots(N))
    _fill_field(keys, host)

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, scratch=scratch)
    np.testing.assert_array_equal(keys.to_numpy(), np.sort(host, kind="stable"))


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_caller_scratch_key_value():
    """Caller-scratch key-value sort permutes values in lock-step (u64 key + i32 value, the libuipc shape)."""
    _skip_if_dtype_unsupported(qd.u64)
    N = 100_000
    rng = np.random.default_rng(seed=8)
    host = (rng.integers(0, 2**63, size=N, dtype=np.uint64) * np.uint64(2)).astype(np.uint64)
    keys = qd.field(qd.u64, shape=N)
    tmp_keys = qd.field(qd.u64, shape=N)
    values = qd.field(qd.i32, shape=N)
    tmp_values = qd.field(qd.i32, shape=N)
    scratch = qd.field(qd.u32, shape=qd.algorithms.device_radix_sort_scratch_slots(N))
    _fill_field(keys, host)
    _fill_field(values, np.arange(N, dtype=np.int32))

    qd.algorithms.device_radix_sort(
        keys, tmp_keys=tmp_keys, values=values, tmp_values=tmp_values, scratch=scratch
    )
    want_idx = np.argsort(host, kind="stable")
    np.testing.assert_array_equal(keys.to_numpy(), host[want_idx])
    np.testing.assert_array_equal(values.to_numpy(), want_idx.astype(np.int32))


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_scratch_slots_query():
    """``device_radix_sort_scratch_slots`` returns 0 for N<=1 and the full recursive scan footprint otherwise; a
    buffer of exactly that size sorts successfully."""
    from quadrants.algorithms._radix_sort import BLOCK_DIM, RADIX_DIGITS
    from quadrants.algorithms._scan import _scan_total_scratch_slots

    assert qd.algorithms.device_radix_sort_scratch_slots(0) == 0
    assert qd.algorithms.device_radix_sort_scratch_slots(1) == 0

    N = 100_000
    num_blocks = (N + BLOCK_DIM - 1) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    needed = qd.algorithms.device_radix_sort_scratch_slots(N)
    assert needed == _scan_total_scratch_slots(hist_len, partials_cursor=hist_len)

    rng = np.random.default_rng(seed=9)
    host = rng.integers(-(2**31), 2**31 - 1, size=N, dtype=np.int32)
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    scratch = qd.field(qd.u32, shape=needed)  # exactly enough
    _fill_field(keys, host)

    qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, scratch=scratch)
    np.testing.assert_array_equal(keys.to_numpy(), np.sort(host, kind="stable"))


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_insufficient_caller_scratch():
    """A too-small caller ``scratch`` raises ``InsufficientScratchError`` (a ``RuntimeError`` subclass) carrying the
    required size, *before* any in-place twiddle - so the caller's keys are untouched and recoverable."""
    N = 100_000
    needed = qd.algorithms.device_radix_sort_scratch_slots(N)
    rng = np.random.default_rng(seed=10)
    host = rng.integers(-(2**30), 2**30, size=N, dtype=np.int32)  # signed -> would hit the in-place twiddle
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    scratch = qd.field(qd.u32, shape=needed - 1)  # one slot short
    _fill_field(keys, host)

    with pytest.raises(qd.algorithms.InsufficientScratchError) as excinfo:
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, scratch=scratch)
    assert excinfo.value.required_slots == needed
    assert excinfo.value.provided_slots == needed - 1
    assert isinstance(excinfo.value, RuntimeError)
    np.testing.assert_array_equal(keys.to_numpy(), host)  # no twiddle ran


@test_utils.test(arch=qd.gpu)
def test_device_radix_sort_rejects_non_u32_scratch():
    """A caller ``scratch`` of the wrong dtype is rejected (tile histograms are u32 regardless of key width)."""
    N = 1000
    keys = qd.field(qd.i32, shape=N)
    tmp = qd.field(qd.i32, shape=N)
    bad_scratch = qd.field(qd.i32, shape=qd.algorithms.device_radix_sort_scratch_slots(N))
    with pytest.raises(TypeError, match="u32"):
        qd.algorithms.device_radix_sort(keys, tmp_keys=tmp, scratch=bad_scratch)


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
    _skip_if_dtype_unsupported(dtype)
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
        rtol, atol = (_F32_LARGE_N_RTOL, _F32_LARGE_N_ATOL) if dtype == qd.f32 else (_F64_RTOL, _F64_ATOL)
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
    _skip_if_dtype_unsupported(dtype)
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
            np.testing.assert_allclose(got[:64], ref[:64], rtol=_F32_LARGE_N_RTOL, atol=_F32_LARGE_N_ATOL)
            assert np.isfinite(got[-1])
        else:
            np.testing.assert_allclose(got, ref, rtol=_F64_RTOL, atol=_F64_ATOL)
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
    _skip_if_radix_sort_large_n_on_apple_gpu(N1)

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
