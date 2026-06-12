"""Tests for ``qd.algorithms.*`` device-wide primitives.

Covers:

- ``qd.algorithms.reduce_{add,min,max}`` - composable tree reduction emitted inside a user ``@qd.kernel``.
- ``qd.algorithms.exclusive_scan_{add,min,max}`` - composable three-pass scan.
- ``qd.algorithms.select`` - composable scan-based stream compaction.
- ``qd.algorithms.sort`` - composable LSB radix sort built on ``block.radix_rank_match_atomic_or``.
- ``qd.algorithms.reduce_by_key_add`` - composable scan + scatter + atomic_add reduce-by-key.

Each test runs across the full ``arch=qd.gpu`` parametrization so the kernels are exercised on CUDA, AMDGPU, Vulkan,
and Metal (where the host supports each).
"""

import math
import platform
import struct

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.util import to_numpy_type

from tests import test_utils

# ---------------------------------------------------------------------------
# Scratch-width helper. ``_FOURBYTE_DTYPES`` selects the u32-vs-u64 scratch element width used by the composable
# composition tests (4-byte dtypes stage through u32 scratch; 8-byte dtypes through u64).
# ---------------------------------------------------------------------------

_FOURBYTE_DTYPES = (qd.i32, qd.u32, qd.f32)


# ---------------------------------------------------------------------------
# Module-level constants: dtype set + numpy-dtype / identity lookup tables used by the kept composition tests.
# ---------------------------------------------------------------------------

# Radix-sort key dtypes (u32 first because that's the natural histogram dtype).
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


# Identities for reduce_min / max (passed by tests that initialize an "all-identity" input). Floats use the
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
def test_bit_cast_round_trips_f32():
    """Smoke: write f32 values into a u32 buffer via qd.bit_cast and read them back. Verifies the bit_cast pattern
    every device algorithm uses to stage 4-byte values through u32 scratch."""
    s = qd.field(qd.u32, shape=64)
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
def test_bit_cast_round_trips_f64():
    """Smoke: feed exact-known f64 bit patterns into the kernel, bit_cast through a u64 buffer, read back. Mirrors
    ``test_bit_cast_round_trips_f32`` for the 8-byte-dtype path used by 64-bit ``reduce_*`` / scan.

    We push the host-computed bit pattern in via a u64 source field rather than arithmetic on f64 literals to dodge
    kernel-side fp-contract / FMA-reassociation that can offset the result by 1 ulp from the host-side value.
    """
    _skip_if_dtype_unsupported(qd.f64)
    N = 64
    s = qd.field(qd.u64, shape=N)
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


@pytest.mark.parametrize("op", _REDUCE_OPS)
@pytest.mark.parametrize("N", [1, 255, 256, 257, 1024, 65537])
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_reduce_composition(op, dtype, N):
    """``reduce_{add,min,max}`` compose at the **top level** of a user ``@qd.kernel`` with a device-resident
    count (``count[0]``) and a compile-time ``LOG256_MAX_N``, matching the host ``reduce_*`` entries. This pins the
    graph-composable path qipc uses: the count flows as a device ``Expr`` while ``LOG256_MAX_N`` fixes the launch
    topology.
    """
    _skip_if_dtype_unsupported(dtype)
    from quadrants.algorithms._reduce import _reduce_depth_for_n

    log256_max_n = _reduce_depth_for_n(N)
    rng = np.random.default_rng(seed=7)
    host = _reduce_host(rng, op, dtype, N)

    arr = qd.field(dtype, shape=N)
    out = qd.field(dtype, shape=1)
    sdt = qd.u32 if dtype in _FOURBYTE_DTYPES else qd.u64
    scratch = qd.field(sdt, shape=max(qd.algorithms.reduce_scratch_slots(N, log256_max_n), 1))
    count = qd.field(qd.i32, shape=1)
    _fill_field(arr, host)
    count.from_numpy(np.asarray([N], dtype=np.int32))

    if op == "add":

        @qd.kernel
        def run(DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
            qd.algorithms.reduce_add(arr, out, scratch, count[0], DTYPE, LOG256_MAX_N)

    elif op == "min":

        @qd.kernel
        def run(DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
            qd.algorithms.reduce_min(arr, out, scratch, count[0], DTYPE, LOG256_MAX_N)

    else:

        @qd.kernel
        def run(DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
            qd.algorithms.reduce_max(arr, out, scratch, count[0], DTYPE, LOG256_MAX_N)

    run(dtype, log256_max_n)
    got = out.to_numpy()[0]

    if op == "add":
        if _is_float(dtype):
            expected = float(np.sum(host.astype(np.float64)))
            rtol, atol = (_F32_REDUCE_RTOL, _F32_REDUCE_ATOL) if dtype == qd.f32 else (_F64_RTOL, _F64_ATOL)
            assert math.isclose(got, expected, rel_tol=rtol, abs_tol=atol), f"{dtype} N={N}: {got} vs {expected}"
        else:
            mask = (1 << 64) - 1 if dtype == qd.u64 else None
            ref = int(np.sum(host.astype(np.uint64 if dtype == qd.u64 else np.int64)))
            got_int = int(got)
            if mask is not None:
                ref &= mask
                got_int &= mask
            assert got_int == ref, f"{dtype} N={N}: {got_int} vs {ref}"
    else:
        expected = host.min() if op == "min" else host.max()
        if _is_float(dtype):
            assert got == pytest.approx(expected, abs=1e-6 if dtype == qd.f32 else 1e-12)
        else:
            assert int(got) == int(expected), f"{dtype} reduce_{op}(N={N}): {got} vs {expected}"


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


def _verify_scan(got, op, dtype, N, host):
    """Assert ``got`` matches ``numpy.<op>.accumulate``-shifted of ``host``.

    Shared by the ``exclusive_scan_{op}`` composition test. Like the
    reduce family, ``add`` accumulates (overflow / precision care) while ``min`` / ``max`` are bitwise-exact in both
    float and int paths.
    """
    np_dt = _DTYPE_TO_NP[dtype]
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
@pytest.mark.parametrize("N", [1, 255, 256, 257, 1024, 65537])
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_exclusive_scan_composition(op, dtype, N):
    """``exclusive_scan_{add,min,max}`` compose at the **top level** of a user ``@qd.kernel`` with a
    device-resident count (``count[0]``) and a compile-time ``LOG256_MAX_N``, matching the host ``exclusive_scan_*``
    entries. This pins the graph-composable path qipc uses: the count flows as a device ``Expr`` while
    ``LOG256_MAX_N`` fixes the launch topology (out-of-place ``arr`` -> ``out`` with a caller-sized partials staircase
    in ``scratch``)."""
    _skip_if_dtype_unsupported(dtype)
    from quadrants.algorithms._reduce import _reduce_depth_for_n

    log256_max_n = _reduce_depth_for_n(N)
    rng = np.random.default_rng(seed=1234)
    host = _scan_host(rng, op, dtype, N)

    arr, out = _alloc_scan_input_out(dtype, N)
    sdt = qd.u32 if dtype in _FOURBYTE_DTYPES else qd.u64
    scratch = qd.field(sdt, shape=max(qd.algorithms.exclusive_scan_scratch_slots(N, log256_max_n), 1))
    count = qd.field(qd.i32, shape=1)
    _fill_field(arr, host)
    count.from_numpy(np.asarray([N], dtype=np.int32))

    if op == "add":

        @qd.kernel
        def run(DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
            qd.algorithms.exclusive_scan_add(arr, out, scratch, count[0], DTYPE, LOG256_MAX_N)

    elif op == "min":

        @qd.kernel
        def run(DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
            qd.algorithms.exclusive_scan_min(arr, out, scratch, count[0], DTYPE, LOG256_MAX_N)

    else:

        @qd.kernel
        def run(DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
            qd.algorithms.exclusive_scan_max(arr, out, scratch, count[0], DTYPE, LOG256_MAX_N)

    run(dtype, log256_max_n)
    _verify_scan(out.to_numpy(), op, dtype, N, host)


# ---------------------------------------------------------------------------
# Device select / compact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 255, 256, 257, 1024, 65537])
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_select_composition(dtype, N):
    """``select`` composes at the **top level** of a user ``@qd.kernel`` with a device-resident count
    (``count[0]``) and a compile-time ``LOG256_MAX_N``, matching the host ``select`` entry. This pins the
    graph-composable compaction path qipc uses: the count flows as a device ``Expr`` while ``LOG256_MAX_N`` fixes the
    launch topology (the scan-of-flags staircase + scatter + count all emit inside one kernel)."""
    _skip_if_dtype_unsupported(dtype)
    from quadrants.algorithms._reduce import _reduce_depth_for_n

    log256_max_n = _reduce_depth_for_n(N)
    rng = np.random.default_rng(seed=1234)
    np_dt = _DTYPE_TO_NP[dtype]
    if dtype in (qd.f32, qd.f64):
        host = rng.uniform(-1.0, 1.0, size=N).astype(np_dt)
    elif dtype in (qd.u32, qd.u64):
        host = rng.integers(0, 10000, size=N, dtype=np_dt)
    else:
        host = rng.integers(-10000, 10000, size=N, dtype=np_dt)
    flags_host = (rng.random(N) < 0.3).astype(np.int32)
    expected = host[flags_host == 1]
    expected_n = int(flags_host.sum())

    arr = qd.field(dtype, shape=N)
    flags = qd.field(qd.i32, shape=N)
    out = qd.field(dtype, shape=max(N, 1))
    num_out = qd.field(qd.i32, shape=1)
    scratch = qd.field(qd.u32, shape=max(qd.algorithms.select_scratch_slots(N), 1))
    count = qd.field(qd.i32, shape=1)
    _fill_field(arr, host)
    _fill_field(flags, flags_host)
    count.from_numpy(np.asarray([N], dtype=np.int32))

    @qd.kernel
    def run(LOG256_MAX_N: qd.template()):
        qd.algorithms.select(arr, flags, out, num_out, scratch, count[0], LOG256_MAX_N)

    run(log256_max_n)
    got_n = int(num_out.to_numpy()[0])
    assert got_n == expected_n, f"{dtype} N={N}: got count {got_n}, expected {expected_n}"
    np.testing.assert_array_equal(out.to_numpy()[:got_n], expected, err_msg=f"{dtype} select(N={N})")


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


@pytest.mark.parametrize("N", [257, 1024, 65536])
@pytest.mark.parametrize("dtype", _RADIX_KEY_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_sort_composition(dtype, N):
    """``sort`` composes at the **top level** of a user ``@qd.kernel`` with a device-resident 0-d count
    (read as ``n[()]``) and compile-time ``KEY_DTYPE`` / ``HAS_VALUES`` / ``END_BIT`` / ``LOG256_MAX_N`` - the exact
    graph-composable contract qipc chains inside its LBVH / broadphase pipelines (see ``sort`` docstring).
    Pins the public func against the reference ``numpy`` argsort: keys come back ascending and the ``u32`` payload
    follows the stable argsort. The reduce / scan / select / reduce-by-key families have matching composition tests."""
    _skip_if_dtype_unsupported(dtype)
    from quadrants.algorithms._radix_sort import _min_log256_for_n

    end_bit = 32 if dtype in _FOURBYTE_DTYPES else 64
    log256_max_n = _min_log256_for_n(N)
    rng = np.random.default_rng(seed=1357)
    host = _gen_keys(rng, dtype, N)

    key_nd = qd.types.ndarray(dtype, ndim=1)
    u32_nd = qd.types.ndarray(qd.u32, ndim=1)
    i32_0d = qd.types.ndarray(qd.i32, ndim=0)

    @qd.kernel
    def run(
        keys: key_nd,
        tmp_keys: key_nd,
        values: u32_nd,
        tmp_values: u32_nd,
        scratch: u32_nd,
        n: i32_0d,
    ):
        # Compile-time params (dtype / end_bit / depth) are captured Python constants, exactly as qipc's graph sort
        # bakes ``qd.u32, True, SORT_END_BIT, SORT_LOG256_MAX_N`` into ``sort``.
        qd.algorithms.sort(keys, tmp_keys, values, tmp_values, scratch, n, dtype, True, end_bit, log256_max_n)

    keys = qd.ndarray(dtype, shape=(N,))
    tmp_keys = qd.ndarray(dtype, shape=(N,))
    values = qd.ndarray(qd.u32, shape=(N,))
    tmp_values = qd.ndarray(qd.u32, shape=(N,))
    scratch = qd.ndarray(qd.u32, shape=(max(qd.algorithms.sort_scratch_slots(N, log256_max_n), 1),))
    n_dev = qd.ndarray(qd.i32, shape=())
    keys.from_numpy(host)
    values.from_numpy(np.arange(N, dtype=np.uint32))
    n_dev.fill(N)

    run(keys, tmp_keys, values, tmp_values, scratch, n_dev)

    want_idx = np.argsort(host, kind="stable")
    np.testing.assert_array_equal(keys.to_numpy(), host[want_idx], err_msg=f"{dtype} keys(N={N})")
    np.testing.assert_array_equal(values.to_numpy(), want_idx.astype(np.uint32), err_msg=f"{dtype} values(N={N})")


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


@pytest.mark.parametrize("N", [1, 255, 256, 257, 1024, 65537])
@pytest.mark.parametrize("key_dtype", [qd.i32, qd.f32])
@pytest.mark.parametrize("val_dtype", [qd.i32, qd.f32])
@test_utils.test(arch=qd.gpu)
def test_reduce_by_key_add_composition(key_dtype, val_dtype, N):
    """``reduce_by_key_add`` composes at the **top level** of a user ``@qd.kernel`` with a device-resident count
    (``count[0]``), a compile-time ``LOG256_MAX_N``, and the values dtype as a template (needed only for the zero-init
    of ``values_out``). Pins the graph-composable reduce-by-key path: count flows as a device ``Expr`` while
    ``LOG256_MAX_N`` fixes the launch topology (head flags + in-place scan + zero + scatter + count all emit inside one
    kernel)."""
    from quadrants.algorithms._reduce import _reduce_depth_for_n

    log256_max_n = _reduce_depth_for_n(N)
    rng = np.random.default_rng(seed=1234)
    keys_host = _gen_run_keys(rng, key_dtype, N)
    val_np = to_numpy_type(val_dtype)
    if val_dtype == qd.f32:
        values_host = rng.uniform(-1.0, 1.0, size=N).astype(val_np)
    else:
        values_host = rng.integers(-100, 100, size=N, dtype=val_np)

    keys_in = qd.field(key_dtype, shape=N)
    values_in = qd.field(val_dtype, shape=N)
    keys_out = qd.field(key_dtype, shape=N)
    values_out = qd.field(val_dtype, shape=N)
    num_runs = qd.field(qd.i32, shape=1)
    scratch = qd.field(qd.u32, shape=max(qd.algorithms.reduce_by_key_scratch_slots(N), 1))
    count = qd.field(qd.i32, shape=1)
    _fill_field(keys_in, keys_host)
    _fill_field(values_in, values_host)
    count.from_numpy(np.asarray([N], dtype=np.int32))

    @qd.kernel
    def run(VALUE_DTYPE: qd.template(), LOG256_MAX_N: qd.template()):
        qd.algorithms.reduce_by_key_add(
            keys_in, values_in, keys_out, values_out, num_runs, scratch, count[0], VALUE_DTYPE, LOG256_MAX_N
        )

    run(val_dtype, log256_max_n)
    nr = int(num_runs.to_numpy()[0])
    want_keys, want_vals = _ref_rbk_add(keys_host, values_host)

    assert nr == len(want_keys), f"{key_dtype}/{val_dtype} N={N}: num_runs {nr} vs {len(want_keys)}"
    np.testing.assert_array_equal(keys_out.to_numpy()[:nr], want_keys, err_msg=f"{key_dtype}/{val_dtype} N={N}: keys")
    if val_dtype == qd.f32:
        np.testing.assert_allclose(
            values_out.to_numpy()[:nr],
            want_vals,
            rtol=_F32_LARGE_N_RTOL,
            atol=_F32_LARGE_N_ATOL,
            err_msg=f"{key_dtype}/{val_dtype} N={N}: values",
        )
    else:
        np.testing.assert_array_equal(
            values_out.to_numpy()[:nr], want_vals, err_msg=f"{key_dtype}/{val_dtype} N={N}: values"
        )


# ---------------------------------------------------------------------------
# Deprecated surfaces
# ---------------------------------------------------------------------------


# --- Deprecation warnings on the legacy executor / parallel_sort surfaces. We added the warnings; assert they
# actually fire so an accidental rebase that drops them is caught.


@test_utils.test(arch=qd.gpu)
def test_prefix_sum_executor_emits_deprecation_warning():
    """`PrefixSumExecutor(N)` must emit `DeprecationWarning` per the migration plan in algorithms.md."""
    with pytest.warns(DeprecationWarning, match="exclusive_scan_add"):
        qd.algorithms.PrefixSumExecutor(64)


@test_utils.test(arch=qd.gpu)
def test_parallel_sort_emits_deprecation_warning():
    """`parallel_sort` must emit `DeprecationWarning` per the migration plan in algorithms.md."""
    keys = qd.field(qd.i32, shape=8)
    with pytest.warns(DeprecationWarning, match="sort"):
        qd.algorithms.parallel_sort(keys)
