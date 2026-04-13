"""Tests for the `qd.precise(...)` per-op IEEE-strict primitive.

`qd.precise(expr)` must protect floating-point arithmetic from
fast-math reassociation/contraction/algebraic simplification, even when
the module is compiled with `fast_math=True`. The canonical workload is
Dekker / Kahan 2Sum: the compensation term `(a - aa) + (b - bb)` is the
*entire point* and silently rounds to zero under fast-math.
"""

import numpy as np

import quadrants as qd

from tests import test_utils

N = 1000


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_protects_fast_math():
    """Run Dekker 2Sum twice under `fast_math=True`: once unprotected (the
    compensation term must be folded to zero — that is the very bug
    `qd.precise` exists to fix) and once with `qd.precise(...)` wrapping
    every FP op (the compensation term must survive).
    """

    @qd.func
    def two_sum_naive(a, b):
        s = a + b
        bb = s - a
        aa = s - bb
        e = (a - aa) + (b - bb)
        return s, e

    @qd.func
    def fast_two_sum_naive(a, b):
        s = a + b
        e = b - (s - a)
        return s, e

    @qd.func
    def two_sum_precise(a, b):
        # Every FP op below is wrapped in `qd.precise`, which transitively
        # tags each underlying BinaryOpStmt as IEEE-strict.
        s = qd.precise(a + b)
        bb = qd.precise(s - a)
        aa = qd.precise(s - bb)
        e = qd.precise((a - aa) + (b - bb))
        return s, e

    @qd.func
    def fast_two_sum_precise(a, b):
        s = qd.precise(a + b)
        e = qd.precise(b - (s - a))
        return s, e

    @qd.kernel
    def df_accum_naive(in_arr: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.f32, ndim=1)):
        for _ in range(1):
            hi = qd.f32(1.0)
            lo = qd.f32(0.0)
            for i in range(N):
                s, e = two_sum_naive(hi, in_arr[i])
                e = e + lo
                hi, lo = fast_two_sum_naive(s, e)
            out[0] = hi
            out[1] = lo

    @qd.kernel
    def df_accum_precise(in_arr: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.f32, ndim=1)):
        for _ in range(1):
            hi = qd.f32(1.0)
            lo = qd.f32(0.0)
            for i in range(N):
                s, e = two_sum_precise(hi, in_arr[i])
                # `e + lo` outside the helpers: also tagged so the accumulator
                # chain stays compensated end-to-end.
                e = qd.precise(e + lo)
                hi, lo = fast_two_sum_precise(s, e)
            out[0] = hi
            out[1] = lo

    in_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    in_arr.from_numpy(np.full(N, 1e-8, dtype=np.float32))
    out_naive = qd.ndarray(dtype=qd.f32, shape=(2,))
    out_precise = qd.ndarray(dtype=qd.f32, shape=(2,))

    # NOTE: defining the naive and precise kernels in the same test also indirectly validates that the
    # offline-cache key generator distinguishes `precise` from non-`precise` BinaryOpExpressions: the two
    # kernels are structurally identical apart from `qd.precise(...)` wrappers, so if the cache key did not
    # account for `precise` (as was the case before), the second kernel compiled would silently reuse the
    # first kernel's compiled artifact and both `out_*` arrays would end up with the same values.
    df_accum_naive(in_arr, out_naive)
    df_accum_precise(in_arr, out_precise)

    hi_naive, lo_naive = out_naive.to_numpy()
    hi_precise, lo_precise = out_precise.to_numpy()

    # Reference values for the assertions below.
    expected_f64 = 1.0 + N * 1e-8
    naive_ref = np.float32(1.0)
    for _ in range(N):
        naive_ref = np.float32(naive_ref + 1e-8)

    # 1. Negative control: without `qd.precise`, the compensation term IS
    #    stripped under `fast_math=True`. If this fails, fast_math has been
    #    silently disabled or one of the backends became more conservative.
    assert abs(float(lo_naive)) < 1e-10 or float(hi_naive) == np.float32(1.0), (
        f"Unexpected: 2Sum compensation survived under fast_math=True without qd.precise "
        f"(hi={hi_naive!r}, lo={lo_naive!r}). Did fast_math get silently disabled?"
    )

    # 2. Positive case: `qd.precise` must restore IEEE semantics locally.
    #    Compensation must be non-trivially non-zero.
    assert abs(float(lo_precise)) > 1e-10, (
        f"qd.precise failed to protect 2Sum: lo={lo_precise!r} (expected |lo| > 1e-10). "
        f"The backend folded `(a - aa) + (b - bb)` to zero — IEEE-strict ordering was not honored."
    )

    # 3. And the compensated sum must beat the naïve f32 sum by orders of magnitude.
    ds_err = abs(float(hi_precise) + float(lo_precise) - expected_f64)
    naive_err = abs(float(naive_ref) - expected_f64)
    assert (
        ds_err < naive_err * 1e-3
    ), f"qd.precise Dekker sum no more accurate than naive f32: ds_err={ds_err:.2e}, naive_err={naive_err:.2e}"
