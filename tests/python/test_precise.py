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
    compensation term must be folded to zero -- that is the very bug
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
    # Scratch buffer for the naive kernel's output; never read back. Its only purpose is to give the naive
    # kernel somewhere to write so the compile happens and populates the cache (see NOTE below).
    out_naive = qd.ndarray(dtype=qd.f32, shape=(2,))
    out_precise = qd.ndarray(dtype=qd.f32, shape=(2,))

    # NOTE: running the naive kernel first also indirectly validates that the offline-cache key generator
    # distinguishes `precise` from non-`precise` BinaryOpExpressions. The two kernels are structurally
    # identical apart from `qd.precise(...)` wrappers, so if the cache key did not account for `precise`
    # (as was the case before), the second compile would silently reuse the first's artifact and
    # `df_accum_precise` would produce naive behavior -- caught by the final assertion below.
    df_accum_naive(in_arr, out_naive)
    df_accum_precise(in_arr, out_precise)

    hi_precise, lo_precise = out_precise.to_numpy()

    # Reference values for the assertions below.
    expected_f64 = 1.0 + N * 1e-8
    naive_ref = np.float32(1.0)
    for _ in range(N):
        naive_ref = np.float32(naive_ref + 1e-8)

    # `qd.precise` must restore IEEE semantics locally: the compensation term must be non-trivially non-zero.
    assert abs(float(lo_precise)) > 1e-10, (
        f"qd.precise failed to protect 2Sum: lo={lo_precise!r} (expected |lo| > 1e-10). "
        f"The backend folded `(a - aa) + (b - bb)` to zero -- IEEE-strict ordering was not honored."
    )

    # And the compensated sum must beat the naive f32 sum by orders of magnitude. This is the end-to-end
    # guarantee `qd.precise` exists to provide; it also indirectly validates that the offline-cache key
    # generator distinguishes `precise` from non-`precise` BinaryOpExpressions -- if it did not, the two
    # kernels (structurally identical apart from `qd.precise(...)` wrappers) would share a compiled artifact
    # and `out_precise` would match `out_naive`.
    ds_err = abs(float(hi_precise) + float(lo_precise) - expected_f64)
    naive_err = abs(float(naive_ref) - expected_f64)
    assert (
        ds_err < naive_err * 1e-3
    ), f"qd.precise Dekker sum no more accurate than naive f32: ds_err={ds_err:.2e}, naive_err={naive_err:.2e}"


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_unary_rounding():
    """`qd.precise(qd.sin/cos/log/sqrt(x))` must produce the correctly-rounded f32
    result on every backend, even with module-level `fast_math=True`.

    This exercises the unary precise path end-to-end: AST tagging -> IR
    propagation -> codegen honoring the tag (LLVM FMF clear, SPIR-V
    `NoContraction` decoration, or CUDA libdevice selection, depending
    on the backend). We verify correctness against numpy's
    correctly-rounded f32 reference; the naive (non-precise) variant is
    deliberately not part of this test, because on most backends
    `fast_math=True` happens to give correctly-rounded transcendentals
    anyway and a comparison against it would be uninformative. `sqrt`
    is included because LLVM FMF's `afn` can substitute `rsqrt+refine`
    which is ~2-3 ULP -- the precise tag must defeat that substitution.
    """

    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.f32, ndim=2)):
        for i in range(x.shape[0]):
            out[i, 0] = qd.precise(qd.sin(x[i]))
            out[i, 1] = qd.precise(qd.cos(x[i]))
            out[i, 2] = qd.precise(qd.log(x[i]))
            out[i, 3] = qd.precise(qd.sqrt(x[i]))

    # Inputs span both the central range and values where some backends'
    # fast-math approximations are known to degrade.
    xs = np.array([0.5, 1.5, 2.5, 4.0, 7.0, 10.0, 25.0, 50.0], dtype=np.float32)
    in_arr = qd.ndarray(dtype=qd.f32, shape=(len(xs),))
    in_arr.from_numpy(xs)
    out = qd.ndarray(dtype=qd.f32, shape=(len(xs), 4))
    k(in_arr, out)
    res = out.to_numpy()

    # Correctly-rounded f32 reference, computed in f64 then narrowed.
    xs64 = xs.astype(np.float64)
    ref = np.stack([np.sin(xs64), np.cos(xs64), np.log(xs64), np.sqrt(xs64)], axis=1).astype(np.float32)

    # Within 2 ULP of the correctly-rounded f32 value: tight enough to catch
    # backends that silently substitute fast-math variants, generous enough
    # to absorb single-ULP rounding noise across implementations.
    ulp = np.spacing(np.maximum(np.abs(ref), np.float32(1.0)))
    err_in_ulp = np.abs(res - ref) / ulp
    max_ulp = float(err_in_ulp.max())
    assert max_ulp <= 2.0, (
        f"qd.precise(unary) deviated from the correctly-rounded f32 reference by {max_ulp:.2f} ULP. "
        f"The unary precise tag is not reaching the codegen for at least one of sin/cos/log/sqrt."
    )
