"""Tests for the `qd.precise(...)` per-op IEEE-strict primitive.

`qd.precise(expr)` must protect floating-point arithmetic from
fast-math reassociation/contraction/algebraic simplification, even when
the module is compiled with `fast_math=True`. The canonical workload is
Dekker / Kahan 2Sum: the compensation term `(a - aa) + (b - bb)` is the
*entire point* and silently rounds to zero under fast-math.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

N = 1000


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_protects_fast_math():
    """Run Dekker 2Sum twice under `fast_math=True`: once unprotected (the
    compensation term must be folded to zero - that is the very bug
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
    # `df_accum_precise` would produce naive behavior - caught by the final assertion below.
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
        f"The backend folded `(a - aa) + (b - bb)` to zero - IEEE-strict ordering was not honored."
    )

    # And the compensated sum must beat the naive f32 sum by orders of magnitude. This is the end-to-end
    # guarantee `qd.precise` exists to provide; it also indirectly validates that the offline-cache key
    # generator distinguishes `precise` from non-`precise` BinaryOpExpressions - if it did not, the two
    # kernels (structurally identical apart from `qd.precise(...)` wrappers) would share a compiled artifact
    # and `out_precise` would match `out_naive`.
    ds_err = abs(float(hi_precise) + float(lo_precise) - expected_f64)
    naive_err = abs(float(naive_ref) - expected_f64)
    assert (
        ds_err < naive_err * 1e-3
    ), f"qd.precise Dekker sum no more accurate than naive f32: ds_err={ds_err:.2e}, naive_err={naive_err:.2e}"


# Restricted to LLVM backends. The SPIR-V spec scopes `NoContraction` to arithmetic instructions, so the
# decoration is ignored on the `OpExtInst GLSL.std.450 Sin/Cos/Log/Sqrt/...` calls used for transcendentals.
# The Vulkan precision requirements for those ExtInsts also leave the driver latitude that exceeds the 2 ULP
# bound below (GLSL.std.450 Sin/Cos: 2^-11 absolute error; Log: 3 ULP outside [0.5, 2.0]; Sqrt: 2.5 ULP), so
# no amount of tagging can force correctly-rounded transcendentals through the driver on SPIR-V. See
# `docs/source/user_guide/precise.md` (Backend coverage) for the backend-specific nuance.
@pytest.mark.parametrize("op_name", ["sin", "cos", "log", "sqrt", "rsqrt"])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], default_fp=qd.f32, fast_math=True)
def test_qd_precise_unary_rounding(op_name):
    """Contract check: on every LLVM backend, `qd.precise(qd.<op>(x))` must produce the correctly-rounded f32 result
    even with module-level `fast_math=True`.

    This pins the precise path end-to-end: AST tagging -> IR propagation -> codegen honoring the tag (LLVM FMF clear
    and CUDA libdevice non-fast selection). Whether the naive (non-precise) path happens to also satisfy the 2 ULP
    bound on a given backend is incidental - libc `sinf` / `__ocml_<fn>f` / hardware `fsqrt` are correctly-rounded
    today regardless, and the test is not comparing against the naive path. The point is to catch the precise path
    regressing: e.g. the CUDA `use_fast = fast_math && !stmt->precise` dispatch at `codegen_cuda.cpp` flipping to
    unconditional `__nv_fast_<fn>f`, or `disable_fast_math()` being dropped so an LLVM upgrade starts substituting
    `sqrt` with `rsqrt+refine` under `afn`. In every such regression the precise path is the one that fails here.

    `sqrt` is included because LLVM FMF's `afn` can substitute `rsqrt+refine` which is ~2-3 ULP - the precise tag
    must defeat that substitution. `rsqrt` exercises the unique multi-instruction codegen path (sqrt intrinsic +
    fdiv) where `disable_fast_math(intermediate)` clears FMF on the sqrt separately from the enclosing fdiv.
    Parametrized per op so each failure reports the specific function that regressed.
    """
    qd_op = getattr(qd, op_name)

    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            out[i] = qd.precise(qd_op(x[i]))

    # Inputs span both the central range and values where some backends' fast-math approximations
    # are known to degrade.
    xs = np.array([0.5, 1.5, 2.5, 4.0, 7.0, 10.0, 25.0, 50.0], dtype=np.float32)
    in_arr = qd.ndarray(dtype=qd.f32, shape=(len(xs),))
    in_arr.from_numpy(xs)
    out = qd.ndarray(dtype=qd.f32, shape=(len(xs),))
    k(in_arr, out)
    res = out.to_numpy()

    # Correctly-rounded f32 reference, computed in f64 then narrowed. NumPy has no rsqrt, so we compute it by hand.
    if op_name == "rsqrt":
        ref = (1.0 / np.sqrt(xs.astype(np.float64))).astype(np.float32)
    else:
        ref = getattr(np, op_name)(xs.astype(np.float64)).astype(np.float32)

    # Within 2 ULP of the correctly-rounded f32 value: tight enough to catch backends that silently
    # substitute fast-math variants, generous enough to absorb single-ULP rounding noise across
    # implementations.
    ulp = np.spacing(np.maximum(np.abs(ref), np.float32(1.0)))
    max_ulp = float(np.max(np.abs(res - ref) / ulp))
    assert max_ulp <= 2.0, (
        f"qd.precise(qd.{op_name}(x)) deviated from the correctly-rounded f32 reference by "
        f"{max_ulp:.2f} ULP. The precise tag for `{op_name}` is not reaching codegen."
    )


@test_utils.test(default_fp=qd.f32)
def test_qd_precise_rejects_quadrants_classes():
    """`qd.precise` is a scalar primitive. Wrapping a `Vector` or `Matrix` must raise so that users who
    intended the scalar form get a clear error instead of a silent no-op.
    """
    with pytest.raises(ValueError, match="Quadrants classes"):
        qd.precise(qd.Vector([1.0, 2.0]))
    with pytest.raises(ValueError, match="Quadrants classes"):
        qd.precise(qd.Matrix([[1.0, 2.0], [3.0, 4.0]]))


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_recurses_through_select():
    """The walker must descend through `qd.select` (TernaryOp) so inner binary ops get tagged.

    Observable via the signed-zero rule: alg_simp rewrites `x + 0.0 -> x` unconditionally unless the add
    is tagged `precise`. When the add lives inside a `qd.select(...)` wrapped by `qd.precise`, the walker
    must reach it for the rewrite to be skipped -- at which point IEEE arithmetic delivers
    `(-0.0) + 0.0 = +0.0`. Without the tag, alg_simp strips the add and `-0.0` survives.
    """

    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.f32, ndim=1)):
        # `x[0]` is a runtime load, so neither operand reduces to a compile-time constant and the
        # ConstantFold pass cannot pre-compute the add. alg_simp's `a + 0 -> a` still matches.
        zero = qd.f32(0.0)
        # Without qd.precise wrap, alg_simp strips the add, leaving `x[0]` itself: bit pattern 0x80000000.
        out[0] = qd.select(qd.i32(1), x[0] + zero, zero)
        # With qd.precise wrap, the walker must recurse through the select and tag the inner add;
        # alg_simp then skips the fold, and IEEE `(-0.0) + 0.0` yields `+0.0`: bit pattern 0x00000000.
        out[1] = qd.precise(qd.select(qd.i32(1), x[0] + zero, zero))

    x_in = qd.ndarray(dtype=qd.f32, shape=(1,))
    x_in.from_numpy(np.array([-0.0], dtype=np.float32))
    out = qd.ndarray(dtype=qd.f32, shape=(2,))
    k(x_in, out)
    naive_bits, precise_bits = (int(v.view(np.uint32)) for v in out.to_numpy())
    assert naive_bits == 0x80000000, (
        f"Expected alg_simp to strip the unprotected `-0.0 + 0.0`, leaving bit pattern 0x80000000, "
        f"got 0x{naive_bits:08x}."
    )
    assert precise_bits == 0x00000000, (
        f"Expected `qd.precise(select(..., -0.0 + 0.0, ...))` to recurse through the select, tag the inner "
        f"add, and let IEEE collapse `-0.0 + 0.0` to `+0.0` (bit pattern 0x00000000); got 0x{precise_bits:08x}. "
        f"The walker may not be descending through TernaryOp."
    )


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_recurses_through_bit_cast():
    """The walker must descend through unary `bit_cast` (a `UnaryOpExpression` with op
    `cast_bits`) so that `qd.precise(qd.bit_cast(a + b, dtype))` tags the inner binary op.

    Observable via the signed-zero rule, as in `test_qd_precise_recurses_through_select`, but
    with the protected add nested inside a unary cast rather than a ternary select: without the
    wrap, alg_simp strips `x[0] + 0.0` and the bit pattern of `-0.0` (0x80000000) survives; with
    the wrap, the walker descends through `bit_cast` (UnaryOp), tags the inner add, alg_simp
    skips the fold, and IEEE `-0.0 + 0.0 = +0.0` yields bit pattern 0x00000000.
    """

    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.i32, ndim=1)):
        zero = qd.f32(0.0)
        # Without wrap: alg_simp strips the add inside the bit_cast; the cast reinterprets -0.0 -> 0x80000000.
        out[0] = qd.bit_cast(x[0] + zero, qd.i32)
        # With wrap: walker descends through bit_cast (UnaryOp) into the inner add and tags it;
        # alg_simp skips the fold, IEEE `(-0.0) + 0.0 = +0.0`, bit_cast yields 0x00000000.
        out[1] = qd.precise(qd.bit_cast(x[0] + zero, qd.i32))

    x_in = qd.ndarray(dtype=qd.f32, shape=(1,))
    x_in.from_numpy(np.array([-0.0], dtype=np.float32))
    out = qd.ndarray(dtype=qd.i32, shape=(2,))
    k(x_in, out)
    naive_bits, precise_bits = (int(v) & 0xFFFFFFFF for v in out.to_numpy())
    assert naive_bits == 0x80000000, (
        f"Expected alg_simp to strip the unprotected `-0.0 + 0.0` inside bit_cast, leaving bit pattern "
        f"0x80000000; got 0x{naive_bits:08x}."
    )
    assert precise_bits == 0x00000000, (
        f"Expected `qd.precise(bit_cast(x + 0.0, i32))` to recurse through the unary cast, tag the inner "
        f"add, and let IEEE collapse `-0.0 + 0.0` to `+0.0` (bit pattern 0x00000000); got 0x{precise_bits:08x}. "
        f"The walker may not be descending through UnaryOp (`cast_bits`)."
    )


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_stops_at_qd_func_call():
    """The walker must stop at `qd.func` call-site expressions: wrapping a call in
    `qd.precise(...)` is a no-op for ops inside the callee that are not directly part of the
    returned expression. Semantics inside a `qd.func` body are governed by the body's own ops.

    `qd.func` is inlined at the frontend, so the call returns whatever Expression the body's
    `return` resolves to. When the body routes its result through a local variable (a common
    pattern for multi-step compensated arithmetic), the returned expression is an
    `IdExpression` (a load from the local's alloca). The walker stops at `IdExpression`, so the
    inner `BinaryOpExpression` stored as the alloca's rvalue is unreachable from the caller.

    Signed-zero observable, with `x[0] = -0.0`:
      (1) naive body, naive call site -> alg_simp strips inside the body -> -0.0 survives.
      (2) naive body, `qd.precise(call(...))` at the caller -> walker stops at the returned
          IdExpression -> body's add is still stripped -> -0.0 still survives.
      (3) body-local `qd.precise(a + 0.0)` -> the body's own tag protects the add -> +0.0.
    """

    @qd.func
    def add_zero_naive(a):
        # Route the result through a local. The `return s` resolves at the inlining site to an
        # IdExpression (load from the alloca backing `s`), not the inner BinaryOp.
        s = a + qd.f32(0.0)
        return s

    @qd.func
    def add_zero_precise(a):
        # Body-local tag: alg_simp must skip the fold, independent of any caller wrap.
        s = qd.precise(a + qd.f32(0.0))
        return s

    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.i32, ndim=1)):
        # (1) Baseline: call site and body both unprotected -> bit pattern 0x80000000.
        out[0] = qd.bit_cast(add_zero_naive(x[0]), qd.i32)
        # (2) Wrap the call in qd.precise at the caller: walker stops at the IdExpression returned
        #     by the inlined body -> inner fold still happens -> bit pattern 0x80000000.
        out[1] = qd.bit_cast(qd.precise(add_zero_naive(x[0])), qd.i32)
        # (3) Body-local precise: only way to reach the inner op -> IEEE -0.0 + 0.0 = +0.0 -> 0x00000000.
        out[2] = qd.bit_cast(add_zero_precise(x[0]), qd.i32)

    x_in = qd.ndarray(dtype=qd.f32, shape=(1,))
    x_in.from_numpy(np.array([-0.0], dtype=np.float32))
    out = qd.ndarray(dtype=qd.i32, shape=(3,))
    k(x_in, out)
    naive_bits, wrapped_bits, inner_bits = (int(v) & 0xFFFFFFFF for v in out.to_numpy())
    assert (
        naive_bits == 0x80000000
    ), f"Expected the naive call to strip `x + 0.0` inside the body; got 0x{naive_bits:08x}."
    assert wrapped_bits == 0x80000000, (
        f"Expected `qd.precise(call(...))` at the caller to be a no-op for the callee's inner ops "
        f"(walker stops at the returned IdExpression); got 0x{wrapped_bits:08x} instead of "
        f"0x80000000. The walker may be descending past the call-site boundary."
    )
    assert inner_bits == 0x00000000, (
        f"Expected body-local `qd.precise(a + 0.0)` to protect the add; got 0x{inner_bits:08x}. "
        f"The inner tag is not reaching codegen."
    )


@test_utils.test(default_fp=qd.f32, fast_math=True)
def test_qd_precise_clones_shared_subexpression():
    """Non-mutation contract: when the same subtree appears twice in a single kernel (shared via an intermediate
    Python variable), wrapping one position in `qd.precise(...)` must not propagate the tag to the other position.

    Under the old in-place-mutation design this test would fail: tagging one alias would reach through the shared
    `BinaryOpExpression` and retroactively tag every other reference to it. The clone-based contract produces a fresh
    subtree for the `qd.precise` side and leaves the raw side bit-exactly untouched.
    """

    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.i32, ndim=1)):
        zero = qd.f32(0.0)
        # Bind the subexpression to a Python name so both subsequent uses alias the same value.
        shared = x[0] + zero
        # Wrap one use in qd.precise; the other must remain unprotected.
        out[0] = qd.bit_cast(qd.precise(shared), qd.i32)
        out[1] = qd.bit_cast(shared, qd.i32)

    x_in = qd.ndarray(dtype=qd.f32, shape=(1,))
    x_in.from_numpy(np.array([-0.0], dtype=np.float32))
    out = qd.ndarray(dtype=qd.i32, shape=(2,))
    k(x_in, out)
    wrapped_bits, raw_bits = (int(v) & 0xFFFFFFFF for v in out.to_numpy())
    # Note: because Python expr_init wraps `x[0] + zero` in an alloca, `shared` is an
    # IdExpression at the Python / AST level. `qd.precise(shared)` walks the IdExpression,
    # passes it through by reference, and returns an unchanged Expr. The observable effect
    # is that NEITHER store gets a precise BinaryOp - the original BinaryOp lives inside the
    # alloca's rvalue and is never reached by the walker. Both stores therefore observe the
    # non-precise path and `-0.0 + 0.0` is stripped by alg_simp to `-0.0` (0x80000000). This
    # shared-through-alloca outcome is what we pin down: qd.precise did NOT reach through and
    # retroactively tag the alloca's rvalue, which is exactly the non-mutation guarantee.
    assert raw_bits == 0x80000000, (
        f"Shared raw use must stay unprotected when the other alias is wrapped in qd.precise; "
        f"got 0x{raw_bits:08x}, expected 0x80000000."
    )
    assert wrapped_bits == 0x80000000, (
        f"qd.precise applied to a Python-aliased expression (IdExpression after expr_init) is a "
        f"no-op: the walker stops at IdExpression and must NOT reach into the alloca's rvalue to "
        f"mutate it; got 0x{wrapped_bits:08x}, expected 0x80000000."
    )


# Restricted to LLVM backends. On SPIR-V backends (Vulkan/Metal) the driver's optimizer retains
# latitude regardless of quadrants' `fast_math` flag - quadrants only emits `NoContraction` when
# `qd.precise` is explicitly set. Thus the "fast_math=False is equivalent to qd.precise everywhere"
# idempotency claim holds on LLVM backends but not on SPIR-V; see `docs/source/user_guide/precise.md`
# (Interaction with fast_math) for the backend-specific nuance.
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], default_fp=qd.f32, fast_math=False)
def test_qd_precise_idempotent_when_fast_math_off():
    """With `fast_math=False`, the reassociation / contraction / approximation rewrites that `qd.precise` gates are
    already globally disabled, so for computations that only depend on those gates, wrapping in `qd.precise(...)` must
    be a bit-exact no-op. Note: `qd.precise` also gates the `a + 0 -> a` fold for FP adds (signed-zero semantics),
    which fires regardless of `fast_math`; this test's Dekker 2Sum workload does not exercise that pattern, so the
    idempotency claim holds here but is not universal.

    The canonical observable is Dekker / Kahan 2Sum: under `fast_math=False`, the compensation term
    `(a - aa) + (b - bb)` is IEEE-preserved without the wrap, and the wrap must not change the result.
    """

    @qd.func
    def two_sum_naive(a, b):
        s = a + b
        bb = s - a
        aa = s - bb
        e = (a - aa) + (b - bb)
        return s, e

    @qd.func
    def two_sum_precise(a, b):
        s = qd.precise(a + b)
        bb = qd.precise(s - a)
        aa = qd.precise(s - bb)
        e = qd.precise((a - aa) + (b - bb))
        return s, e

    @qd.kernel
    def k(
        a: qd.types.ndarray(qd.f32, ndim=1), b: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.i32, ndim=2)
    ):
        s_n, e_n = two_sum_naive(a[0], b[0])
        s_p, e_p = two_sum_precise(a[0], b[0])
        out[0, 0] = qd.bit_cast(s_n, qd.i32)
        out[0, 1] = qd.bit_cast(e_n, qd.i32)
        out[1, 0] = qd.bit_cast(s_p, qd.i32)
        out[1, 1] = qd.bit_cast(e_p, qd.i32)

    # Pick an `(a, b)` pair where `a + b` rounds and produces a non-trivial compensation: a large
    # magnitude plus a small ULP-scale addend.
    a_in = qd.ndarray(dtype=qd.f32, shape=(1,))
    b_in = qd.ndarray(dtype=qd.f32, shape=(1,))
    a_in.from_numpy(np.array([1.0], dtype=np.float32))
    b_in.from_numpy(np.array([1e-8], dtype=np.float32))
    out = qd.ndarray(dtype=qd.i32, shape=(2, 2))
    k(a_in, b_in, out)
    bits = out.to_numpy()
    assert bits[0, 0] == bits[1, 0], (
        f"qd.precise must be bit-exactly idempotent under fast_math=False (sum term): "
        f"naive=0x{int(bits[0, 0]) & 0xFFFFFFFF:08x}, precise=0x{int(bits[1, 0]) & 0xFFFFFFFF:08x}."
    )
    assert bits[0, 1] == bits[1, 1], (
        f"qd.precise must be bit-exactly idempotent under fast_math=False (compensation term): "
        f"naive=0x{int(bits[0, 1]) & 0xFFFFFFFF:08x}, precise=0x{int(bits[1, 1]) & 0xFFFFFFFF:08x}."
    )
    # Sanity: the compensation is genuinely non-zero - i.e. the test is actually exercising the
    # rewrites that qd.precise gates. If `fast_math=False` were silently upgraded somewhere and
    # the compensation collapsed to 0, the idempotency assertion above would pass vacuously.
    assert (int(bits[0, 1]) & 0xFFFFFFFF) != 0, (
        "Under fast_math=False the compensation term must be IEEE-preserved (non-zero); "
        "if it is zero, the idempotency check is vacuous."
    )


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], default_fp=qd.f32, fast_math=True)
def test_qd_precise_floordiv_rounding():
    """Contract check: `qd.precise(a // b)` must produce `floor(a / b)` correctly on LLVM backends, even with
    module-level `fast_math=True`.

    `demote_operations.cpp::demote_ffloor` lowers FP floordiv into a synthesized `div + floor` chain. The PR
    propagates `stmt->precise` onto both stmts so codegen clears FMF on the div (defeating `arcp` / approximate
    reciprocal substitution) and on the floor. This test pins that contract: if someone removes the `div->precise`
    or `floor->precise` propagation in `demote_ffloor`, AND LLVM's `arcp` / `afn` alters the division near an
    integer boundary, the bit-exact assertion catches the regression.
    """

    @qd.kernel
    def k(
        a: qd.types.ndarray(qd.f32, ndim=1), b: qd.types.ndarray(qd.f32, ndim=1), out: qd.types.ndarray(qd.f32, ndim=1)
    ):
        for i in range(a.shape[0]):
            out[i] = qd.precise(a[i] // b[i])

    # Inputs chosen around integer-quotient boundaries where approximate reciprocal division (`arcp`) or
    # fused-multiply-reciprocal could round the quotient to the wrong side of the floor.
    a_vals = np.array([10.0, 7.0, -7.0, 1.0, 100.0, 0.1, 1e10], dtype=np.float32)
    b_vals = np.array([3.0, 2.0, 2.0, 3.0, 7.0, 0.03, 3.0], dtype=np.float32)
    a_in = qd.ndarray(dtype=qd.f32, shape=(len(a_vals),))
    a_in.from_numpy(a_vals)
    b_in = qd.ndarray(dtype=qd.f32, shape=(len(b_vals),))
    b_in.from_numpy(b_vals)
    out = qd.ndarray(dtype=qd.f32, shape=(len(a_vals),))
    k(a_in, b_in, out)
    res = out.to_numpy()

    # Reference: floor(a/b) computed in f32 (matching IEEE semantics of the precise div + floor chain).
    ref = np.floor(a_vals / b_vals)
    np.testing.assert_array_equal(res, ref, err_msg="qd.precise(a // b) did not match floor(a / b) reference")


# NOTE: a behavioral test for `pow` precise-propagation (alg_simp.cpp pow branch, ~line 485) is deliberately omitted.
# The rewrites `a**1 -> a`, `a**0 -> 1`, `a**0.5 -> sqrt(a)`, and `a**n -> (a*a)...` are all IEEE-equivalent to the
# original `pow()` call on the inputs exposed by any plain-pytest kernel, so there is no observable difference between
# `qd.precise(x ** n)` and `x ** n` at runtime today. Propagating `stmt->precise` onto the synthesized sqrt / mul / div
# chain remains valuable as future-proofing (keeps the rewritten chain tagged consistently with what the user wrote).
