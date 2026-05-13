"""Tests for the hybrid loop-unrolling path of ``Matrix.__matmul__``.

For matrix products with ``M*K*N`` above ``_MATMUL_UNROLL_THRESHOLD`` (= 64),
``_matmul_helper`` promotes the largest of ``M / K / N`` to a runtime ``range`` and keeps the other
two as ``static(range)``. This file pins down:

* numerical correctness for all three runtime branches (K, M, and N as the runtime dim);
* the qipc 3-way chain ``(9×12) · (12×12) · (12×9) → 9×9``;
* a kernel pattern with **no top-level outer loop** at N=12 — the ``loop_config(serialize=True)``
  inside ``_matmul_helper`` must keep the outer runtime loop sequential per thread, otherwise the
  kernel would parallelize it as the grid (see
  ``perso_hugh/doc/quadrants_runtime_range_in_func_parallelized_gotcha_20260510.md``).

All tests are parametrized over ``qd.gpu`` so they run on CUDA / AMDGPU / Vulkan / Metal.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _shape_inputs(M, K, N, dt, seed):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((M, K)).astype(np_dt)
    B_np = rng.standard_normal((K, N)).astype(np_dt)
    return A_np, B_np


def _test_matmul_shape(M, K, N, dt):
    """Numerical round-trip vs numpy for ``A @ B`` at the given shape."""
    A_np, B_np = _shape_inputs(M, K, N, dt, seed=0xCAFE + 17 * M + 31 * K + 53 * N)

    A = qd.Matrix.field(M, K, dtype=dt, shape=())
    B = qd.Matrix.field(K, N, dtype=dt, shape=())
    C = qd.Matrix.field(M, N, dtype=dt, shape=())

    A.from_numpy(A_np)
    B.from_numpy(B_np)

    @qd.kernel
    def run():
        C[None] = A[None] @ B[None]

    run()

    expected = A_np @ B_np
    tol = 5e-4 if dt == qd.f32 else 1e-10
    np.testing.assert_allclose(C.to_numpy(), expected, rtol=tol, atol=tol)


# Each tuple covers a different hybrid branch in _matmul_helper:
#   (12, 12, 12) — K-runtime (rank-1 update; tie broken in favour of K).
#   (12,  9,  9) — M-runtime (M is strict max; row-major form).
#   ( 9,  9, 12) — N-runtime (N is strict max; column-major form).
# Plus a few qipc shapes and edge cases.
_HYBRID_SHAPES = [
    (12, 12, 12),
    (12, 9, 9),
    (9, 9, 12),
    (9, 12, 12),  # K=N tied at max — K wins, rank-1 form
    (12, 12, 9),  # M=K tied at max — K wins, rank-1 form
    (9, 12, 9),  # K is strict max — rank-1 form
    (5, 5, 5),  # smallest size that triggers hybrid (125 > 64)
    (4, 4, 4),  # boundary: stays fully unrolled (64 == threshold)
]


@pytest.mark.parametrize("M,K,N", _HYBRID_SHAPES)
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_matmul_hybrid_shapes_f32(M, K, N):
    _test_matmul_shape(M, K, N, qd.f32)


@pytest.mark.parametrize("M,K,N", _HYBRID_SHAPES)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_matmul_hybrid_shapes_f64(M, K, N):
    _test_matmul_shape(M, K, N, qd.f64)


def _test_matmul_chain(dt):
    """qipc 3-way chain (9×12) · (12×12) · (12×9) → 9×9, both chained and staged."""
    A_np, B_np = _shape_inputs(9, 12, 12, dt, seed=0xCA70)
    _, C_np = _shape_inputs(12, 12, 9, dt, seed=0xCA72)

    A = qd.Matrix.field(9, 12, dtype=dt, shape=())
    B = qd.Matrix.field(12, 12, dtype=dt, shape=())
    C = qd.Matrix.field(12, 9, dtype=dt, shape=())
    AB = qd.Matrix.field(9, 12, dtype=dt, shape=())
    ABC_chained = qd.Matrix.field(9, 9, dtype=dt, shape=())
    ABC_staged = qd.Matrix.field(9, 9, dtype=dt, shape=())

    A.from_numpy(A_np)
    B.from_numpy(B_np)
    C.from_numpy(C_np)

    @qd.kernel
    def run():
        ABC_chained[None] = A[None] @ B[None] @ C[None]
        AB[None] = A[None] @ B[None]
        ABC_staged[None] = AB[None] @ C[None]

    run()

    expected = A_np @ B_np @ C_np
    tol = 5e-4 if dt == qd.f32 else 1e-10

    np.testing.assert_allclose(ABC_chained.to_numpy(), expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(ABC_staged.to_numpy(), expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(AB.to_numpy(), A_np @ B_np, rtol=tol, atol=tol)
    np.testing.assert_allclose(ABC_chained.to_numpy(), ABC_staged.to_numpy(), rtol=tol, atol=tol)


@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_matmul_chain_qipc_sizes_f32():
    _test_matmul_chain(qd.f32)


@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_matmul_chain_qipc_sizes_f64():
    _test_matmul_chain(qd.f64)


def _test_matmul_no_toplevel_loop(M, K, N, dt):
    """Kernel that calls ``A @ B`` at hybrid sizes with no top-level ``range`` of its own.

    Without ``loop_config(serialize=True)`` inside ``_matmul_helper``, this kernel would parallelize
    ``_matmul_helper``'s outer runtime loop as the grid — every thread would execute one slice of the
    matmul instead of the full product. With ``serialize=True``, the runtime loop runs sequentially
    on a single thread and the result matches numpy.
    """
    A_np, B_np = _shape_inputs(M, K, N, dt, seed=0xBADCAFE + M * 17 + K * 31 + N * 53)

    A = qd.Matrix.field(M, K, dtype=dt, shape=())
    B = qd.Matrix.field(K, N, dtype=dt, shape=())
    C = qd.Matrix.field(M, N, dtype=dt, shape=())

    A.from_numpy(A_np)
    B.from_numpy(B_np)

    # Crucially, no `for ... in range(...)` outside the matmul. The kernel body is a single
    # statement that calls __matmul__ on a value-type Matrix.
    @qd.kernel
    def run():
        C[None] = A[None] @ B[None]

    run()

    expected = A_np @ B_np
    tol = 5e-4 if dt == qd.f32 else 1e-10
    np.testing.assert_allclose(C.to_numpy(), expected, rtol=tol, atol=tol)


# One representative shape per hybrid branch.
@pytest.mark.parametrize("M,K,N", [(12, 12, 12), (12, 9, 9), (9, 9, 12)])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_matmul_no_toplevel_loop_f32(M, K, N):
    _test_matmul_no_toplevel_loop(M, K, N, qd.f32)


@pytest.mark.parametrize("M,K,N", [(12, 12, 12), (12, 9, 9), (9, 9, 12)])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_matmul_no_toplevel_loop_f64(M, K, N):
    _test_matmul_no_toplevel_loop(M, K, N, qd.f64)


def _test_matmul_with_toplevel_grid(M, K, N, dt):
    """Kernel that *does* parallelize a top-level ``range`` over a batch of matmuls.

    This ensures the hybrid path nests correctly when the kernel grid is the outer parallel loop:
    each thread does one full matmul of its own data, and ``_matmul_helper``'s internal runtime loop
    runs sequentially within that thread.
    """
    np_dt = np.float32 if dt == qd.f32 else np.float64
    BATCH = 8
    rng = np.random.default_rng(0xBA7C4 + M * 17 + K * 31 + N * 53)
    A_np = rng.standard_normal((BATCH, M, K)).astype(np_dt)
    B_np = rng.standard_normal((BATCH, K, N)).astype(np_dt)

    A = qd.Matrix.field(M, K, dtype=dt, shape=BATCH)
    B = qd.Matrix.field(K, N, dtype=dt, shape=BATCH)
    C = qd.Matrix.field(M, N, dtype=dt, shape=BATCH)

    A.from_numpy(A_np)
    B.from_numpy(B_np)

    @qd.kernel
    def run():
        for b in range(BATCH):
            C[b] = A[b] @ B[b]

    run()

    expected = np.einsum("bij,bjk->bik", A_np, B_np)
    tol = 5e-4 if dt == qd.f32 else 1e-10
    np.testing.assert_allclose(C.to_numpy(), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("M,K,N", [(12, 12, 12), (12, 9, 9), (9, 9, 12)])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_matmul_with_toplevel_grid_f32(M, K, N):
    _test_matmul_with_toplevel_grid(M, K, N, qd.f32)


@pytest.mark.parametrize("M,K,N", [(12, 12, 12), (12, 9, 9), (9, 9, 12)])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_matmul_with_toplevel_grid_f64(M, K, N):
    _test_matmul_with_toplevel_grid(M, K, N, qd.f64)
