"""Symmetric eigenvalue decomposition for ``2 <= N <= 12`` via cyclic Jacobi.

Cyclic Jacobi is the textbook robust algorithm for small symmetric EVD: it iteratively zeros out off-diagonal entries
via Givens rotations until the matrix is (numerically) diagonal. Complexity is O(N^3) per sweep and convergence is
quadratic near the solution, so ``MAX_SWEEPS = 12`` is comfortably enough to hit f64 precision for any ``N <= 12``.

Algorithm (Golub & Van Loan, §8.5):

* ``A_work = A``, ``Q = I``.
* Repeat ``MAX_SWEEPS`` sweeps, for each ``(p, q)`` with ``p < q``:
    * Compute Jacobi rotation ``J`` that zeros ``A_work[p, q]``.
    * Apply ``A_work := J^T A_work J`` (rank-2 update).
    * Accumulate ``Q := Q J``.
* Eigenvalues are ``diag(A_work)``; eigenvectors are columns of ``Q``.
* Sort ascending.

The outer sweep loop is a runtime ``range`` so compile time stays bounded even at N=12 (the inner ``(p, q)`` and
per-row ``static(range)`` updates already give straight-line code per Givens step). The sweep loop is explicitly tagged
``loop_config(serialize=True)``, so a calling ``@qd.kernel`` without its own outermost ``for ... in range(...)`` still
executes the sweeps sequentially on a single thread instead of parallelizing them — see
``perso_hugh/doc/quadrants_runtime_range_in_func_parallelized_gotcha_20260510.md`` for the underlying gotcha that this
directive sidesteps.
"""

from quadrants.lang import ops
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func
from quadrants.lang.matrix import Matrix, Vector
from quadrants.lang.misc import loop_config

_CONSIDER_AS_ZERO = 1e-30
_MAX_SWEEPS = 12


@func
def sym_eig_general(A, dt):
    """Symmetric eigendecomposition ``A = Q diag(λ) Qᵀ`` via cyclic Jacobi.

    Args:
        A: symmetric :class:`~quadrants.Matrix` (N × N). Not mutated.
        dt: element dtype (``qd.f32`` or ``qd.f64``).

    Returns:
        ``(eigenvalues, eigenvectors)`` — a length-N :class:`~quadrants.Vector` sorted ascending and an N × N
        :class:`~quadrants.Matrix` whose ``i``-th column is the eigenvector for ``eigenvalues[i]``.
    """
    N = static(A.n)
    A_work = Matrix.zero(dt, N, N)
    eigvecs = Matrix.zero(dt, N, N)
    eigvals = Vector.zero(dt, N)

    zero = A[0, 0] * 0.0
    one = zero + 1.0

    for i in static(range(N)):
        for j in static(range(N)):
            A_work[i, j] = A[i, j]
            eigvecs[i, j] = zero
        eigvecs[i, i] = one

    loop_config(serialize=True)
    for _sweep in range(_MAX_SWEEPS):
        for p in static(range(N)):
            for q in static(range(N)):
                if static(p < q):
                    apq = A_work[p, q]
                    if ops.abs(apq) > _CONSIDER_AS_ZERO:
                        app = A_work[p, p]
                        aqq = A_work[q, q]
                        diff = aqq - app

                        tau = one
                        if ops.abs(diff) < _CONSIDER_AS_ZERO:
                            if apq < 0.0:
                                tau = -one
                        else:
                            theta = diff / (2.0 * apq)
                            tau = 1.0 / (ops.abs(theta) + ops.sqrt(1.0 + theta * theta))
                            if theta < 0.0:
                                tau = -tau

                        gc = 1.0 / ops.sqrt(1.0 + tau * tau)
                        gs = tau * gc

                        for r in static(range(N)):
                            arp = A_work[r, p]
                            arq = A_work[r, q]
                            A_work[r, p] = gc * arp - gs * arq
                            A_work[r, q] = gs * arp + gc * arq

                        for k in static(range(N)):
                            akp = A_work[p, k]
                            akq = A_work[q, k]
                            A_work[p, k] = gc * akp - gs * akq
                            A_work[q, k] = gs * akp + gc * akq

                        A_work[p, q] = zero
                        A_work[q, p] = zero

                        for r in static(range(N)):
                            vrp = eigvecs[r, p]
                            vrq = eigvecs[r, q]
                            eigvecs[r, p] = gc * vrp - gs * vrq
                            eigvecs[r, q] = gs * vrp + gc * vrq

    for i in static(range(N)):
        eigvals[i] = A_work[i, i]

    for i in static(range(N)):
        min_idx = i
        min_val = eigvals[i]
        for j in static(range(N)):
            if static(j > i):
                if eigvals[j] < min_val:
                    min_val = eigvals[j]
                    min_idx = j
        if min_idx != i:
            eigvals[min_idx] = eigvals[i]
            eigvals[i] = min_val
            for k in static(range(N)):
                tmp = eigvecs[k, i]
                eigvecs[k, i] = eigvecs[k, min_idx]
                eigvecs[k, min_idx] = tmp

    return eigvals, eigvecs


@func
def make_spd(A, dt):
    """Project a symmetric matrix ``A`` to the nearest positive semi-definite matrix in the Frobenius norm sense,
    by clamping its eigenvalues to ``≥ 0``.

    Implemented as ``Q · diag(max(λ, 0)) · Qᵀ`` where ``A = Q diag(λ) Qᵀ`` is the symmetric eigendecomposition
    computed by :func:`sym_eig_general`.
    """
    N = static(A.n)
    eigvals, eigvecs = sym_eig_general(A, dt)
    for i in static(range(N)):
        if eigvals[i] < 0.0:
            eigvals[i] = eigvals[i] * 0.0
    out = Matrix.zero(dt, N, N)
    for i in static(range(N)):
        for j in static(range(N)):
            acc = A[0, 0] * 0.0
            for k in static(range(N)):
                acc = acc + eigvecs[i, k] * eigvals[k] * eigvecs[j, k]
            out[i, j] = acc
    return out
