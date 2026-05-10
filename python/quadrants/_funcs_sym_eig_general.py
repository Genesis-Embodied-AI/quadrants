"""Symmetric eigenvalue decomposition for N > 3 via Householder tridiagonalisation
+ implicit QR with Wilkinson shift.

Faithful port of Eigen 3.4's ``Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h`` /
``Tridiagonalization.h``, originally written in ``qipc``
(``qipc/_src/core/linalg/evd.py``). Re-exposed here via
:func:`quadrants.sym_eig` so qipc can drop its private copy and upstream
consumers (e.g. ABD ``make_spd``) get a portable implementation.

The full algorithm is inlined into a single :func:`sym_eig_general` for two
reasons:

1. quadrants' value-typed ``Matrix`` / ``Vector`` mutations don't propagate
   back through ``@func`` boundaries via ``template()`` parameters when the
   caller created them with ``Matrix.zero`` / ``Vector.zero`` — qipc sidesteps
   this by using kernel-level ``ndarray`` of compound types, but :func:`sym_eig`
   needs to be callable on a ``Matrix`` value (not just on storage). Inlining
   is the smallest contact-area workaround.
2. Code is fully :func:`static(range)` so all loop bounds unroll at compile
   time; no template-passing dance is needed.
"""

from quadrants.lang import ops
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func
from quadrants.lang.matrix import Matrix, Vector

_CONSIDER_AS_ZERO = 2.3e-308
_EPSILON = 2.220446049250313e-16
_PRECISION_INV = 1.0 / _EPSILON


@func
def sym_eig_general(A, dt):
    """Symmetric eigendecomposition ``A = Q diag(λ) Qᵀ`` for square symmetric
    ``A`` of any size ``N``.

    Algorithm: Householder tridiagonalisation (Golub algorithm 8.3.1) followed
    by implicit QR with Wilkinson shift, then ascending sort. This is the same
    path Eigen 3.4's ``SelfAdjointEigenSolver`` uses for ``N > 3``.

    Args:
        A: symmetric :class:`~quadrants.Matrix` (N × N). Not mutated.
        dt: element dtype (``qd.f32`` or ``qd.f64``).

    Returns:
        ``(eigenvalues, eigenvectors)`` — a length-N :class:`~quadrants.Vector`
        sorted ascending and an N × N :class:`~quadrants.Matrix` whose ``i``-th
        column is the eigenvector for ``eigenvalues[i]``.
    """
    N = static(A.n)
    A_work = Matrix.zero(dt, N, N)
    eigvals = Vector.zero(dt, N)
    eigvecs = Matrix.zero(dt, N, N)
    workspace = Vector.zero(dt, N)

    abs_max = A[0, 0] * 0.0
    for r in static(range(N)):
        for cc in static(range(N)):
            if static(r >= cc):
                v = ops.abs(A[r, cc])
                if v > abs_max:
                    abs_max = v

    scale = abs_max
    if abs_max == 0.0:
        scale = A[0, 0] * 0.0 + 1.0
    inv_scale = 1.0 / scale
    for i in static(range(N)):
        for j in static(range(N)):
            A_work[i, j] = A[i, j] * inv_scale

    for i in static(range(N - 1)):
        tail_sq_norm = A[0, 0] * 0.0
        for j in static(range(i + 2, N)):
            tail_sq_norm = tail_sq_norm + A_work[j, i] * A_work[j, i]

        c0 = A_work[i + 1, i]
        h = A[0, 0] * 0.0
        beta = c0

        if tail_sq_norm > _CONSIDER_AS_ZERO:
            beta = ops.sqrt(c0 * c0 + tail_sq_norm)
            if c0 >= 0.0:
                beta = -beta
            inv_denom = 1.0 / (c0 - beta)
            for j in static(range(i + 2, N)):
                A_work[j, i] = A_work[j, i] * inv_denom
            h = (beta - c0) / beta

        A_work[i + 1, i] = A[0, 0] * 0.0 + 1.0

        if h != 0.0:
            for r in static(range(i + 1, N)):
                acc = A[0, 0] * 0.0
                for cc in static(range(i + 1, N)):
                    a_val = A_work[r, cc]
                    if static(r < cc):
                        a_val = A_work[cc, r]
                    acc = acc + a_val * A_work[cc, i]
                workspace[r - 1] = h * acc

            dot_pv = A[0, 0] * 0.0
            for k in static(range(i + 1, N)):
                dot_pv = dot_pv + workspace[k - 1] * A_work[k, i]
            correction = h * (-0.5) * dot_pv
            for k in static(range(i + 1, N)):
                workspace[k - 1] = workspace[k - 1] + correction * A_work[k, i]

            for r in static(range(i + 1, N)):
                for cc in static(range(i + 1, N)):
                    if static(r >= cc):
                        A_work[r, cc] = A_work[r, cc] - A_work[r, i] * workspace[cc - 1] - workspace[r - 1] * A_work[cc, i]
        else:
            for k in static(range(i + 1, N)):
                workspace[k - 1] = A[0, 0] * 0.0

        A_work[i + 1, i] = beta
        workspace[i] = h

    for r in static(range(N)):
        for cc in static(range(N)):
            if static(r == cc):
                eigvecs[r, cc] = A[0, 0] * 0.0 + 1.0
            else:
                eigvecs[r, cc] = A[0, 0] * 0.0

    for i_rev in static(range(N - 1)):
        i = static(N - 2 - i_rev)
        h = workspace[i]
        if h != 0.0:
            for j in static(range(N)):
                dot_v = eigvecs[i + 1, j]
                for k in static(range(i + 2, N)):
                    dot_v = dot_v + A_work[k, i] * eigvecs[k, j]
                eigvecs[i + 1, j] = eigvecs[i + 1, j] - h * dot_v
                for k in static(range(i + 2, N)):
                    eigvecs[k, j] = eigvecs[k, j] - h * dot_v * A_work[k, i]

    for i in static(range(N)):
        eigvals[i] = A_work[i, i]
    for i in static(range(N - 1)):
        workspace[i] = A_work[i + 1, i]

    end_val = static(N - 1)
    end_val_v = end_val
    for _outer in range(30 * N):
        if end_val_v > 0:
            for ii in static(range(N - 1)):
                if ii < end_val_v:
                    abs_sub = ops.abs(workspace[ii])
                    if abs_sub < _CONSIDER_AS_ZERO:
                        workspace[ii] = workspace[0] * 0.0
                    else:
                        scaled = _PRECISION_INV * workspace[ii]
                        if scaled * scaled <= ops.abs(eigvals[ii]) + ops.abs(eigvals[ii + 1]):
                            workspace[ii] = workspace[0] * 0.0

            still_going = 1
            for _d in static(range(N)):
                if still_going == 1:
                    if end_val_v > 0:
                        if workspace[end_val_v - 1] == 0.0:
                            end_val_v = end_val_v - 1
                        else:
                            still_going = 0
                    else:
                        still_going = 0

            if end_val_v > 0:
                start_val = end_val_v - 1
                for _s in static(range(N)):
                    if start_val > 0:
                        if workspace[start_val - 1] != 0.0:
                            start_val = start_val - 1

                td = (eigvals[end_val_v - 1] - eigvals[end_val_v]) * 0.5
                e = workspace[end_val_v - 1]
                mu = eigvals[end_val_v]
                if td == 0.0:
                    mu = mu - ops.abs(e)
                else:
                    if e != 0.0:
                        e2 = e * e
                        h_val = ops.sqrt(td * td + e2)
                        denom = td + h_val
                        if td < 0.0:
                            denom = td - h_val
                        if e2 == 0.0:
                            mu = mu - e / (denom / e)
                        else:
                            mu = mu - e2 / denom

                x = eigvals[start_val] - mu
                z = workspace[start_val]

                for k in static(range(N - 1)):
                    if k >= start_val:
                        if k < end_val_v:
                            if z != 0.0:
                                abs_x = ops.abs(x)
                                abs_z = ops.abs(z)
                                gc = x
                                gs = z
                                if z == 0.0:
                                    gc = eigvals[0] * 0.0 + 1.0
                                    if x < 0.0:
                                        gc = eigvals[0] * 0.0 - 1.0
                                    gs = eigvals[0] * 0.0
                                else:
                                    if x == 0.0:
                                        gc = eigvals[0] * 0.0
                                        gs = eigvals[0] * 0.0 - 1.0
                                        if z < 0.0:
                                            gs = eigvals[0] * 0.0 + 1.0
                                    else:
                                        if abs_x > abs_z:
                                            t = z / x
                                            u = ops.sqrt(1.0 + t * t)
                                            if x < 0.0:
                                                u = -u
                                            gc = 1.0 / u
                                            gs = -t * gc
                                        else:
                                            t = x / z
                                            u = ops.sqrt(1.0 + t * t)
                                            if z < 0.0:
                                                u = -u
                                            gs = -1.0 / u
                                            gc = -t * gs

                                sdk = gs * eigvals[k] + gc * workspace[k]
                                dkp1 = gs * workspace[k] + gc * eigvals[k + 1]

                                eigvals[k] = gc * (gc * eigvals[k] - gs * workspace[k]) - gs * (gc * workspace[k] - gs * eigvals[k + 1])
                                eigvals[k + 1] = gs * sdk + gc * dkp1
                                workspace[k] = gc * sdk - gs * dkp1

                                if static(k > 0):
                                    if k > start_val:
                                        workspace[k - 1] = gc * workspace[k - 1] - gs * z

                                x = workspace[k]
                                z = eigvals[0] * 0.0
                                if static(k + 1 < N - 1):
                                    if k < end_val_v - 1:
                                        z = -gs * workspace[k + 1]
                                        workspace[k + 1] = gc * workspace[k + 1]

                                for r in static(range(N)):
                                    qk = eigvecs[r, k]
                                    qk1 = eigvecs[r, k + 1]
                                    eigvecs[r, k] = gc * qk - gs * qk1
                                    eigvecs[r, k + 1] = gs * qk + gc * qk1

    for i in static(range(N)):
        eigvals[i] = eigvals[i] * scale

    for i in static(range(N)):
        min_idx = i
        min_val = eigvals[i]
        for j in static(range(N)):
            if j > i:
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
    """Project a symmetric matrix ``A`` to the nearest positive semi-definite
    matrix in the Frobenius norm sense, by clamping its eigenvalues to ``≥ 0``.

    Implemented as ``Q · diag(max(λ, 0)) · Qᵀ`` where ``A = Q diag(λ) Qᵀ`` is
    the symmetric eigendecomposition computed by :func:`sym_eig_general`.
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
