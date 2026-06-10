import math

from quadrants.lang import impl, ops
from quadrants.lang.impl import get_runtime, grouped, static
from quadrants.lang.kernel_impl import func
from quadrants.lang.matrix import Matrix, Vector
from quadrants.types import f32, f64
from quadrants.types.annotations import template


@func
def _randn(dt):
    """
    Generate a random float sampled from univariate standard normal
    (Gaussian) distribution of mean 0 and variance 1, using the
    Box-Muller transformation.
    """
    assert dt == f32 or dt == f64
    u1 = ops.cast(1.0, dt) - ops.random(dt)
    u2 = ops.random(dt)
    r = ops.sqrt(-2 * ops.log(u1))
    c = ops.cos(math.tau * u2)
    return r * c


def randn(dt=None):
    """Generate a random float sampled from univariate standard normal
    (Gaussian) distribution of mean 0 and variance 1, using the
    Box-Muller transformation. Must be called in Quadrants scope.

    Args:
        dt (DataType): Data type of the required random number. Default to `None`.
            If set to `None` `dt` will be determined dynamically in runtime.

    Returns:
        The generated random float.

    Example::

        >>> @qd.kernel
        >>> def main():
        >>>     print(qd.randn())
        >>>
        >>> main()
        -0.463608
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    return _randn(dt)


@func
def _polar_decompose2d(A, dt):
    """Perform polar decomposition (A=UP) for 2x2 matrix.
    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        A (qd.Matrix(2, 2)): input 2x2 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        Decomposed 2x2 matrices `U` and `P`. `U` is a 2x2 orthogonal matrix
        and `P` is a 2x2 positive or semi-positive definite matrix.
    """
    U = Matrix.identity(dt, 2)
    P = ops.cast(A, dt)
    zero = ops.cast(0.0, dt)
    # if A is a zero matrix we simply return the pair (I, A)
    if A[0, 0] == zero and A[0, 1] == zero and A[1, 0] == zero and A[1, 1] == zero:
        pass
    else:
        detA = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
        adetA = abs(detA)
        B = Matrix(
            [
                [A[0, 0] + A[1, 1], A[0, 1] - A[1, 0]],
                [A[1, 0] - A[0, 1], A[1, 1] + A[0, 0]],
            ],
            dt,
        )

        if detA < zero:
            B = Matrix(
                [
                    [A[0, 0] - A[1, 1], A[0, 1] + A[1, 0]],
                    [A[1, 0] + A[0, 1], A[1, 1] - A[0, 0]],
                ],
                dt,
            )
        # here det(B) != 0 if A is not the zero matrix
        adetB = abs(B[0, 0] * B[1, 1] - B[1, 0] * B[0, 1])
        k = ops.cast(1.0, dt) / ops.sqrt(adetB)
        U = B * k
        P = (A.transpose() @ A + adetA * Matrix.identity(dt, 2)) * k

    return U, P


@func
def _polar_decompose3d(A, dt):
    """Perform polar decomposition (A=UP) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        A (qd.Matrix(3, 3)): input 3x3 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        Decomposed 3x3 matrices `U` and `P`.
    """
    U, sig, V = _svd3d(A, dt)
    return U @ V.transpose(), V @ sig @ V.transpose()


# https://www.seas.upenn.edu/~cffjiang/research/svd/svd.pdf
@func
def _svd2d(A, dt):
    """Perform singular value decomposition (A=USV^T) for 2x2 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.

    Args:
        A (qd.Matrix(2, 2)): input 2x2 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        Decomposed 2x2 matrices `U`, 'S' and `V`.
    """
    R, S = _polar_decompose2d(A, dt)
    c, s = ops.cast(0.0, dt), ops.cast(0.0, dt)
    s1, s2 = ops.cast(0.0, dt), ops.cast(0.0, dt)
    if abs(S[0, 1]) < 1e-5:  # type: ignore
        c, s = 1, 0
        s1, s2 = S[0, 0], S[1, 1]  # type: ignore
    else:
        tao = ops.cast(0.5, dt) * (S[0, 0] - S[1, 1])  # type: ignore
        w = ops.sqrt(tao**2 + S[0, 1] ** 2)  # type: ignore
        t = ops.cast(0.0, dt)
        if tao > 0:
            t = S[0, 1] / (tao + w)  # type: ignore
        else:
            t = S[0, 1] / (tao - w)  # type: ignore
        c = 1 / ops.sqrt(t**2 + 1)
        s = -t * c
        s1 = c**2 * S[0, 0] - 2 * c * s * S[0, 1] + s**2 * S[1, 1]  # type: ignore
        s2 = s**2 * S[0, 0] + 2 * c * s * S[0, 1] + c**2 * S[1, 1]  # type: ignore
    V = Matrix.zero(dt, 2, 2)
    if s1 < s2:
        tmp = s1
        s1 = s2
        s2 = tmp
        V = Matrix([[-s, c], [-c, -s]], dt=dt)
    else:
        V = Matrix([[c, s], [-s, c]], dt=dt)
    U = R @ V  # type: ignore
    return U, Matrix([[s1, ops.cast(0, dt)], [ops.cast(0, dt), s2]], dt=dt), V


def _svd3d(A, dt, iters=None):
    """Perform singular value decomposition (A=USV^T) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.

    Args:
        A (qd.Matrix(3, 3)): input 3x3 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.
        iters (int): iteration number to control algorithm precision.

    Returns:
        Decomposed 3x3 matrices `U`, 'S' and `V`.
    """
    assert A.n == 3 and A.m == 3
    assert dt in [f32, f64]
    if iters is None:
        if dt == f32:
            iters = 5
        else:
            iters = 8
    if dt == f32:
        rets = get_runtime().compiling_callable.ast_builder().sifakis_svd_f32(A.ptr, iters)
    else:
        rets = get_runtime().compiling_callable.ast_builder().sifakis_svd_f64(A.ptr, iters)
    assert len(rets) == 21
    U_entries = rets[:9]
    V_entries = rets[9:18]
    sig_entries = rets[18:]

    @func
    def get_result():
        U = Matrix.zero(dt, 3, 3)
        V = Matrix.zero(dt, 3, 3)
        sigma = Matrix.zero(dt, 3, 3)
        sig_v = Vector.zero(dt, 3)
        for i in static(range(3)):
            for j in static(range(3)):
                U[i, j] = U_entries[i * 3 + j]
                V[i, j] = V_entries[i * 3 + j]
            sig_v[i] = sig_entries[i]
        # Sort sig_v descending via selection sort, swapping matching columns of U and V together so
        # A = U · diag(sig_v) · Vᵀ is preserved across each swap. Sifakis already gives det(U) = det(V) = +1 (the sign
        # of det(A) is absorbed into σ); each pairwise column swap flips both determinants, so an odd total number of
        # swaps requires negating column 0 of U and V at the end to restore det(U) = det(V) = +1. That fix-up preserves
        # A because the two negations of column 0 cancel out in U_j0 · σ_0 · V_k0.
        swap_parity = 0
        for i in static(range(3)):
            max_idx = i
            max_val = sig_v[i]
            for j in static(range(3)):
                if static(j > i):
                    if sig_v[j] > max_val:
                        max_val = sig_v[j]
                        max_idx = j
            if max_idx != i:
                sig_v[max_idx] = sig_v[i]
                sig_v[i] = max_val
                for r in static(range(3)):
                    tmp_u = U[r, i]
                    U[r, i] = U[r, max_idx]
                    U[r, max_idx] = tmp_u
                    tmp_v = V[r, i]
                    V[r, i] = V[r, max_idx]
                    V[r, max_idx] = tmp_v
                swap_parity = 1 - swap_parity
        if swap_parity == 1:
            for r in static(range(3)):
                U[r, 0] = -U[r, 0]
                V[r, 0] = -V[r, 0]
        for i in static(range(3)):
            sigma[i, i] = sig_v[i]
        return U, sigma, V

    return get_result()


@func
def _eig2x2(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (qd.Matrix(2, 2)): input 2x2 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        eigenvalues (qd.Matrix(2, 2)): The eigenvalues in complex form. Each row stores one eigenvalue. The first number of the eigenvalue represents the real part and the second number represents the imaginary part.
        eigenvectors: (qd.Matrix(4, 2)): The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector consists of 2 entries, each of which is represented by two numbers for its real part and imaginary part.
    """
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    lambda1 = Vector.zero(dt, 2)
    lambda2 = Vector.zero(dt, 2)
    v1 = Vector.zero(dt, 4)
    v2 = Vector.zero(dt, 4)
    if gap > 0:
        lambda1 = Vector([tr + ops.sqrt(gap), 0.0], dt=dt) * 0.5
        lambda2 = Vector([tr - ops.sqrt(gap), 0.0], dt=dt) * 0.5
        A1 = A - lambda1[0] * Matrix.identity(dt, 2)  # type: ignore
        A2 = A - lambda2[0] * Matrix.identity(dt, 2)  # type: ignore
        if all(A1 == Matrix.zero(dt, 2, 2)) and all(A1 == Matrix.zero(dt, 2, 2)):
            v1 = Vector([0.0, 0.0, 1.0, 0.0]).cast(dt)
            v2 = Vector([1.0, 0.0, 0.0, 0.0]).cast(dt)
        else:
            v1 = Vector([A2[0, 0], 0.0, A2[1, 0], 0.0], dt=dt).normalized()
            v2 = Vector([A1[0, 0], 0.0, A1[1, 0], 0.0], dt=dt).normalized()
    else:
        lambda1 = Vector([tr, ops.sqrt(-gap)], dt=dt) * 0.5
        lambda2 = Vector([tr, -ops.sqrt(-gap)], dt=dt) * 0.5
        A1r = A - lambda1[0] * Matrix.identity(dt, 2)  # type: ignore
        A1i = -lambda1[1] * Matrix.identity(dt, 2)  # type: ignore
        A2r = A - lambda2[0] * Matrix.identity(dt, 2)  # type: ignore
        A2i = -lambda2[1] * Matrix.identity(dt, 2)  # type: ignore
        v1 = Vector([A2r[0, 0], A2i[0, 0], A2r[1, 0], A2i[1, 0]], dt=dt).normalized()
        v2 = Vector([A1r[0, 0], A1i[0, 0], A1r[1, 0], A1i[1, 0]], dt=dt).normalized()
    eigenvalues = Matrix.rows([lambda1, lambda2])
    eigenvectors = Matrix.cols([v1, v2])

    return eigenvalues, eigenvectors


@func
def _sym_eig2x2(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real symmetric matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (qd.Matrix(2, 2)): input 2x2 symmetric matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        eigenvalues (qd.Vector(2)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (qd.Matrix(2, 2)): The eigenvectors. Each column stores one eigenvector.
    """
    assert all(A == A.transpose()), "A needs to be symmetric"
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    # `gap >= 0` for symmetric A, so `lambda_hi >= lambda_lo`. Emit them as `(lambda_lo, lambda_hi)` so the result is
    # sorted ascending — matches the >=3x3 paths and NumPy / LAPACK convention for symmetric EVD.
    lambda_hi = (tr + ops.sqrt(gap)) * 0.5
    lambda_lo = (tr - ops.sqrt(gap)) * 0.5
    eigenvalues = Vector([lambda_lo, lambda_hi], dt=dt)

    A_hi = A - lambda_hi * Matrix.identity(dt, 2)
    A_lo = A - lambda_lo * Matrix.identity(dt, 2)
    v_hi = Vector.zero(dt, 2)
    v_lo = Vector.zero(dt, 2)
    if all(A_hi == Matrix.zero(dt, 2, 2)) and all(A_hi == Matrix.zero(dt, 2, 2)):
        v_hi = Vector([0.0, 1.0]).cast(dt)
        v_lo = Vector([1.0, 0.0]).cast(dt)
    else:
        v_hi = Vector([A_lo[0, 0], A_lo[1, 0]], dt=dt).normalized()
        v_lo = Vector([A_hi[0, 0], A_hi[1, 0]], dt=dt).normalized()
    eigenvectors = Matrix.cols([v_lo, v_hi])
    return eigenvalues, eigenvectors


def polar_decompose(A, dt=None):
    """Perform polar decomposition (A=UP) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        A (qd.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        Decomposed nxn matrices `U` and `P`.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n == 2:
        return _polar_decompose2d(A, dt)
    if A.n == 3:
        return _polar_decompose3d(A, dt)
    raise Exception("Polar decomposition only supports 2×2 and 3×3 matrices.")


def svd(A, dt=None):
    """Perform singular value decomposition (A=USV^T) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.

    Args:
        A (qd.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts qd.f32 or qd.f64.

    Returns:
        Decomposed nxn matrices `U`, 'S' and `V`.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n == 2:
        return _svd2d(A, dt)
    if A.n == 3:
        return _svd3d(A, dt)
    raise Exception("SVD only supports 2×2 and 3×3 matrices.")


def eig(A, dt=None):
    """Compute the eigenvalues and right eigenvectors of a real matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (qd.Matrix(n, n)): 2D Matrix for which the eigenvalues and right eigenvectors will be computed.
        dt (DataType): The datatype for the eigenvalues and right eigenvectors.

    Returns:
        eigenvalues (qd.Matrix(n, 2)): The eigenvalues in complex form. Each row stores one eigenvalue. The first number of the eigenvalue represents the real part and the second number represents the imaginary part.
        eigenvectors (qd.Matrix(n*2, n)): The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector consists of n entries, each of which is represented by two numbers for its real part and imaginary part.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n == 2:
        return _eig2x2(A, dt)
    raise Exception("Eigen solver only supports 2×2 matrices.")


def sym_eig(A, dt=None):
    """Compute the eigenvalues and right eigenvectors of a real symmetric matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Size ``A.n == 2`` uses a closed-form (trace / determinant) path. Sizes ``3 ≤ A.n ≤ 12`` use cyclic Jacobi
    (:func:`quadrants._funcs_sym_eig_general.sym_eig_general`).

    Args:
        A (qd.Matrix(n, n)): Symmetric Matrix for which the eigenvalues and right eigenvectors will be computed.
        dt (DataType): The datatype for the eigenvalues and right eigenvectors.

    Returns:
        eigenvalues (qd.Vector(n)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (qd.Matrix(n, n)): The eigenvectors. Each column stores one eigenvector.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n == 2:
        return _sym_eig2x2(A, dt)
    # pylint: disable=C0415
    from quadrants._funcs_sym_eig_general import sym_eig_general

    if A.n <= 12:
        return sym_eig_general(A, dt)
    raise Exception("Symmetric eigen solver currently supports sizes up to 12×12.")


def make_spd(A, dt=None):
    """Project a symmetric matrix ``A`` to the nearest positive semi-definite matrix in the Frobenius norm sense,
    by clamping its eigenvalues to ``≥ 0``.

    Implemented as ``Q · diag(max(λ, 0)) · Qᵀ`` where ``A = Q diag(λ) Qᵀ`` is the symmetric eigendecomposition
    computed by :func:`sym_eig`.

    Args:
        A (qd.Matrix(n, n)): Symmetric matrix.
        dt (DataType): Element dtype.

    Returns:
        qd.Matrix(n, n): the SPD projection of ``A``.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    # pylint: disable=C0415
    from quadrants._funcs_sym_eig_general import make_spd as _make_spd

    return _make_spd(A, dt)


@func
def _gauss_elimination_2x2(Ab, dt):
    if ops.abs(Ab[0, 0]) < ops.abs(Ab[1, 0]):
        Ab[0, 0], Ab[1, 0] = Ab[1, 0], Ab[0, 0]
        Ab[0, 1], Ab[1, 1] = Ab[1, 1], Ab[0, 1]
        Ab[0, 2], Ab[1, 2] = Ab[1, 2], Ab[0, 2]
    assert Ab[0, 0] != 0.0, "Matrix is singular in linear solve."
    scale = Ab[1, 0] / Ab[0, 0]
    Ab[1, 0] = 0.0
    for k in static(range(1, 3)):
        Ab[1, k] -= Ab[0, k] * scale
    x = Vector.zero(dt, 2)
    # Back substitution
    x[1] = Ab[1, 2] / Ab[1, 1]
    x[0] = (Ab[0, 2] - Ab[0, 1] * x[1]) / Ab[0, 0]
    return x


@func
def _gauss_elimination_3x3(Ab, dt):
    for i in static(range(3)):
        max_row = i
        max_v = ops.abs(Ab[i, i])
        for j in static(range(i + 1, 3)):
            if ops.abs(Ab[j, i]) > max_v:
                max_row = j
                max_v = ops.abs(Ab[j, i])
        assert max_v != 0.0, "Matrix is singular in linear solve."
        if i != max_row:
            if max_row == 1:
                for col in static(range(4)):
                    Ab[i, col], Ab[1, col] = Ab[1, col], Ab[i, col]
            else:
                for col in static(range(4)):
                    Ab[i, col], Ab[2, col] = Ab[2, col], Ab[i, col]
        assert Ab[i, i] != 0.0, "Matrix is singular in linear solve."
        for j in static(range(i + 1, 3)):
            scale = Ab[j, i] / Ab[i, i]
            Ab[j, i] = 0.0
            for k in static(range(i + 1, 4)):
                Ab[j, k] -= Ab[i, k] * scale
    # Back substitution
    x = Vector.zero(dt, 3)
    for i in static(range(2, -1, -1)):
        x[i] = Ab[i, 3]
        for k in static(range(i + 1, 3)):
            x[i] -= Ab[i, k] * x[k]
        x[i] = x[i] / Ab[i, i]
    return x


@func
def _combine(A, b, dt):
    n = static(A.n)
    Ab = Matrix.zero(dt, n, n + 1)
    for i in static(range(n)):
        for j in static(range(n)):
            Ab[i, j] = A[i, j]
    for i in static(range(n)):
        Ab[i, n] = b[i]
    return Ab


def solve(A, b, dt=None):
    """Solve a matrix using Gauss elimination method.

    Args:
        A (qd.Matrix(n, n)): input nxn matrix `A`.
        b (qd.Vector(n, 1)): input nx1 vector `b`.
        dt (DataType): The datatype for the `A` and `b`.

    Returns:
        x (qd.Vector(n, 1)): the solution of Ax=b.
    """
    assert A.n == A.m, "Only square matrix is supported"
    assert A.n >= 2 and A.n <= 3, "Only 2×2 and 3×3 matrices are supported"
    assert A.m == b.n, "Matrix and Vector dimension dismatch"
    if dt is None:
        dt = impl.get_runtime().default_fp
    Ab = _combine(A, b, dt)
    if A.n == 2:
        return _gauss_elimination_2x2(Ab, dt)
    if A.n == 3:
        return _gauss_elimination_3x3(Ab, dt)
    raise Exception("Solver only supports 2×2 and 3×3 matrices.")


@func
def field_fill_quadrants_scope(F: template(), val: template()):
    for I in grouped(F):
        F[I] = val


__all__ = ["randn", "polar_decompose", "eig", "sym_eig", "make_spd", "svd", "solve"]
