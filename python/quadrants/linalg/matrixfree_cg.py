# type: ignore

from math import sqrt

from quadrants.lang import misc
from quadrants.lang.exception import QuadrantsRuntimeError, QuadrantsTypeError
from quadrants.lang.impl import field, fields_builder, grouped
from quadrants.lang.kernel_impl import data_oriented, kernel
from quadrants.types import primitive_types, template


@data_oriented
class LinearOperator:
    def __init__(self, matvec_kernel):
        self._matvec = matvec_kernel

    def matvec(self, x, Ax):
        if x.shape != Ax.shape:
            raise QuadrantsRuntimeError(f"Dimension mismatch x.shape{x.shape} != Ax.shape{Ax.shape}.")
        self._matvec(x, Ax)


def MatrixFreeCG(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free conjugate-gradient solver.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
        b (Field): The right-hand side of the linear system.
        x (Field): The initial guess for the solution.
        maxiter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
        quiet (bool): Switch to turn on/off iteration log.
    """

    if b.dtype != x.dtype:
        raise QuadrantsTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if str(b.dtype) == "f32":
        solver_dtype = primitive_types.f32
    elif str(b.dtype) == "f64":
        solver_dtype = primitive_types.f64
    else:
        raise QuadrantsTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise QuadrantsRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")

    size = b.shape
    vector_fields_builder = fields_builder.FieldsBuilder()
    p = field(dtype=solver_dtype)
    r = field(dtype=solver_dtype)
    Ap = field(dtype=solver_dtype)
    Ax = field(dtype=solver_dtype)
    if len(size) == 1:
        axes = misc.i
    elif len(size) == 2:
        axes = misc.ij
    elif len(size) == 3:
        axes = misc.ijk
    else:
        raise QuadrantsRuntimeError(f"MatrixFreeCG only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D.")
    vector_fields_builder.dense(axes, size).place(p, r, Ap, Ax)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = fields_builder.FieldsBuilder()
    alpha = field(dtype=solver_dtype)
    beta = field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()

    @kernel
    def init():
        for I in grouped(x):
            r[I] = b[I] - Ax[I]
            p[I] = 0.0
            Ap[I] = 0.0

    @kernel
    def reduce(p: template(), q: template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in grouped(p):
            result += p[I] * q[I]
        return result

    @kernel
    def update_x():
        for I in grouped(x):
            x[I] += alpha[None] * p[I]

    @kernel
    def update_r():
        for I in grouped(r):
            r[I] -= alpha[None] * Ap[I]

    @kernel
    def update_p():
        for I in grouped(p):
            p[I] = r[I] + beta[None] * p[I]

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init()
        initial_rTr = reduce(r, r)
        if not quiet:
            print(f">>> Initial residual = {initial_rTr:e}")
        old_rTr = initial_rTr
        new_rTr = initial_rTr
        update_p()
        if sqrt(initial_rTr) >= tol:  # Do nothing if the initial residual is small enough
            # -- Main loop --
            for i in range(maxiter):
                A._matvec(p, Ap)  # compute Ap = A x p
                pAp = reduce(p, Ap)
                alpha[None] = old_rTr / pAp
                update_x()
                update_r()
                new_rTr = reduce(r, r)
                if sqrt(new_rTr) < tol:
                    if not quiet:
                        print(">>> Conjugate Gradient method converged.")
                        print(f">>> #iterations {i}")
                    break
                beta[None] = new_rTr / old_rTr
                update_p()
                old_rTr = new_rTr
                if not quiet:
                    print(f">>> Iter = {i+1:4}, Residual = {sqrt(new_rTr):e}")
        if new_rTr >= tol:
            if not quiet:
                print(
                    f">>> Conjugate Gradient method failed to converge in {maxiter} iterations: Residual = {sqrt(new_rTr):e}"
                )
            succeeded = False
        return succeeded

    succeeded = solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
    return succeeded


def MatrixFreeBICGSTAB(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free biconjugate-gradient stabilized solver (BiCGSTAB).

    Use BiCGSTAB method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
        b (Field): The right-hand side of the linear system.
        x (Field): The initial guess for the solution.
        maxiter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
        quiet (bool): Switch to turn on/off iteration log.
    """

    if b.dtype != x.dtype:
        raise QuadrantsTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if str(b.dtype) == "f32":
        solver_dtype = primitive_types.f32
    elif str(b.dtype) == "f64":
        solver_dtype = primitive_types.f64
    else:
        raise QuadrantsTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise QuadrantsRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")

    size = b.shape
    vector_fields_builder = fields_builder.FieldsBuilder()
    p = field(dtype=solver_dtype)
    p_hat = field(dtype=solver_dtype)
    r = field(dtype=solver_dtype)
    r_tld = field(dtype=solver_dtype)
    s = field(dtype=solver_dtype)
    s_hat = field(dtype=solver_dtype)
    t = field(dtype=solver_dtype)
    Ap = field(dtype=solver_dtype)
    Ax = field(dtype=solver_dtype)
    Ashat = field(dtype=solver_dtype)
    if len(size) == 1:
        axes = misc.i
    elif len(size) == 2:
        axes = misc.ij
    elif len(size) == 3:
        axes = misc.ijk
    else:
        raise QuadrantsRuntimeError(f"MatrixFreeBICGSTAB only support 1D, 2D, 3D inputs; your inputs is {len(size)}-D.")
    vector_fields_builder.dense(axes, size).place(p, p_hat, r, r_tld, s, s_hat, t, Ap, Ax, Ashat)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = fields_builder.FieldsBuilder()
    alpha = field(dtype=solver_dtype)
    beta = field(dtype=solver_dtype)
    omega = field(dtype=solver_dtype)
    rho = field(dtype=solver_dtype)
    rho_1 = field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta, omega, rho, rho_1)
    scalar_snode_tree = scalar_builder.finalize()
    succeeded = True

    @kernel
    def init():
        for I in grouped(x):
            r[I] = b[I] - Ax[I]
            r_tld[I] = b[I]
            p[I] = 0.0
            Ap[I] = 0.0
            Ashat[I] = 0.0
        rho[None] = 0.0
        rho_1[None] = 1.0
        alpha[None] = 1.0
        beta[None] = 1.0
        omega[None] = 1.0

    @kernel
    def reduce(p: template(), q: template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in grouped(p):
            result += p[I] * q[I]
        return result

    @kernel
    def copy(orig: template(), dest: template()):
        for I in grouped(orig):
            dest[I] = orig[I]

    @kernel
    def update_p():
        for I in grouped(p):
            p[I] = r[I] + beta[None] * (p[I] - omega[None] * Ap[I])

    @kernel
    def update_phat():
        for I in grouped(p_hat):
            p_hat[I] = p[I]

    @kernel
    def update_s():
        for I in grouped(s):
            s[I] = r[I] - alpha[None] * Ap[I]

    @kernel
    def update_shat():
        for I in grouped(s_hat):
            s_hat[I] = s[I]

    @kernel
    def update_x():
        for I in grouped(x):
            x[I] += alpha[None] * p_hat[I] + omega[None] * s_hat[I]

    @kernel
    def update_r():
        for I in grouped(r):
            r[I] = s[I] - omega[None] * t[I]

    def solve():
        succeeded = True
        A._matvec(x, Ax)
        init()
        initial_rTr = reduce(r, r)
        rTr = initial_rTr
        if not quiet:
            print(f">>> Initial residual = {initial_rTr:e}")
        if sqrt(initial_rTr) >= tol:  # Do nothing if the initial residual is small enough
            for i in range(maxiter):
                rho[None] = reduce(r, r_tld)
                if rho[None] == 0.0:
                    if not quiet:
                        print(">>> BICGSTAB failed because r@r_tld = 0.")
                    succeeded = False
                    break
                if i == 0:
                    copy(orig=r, dest=p)
                else:
                    beta[None] = (rho[None] / rho_1[None]) * (alpha[None] / omega[None])
                    update_p()
                update_phat()
                A._matvec(p, Ap)
                alpha_lower = reduce(r_tld, Ap)
                alpha[None] = rho[None] / alpha_lower
                update_s()
                update_shat()
                A._matvec(s_hat, Ashat)
                copy(orig=Ashat, dest=t)
                omega_upper = reduce(t, s)
                omega_lower = reduce(t, t)
                omega[None] = omega_upper / (omega_lower + 1e-16) if omega_lower == 0.0 else omega_upper / omega_lower
                update_x()
                update_r()
                rTr = reduce(r, r)
                if not quiet:
                    print(f">>> Iter = {i+1:4}, Residual = {sqrt(rTr):e}")
                if sqrt(rTr) < tol:
                    if not quiet:
                        print(f">>> BICGSTAB method converged at #iterations {i}")
                    break
                rho_1[None] = rho[None]
        if rTr >= tol:
            if not quiet:
                print(f">>> BICGSTAB failed to converge in {maxiter} iterations: Residual = {sqrt(rTr):e}")
            succeeded = False
        return succeeded

    succeeded = solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
    return succeeded
