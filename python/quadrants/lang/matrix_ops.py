# type: ignore

import quadrants.lang.ops as ops_mod
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func, pyfunc
from quadrants.lang.matrix import Matrix, Vector
from quadrants.lang.matrix_ops_utils import (
    arg_at,
    arg_foreach_check,
    assert_list,
    assert_tensor,
    assert_vector,
    check_matmul,
    check_transpose,
    dim_lt,
    is_int_const,
    preconditions,
    same_shapes,
    square_matrix,
)
from quadrants.lang.misc import loop_config
from quadrants.types.annotations import template

_MATMUL_UNROLL_THRESHOLD = 64


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def _reduce(mat, fun: template()):
    shape = static(mat.get_shape())
    if static(len(shape) == 1):
        result = mat[0]
        for i in static(range(1, shape[0])):
            result = fun(result, mat[i])
        return result
    result = mat[0, 0]
    for i in static(range(shape[0])):
        for j in static(range(shape[1])):
            if static(i != 0 or j != 0):
                result = fun(result, mat[i, j])
    return result


@pyfunc
def _filled_vector(n: template(), dtype: template(), val: template()):
    return Vector([val for _ in static(range(n))], dtype)


@pyfunc
def _filled_matrix(n: template(), m: template(), dtype: template(), val: template()):
    return Matrix([[val for _ in static(range(m))] for _ in static(range(n))], dtype)


@pyfunc
def _unit_vector(n: template(), i: template(), dtype: template()):
    return Vector([i == j for j in static(range(n))], dtype)


@pyfunc
def _identity_matrix(n: template(), dtype: template()):
    return Matrix([[i == j for j in static(range(n))] for i in static(range(n))], dtype)


@preconditions(
    arg_at(0, lambda xs: same_shapes(*xs)),
    arg_foreach_check(
        0,
        fns=[assert_vector(), assert_list],
        logic="or",
        msg="Cols/rows must be a list of lists, or a list of vectors",
    ),
)
@pyfunc
def rows(rows):  # pylint: disable=W0621
    return Matrix([[x for x in row] for row in rows])


@pyfunc
def cols(cols):  # pylint: disable=W0621
    return rows(cols).transpose()


@pyfunc
def E(mat: template(), x: template(), y: template(), n: template()):
    return mat[x % n, y % n]


@preconditions(square_matrix, dim_lt(0, 5))
@pyfunc
def determinant(mat):
    shape = static(mat.get_shape())
    if static(shape[0] == 1):
        return mat[0, 0]
    if static(shape[0] == 2):
        return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    if static(shape[0] == 3):
        return (
            mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[2, 1] * mat[1, 2])
            - mat[1, 0] * (mat[0, 1] * mat[2, 2] - mat[2, 1] * mat[0, 2])
            + mat[2, 0] * (mat[0, 1] * mat[1, 2] - mat[1, 1] * mat[0, 2])
        )
    if static(shape[0] == 4):
        det = mat[0, 0] * 0  # keep type
        for i in static(range(4)):
            det = det + (-1) ** i * (
                mat[i, 0]
                * (
                    E(mat, i + 1, 1, 4)
                    * (E(mat, i + 2, 2, 4) * E(mat, i + 3, 3, 4) - E(mat, i + 3, 2, 4) * E(mat, i + 2, 3, 4))
                    - E(mat, i + 2, 1, 4)
                    * (E(mat, i + 1, 2, 4) * E(mat, i + 3, 3, 4) - E(mat, i + 3, 2, 4) * E(mat, i + 1, 3, 4))
                    + E(mat, i + 3, 1, 4)
                    * (E(mat, i + 1, 2, 4) * E(mat, i + 2, 3, 4) - E(mat, i + 2, 2, 4) * E(mat, i + 1, 3, 4))
                )
            )
        return det
    # unreachable
    return None


@preconditions(square_matrix, dim_lt(0, 5))
@pyfunc
def inverse(mat):
    shape = static(mat.get_shape())
    if static(shape[0] == 1):
        return Matrix([[1.0 / mat[0, 0]]])
    inv_determinant = 1.0 / determinant(mat)
    if static(shape[0] == 2):
        return inv_determinant * Matrix([[mat[1, 1], -mat[0, 1]], [-mat[1, 0], mat[0, 0]]])
    if static(shape[0] == 3):
        return inv_determinant * Matrix(
            [
                [
                    E(mat, i + 1, j + 1, 3) * E(mat, i + 2, j + 2, 3)
                    - E(mat, i + 2, j + 1, 3) * E(mat, i + 1, j + 2, 3)
                    for i in static(range(3))
                ]
                for j in static(range(3))
            ]
        )
    if static(shape[0] == 4):
        return inv_determinant * Matrix(
            [
                [
                    (-1) ** (i + j)
                    * (
                        (
                            E(mat, i + 1, j + 1, 4)
                            * (
                                E(mat, i + 2, j + 2, 4) * E(mat, i + 3, j + 3, 4)
                                - E(mat, i + 3, j + 2, 4) * E(mat, i + 2, j + 3, 4)
                            )
                            - E(mat, i + 2, j + 1, 4)
                            * (
                                E(mat, i + 1, j + 2, 4) * E(mat, i + 3, j + 3, 4)
                                - E(mat, i + 3, j + 2, 4) * E(mat, i + 1, j + 3, 4)
                            )
                            + E(mat, i + 3, j + 1, 4)
                            * (
                                E(mat, i + 1, j + 2, 4) * E(mat, i + 2, j + 3, 4)
                                - E(mat, i + 2, j + 2, 4) * E(mat, i + 1, j + 3, 4)
                            )
                        )
                    )
                    for i in static(range(4))
                ]
                for j in static(range(4))
            ]
        )
    # unreachable
    return None


@preconditions(check_transpose)
@pyfunc
def transpose(mat):
    shape = static(mat.get_shape())
    return Matrix([[mat[i, j] for i in static(range(shape[0]))] for j in static(range(shape[1]))])


@preconditions(arg_at(0, is_int_const))
@pyfunc
def diag(dim: template(), val: template()):
    return Matrix([[val if i == j else 0 for j in static(range(dim))] for i in static(range(dim))])


@preconditions(assert_tensor)
@pyfunc
def sum(mat):  # pylint: disable=W0622
    return _reduce(mat, ops_mod.add)


@preconditions(assert_tensor)
@pyfunc
def norm_sqr(mat):
    return sum(mat * mat)


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def norm(mat, eps=0.0):
    return ops_mod.sqrt(norm_sqr(mat) + eps)


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def norm_inv(mat, eps=0.0):
    return ops_mod.rsqrt(norm_sqr(mat) + eps)


@preconditions(arg_at(0, assert_vector()))
@pyfunc
def normalized(vec, eps=0.0):
    invlen = 1 / (norm(vec) + eps)
    return invlen * vec


@preconditions(assert_tensor)
@pyfunc
def any(mat):  # pylint: disable=W0622
    return _reduce(mat != 0, ops_mod.logical_or) and True


@preconditions(assert_tensor)
@pyfunc
def all(mat):  # pylint: disable=W0622
    return _reduce(mat != 0, ops_mod.logical_and) and True


@preconditions(assert_tensor)
@pyfunc
def max(mat):  # pylint: disable=W0622
    return _reduce(mat, ops_mod.max_impl)


@preconditions(assert_tensor)
@pyfunc
def min(mat):  # pylint: disable=W0622
    return _reduce(mat, ops_mod.min_impl)


@preconditions(square_matrix)
@pyfunc
def trace(mat):
    shape = static(mat.get_shape())
    result = mat[0, 0]
    # TODO: get rid of static when
    # CHI IR Tensor repr is ready stable
    for i in static(range(1, shape[0])):
        result = result + mat[i, i]
    return result


@preconditions(arg_at(0, assert_tensor))
@pyfunc
def fill(mat: template(), val):
    shape = static(mat.get_shape())
    if static(len(shape) == 1):
        for i in static(range(shape[0])):
            mat[i] = val
    elif static(len(shape) == 0):
        mat[()] = val
    else:
        for i in static(range(shape[0])):
            for j in static(range(shape[1])):
                mat[i, j] = val


@preconditions(check_matmul)
@pyfunc
def _matmul_helper(mat_x, mat_y):
    """Compute ``mat_x @ mat_y`` for vectors and matrices.

    Loop unrolling strategy for the matrix–matrix path: when the unrolled FMA count ``M*K*N`` is at most
    :data:`_MATMUL_UNROLL_THRESHOLD` (= 64, i.e. up to ``4×4·4×4``), all three loops stay
    ``static(range)`` and the kernel is fully straight-line. Above the threshold, the largest of the three
    dimensions is promoted to a runtime ``range`` so the unrolled body stays bounded by the product of the
    *other* two dims; the runtime loop is tagged ``loop_config(serialize=True)`` so a calling
    ``@qd.kernel`` with no top-level outer loop does not parallelize the matmul (see
    ``perso_hugh/doc/quadrants_runtime_range_in_func_parallelized_gotcha_20260510.md``). Tie-break on the
    largest dim prefers ``K`` (rank-1-update form, uniform per-iteration access pattern), then ``M``,
    then ``N``.

    Empirically on cluster CUDA, the cold-compile time of a square ``N×N · N×N`` matmul drops from ~12 s
    at ``N=12`` (fully unrolled, 1728 FMAs) to <1 s with the hybrid path (144 FMAs unrolled in the inner
    block, 12 runtime iterations). Vector dot and matrix-vector cases are unaffected and stay fully unrolled.
    """
    shape_x = static(mat_x.get_shape())
    shape_y = static(mat_y.get_shape())
    if static(len(shape_x) == 1 and len(shape_y) == 1):
        return dot(mat_x, mat_y)
    if static(len(shape_y) == 1):
        zero_elem = mat_x[0, 0] * mat_y[0] * 0  # for correct return type
        vec_z = _filled_vector(shape_x[0], None, zero_elem)
        for i in static(range(shape_x[0])):
            for j in static(range(shape_x[1])):
                vec_z[i] = vec_z[i] + mat_x[i, j] * mat_y[j]
        return vec_z

    M = static(shape_x[0])
    K = static(shape_x[1])
    N = static(shape_y[1])
    zero_elem = mat_x[0, 0] * mat_y[0, 0] * 0  # for correct return type
    mat_z = _filled_matrix(M, N, None, zero_elem)
    if static(M * K * N <= _MATMUL_UNROLL_THRESHOLD):
        for i in static(range(M)):
            for j in static(range(N)):
                for k in static(range(K)):
                    mat_z[i, j] = mat_z[i, j] + mat_x[i, k] * mat_y[k, j]
    elif static(K >= M and K >= N):
        # K-runtime (rank-1 update): each iteration accumulates one outer product into mat_z.
        loop_config(serialize=True)
        for k in range(K):
            for i in static(range(M)):
                for j in static(range(N)):
                    mat_z[i, j] = mat_z[i, j] + mat_x[i, k] * mat_y[k, j]
    elif static(M >= N):
        # M-runtime (row-major): each iteration computes one row of mat_z.
        loop_config(serialize=True)
        for i in range(M):
            for j in static(range(N)):
                for k in static(range(K)):
                    mat_z[i, j] = mat_z[i, j] + mat_x[i, k] * mat_y[k, j]
    else:
        # N-runtime (column-major): each iteration computes one column of mat_z.
        loop_config(serialize=True)
        for j in range(N):
            for i in static(range(M)):
                for k in static(range(K)):
                    mat_z[i, j] = mat_z[i, j] + mat_x[i, k] * mat_y[k, j]
    return mat_z


@pyfunc
def matmul(mat_x, mat_y):
    shape_x = static(mat_x.get_shape())
    shape_y = static(mat_y.get_shape())
    if static(len(shape_x) == 1 and len(shape_y) == 2):
        return _matmul_helper(transpose(mat_y), mat_x)
    return _matmul_helper(mat_x, mat_y)


@preconditions(
    arg_at(0, assert_vector("lhs for dot is not a vector")),
    arg_at(1, assert_vector("rhs for dot is not a vector")),
)
@pyfunc
def dot(vec_x, vec_y):
    return sum(vec_x * vec_y)


@preconditions(
    arg_at(0, assert_vector("lhs for cross is not a vector")),
    arg_at(1, assert_vector("rhs for cross is not a vector")),
    same_shapes,
    arg_at(0, dim_lt(0, 4)),
)
@pyfunc
def cross(vec_x, vec_y):
    shape = static(vec_x.get_shape())
    if static(shape[0] == 2):
        return vec_x[0] * vec_y[1] - vec_x[1] * vec_y[0]
    if static(shape[0] == 3):
        return Vector(
            [
                vec_x[1] * vec_y[2] - vec_x[2] * vec_y[1],
                vec_x[2] * vec_y[0] - vec_x[0] * vec_y[2],
                vec_x[0] * vec_y[1] - vec_x[1] * vec_y[0],
            ]
        )
    return None


@preconditions(
    arg_at(0, assert_vector("lhs for outer_product is not a vector")),
    arg_at(1, assert_vector("rhs for outer_product is not a vector")),
)
@pyfunc
def outer_product(vec_x, vec_y):
    shape_x = static(vec_x.get_shape())
    shape_y = static(vec_y.get_shape())
    return Matrix([[vec_x[i] * vec_y[j] for j in static(range(shape_y[0]))] for i in static(range(shape_x[0]))])


@preconditions(assert_tensor)
@func
def cast(mat, dtype: template()):
    return ops_mod.cast(mat, dtype)
