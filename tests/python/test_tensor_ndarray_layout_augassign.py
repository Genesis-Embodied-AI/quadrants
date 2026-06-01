"""Extended AugAssign coverage for layout-tagged ndarrays.

an earlier change already covered the basic ``x[i, j] += scalar`` case. This file exercises the trickier paths:
- every augmented operator (+=, -=, *=, //=, etc.)
- read-and-write on the same call (``x[i, j] = x[i, j] * 2 + x[i, j]``)
- dependence on neighbouring canonical indices (``x[i, j] += x[i, j-1]``)
- accumulating into one canonical cell from multiple iterations
- chained subscripts inside expression trees with layout on every operand
"""

import numpy as np

import quadrants as qd
from quadrants._tensor import _with_layout

from tests import test_utils


def _allocate_layout10(M, N, dtype=qd.i32):
    """Allocate an ndarray of physical shape (N, M) tagged layout=(1, 0).

    Canonical user-visible iteration is over (M, N).
    """
    a = qd.tensor(dtype, shape=(N, M), backend=qd.Backend.NDARRAY)
    _with_layout(a, (1, 0))
    return a


# ----------------------------------------------------------------------------
# Every common AugAssign operator
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_add_sub_mul_floordiv():
    M, N = 3, 4
    a = _allocate_layout10(M, N)

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = (i + 1) * 100 + j  # always positive, > 0

    @qd.kernel
    def add(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] += 7

    @qd.kernel
    def sub(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] -= 3

    @qd.kernel
    def mul(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] *= 2

    @qd.kernel
    def floordiv(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] //= 4

    init(a)
    add(a)
    sub(a)
    mul(a)
    floordiv(a)

    arr = a.to_numpy()
    # to_numpy() returns the canonical view; index canonically.
    for i in range(M):
        for j in range(N):
            expected = (((i + 1) * 100 + j + 7 - 3) * 2) // 4
            assert arr[i, j] == expected, (i, j, arr[i, j], expected)


# ----------------------------------------------------------------------------
# Read-and-write of the same canonical index in one statement
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_read_write_same_index():
    M, N = 3, 4
    a = _allocate_layout10(M, N)

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 10 + j

    @qd.kernel
    def transform(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = x[i, j] * 2 + x[i, j]

    init(a)
    transform(a)

    arr = a.to_numpy()
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == (i * 10 + j) * 3


# ----------------------------------------------------------------------------
# Neighbour dependence: ``x[i, j] += x[i, j-1]`` etc.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_neighbour_dependence_along_canonical_j():
    """A reduction-style kernel along the canonical j axis."""
    M, N = 3, 5
    a = _allocate_layout10(M, N)

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = j + 1  # 1, 2, 3, 4, 5 along j

    @qd.kernel
    def cumulate(x: qd.types.ndarray()):
        # serial scan over j, parallel over i
        for i in range(M):
            for j in range(1, N):
                x[i, j] += x[i, j - 1]

    init(a)
    cumulate(a)

    arr = a.to_numpy()
    # to_numpy() returns the canonical view; index canonically.
    expected_j = [1, 3, 6, 10, 15]
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == expected_j[j]


# ----------------------------------------------------------------------------
# Mixed layout / no-layout in the same kernel
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_mixed_with_untagged_in_same_kernel():
    """Two ndarrays in one kernel: one layout-tagged, one not. The rewrite must apply only to the tagged one."""
    M, N = 3, 4
    src = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)  # untagged
    dst = _allocate_layout10(M, N)  # tagged layout=(1, 0)

    @qd.kernel
    def init_src(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    @qd.kernel
    def copy_canonical(s: qd.types.ndarray(), d: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            d[i, j] = s[i, j]  # rewrites only on d

    init_src(src)
    copy_canonical(src, dst)

    src_np = src.to_numpy()
    dst_np = dst.to_numpy()
    # Both ``to_numpy()`` return canonical views, so the two arrays compare equal element-for-element regardless of
    # dst's tag.
    assert src_np.shape == (M, N)
    assert dst_np.shape == (M, N)
    np.testing.assert_array_equal(src_np, dst_np)


# ----------------------------------------------------------------------------
# Chained subscripts in the same expression: a[i,j] + b[i,j]*c[i,j] all tagged
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_three_tagged_operands_one_expression():
    M, N = 2, 3
    a = _allocate_layout10(M, N)
    b = _allocate_layout10(M, N)
    c = _allocate_layout10(M, N)
    out = _allocate_layout10(M, N)

    @qd.kernel
    def init(x: qd.types.ndarray(), v: qd.types.ndarray(), w: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i + 1
            v[i, j] = j + 1
            w[i, j] = (i + 1) * (j + 1)

    @qd.kernel
    def expr(
        x: qd.types.ndarray(),
        v: qd.types.ndarray(),
        w: qd.types.ndarray(),
        o: qd.types.ndarray(),
    ):
        for i, j in qd.ndrange(M, N):
            o[i, j] = x[i, j] + v[i, j] * w[i, j]

    init(a, b, c)
    expr(a, b, c, out)

    arr = out.to_numpy()
    assert arr.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == (i + 1) + (j + 1) * (i + 1) * (j + 1)


# ----------------------------------------------------------------------------
# All AugAssign operators on layout-tagged ndarrays, one kernel each
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_iadd():
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 10
            x[i, j] += 3

    k(a)
    assert (a.to_numpy() == 13).all()


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_isub():
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 10
            x[i, j] -= 3

    k(a)
    assert (a.to_numpy() == 7).all()


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_imul():
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 10
            x[i, j] *= 3

    k(a)
    assert (a.to_numpy() == 30).all()


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_ifloordiv():
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 13
            x[i, j] //= 3

    k(a)
    assert (a.to_numpy() == 4).all()


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_imod():
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 13
            x[i, j] %= 3

    k(a)
    assert (a.to_numpy() == 1).all()


@test_utils.test(arch=qd.cpu)
def test_layout_augassign_iand_ior_ixor():
    M, N = 2, 3
    a = _allocate_layout10(M, N)
    b = _allocate_layout10(M, N)
    c = _allocate_layout10(M, N)

    @qd.kernel
    def k(
        x: qd.types.ndarray(),
        y: qd.types.ndarray(),
        z: qd.types.ndarray(),
    ):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 0b1100
            x[i, j] &= 0b1010
            y[i, j] = 0b1100
            y[i, j] |= 0b1010
            z[i, j] = 0b1100
            z[i, j] ^= 0b1010

    k(a, b, c)
    assert (a.to_numpy() == (0b1100 & 0b1010)).all()
    assert (b.to_numpy() == (0b1100 | 0b1010)).all()
    assert (c.to_numpy() == (0b1100 ^ 0b1010)).all()
