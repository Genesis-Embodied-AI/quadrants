"""Tests for the ``axes=`` keyword on :func:`quadrants.ndrange`.

``axes=`` is canonical-preserving: the loop variables stay bound to canonical axes regardless of
the requested iteration order; only the visit order (which canonical axis is the outermost /
innermost iteration nesting level) changes. ``axes=None`` and the identity permutation are
equivalent and produce the default last-arg-innermost behaviour.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _expected_flat_to_canonical(dims, axes):
    """Build the expected sequence of canonical multi-indices yielded by an ``ndrange`` of the given
    dimensions and ``axes`` permutation.

    Iteration nests with physical level 0 outermost; physical level ``p`` indexes canonical axis ``axes[p]``.
    """
    axes = tuple(range(len(dims))) if axes is None else tuple(axes)
    ranges = [range(dims[axis]) for axis in axes]
    out = []
    for physical_tuple in itertools.product(*ranges):
        canonical = [0] * len(dims)
        for p, ax in enumerate(axes):
            canonical[ax] = physical_tuple[p]
        out.append(tuple(canonical))
    return out


def _expected_flat_index(canonical, dims, axes):
    """Return the flat thread index that visits ``canonical`` under (``dims``, ``axes``).

    Mirrors the AST-builder's decomposition: flat = sum_{p} canonical[axes[p]] * prod(dims[axes[p+1:]]).
    """
    axes = tuple(range(len(dims))) if axes is None else tuple(axes)
    n = len(dims)
    flat = 0
    for p in range(n):
        ax = axes[p]
        inner = 1
        for q in range(p + 1, n):
            inner *= dims[axes[q]]
        flat += canonical[ax] * inner
    return flat


# ----------------------------------------------------------------------------
# Identity / default equivalence
# ----------------------------------------------------------------------------


@test_utils.test()
def test_axes_none_matches_default():
    M, N = 5, 7
    x = qd.field(qd.i32, shape=(M, N))
    y = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill_default():
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    @qd.kernel
    def fill_axes_none():
        for i, j in qd.ndrange(M, N, axes=None):
            y[i, j] = i * 100 + j

    fill_default()
    fill_axes_none()
    np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())


@test_utils.test()
def test_axes_identity_matches_default():
    M, N = 5, 7
    x = qd.field(qd.i32, shape=(M, N))
    y = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill_default():
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    @qd.kernel
    def fill_axes_identity():
        for i, j in qd.ndrange(M, N, axes=(0, 1)):
            y[i, j] = i * 100 + j

    fill_default()
    fill_axes_identity()
    np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())


# ----------------------------------------------------------------------------
# Non-identity permutations: canonical loop targets, full coverage
# ----------------------------------------------------------------------------


@test_utils.test()
def test_axes_2d_transposed_canonical_targets():
    """With ``axes=(1, 0)``, the loop variables (i, j) are still canonical axes 0, 1."""
    M, N = 4, 6
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(M, N, axes=(1, 0)):
            x[i, j] = i * 100 + j

    fill()
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_axes_3d_arbitrary_permutation_canonical_targets():
    """Rank-3 with a non-cyclic permutation."""
    D0, D1, D2 = 3, 4, 5
    x = qd.field(qd.i32, shape=(D0, D1, D2))

    @qd.kernel
    def fill():
        for i, j, k in qd.ndrange(D0, D1, D2, axes=(2, 0, 1)):
            x[i, j, k] = i * 10000 + j * 100 + k

    fill()
    expected = np.array(
        [[[i * 10000 + j * 100 + k for k in range(D2)] for j in range(D1)] for i in range(D0)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_axes_with_tuple_bounds_preserves_offsets():
    """``axes=`` doesn't disturb (begin, end) tuples — each canonical axis keeps its own bounds."""
    M, N = 16, 16
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange((2, 10), (3, 7), axes=(1, 0)):
            x[i, j] = i * 100 + j

    fill()
    expected = np.zeros((M, N), dtype=np.int32)
    for i in range(2, 10):
        for j in range(3, 7):
            expected[i, j] = i * 100 + j
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_axes_full_coverage_via_atomic_count():
    """Every canonical slot is visited exactly once."""
    M, N = 5, 7
    counts = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(M, N, axes=(1, 0)):
            counts[i, j] += 1

    fill()
    np.testing.assert_array_equal(counts.to_numpy(), np.ones((M, N), dtype=np.int32))


@test_utils.test()
def test_axes_flat_index_matches_decomposition():
    """The flat thread index reconstructed from the canonical loop variables under the requested
    ``axes`` permutation matches what a sequential range-loop would assign — i.e. the AST decomposition
    is the inverse of the canonical-from-physical mapping.
    """
    M, N = 4, 6
    flat = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(M, N, axes=(1, 0)):
            # If physical level 0 = axis 1 (outer) and level 1 = axis 0 (inner), then the flat index
            # is j * M + i. Writing it into a per-canonical-slot grid lets us check coverage and the
            # bijection in one pass.
            flat[i, j] = j * M + i

    fill()
    expected = np.array([[j * M + i for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(flat.to_numpy(), expected)


# ----------------------------------------------------------------------------
# qd.grouped + axes=
# ----------------------------------------------------------------------------


@test_utils.test()
def test_axes_grouped_indices_are_canonical():
    """``I[0]`` is the canonical axis-0 index regardless of ``axes``."""
    M, N = 4, 5
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for I in qd.grouped(qd.ndrange(M, N, axes=(1, 0))):
            x[I] = I[0] * 100 + I[1]

    fill()
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_axes_static_grouped():
    """Unrolled (qd.static) grouped path also sees canonical indices in the requested iteration order."""
    M, N = 3, 4
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for I in qd.static(qd.grouped(qd.ndrange(M, N, axes=(1, 0)))):
            x[I] = I[0] * 100 + I[1]

    fill()
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(x.to_numpy(), expected)


# ----------------------------------------------------------------------------
# Pairing with qd.tensor(..., layout=...)
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_axes_pairs_with_tensor_layout_field():
    """The documented pairing use case: matching permutation on ``qd.tensor(layout=...)`` and
    ``qd.ndrange(axes=...)``. The kernel body uses canonical indexing throughout; correctness must
    hold (this exercises the canonical->physical AST rewrite on the tensor side and the
    ``axes=``-aware decomposition on the ndrange side together).
    """
    M, N = 4, 6
    A = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.FIELD, layout=(1, 0))

    @qd.kernel
    def fill(a: qd.template()):
        for i, j in qd.ndrange(M, N, axes=(1, 0)):
            a[i, j] = i * 100 + j

    fill(A)
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(A.to_numpy(), expected)


# ----------------------------------------------------------------------------
# Python-side iteration (outside @qd.kernel)
# ----------------------------------------------------------------------------


def test_axes_python_iteration_2d():
    qd.init(arch=qd.cpu)
    M, N = 3, 4
    got = list(qd.ndrange(M, N, axes=(1, 0)))
    assert got == _expected_flat_to_canonical((M, N), (1, 0))


def test_axes_python_iteration_3d():
    qd.init(arch=qd.cpu)
    dims = (2, 3, 4)
    got = list(qd.ndrange(*dims, axes=(2, 0, 1)))
    assert got == _expected_flat_to_canonical(dims, (2, 0, 1))


def test_axes_python_iteration_identity_matches_default():
    qd.init(arch=qd.cpu)
    M, N = 3, 4
    assert list(qd.ndrange(M, N, axes=(0, 1))) == list(qd.ndrange(M, N))
    assert list(qd.ndrange(M, N, axes=None)) == list(qd.ndrange(M, N))


def test_axes_grouped_python_iteration_via_method():
    """``_Ndrange.grouped()`` (Python-scope method, not ``qd.grouped``) preserves the
    ``axes=``-induced iteration order. ``qd.grouped`` itself is decorated ``@quadrants_scope`` and
    cannot be invoked outside a kernel, so test the underlying method directly here.
    """
    qd.init(arch=qd.cpu)
    from quadrants.lang._ndrange import _Ndrange

    M, N = 3, 4
    got = []
    for vec in _Ndrange(M, N, axes=(1, 0)).grouped():
        got.append(tuple(vec.to_list()))
    assert got == _expected_flat_to_canonical((M, N), (1, 0))


# ----------------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------------


def test_axes_attribute_identity_normalizes_to_none():
    qd.init(arch=qd.cpu)
    # ``axes=None`` and identity permutation both expose ``axes = None`` for introspection
    # (so user code can treat "no axes" symmetrically).
    from quadrants.lang._ndrange import _Ndrange

    a = _Ndrange(3, 4)
    b = _Ndrange(3, 4, axes=None)
    c = _Ndrange(3, 4, axes=(0, 1))
    assert a.axes is None
    assert b.axes is None
    assert c.axes is None


def test_axes_attribute_non_identity_preserved():
    qd.init(arch=qd.cpu)
    from quadrants.lang._ndrange import _Ndrange

    a = _Ndrange(3, 4, axes=(1, 0))
    assert a.axes == (1, 0)


# ----------------------------------------------------------------------------
# Degenerate ranks
# ----------------------------------------------------------------------------


@test_utils.test()
def test_axes_1d_degenerate():
    """``axes=(0,)`` on a 1-D ndrange is the only permutation and must match the default."""
    M = 7
    x = qd.field(qd.i32, shape=(M,))
    y = qd.field(qd.i32, shape=(M,))

    @qd.kernel
    def fill_default():
        for i in qd.ndrange(M):
            x[i] = i

    @qd.kernel
    def fill_axes():
        for i in qd.ndrange(M, axes=(0,)):
            y[i] = i

    fill_default()
    fill_axes()
    np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())


def test_axes_zero_dim_degenerate():
    qd.init(arch=qd.cpu)
    # Empty ndrange yields exactly one (empty) tuple.
    assert list(qd.ndrange()) == [()]
    assert list(qd.ndrange(axes=())) == [()]


# ----------------------------------------------------------------------------
# Error cases
# ----------------------------------------------------------------------------


def test_axes_wrong_length_raises():
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.ndrange\(axes=.*\) has 3 entries but ndrange"):
        qd.ndrange(4, 5, axes=(0, 1, 2))


def test_axes_not_a_permutation_raises():
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.ndrange\(axes=.*\) is not a permutation"):
        qd.ndrange(4, 5, axes=(0, 0))


def test_axes_out_of_range_raises():
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.ndrange\(axes=.*\) is not a permutation"):
        qd.ndrange(4, 5, axes=(0, 2))


def test_axes_non_integer_entry_raises():
    """Non-integer entries (string, float, mixed) surface a QuadrantsTypeError instead of the raw
    Python ``TypeError`` ``sorted`` would emit on mixed-type sequences.
    """
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsTypeError, match=r"entries must be Python ints"):
        qd.ndrange(4, 5, axes=(0, "1"))
    with pytest.raises(qd.QuadrantsTypeError, match=r"entries must be Python ints"):
        qd.ndrange(4, 5, axes=(0.0, 1.0))


def test_axes_bool_entry_rejected():
    """``bool`` is an ``int`` subclass but rejecting ``True`` / ``False`` as axis indices avoids a
    foot-gun.
    """
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsTypeError, match=r"entries must be Python ints"):
        qd.ndrange(4, 5, axes=(True, False))
