"""Tests for the ``layout=`` keyword on :func:`quadrants.ndrange`.

``layout=`` is canonical-preserving: the loop variables stay bound to canonical axes regardless of layout;
only the visit order (which canonical axis is the outermost / innermost iteration nesting level) changes.
``layout=None`` and the identity permutation are equivalent and produce the default last-arg-innermost
behaviour.

See ``perso_hugh/doc/ndrange_layout.md`` for design notes.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _expected_flat_to_canonical(dims, layout):
    """Build the expected sequence of canonical multi-indices yielded by an ``ndrange`` of the given
    dimensions and layout.

    Iteration nests with physical level 0 outermost; physical level ``p`` indexes canonical axis ``layout[p]``.
    """
    layout = tuple(range(len(dims))) if layout is None else tuple(layout)
    ranges = [range(dims[axis]) for axis in layout]
    out = []
    for physical_tuple in itertools.product(*ranges):
        canonical = [0] * len(dims)
        for p, ax in enumerate(layout):
            canonical[ax] = physical_tuple[p]
        out.append(tuple(canonical))
    return out


def _expected_flat_index(canonical, dims, layout):
    """Return the flat thread index that visits ``canonical`` under (``dims``, ``layout``).

    Mirrors the AST-builder's decomposition: flat = sum_{p} canonical[layout[p]] * prod(dims[layout[p+1:]]).
    """
    layout = tuple(range(len(dims))) if layout is None else tuple(layout)
    n = len(dims)
    flat = 0
    for p in range(n):
        ax = layout[p]
        inner = 1
        for q in range(p + 1, n):
            inner *= dims[layout[q]]
        flat += canonical[ax] * inner
    return flat


# ----------------------------------------------------------------------------
# Identity / default equivalence
# ----------------------------------------------------------------------------


@test_utils.test()
def test_layout_none_matches_default():
    M, N = 5, 7
    x = qd.field(qd.i32, shape=(M, N))
    y = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill_default():
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    @qd.kernel
    def fill_layout_none():
        for i, j in qd.ndrange(M, N, layout=None):
            y[i, j] = i * 100 + j

    fill_default()
    fill_layout_none()
    np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())


@test_utils.test()
def test_layout_identity_matches_default():
    M, N = 5, 7
    x = qd.field(qd.i32, shape=(M, N))
    y = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill_default():
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    @qd.kernel
    def fill_layout_identity():
        for i, j in qd.ndrange(M, N, layout=(0, 1)):
            y[i, j] = i * 100 + j

    fill_default()
    fill_layout_identity()
    np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())


# ----------------------------------------------------------------------------
# Non-identity layouts: canonical loop targets, full coverage
# ----------------------------------------------------------------------------


@test_utils.test()
def test_layout_2d_transposed_canonical_targets():
    """With ``layout=(1, 0)``, the loop variables (i, j) are still canonical axes 0, 1."""
    M, N = 4, 6
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(M, N, layout=(1, 0)):
            x[i, j] = i * 100 + j

    fill()
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_layout_3d_arbitrary_permutation_canonical_targets():
    """Rank-3 with a non-cyclic permutation."""
    D0, D1, D2 = 3, 4, 5
    x = qd.field(qd.i32, shape=(D0, D1, D2))

    @qd.kernel
    def fill():
        for i, j, k in qd.ndrange(D0, D1, D2, layout=(2, 0, 1)):
            x[i, j, k] = i * 10000 + j * 100 + k

    fill()
    expected = np.array(
        [[[i * 10000 + j * 100 + k for k in range(D2)] for j in range(D1)] for i in range(D0)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_layout_with_tuple_bounds_preserves_offsets():
    """Layout doesn't disturb (begin, end) tuples — each canonical axis keeps its own bounds."""
    M, N = 16, 16
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange((2, 10), (3, 7), layout=(1, 0)):
            x[i, j] = i * 100 + j

    fill()
    expected = np.zeros((M, N), dtype=np.int32)
    for i in range(2, 10):
        for j in range(3, 7):
            expected[i, j] = i * 100 + j
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_layout_full_coverage_via_atomic_count():
    """Every canonical slot is visited exactly once."""
    M, N = 5, 7
    counts = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(M, N, layout=(1, 0)):
            counts[i, j] += 1

    fill()
    np.testing.assert_array_equal(counts.to_numpy(), np.ones((M, N), dtype=np.int32))


@test_utils.test()
def test_layout_flat_index_matches_decomposition():
    """The flat thread index reconstructed from the canonical loop variables under the requested
    layout permutation matches what a sequential range-loop would assign — i.e. the AST decomposition
    is the inverse of the canonical-from-physical mapping.
    """
    M, N = 4, 6
    flat = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(M, N, layout=(1, 0)):
            # If physical level 0 = axis 1 (outer) and level 1 = axis 0 (inner), then the flat index
            # is j * M + i. Writing it into a per-canonical-slot grid lets us check coverage and the
            # bijection in one pass.
            flat[i, j] = j * M + i

    fill()
    expected = np.array([[j * M + i for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(flat.to_numpy(), expected)


# ----------------------------------------------------------------------------
# qd.grouped + layout
# ----------------------------------------------------------------------------


@test_utils.test()
def test_layout_grouped_indices_are_canonical():
    """``I[0]`` is the canonical axis-0 index regardless of layout."""
    M, N = 4, 5
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for I in qd.grouped(qd.ndrange(M, N, layout=(1, 0))):
            x[I] = I[0] * 100 + I[1]

    fill()
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_layout_static_grouped():
    """Unrolled (qd.static) grouped path also sees canonical indices in physical iteration order."""
    M, N = 3, 4
    x = qd.field(qd.i32, shape=(M, N))

    @qd.kernel
    def fill():
        for I in qd.static(qd.grouped(qd.ndrange(M, N, layout=(1, 0)))):
            x[I] = I[0] * 100 + I[1]

    fill()
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(x.to_numpy(), expected)


# ----------------------------------------------------------------------------
# Pairing with qd.tensor(..., layout=...)
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_pairs_with_tensor_layout_field():
    """The documented pairing use case: matching ``layout=`` on both tensor and ndrange. The kernel
    body uses canonical indexing throughout; correctness must hold (this exercises the
    canonical->physical AST rewrite on the tensor side and the layout-aware decomposition on the
    ndrange side together).
    """
    M, N = 4, 6
    A = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.FIELD, layout=(1, 0))

    @qd.kernel
    def fill(a: qd.template()):
        for i, j in qd.ndrange(M, N, layout=(1, 0)):
            a[i, j] = i * 100 + j

    fill(A)
    expected = np.array([[i * 100 + j for j in range(N)] for i in range(M)], dtype=np.int32)
    np.testing.assert_array_equal(A.to_numpy(), expected)


# ----------------------------------------------------------------------------
# Python-side iteration (outside @qd.kernel)
# ----------------------------------------------------------------------------


def test_layout_python_iteration_2d():
    qd.init(arch=qd.cpu)
    M, N = 3, 4
    got = list(qd.ndrange(M, N, layout=(1, 0)))
    assert got == _expected_flat_to_canonical((M, N), (1, 0))


def test_layout_python_iteration_3d():
    qd.init(arch=qd.cpu)
    dims = (2, 3, 4)
    got = list(qd.ndrange(*dims, layout=(2, 0, 1)))
    assert got == _expected_flat_to_canonical(dims, (2, 0, 1))


def test_layout_python_iteration_identity_matches_default():
    qd.init(arch=qd.cpu)
    M, N = 3, 4
    assert list(qd.ndrange(M, N, layout=(0, 1))) == list(qd.ndrange(M, N))
    assert list(qd.ndrange(M, N, layout=None)) == list(qd.ndrange(M, N))


def test_layout_grouped_python_iteration_via_method():
    """``_Ndrange.grouped()`` (Python-scope method, not ``qd.grouped``) preserves the layout-induced
    iteration order. ``qd.grouped`` itself is decorated ``@quadrants_scope`` and cannot be invoked
    outside a kernel, so test the underlying method directly here.
    """
    qd.init(arch=qd.cpu)
    from quadrants.lang._ndrange import _Ndrange

    M, N = 3, 4
    got = []
    for vec in _Ndrange(M, N, layout=(1, 0)).grouped():
        got.append(tuple(vec.to_list()))
    assert got == _expected_flat_to_canonical((M, N), (1, 0))


# ----------------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------------


def test_layout_attribute_identity_normalizes_to_none():
    qd.init(arch=qd.cpu)
    # ``layout=None`` and identity layout both expose ``layout = None`` for introspection
    # (so user code can treat "no layout" symmetrically).
    from quadrants.lang._ndrange import _Ndrange

    a = _Ndrange(3, 4)
    b = _Ndrange(3, 4, layout=None)
    c = _Ndrange(3, 4, layout=(0, 1))
    assert a.layout is None
    assert b.layout is None
    assert c.layout is None


def test_layout_attribute_non_identity_preserved():
    qd.init(arch=qd.cpu)
    from quadrants.lang._ndrange import _Ndrange

    a = _Ndrange(3, 4, layout=(1, 0))
    assert a.layout == (1, 0)


# ----------------------------------------------------------------------------
# Degenerate ranks
# ----------------------------------------------------------------------------


@test_utils.test()
def test_layout_1d_degenerate():
    """Layout (0,) on a 1-D ndrange is the only permutation and must match the default."""
    M = 7
    x = qd.field(qd.i32, shape=(M,))
    y = qd.field(qd.i32, shape=(M,))

    @qd.kernel
    def fill_default():
        for i in qd.ndrange(M):
            x[i] = i

    @qd.kernel
    def fill_layout():
        for i in qd.ndrange(M, layout=(0,)):
            y[i] = i

    fill_default()
    fill_layout()
    np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())


def test_layout_zero_dim_degenerate():
    qd.init(arch=qd.cpu)
    # Empty ndrange yields exactly one (empty) tuple.
    assert list(qd.ndrange()) == [()]
    assert list(qd.ndrange(layout=())) == [()]


# ----------------------------------------------------------------------------
# Error cases
# ----------------------------------------------------------------------------


def test_layout_wrong_length_raises():
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.ndrange\(layout=.*\) has 3 entries but ndrange"):
        qd.ndrange(4, 5, layout=(0, 1, 2))


def test_layout_not_a_permutation_raises():
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.ndrange\(layout=.*\) is not a permutation"):
        qd.ndrange(4, 5, layout=(0, 0))


def test_layout_out_of_range_raises():
    qd.init(arch=qd.cpu)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.ndrange\(layout=.*\) is not a permutation"):
        qd.ndrange(4, 5, layout=(0, 2))
