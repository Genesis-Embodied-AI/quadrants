"""Tests for ``qd.tensor(..., backend=NDARRAY, layout=...)``.

an earlier change added the AST subscript-rewrite plumbing for layout-tagged ``AnyArray``s; an earlier change wires it
through the public factory so users no longer need the internal ``_with_layout`` helper.

Contract verified here:

- ``shape`` is the **canonical** shape the user indexes inside kernels; ``Ndarray.shape`` inverts the layout permutation
  when ``_qd_layout`` is set so the user-facing contract is consistent across backends.
- The underlying allocation is sized to the *physical* (permuted) shape (``_physical_shape``), but ``to_numpy()``
  permutes back to the canonical view so callers never have to reason about the layout.
- The instance is auto-tagged with ``_qd_layout`` so kernel subscripts ``x[i, j, ...]`` are rewritten correctly.
- ``order=`` is still rejected as a keyword.
- Identity layouts produce no tag and behave exactly like a plain untagged ndarray.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

# ----------------------------------------------------------------------------
# Identity / no-layout: behaviour is unchanged from an earlier change.
# ----------------------------------------------------------------------------


def test_factory_no_layout_does_not_tag():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY)
    assert getattr(a._unwrap(), "_qd_layout", None) is None
    assert tuple(a.shape) == (3, 4)


def test_factory_identity_layout_does_not_tag():
    """Identity layout collapses to ``None`` (matches the FIELD path)."""
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY, layout=(0, 1))
    assert getattr(a._unwrap(), "_qd_layout", None) is None
    assert tuple(a.shape) == (3, 4)


def test_factory_rejects_order_kwarg():
    qd.init(arch=qd.x64)
    with pytest.raises(TypeError, match="order="):
        qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY, order="ji")


# ----------------------------------------------------------------------------
# Non-identity layout: factory allocates physical shape and tags.
# ----------------------------------------------------------------------------


def test_factory_non_identity_layout_allocates_physical_and_tags():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY, layout=(1, 0))
    # Canonical shape is what the user passed; physical buffer is (4, 3).
    assert tuple(a.shape) == (3, 4)
    impl = a._unwrap()
    assert tuple(impl._physical_shape) == (4, 3)
    assert impl._qd_layout == (1, 0)


def test_factory_non_identity_rank3_layout_allocates_physical_and_tags():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.i32, shape=(2, 3, 4), backend=qd.Backend.NDARRAY, layout=(2, 0, 1))
    # Canonical (2, 3, 4); physical buffer = (4, 2, 3) per layout (2, 0, 1).
    assert tuple(a.shape) == (2, 3, 4)
    impl = a._unwrap()
    assert tuple(impl._physical_shape) == (4, 2, 3)
    assert impl._qd_layout == (2, 0, 1)


def test_factory_validates_layout_length():
    qd.init(arch=qd.x64)
    with pytest.raises(ValueError):
        qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY, layout=(0, 1, 2))


def test_factory_validates_layout_is_permutation():
    qd.init(arch=qd.x64)
    with pytest.raises(ValueError):
        qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY, layout=(0, 0))


# ----------------------------------------------------------------------------
# End-to-end: factory-allocated ndarrays match the _with_layout reference.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_factory_layout_rank2_transpose_matches_direct_canonical():
    """Same kernel + same canonical shape, two ndarrays:
    - direct (no layout): canonical == physical == (3, 4).
    - factory-tagged (layout=(1, 0)): canonical == (3, 4), physical == (4, 3).
    ``to_numpy()`` returns the canonical view in both cases, so the two numpy arrays must compare equal
    element-for-element.
    """
    M, N = 3, 4
    direct = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)
    tagged = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY, layout=(1, 0))

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    fill(direct)
    fill(tagged)

    np.testing.assert_array_equal(direct.to_numpy(), tagged.to_numpy())


@test_utils.test(arch=qd.cpu)
def test_factory_layout_rank2_value_check():
    M, N = 3, 4
    a = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY, layout=(1, 0))

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    fill(a)
    arr = a.to_numpy()
    # to_numpy() returns the canonical view: shape and indices match what the kernel wrote, regardless of physical
    # layout.
    assert arr.shape == (M, N)
    assert arr[2, 3] == 203
    assert arr[0, 1] == 1
    assert arr[1, 0] == 100


@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_factory_layout_rank3_all_permutations(layout):
    qd.init(arch=qd.x64)
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)

    # User-facing shape is canonical regardless of layout.
    assert tuple(a.shape) == canonical
    physical = tuple(canonical[axis] for axis in layout)
    impl = a._unwrap()
    assert tuple(impl._physical_shape) == physical
    # Identity layout collapses to no tag (matches the FIELD path).
    if layout == tuple(range(3)):
        assert getattr(impl, "_qd_layout", None) is None
    else:
        assert impl._qd_layout == layout

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j, k in qd.ndrange(2, 3, 4):
            x[i, j, k] = i * 100 + j * 10 + k

    fill(a)
    arr = a.to_numpy()
    # to_numpy() returns the canonical view, so canonical indices work directly regardless of layout.
    assert arr.shape == canonical
    canonical_idx = (1, 2, 3)
    expected = canonical_idx[0] * 100 + canonical_idx[1] * 10 + canonical_idx[2]
    assert arr[canonical_idx] == expected


# ----------------------------------------------------------------------------
# AugAssign through the public factory.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_factory_layout_augassign():
    M, N = 2, 3
    a = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY, layout=(1, 0))
    # Sanity-check the canonical shape contract before exercising augassign.
    assert tuple(a.shape) == (M, N)

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 10 + j

    @qd.kernel
    def add(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] += 1000

    init(a)
    add(a)
    arr = a.to_numpy()
    # to_numpy() returns the canonical view.
    assert arr.shape == (M, N)
    assert arr[1, 2] == 1012
    assert arr[0, 0] == 1000


# ----------------------------------------------------------------------------
# needs_grad propagates and grad inherits the layout.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_factory_layout_needs_grad_inherits_layout():
    M, N = 2, 3
    a = qd.tensor(
        qd.f32,
        shape=(M, N),
        backend=qd.Backend.NDARRAY,
        layout=(1, 0),
        needs_grad=True,
    )

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = float(i * 10 + j)
            x.grad[i, j] = float(i * 100 + j * 10)

    # Canonical shape is (M, N) on both primal and grad; grad inherits the same _qd_layout tag.
    assert tuple(a.shape) == tuple(a.grad.shape) == (M, N)
    fill(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    # to_numpy() returns the canonical view; both primal and grad report canonical shape and canonical-indexed values.
    assert primal.shape == grad.shape == (M, N)
    assert primal[1, 2] == 12.0
    assert grad[1, 2] == 120.0


# ----------------------------------------------------------------------------
# Cache key still distinguishes layouts when the factory does the tagging.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_factory_layout_distinguishes_cache_entries():
    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    M, N = 2, 3
    a_id = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)
    a_swap = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY, layout=(1, 0))

    k(a_id)
    assert len(k._primal.mapper.mapping) == 1

    k(a_swap)
    assert len(k._primal.mapper.mapping) == 2
