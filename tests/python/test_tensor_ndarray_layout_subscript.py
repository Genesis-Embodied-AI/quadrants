"""Tests for the AnyArray subscript-rewrite.

Covers the metadata-flow + AST-rewrite plumbing only. To exercise the rewrite end-to-end without going through the
public ``qd.tensor`` factory, this file uses the internal ``_with_layout`` helper to tag an ndarray allocated at the
*physical* shape with a canonical-axis layout.

Conventions:
- The ndarray is allocated at the **physical** shape, then tagged.
- ``ndarray.shape`` reports the **canonical** shape (the inverse permutation of the physical shape under the tag).
- Inside a kernel, indices are interpreted as **canonical**: ``x[i, j]`` means logical index ``(i, j)`` and the
  rewrite turns it into physical ``(j, i)`` for ``layout=(1, 0)``.
- ``to_numpy()`` returns the **canonical** view (a transposed view of the underlying physical buffer; no data movement
  on the kernel side).
"""

import itertools

import numpy as np
import pytest

import quadrants as qd
from quadrants._tensor import _with_layout

from tests import test_utils

# ----------------------------------------------------------------------------
# Identity / no-tag: behaviour must be unchanged from legacy
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_subscript_untagged_ndarray_unaffected():
    a = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(3, 4):
            x[i, j] = i * 100 + j

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == (3, 4)
    assert arr[2, 3] == 203


@test_utils.test(arch=qd.cpu)
def test_subscript_identity_layout_is_byte_identical():
    """Tagging with the identity permutation must behave like no tag."""
    a = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY)
    b = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.NDARRAY)
    _with_layout(b, (0, 1))

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(3, 4):
            x[i, j] = i * 100 + j

    fill(a)
    fill(b)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


# ----------------------------------------------------------------------------
# Non-identity layout: AST rewrite turns canonical indexing into permuted physical indexing
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_subscript_rank2_transpose_layout_matches_transposed_storage():
    """layout=(1, 0): canonical x[i, j] -> physical[j, i].

    Allocate two ndarrays:
    - direct: canonical (3, 4); kernel writes x[i, j] in canonical order.
    - tagged: physical (4, 3); tagged with layout=(1, 0) so canonical shape is (3, 4). Kernel iterates the canonical
      (3, 4) range and writes x[i, j]. Rewrite turns this into physical[j, i].
    Both ``to_numpy()`` calls return the canonical view, so the two arrays must compare equal element-for-element.
    """
    M, N = 3, 4
    direct = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)
    tagged_phys = qd.tensor(qd.i32, shape=(N, M), backend=qd.Backend.NDARRAY)
    _with_layout(tagged_phys, (1, 0))

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    fill(direct)
    fill(tagged_phys)
    np.testing.assert_array_equal(direct.to_numpy(), tagged_phys.to_numpy())


@test_utils.test(arch=qd.cpu)
def test_subscript_rank2_transpose_layout_explicit_value_check():
    """Spot-check exact canonical positions after a layout=(1, 0) rewrite."""
    M, N = 3, 4
    a_phys = qd.tensor(qd.i32, shape=(N, M), backend=qd.Backend.NDARRAY)
    _with_layout(a_phys, (1, 0))

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j  # rewritten to x[j, i] physically

    fill(a_phys)
    arr = a_phys.to_numpy()
    # Canonical view: shape and indices match what the kernel wrote.
    assert arr.shape == (M, N)
    assert arr[2, 3] == 203
    assert arr[0, 1] == 1
    assert arr[1, 0] == 100


@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_subscript_rank3_all_permutations(layout):
    """Every rank-3 permutation: canonical access produces correct physical layout."""
    qd.init(arch=qd.x64)
    canonical_shape = (2, 3, 4)
    # physical_shape[k] = canonical_shape[layout[k]]
    physical_shape = tuple(canonical_shape[axis] for axis in layout)

    a_phys = qd.tensor(qd.i32, shape=physical_shape, backend=qd.Backend.NDARRAY)
    _with_layout(a_phys, layout)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j, k in qd.ndrange(2, 3, 4):
            x[i, j, k] = i * 100 + j * 10 + k

    fill(a_phys)
    arr = a_phys.to_numpy()
    # to_numpy() returns the canonical view regardless of layout.
    assert arr.shape == canonical_shape
    canonical_idx = (1, 2, 3)
    expected = canonical_idx[0] * 100 + canonical_idx[1] * 10 + canonical_idx[2]
    assert arr[canonical_idx] == expected


# ----------------------------------------------------------------------------
# Aug-assign and read-modify-write
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_subscript_layout_augassign_rank2():
    """``x[i, j] += ...`` must be rewritten consistently for both subscripts."""
    M, N = 2, 3
    a_phys = qd.tensor(qd.i32, shape=(N, M), backend=qd.Backend.NDARRAY)
    _with_layout(a_phys, (1, 0))

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 10 + j

    @qd.kernel
    def add(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] += 1000

    init(a_phys)
    add(a_phys)
    arr = a_phys.to_numpy()
    # Canonical view; canonical (i=1, j=2) initially 12, after += 1000 -> 1012.
    assert arr.shape == (M, N)
    assert arr[1, 2] == 1012
    assert arr[0, 0] == 1000


# ----------------------------------------------------------------------------
# Grad propagation: AnyArray.grad inherits the layout
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_subscript_layout_grad_inherits_layout():
    """The .grad accessor on a layout-tagged AnyArray must use the same layout."""
    M, N = 2, 3
    a_phys = qd.tensor(qd.f32, shape=(N, M), backend=qd.Backend.NDARRAY, needs_grad=True)
    _with_layout(a_phys, (1, 0))

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = float(i * 10 + j)
            x.grad[i, j] = float(i * 100 + j * 10)

    fill(a_phys)
    primal = a_phys.to_numpy()
    grad = a_phys.grad.to_numpy()
    # Canonical view on both primal and grad.
    assert primal.shape == grad.shape == (M, N)
    assert primal[1, 2] == 12.0
    assert grad[1, 2] == 120.0


# ----------------------------------------------------------------------------
# _with_layout validation
# ----------------------------------------------------------------------------


def test_with_layout_wrong_length_raises():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.NDARRAY)
    with pytest.raises(ValueError, match="layout has"):
        _with_layout(a, (0, 1, 2))


def test_with_layout_not_permutation_raises():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.NDARRAY)
    with pytest.raises(ValueError, match="not a permutation"):
        _with_layout(a, (0, 0))
