"""Higher-rank coverage for layout-tagged ndarrays.

an earlier change's parametrized rank-3 test exercises every permutation, but only on a single canonical cell.
an earlier change widens that to:

- Rank 4: every permutation, full-grid value comparison.
- Rank 5 and 6: spot checks (24 / 720 perms is too many to enumerate).
- AugAssign + grad on a rank-4 layout-tagged ndarray.

The Quadrants ``quadrants_max_num_indices`` is 12, so up to 12-D should work in principle; in practice ndrange of
higher dimensions becomes expensive, and 6-D is enough to demonstrate the rewrite scales linearly with rank.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd
from quadrants._tensor import _with_layout

from tests import test_utils


def _allocate_with_layout(canonical_shape, layout, dtype=qd.i32, needs_grad=False):
    """Helper: allocate at the *physical* shape implied by canonical_shape + layout, then tag."""
    physical_shape = tuple(canonical_shape[axis] for axis in layout)
    a = qd.tensor(dtype, shape=physical_shape, backend=qd.Backend.NDARRAY, needs_grad=needs_grad)
    _with_layout(a, layout)
    return a, physical_shape


def _canonical_to_physical(idx, layout):
    return tuple(idx[axis] for axis in layout)


# ----------------------------------------------------------------------------
# Rank 4: every permutation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", list(itertools.permutations(range(4))))
def test_layout_rank4_all_permutations_full_grid(layout):
    qd.init(arch=qd.x64)
    canonical = (2, 2, 3, 2)
    a, _ = _allocate_with_layout(canonical, layout)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j, k, l in qd.ndrange(2, 2, 3, 2):
            x[i, j, k, l] = i * 1000 + j * 100 + k * 10 + l

    fill(a)
    arr = a.to_numpy()
    # to_numpy() returns the canonical view regardless of layout.
    assert arr.shape == canonical
    for ci in itertools.product(*[range(d) for d in canonical]):
        expected = ci[0] * 1000 + ci[1] * 100 + ci[2] * 10 + ci[3]
        assert arr[ci] == expected, (layout, ci, arr[ci], expected)


# ----------------------------------------------------------------------------
# Rank 5: spot checks on a few representative permutations
# ----------------------------------------------------------------------------


_RANK5_LAYOUTS = [
    (0, 1, 2, 3, 4),  # identity
    (4, 3, 2, 1, 0),  # full reverse
    (0, 1, 2, 4, 3),  # innermost swap
    (4, 0, 1, 2, 3),  # cyclic shift
    (1, 0, 3, 2, 4),  # adjacent pair swaps
]


@pytest.mark.parametrize("layout", _RANK5_LAYOUTS)
def test_layout_rank5_spot_checks(layout):
    qd.init(arch=qd.x64)
    canonical = (2, 2, 2, 2, 2)
    a, _ = _allocate_with_layout(canonical, layout)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j, k, l, m in qd.ndrange(2, 2, 2, 2, 2):
            x[i, j, k, l, m] = i * 10000 + j * 1000 + k * 100 + l * 10 + m

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    for ci in itertools.product(*[range(d) for d in canonical]):
        expected = ci[0] * 10000 + ci[1] * 1000 + ci[2] * 100 + ci[3] * 10 + ci[4]
        assert arr[ci] == expected


# ----------------------------------------------------------------------------
# Rank 6
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout",
    [
        (0, 1, 2, 3, 4, 5),  # identity
        (5, 4, 3, 2, 1, 0),  # full reverse
        (1, 0, 3, 2, 5, 4),  # adjacent pair swaps
    ],
)
def test_layout_rank6_spot_checks(layout):
    qd.init(arch=qd.x64)
    canonical = (2, 2, 2, 2, 2, 2)
    a, _ = _allocate_with_layout(canonical, layout)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j, k, l, m, n in qd.ndrange(2, 2, 2, 2, 2, 2):
            x[i, j, k, l, m, n] = i * 100000 + j * 10000 + k * 1000 + l * 100 + m * 10 + n

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    for ci in itertools.product(*[range(d) for d in canonical]):
        expected = ci[0] * 100000 + ci[1] * 10000 + ci[2] * 1000 + ci[3] * 100 + ci[4] * 10 + ci[5]
        assert arr[ci] == expected


# ----------------------------------------------------------------------------
# Rank 4 + AugAssign + grad
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_rank4_augassign_and_grad():
    layout = (3, 0, 2, 1)
    canonical = (2, 3, 2, 2)
    a, _ = _allocate_with_layout(canonical, layout, dtype=qd.f32, needs_grad=True)

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j, k, l in qd.ndrange(2, 3, 2, 2):
            x[i, j, k, l] = float(i * 1000 + j * 100 + k * 10 + l)
            x.grad[i, j, k, l] = float(i * 10 + j)

    @qd.kernel
    def add_one_everywhere(x: qd.types.ndarray()):
        for i, j, k, l in qd.ndrange(2, 3, 2, 2):
            x[i, j, k, l] += 1.0
            x.grad[i, j, k, l] += 100.0

    init(a)
    add_one_everywhere(a)

    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == canonical
    for ci in itertools.product(*[range(d) for d in canonical]):
        expected_p = ci[0] * 1000 + ci[1] * 100 + ci[2] * 10 + ci[3] + 1
        expected_g = ci[0] * 10 + ci[1] + 100
        assert primal[ci] == expected_p
        assert grad[ci] == expected_g


# ----------------------------------------------------------------------------
# Rank-4 cross-check: tagged-with-layout matches direct-with-permuted-iteration
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_rank4_tagged_matches_direct_permuted():
    canonical = (2, 3, 2, 2)
    layout = (2, 0, 3, 1)
    physical = tuple(canonical[axis] for axis in layout)

    tagged, _ = _allocate_with_layout(canonical, layout)
    direct = qd.tensor(qd.i32, shape=physical, backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill_tagged(x: qd.types.ndarray()):
        for i, j, k, l in qd.ndrange(*canonical):
            x[i, j, k, l] = i * 1000 + j * 100 + k * 10 + l

    @qd.kernel
    def fill_direct(x: qd.types.ndarray()):
        # Iterate over *physical* axes; map back to canonical for the value. We index physical[a, b, c, d] ->
        # canonical position is the inverse. For layout=(2, 0, 3, 1): physical axis 0 is canonical axis 2, etc.
        # So canonical i (axis 0) is physical axis 1, canonical j (axis 1) is physical axis 3,
        # canonical k (axis 2) is physical axis 0, canonical l (axis 3) is physical axis 2.
        for a, b, c, d in qd.ndrange(*physical):
            i = b
            j = d
            k = a
            l = c
            x[a, b, c, d] = i * 1000 + j * 100 + k * 10 + l

    fill_tagged(tagged)
    fill_direct(direct)
    # ``tagged.to_numpy()`` is canonical-shaped (M, N, ...); ``direct`` holds the same canonical data but laid out at
    # the *physical* shape. Transposing ``direct`` by the inverse permutation recovers the canonical view, which must
    # then equal ``tagged.to_numpy()``.
    invperm = [0] * len(layout)
    for src, dst in enumerate(layout):
        invperm[dst] = src
    np.testing.assert_array_equal(tagged.to_numpy(), np.transpose(direct.to_numpy(), invperm))
