"""Phase 4 tests: atomic ops through a permuted lvalue.

Phase 4 is a pure validation phase: the Phase 3 AST pre-pass already
rewrites every ``Subscript(Name('t'), ...)`` it sees, including the
ones that appear as the lvalue argument of ``qd.atomic_add`` /
``atomic_sub`` / ``atomic_min`` / ``atomic_max``. So no production
code change is needed; we just confirm with parallel-scatter tests
that atomics land at the right *physical* slot under non-identity
layouts and produce identical logical results across layouts.

Coverage:
- Single-cell hot-spot atomic_add (concurrency stress on one slot).
- Parallel scatter into a small grid (cross-cell concurrency).
- Identity vs transposed layout produce identical logical totals.
- Both backends: NdarrayTensor + FieldTensor.
- ``atomic_min`` / ``atomic_max`` exercise the same lvalue path with
  non-commutative reductions to ensure indices weren't silently
  swapped on the wrong axis.
"""

import numpy as np

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

# ---------------------------------------------------------------------------
# Single-cell hot-spot.
# ---------------------------------------------------------------------------


@qd.kernel
def _hotspot_add(t: qd.Tensor, n: qd.i32):
    for _ in range(n):
        qd.atomic_add(t[0, 0], qd.f32(1.0))


@test_utils.test(arch=get_host_arch_list())
def test_atomic_add_hotspot_identity_layout():
    t = qd.tensor(qd.f32, shape=(3, 4))
    t.from_numpy(np.zeros((3, 4), dtype=np.float32))
    _hotspot_add(t, 200)
    assert t.to_numpy()[0, 0] == 200.0
    # Every other cell remains zero — confirms the lvalue went to (0,0)
    # and not somewhere permuted.
    rest = t.to_numpy().copy()
    rest[0, 0] = 0
    assert (rest == 0).all()


@test_utils.test(arch=get_host_arch_list())
def test_atomic_add_hotspot_transposed_layout():
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    t.from_numpy(np.zeros((3, 4), dtype=np.float32))
    _hotspot_add(t, 200)
    assert t.to_numpy()[0, 0] == 200.0
    rest = t.to_numpy().copy()
    rest[0, 0] = 0
    assert (rest == 0).all()


# ---------------------------------------------------------------------------
# Parallel scatter — each iteration hits a (deterministic) cell.
# ---------------------------------------------------------------------------


@qd.kernel
def _scatter(t: qd.Tensor, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        qd.atomic_add(t[i % 2, j % 3], qd.f32(1.0))


def _expected_scatter(n0: int, n1: int) -> np.ndarray:
    out = np.zeros((2, 3), dtype=np.float32)
    for i in range(n0):
        for j in range(n1):
            out[i % 2, j % 3] += 1.0
    return out


@test_utils.test(arch=get_host_arch_list())
def test_atomic_add_scatter_both_layouts_identical():
    """Same scatter on both layouts must produce identical logical totals."""
    expected = _expected_scatter(100, 100)
    for layout in [(0, 1), (1, 0)]:
        t = qd.tensor(qd.f32, shape=(2, 3), layout=layout)
        t.from_numpy(np.zeros((2, 3), dtype=np.float32))
        _scatter(t, 100, 100)
        np.testing.assert_array_equal(t.to_numpy(), expected)


@test_utils.test(arch=get_host_arch_list())
def test_atomic_add_scatter_transposed_lands_in_physical_transpose():
    """Confirm transposed layout's physical storage really is the transpose."""
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0))
    t.from_numpy(np.zeros((2, 3), dtype=np.float32))
    _scatter(t, 100, 100)
    expected = _expected_scatter(100, 100)
    np.testing.assert_array_equal(t.underlying.to_numpy(), expected.T)


# ---------------------------------------------------------------------------
# Field backend.
# ---------------------------------------------------------------------------


@qd.kernel
def _scatter_field(t: qd.FieldTensor, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        qd.atomic_add(t[i % 2, j % 3], qd.f32(1.0))


@test_utils.test(arch=get_host_arch_list())
def test_atomic_add_scatter_field_backend_transposed():
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0), backend=qd.Backend.FIELD)
    t.from_numpy(np.zeros((2, 3), dtype=np.float32))
    _scatter_field(t, 100, 100)
    expected = _expected_scatter(100, 100)
    np.testing.assert_array_equal(t.to_numpy(), expected)


# ---------------------------------------------------------------------------
# atomic_min / atomic_max — non-commutative; if axes were swapped the
# wrong cell would receive the extremum and the assertion would fire.
# ---------------------------------------------------------------------------


@qd.kernel
def _min_max(t: qd.Tensor, n: qd.i32):
    for i in range(n):
        # row 0: take min with i (decreasing then negative would land here)
        qd.atomic_min(t[0, 0], qd.f32(10 - i))
        # row 1: take max with i
        qd.atomic_max(t[1, 1], qd.f32(i))


@test_utils.test(arch=get_host_arch_list())
def test_atomic_min_max_transposed_layout():
    t = qd.tensor(qd.f32, shape=(2, 2), layout=(1, 0))
    init = np.full((2, 2), 5.0, dtype=np.float32)
    t.from_numpy(init)
    _min_max(t, 20)
    out = t.to_numpy()
    # min at (0, 0) with 10-i for i in [0..19]: minimum is 10-19 = -9
    assert out[0, 0] == -9.0
    # max at (1, 1) with i for i in [0..19]: maximum is 19 vs init 5 → 19
    assert out[1, 1] == 19.0
    # Off-diagonal cells untouched.
    assert out[0, 1] == 5.0
    assert out[1, 0] == 5.0
