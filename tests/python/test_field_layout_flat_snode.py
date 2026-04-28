"""Tests for the flat-SNode field layout implementation.

Validates that ``qd.field(..., order=...)`` produces a single rank-N dense SNode (not N nested rank-1 SNodes), uses
AST subscript rewrites instead of nested SNode lookup, and preserves identical semantics for host and kernel access,
``get_addr``, ``to_dlpack``, and autograd.
"""

import itertools

import pytest

import quadrants as qd

from tests import test_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order_string(layout):
    """Convert an integer layout tuple to the axis-char order string."""
    return "".join(chr(ord("i") + a) for a in layout)


def _all_non_identity_perms(dim):
    """All permutations of range(dim) that are NOT the identity."""
    identity = tuple(range(dim))
    return [p for p in itertools.permutations(range(dim)) if p != identity]


# ---------------------------------------------------------------------------
# IR-level test: flat rank-N SNode, single linearize+lookup per subscript
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 0), (0, 1)])
@test_utils.test(arch=qd.cpu)
def test_flat_snode_rank2(layout):
    """A field with order= must produce exactly ONE dense SNode above its place node — not ``dim`` nested rank-1
    SNodes."""
    X, Y = 5, 7
    order = _order_string(layout)
    f = qd.field(qd.f32, shape=(X, Y), order=order)
    qd.sync()
    # Walk the SNode tree: place -> parent (should be the only dense) -> root.
    place_snode = f._snode
    dense_snode = place_snode.parent()
    assert dense_snode.parent() is not None, "dense SNode should have a parent (root)"
    root_snode = dense_snode.parent()
    # The root's parent is None.
    assert root_snode.parent() is None, (
        "Expected root's parent to be None — got an extra SNode level, "
        "indicating nested rank-1 allocation instead of flat rank-N."
    )


@pytest.mark.parametrize("layout", _all_non_identity_perms(3))
@test_utils.test(arch=qd.cpu)
def test_flat_snode_rank3(layout):
    """Rank-3 variant: still only one dense SNode above the place node."""
    shape = (3, 4, 5)
    order = _order_string(layout)
    f = qd.field(qd.f32, shape=shape, order=order)
    qd.sync()
    place_snode = f._snode
    dense_snode = place_snode.parent()
    root_snode = dense_snode.parent()
    assert root_snode.parent() is None


# ---------------------------------------------------------------------------
# Semantic test: host write + kernel read round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 0)])
@test_utils.test(arch=qd.cpu)
def test_host_kernel_roundtrip_rank2(layout):
    X, Y = 4, 6
    order = _order_string(layout)
    f = qd.field(qd.i32, shape=(X, Y), order=order)

    for i in range(X):
        for j in range(Y):
            f[i, j] = i * 100 + j

    @qd.kernel
    def read_back(f: qd.template(), i: qd.i32, j: qd.i32) -> qd.i32:
        return f[i, j]

    for i in range(X):
        for j in range(Y):
            assert read_back(f, i, j) == i * 100 + j


# ---------------------------------------------------------------------------
# Semantic test: kernel write + host read round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 0)])
@test_utils.test(arch=qd.cpu)
def test_kernel_host_roundtrip_rank2(layout):
    X, Y = 4, 6
    order = _order_string(layout)
    f = qd.field(qd.i32, shape=(X, Y), order=order)

    @qd.kernel
    def fill(f: qd.template()):
        for i, j in qd.ndrange(X, Y):
            f[i, j] = i * 100 + j

    fill(f)

    for i in range(X):
        for j in range(Y):
            assert f[i, j] == i * 100 + j


# ---------------------------------------------------------------------------
# get_addr test: layout-tagged field addresses match default-layout field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 0)])
@test_utils.test(arch=qd.cpu)
def test_get_addr_matches_default_layout(layout):
    """Physical address pattern of layout-tagged field must match a default-layout field whose shape is the physical
    (permuted) shape."""
    X, Y = 5, 7
    order = _order_string(layout)
    f_layout = qd.field(qd.i32, shape=(X, Y), order=order)
    f_default = qd.field(qd.i32, shape=(Y, X))

    @qd.kernel
    def fill_layout(f: qd.template()):
        for i, j in qd.ndrange(X, Y):
            f[i, j] = i * 100 + j

    @qd.kernel
    def fill_default(f: qd.template()):
        for j, i in qd.ndrange(Y, X):
            f[j, i] = i * 100 + j

    fill_layout(f_layout)
    fill_default(f_default)

    @qd.kernel
    def addr(f: qd.template(), a: qd.i32, b: qd.i32) -> qd.u64:
        return qd.get_addr(f, [a, b])

    base_l = addr(f_layout, 0, 0)
    base_d = addr(f_default, 0, 0)

    for i in range(X):
        for j in range(Y):
            off_l = addr(f_layout, i, j) - base_l
            off_d = addr(f_default, j, i) - base_d
            assert off_l == off_d, f"addr offset mismatch at ({i},{j}): layout={off_l}, default={off_d}"


# ---------------------------------------------------------------------------
# to_dlpack canonical-view test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 0)])
@test_utils.test(arch=qd.cpu)
def test_to_dlpack_canonical_shape(layout):
    """to_dlpack must expose canonical shape, not physical."""
    torch = pytest.importorskip("torch")
    X, Y = 5, 7
    order = _order_string(layout)
    f = qd.field(qd.f32, shape=(X, Y), order=order)

    @qd.kernel
    def fill(f: qd.template()):
        for i, j in qd.ndrange(X, Y):
            f[i, j] = float(i * 100 + j)

    fill(f)
    qd.sync()

    t = torch.utils.dlpack.from_dlpack(f.to_dlpack())
    assert tuple(t.shape) == (X, Y), f"Expected canonical shape {(X,Y)}, got {tuple(t.shape)}"

    for i in range(X):
        for j in range(Y):
            assert float(t[i, j]) == float(i * 100 + j)


# ---------------------------------------------------------------------------
# _qd_layout attribute test
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_qd_layout_attribute_set():
    """Fields created with order= must have _qd_layout set."""
    f = qd.field(qd.f32, shape=(5, 7), order="ji")
    assert hasattr(f, "_qd_layout")
    assert f._qd_layout == (1, 0)
    assert f.layout == (1, 0)


@test_utils.test(arch=qd.cpu)
def test_qd_layout_attribute_none_default():
    """Fields created without order= must NOT have _qd_layout set."""
    f = qd.field(qd.f32, shape=(5, 7))
    assert not hasattr(f, "_qd_layout") or f._qd_layout is None


# ---------------------------------------------------------------------------
# shape property test
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_canonical_shape_with_layout():
    """field.shape must report canonical shape even when physical is permuted."""
    X, Y = 5, 7
    f = qd.field(qd.f32, shape=(X, Y), order="ji")
    assert f.shape == (X, Y), f"Expected canonical shape {(X,Y)}, got {f.shape}"


# ---------------------------------------------------------------------------
# Vector field with layout
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_vector_field_layout_roundtrip():
    """Vector.field with order= should work correctly."""
    X, Y = 4, 5
    v = qd.Vector.field(2, dtype=qd.f32, shape=(X, Y), order="ji")

    @qd.kernel
    def fill(v: qd.template()):
        for i, j in qd.ndrange(X, Y):
            v[i, j] = [float(i), float(j)]

    fill(v)

    for i in range(X):
        for j in range(Y):
            val = v[i, j]
            assert val[0] == float(i) and val[1] == float(
                j
            ), f"Mismatch at [{i},{j}]: got {val}, expected [{float(i)}, {float(j)}]"


# ---------------------------------------------------------------------------
# qd.tensor factory with Backend.FIELD and layout
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_field_backend_layout():
    """qd.tensor(backend=FIELD, layout=...) should produce correct results."""
    X, Y = 5, 7
    t = qd.tensor(qd.f32, shape=(X, Y), backend=qd.Backend.FIELD, layout=(1, 0))
    assert t.shape == (X, Y)

    for i in range(X):
        for j in range(Y):
            t[i, j] = float(i * 100 + j)

    for i in range(X):
        for j in range(Y):
            assert float(t[i, j]) == float(i * 100 + j)


# ---------------------------------------------------------------------------
# Autograd: grad field also gets flat SNode + layout tag
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_grad_field_has_flat_snode():
    """The gradient field of a layout-tagged field should also have a flat SNode (same depth) and be tagged with
    _qd_layout."""
    X, Y = 5, 7
    f = qd.field(qd.f32, shape=(X, Y), order="ji", needs_grad=True)
    qd.sync()

    assert hasattr(f.grad, "_qd_layout")
    assert f.grad._qd_layout == (1, 0)

    place_snode = f.grad._snode
    dense_snode = place_snode.parent()
    root_snode = dense_snode.parent()
    assert root_snode.parent() is None
