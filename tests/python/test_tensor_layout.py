"""Tests for ``layout=`` on the ``qd.tensor()`` factory.

Phase 2 ships layout support for the field backend only. Non-identity
layouts on the ndarray backend raise NotImplementedError until the AST
rewrite lands in an earlier change.
"""

import itertools

import pytest

import quadrants as qd

from tests import test_utils

# ----------------------------------------------------------------------------
# Identity / default layouts
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_none_is_default():
    a = qd.tensor(qd.f32, shape=(4, 5))
    b = qd.tensor(qd.f32, shape=(4, 5), layout=None)
    assert a.shape == b.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_layout_identity_field_works():
    a = qd.tensor(qd.f32, shape=(4, 5), layout=(0, 1))
    assert a.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_layout_identity_ndarray_works():
    """Identity layout on ndarray must work — only non-identity is gated."""
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY, layout=(0, 1))
    assert a.shape == (4, 5)


# ----------------------------------------------------------------------------
# Non-identity layouts on the field backend
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_field_rank2_transposed():
    a = qd.tensor(qd.f32, shape=(4, 5), layout=(1, 0))
    # Field.shape stays canonical regardless of physical layout (Q1 of pre-impl POC).
    assert a.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_layout_field_kernel_canonical_indexing_rank2():
    """Writing canonical indices through a kernel works, regardless of layout."""
    a = qd.tensor(qd.i32, shape=(3, 4), layout=(1, 0))

    @qd.kernel
    def fill(x: qd.template()):
        for i, j in qd.ndrange(3, 4):
            x[i, j] = i * 10 + j

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == (3, 4)
    assert arr[2, 1] == 21
    assert arr[0, 3] == 3


BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _expected_factory_shape(canonical, layout, backend):
    """``a.shape`` after allocation, accounting for FIELD canonical vs
    NDARRAY physical reporting (the canonical-vs-physical bug is tracked
    separately; this helper localises the asymmetry)."""
    if backend is qd.Backend.FIELD:
        return canonical
    return tuple(canonical[axis] for axis in layout)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_layout_rank3_all_permutations(layout, backend):
    """Every rank-3 permutation must allocate without error on both backends."""
    qd.init(arch=qd.x64)
    canonical = (2, 3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    assert tuple(a.shape) == _expected_factory_shape(canonical, layout, backend)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", list(itertools.permutations(range(4))))
def test_layout_rank4_all_permutations(layout, backend):
    """Every rank-4 permutation must allocate without error on both backends."""
    qd.init(arch=qd.x64)
    canonical = (2, 3, 4, 5)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    assert tuple(a.shape) == _expected_factory_shape(canonical, layout, backend)


# ----------------------------------------------------------------------------
# Validation errors
# ----------------------------------------------------------------------------


def test_layout_wrong_length_raises():
    qd.init(arch=qd.x64)
    with pytest.raises(ValueError, match="layout has"):
        qd.tensor(qd.f32, shape=(4, 5), layout=(0, 1, 2))


def test_layout_not_permutation_raises():
    qd.init(arch=qd.x64)
    with pytest.raises(ValueError, match="not a permutation"):
        qd.tensor(qd.f32, shape=(4, 5), layout=(0, 0))
    with pytest.raises(ValueError, match="not a permutation"):
        qd.tensor(qd.f32, shape=(4, 5), layout=(1, 2))


def test_order_kwarg_rejected():
    """order= is forbidden — users must say layout=."""
    qd.init(arch=qd.x64)
    with pytest.raises(TypeError, match="layout="):
        qd.tensor(qd.f32, shape=(4, 5), order="ji")


# ----------------------------------------------------------------------------
# Ndarray non-identity layout: enabled in an earlier change.
# These cases are exercised in depth in test_tensor_factory_layout_ndarray.py;
# this file just pins the smoke "factory does not raise" contract that
# replaces the PR-6-era NotImplementedError gating.
# ----------------------------------------------------------------------------


def test_layout_nonidentity_ndarray_accepted():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY, layout=(1, 0))
    # Allocated at the physical (permuted) shape.
    assert tuple(a.shape) == (5, 4)
    assert a._qd_layout == (1, 0)


def test_layout_nonidentity_ndarray_rank3_accepted():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(2, 3, 4), backend=qd.Backend.NDARRAY, layout=(2, 0, 1))
    assert tuple(a.shape) == (4, 2, 3)
    assert a._qd_layout == (2, 0, 1)


# ----------------------------------------------------------------------------
# layout + needs_grad combination
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_layout_with_needs_grad_allocates_grad(backend):
    a = qd.tensor(qd.f32, shape=(4, 5), backend=backend, layout=(1, 0), needs_grad=True)
    assert a.grad is not None
    assert tuple(a.grad.shape) == tuple(a.shape)
