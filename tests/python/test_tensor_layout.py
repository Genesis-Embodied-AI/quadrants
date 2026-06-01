"""Tests for ``layout=`` on the ``qd.tensor()`` factory.

Phase 2 ships layout support for the field backend only. Non-identity layouts on the ndarray backend raise
NotImplementedError until the AST rewrite lands in an earlier change.
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
    a = qd.tensor(qd.i32, shape=(3, 4), backend=qd.Backend.FIELD, layout=(1, 0))

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


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_layout_rank3_all_permutations(layout, backend):
    """Every rank-3 permutation must allocate without error on both backends, and ``shape`` is canonical regardless of
    layout."""
    qd.init(arch=qd.x64)
    canonical = (2, 3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    assert tuple(a.shape) == canonical


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", list(itertools.permutations(range(4))))
def test_layout_rank4_all_permutations(layout, backend):
    """Every rank-4 permutation must allocate without error on both backends, and ``shape`` is canonical regardless of
    layout."""
    qd.init(arch=qd.x64)
    canonical = (2, 3, 4, 5)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    assert tuple(a.shape) == canonical


# Higher-rank sampled layouts: full enumeration is infeasible (rank 12 alone is 12! ≈ 479M permutations). The
# rank-3/rank-4 sweeps above already enumerate exhaustively; the cases below random-sample the higher-rank space to
# back the user-guide claim that "any permutation up to quadrants_max_num_indices (12) is supported". A fixed seed per
# (rank, trial) keeps the suite deterministic so a regression on a particular permutation always reproduces.
def _sampled_layouts(rank, num_samples, seed_base=0):
    import random  # pylint: disable=import-outside-toplevel

    out = [tuple(range(rank))]  # always include identity
    rng = random.Random(seed_base + rank)
    seen = {out[0]}
    while len(out) < num_samples + 1:
        candidate = list(range(rank))
        rng.shuffle(candidate)
        candidate = tuple(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("rank", [5, 8, 12])
@pytest.mark.parametrize("trial", list(range(4)))
def test_layout_higher_rank_sampled_permutations(rank, trial, backend):
    """Random-sample the rank-5 / rank-8 / rank-12 layout space on both backends and check that every sampled
    permutation allocates and reports the canonical shape unchanged. Backs the user-guide claim of 'any permutation
    up to quadrants_max_num_indices (12)'."""
    qd.init(arch=qd.x64)
    layouts = _sampled_layouts(rank, num_samples=5)
    layout = layouts[trial % len(layouts)]
    canonical = tuple(2 + (i % 3) for i in range(rank))  # small distinct-ish dims
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    assert tuple(a.shape) == canonical


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
# These cases are exercised in depth in test_tensor_factory_layout_ndarray.py; this file just pins the smoke "factory
# does not raise" contract that replaces the PR-6-era NotImplementedError gating.
# ----------------------------------------------------------------------------


def test_layout_nonidentity_ndarray_accepted():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY, layout=(1, 0))
    # Canonical shape regardless of layout; physical buffer is permuted.
    assert tuple(a.shape) == (4, 5)
    assert a.layout == (1, 0)
    impl = a._unwrap()
    assert tuple(impl._physical_shape) == (5, 4)
    assert impl._qd_layout == (1, 0)


def test_layout_nonidentity_ndarray_rank3_accepted():
    qd.init(arch=qd.x64)
    a = qd.tensor(qd.f32, shape=(2, 3, 4), backend=qd.Backend.NDARRAY, layout=(2, 0, 1))
    assert tuple(a.shape) == (2, 3, 4)
    assert a.layout == (2, 0, 1)
    impl = a._unwrap()
    assert tuple(impl._physical_shape) == (4, 2, 3)
    assert impl._qd_layout == (2, 0, 1)


# ----------------------------------------------------------------------------
# layout + needs_grad combination
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_layout_with_needs_grad_allocates_grad(backend):
    a = qd.tensor(qd.f32, shape=(4, 5), backend=backend, layout=(1, 0), needs_grad=True)
    assert a.grad is not None
    assert tuple(a.grad.shape) == tuple(a.shape)


# ----------------------------------------------------------------------------
# Compound dtype (vector / matrix) + layout= must be rejected
# ----------------------------------------------------------------------------


def test_layout_rejected_for_vector_dtype():
    """qd.tensor() with a compound vector dtype must reject layout=."""
    qd.init(arch=qd.x64)
    vec3 = qd.types.vector(3, qd.f32)
    with pytest.raises(TypeError, match="layout.*not supported.*compound"):
        qd.tensor(vec3, shape=(4,), layout=(0,))


def test_layout_rejected_for_matrix_dtype():
    """qd.tensor() with a compound matrix dtype must reject layout=."""
    qd.init(arch=qd.x64)
    mat2x2 = qd.types.matrix(2, 2, qd.f32)
    with pytest.raises(TypeError, match="layout.*not supported.*compound"):
        qd.tensor(mat2x2, shape=(4,), layout=(0,))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_layout_rejected_for_vector_dtype_both_backends(backend):
    """layout= rejection applies to both field and ndarray backends."""
    qd.init(arch=qd.x64)
    vec3 = qd.types.vector(3, qd.f32)
    with pytest.raises(TypeError, match="layout.*not supported.*compound"):
        qd.tensor(vec3, shape=(5,), backend=backend, layout=(0,))


def test_vector_dtype_without_layout_still_works():
    """Compound vector dtype without layout= must still allocate fine."""
    qd.init(arch=qd.x64)
    vec3 = qd.types.vector(3, qd.f32)
    a = qd.tensor(vec3, shape=(4,))
    assert tuple(a.shape) == (4,)


def test_matrix_dtype_without_layout_still_works():
    """Compound matrix dtype without layout= must still allocate fine."""
    qd.init(arch=qd.x64)
    mat2x2 = qd.types.matrix(2, 2, qd.f32)
    a = qd.tensor(mat2x2, shape=(4,))
    assert tuple(a.shape) == (4,)
