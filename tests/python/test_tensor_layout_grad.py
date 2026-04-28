"""Layout + needs_grad combination: canonical indexing on the grad buffer must keep working for non-identity physical
layouts on both backends.

Pre-impl POC Q3b established this works on FIELD (axis_seq propagation to the grad SNode); the NDARRAY equivalent is
the ``_qd_layout`` tag being copied onto the companion grad ndarray. These tests pin both contracts in the suite so
an upstream regression surfaces immediately.
"""

import itertools

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _to_numpy_shape(canonical, layout, backend):
    """Shape of ``a.to_numpy()`` after a layout-tagged allocation.

    Both FIELD and NDARRAY return the canonical view from ``to_numpy()``; the layout is purely an internal storage
    hint. The signature is kept for backwards compatibility with the rest of this file.
    """
    del layout, backend  # the canonical view is the same on both backends
    return canonical


def _to_numpy_idx(canonical_idx, layout, backend):
    """Translate a canonical multi-index to the numpy-readback index.

    Both backends return canonical views, so this is the identity.
    """
    del layout, backend
    return canonical_idx


# ----------------------------------------------------------------------------
# Rank-2 canonical kernel roundtrip on a transposed-storage tensor.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_layout_grad_canonical_kernel_roundtrip_rank2(backend):
    canonical = (4, 5)
    layout = (1, 0)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout, needs_grad=True)
    # Canonical shape on both primal and grad regardless of backend or layout.
    assert tuple(a.shape) == canonical
    assert tuple(a.grad.shape) == canonical

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def write_primal(x: qd.template()):
            for i, j in qd.ndrange(4, 5):
                x[i, j] = i * 10.0 + j

        @qd.kernel
        def write_grad(x: qd.template()):
            for i, j in qd.ndrange(4, 5):
                x.grad[i, j] = i * 100.0 + j * 10.0

    else:

        @qd.kernel
        def write_primal(x: qd.types.ndarray()):
            for i, j in qd.ndrange(4, 5):
                x[i, j] = i * 10.0 + j

        @qd.kernel
        def write_grad(x: qd.types.ndarray()):
            for i, j in qd.ndrange(4, 5):
                x.grad[i, j] = i * 100.0 + j * 10.0

    write_primal(a)
    write_grad(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == _to_numpy_shape(canonical, layout, backend)
    for canonical_idx, expected_primal, expected_grad in [
        ((1, 2), 12.0, 120.0),
        ((3, 4), 34.0, 340.0),
    ]:
        physical = _to_numpy_idx(canonical_idx, layout, backend)
        assert primal[physical] == expected_primal
        assert grad[physical] == expected_grad


# ----------------------------------------------------------------------------
# Rank-3 canonical kernel roundtrip — single non-trivial permutation.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_layout_grad_canonical_kernel_roundtrip_rank3_kij(backend):
    canonical = (2, 3, 4)
    layout = (2, 0, 1)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout, needs_grad=True)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def write_primal(x: qd.template()):
            for i, j, k in qd.ndrange(2, 3, 4):
                x[i, j, k] = i * 100.0 + j * 10.0 + k

        @qd.kernel
        def write_grad(x: qd.template()):
            for i, j, k in qd.ndrange(2, 3, 4):
                x.grad[i, j, k] = i * 1000.0 + j * 100.0 + k * 10.0

    else:

        @qd.kernel
        def write_primal(x: qd.types.ndarray()):
            for i, j, k in qd.ndrange(2, 3, 4):
                x[i, j, k] = i * 100.0 + j * 10.0 + k

        @qd.kernel
        def write_grad(x: qd.types.ndarray()):
            for i, j, k in qd.ndrange(2, 3, 4):
                x.grad[i, j, k] = i * 1000.0 + j * 100.0 + k * 10.0

    write_primal(a)
    write_grad(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == _to_numpy_shape(canonical, layout, backend)
    physical = _to_numpy_idx((1, 2, 3), layout, backend)
    assert primal[physical] == 123.0
    assert grad[physical] == 1230.0


# ----------------------------------------------------------------------------
# Rank-3 perm sweep: every layout produces a canonical-indexable primal+grad.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
def test_layout_grad_all_rank3_permutations(layout, backend):
    qd.init(arch=qd.x64)
    canonical = (2, 3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout, needs_grad=True)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i, j, k in qd.ndrange(2, 3, 4):
                x[i, j, k] = i * 100.0 + j * 10.0 + k
                x.grad[i, j, k] = 1000.0 + i * 100.0 + j * 10.0 + k

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i, j, k in qd.ndrange(2, 3, 4):
                x[i, j, k] = i * 100.0 + j * 10.0 + k
                x.grad[i, j, k] = 1000.0 + i * 100.0 + j * 10.0 + k

    fill(a)
    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    expected_shape = _to_numpy_shape(canonical, tuple(layout), backend)
    assert primal.shape == expected_shape
    assert grad.shape == expected_shape
    physical = _to_numpy_idx((1, 2, 3), tuple(layout), backend)
    assert primal[physical] == 123.0
    assert grad[physical] == 1123.0


# ----------------------------------------------------------------------------
# NDARRAY-only: the _qd_layout tag is the NDARRAY-side mechanism for carrying the layout to kernels and to the
# companion grad. FIELD uses a different storage mechanism (axis_seq inside the SNode tree); see the rank-2/3
# kernel-roundtrip tests above for the FIELD-side equivalent of this contract.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_ndarray_grad_tag_propagates_rank2():
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY, layout=(1, 0), needs_grad=True)
    assert a.grad is not None
    impl = a._unwrap()
    assert getattr(impl, "_qd_layout", None) == (1, 0)
    assert getattr(a.grad._unwrap(), "_qd_layout", None) == (1, 0)
