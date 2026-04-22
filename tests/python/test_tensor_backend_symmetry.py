"""Cross-backend symmetry tests for ``qd.tensor()``.

These tests pin the contract that the entire user-facing tensor surface
behaves identically on ``Backend.FIELD`` and ``Backend.NDARRAY`` for any
layout (identity or non-identity). Downstream code (Genesis-style
``GS_ENABLE_NDARRAY`` switching, ``layout=`` perf tuning) should never
have to special-case the backend or the layout.

Each test parametrises over both backends and at least one non-identity
layout. If a future change re-introduces an asymmetry (e.g. one backend
gains a kwarg the other doesn't), one of these tests will fail.

See ``perso_hugh/doc/quadrants-tensor.md`` §8.9 for the design rationale
and the per-asymmetry decision matrix.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]
_LAYOUTS_RANK2 = list(itertools.permutations(range(2)))


# ----------------------------------------------------------------------------
# tensor.layout: must report the user-supplied permutation (or None for
# identity / no layout) on both backends.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_layout_property_reports_user_layout(backend, layout):
    a = qd.tensor(qd.f32, shape=(3, 4), backend=backend, layout=layout)
    if layout == (0, 1):
        assert a.layout is None
    else:
        assert a.layout == layout


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_layout_property_none_when_layout_omitted(backend):
    a = qd.tensor(qd.f32, shape=(3, 4), backend=backend)
    assert a.layout is None


# ----------------------------------------------------------------------------
# to_torch() / from_torch(): both backends expose the same method with the
# same canonical-view contract under any layout.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_to_torch_canonical_view_round_trips(backend, layout):
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i, j in qd.ndrange(*canonical):
                x[i, j] = i * 100 + j

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i, j in qd.ndrange(*canonical):
                x[i, j] = i * 100 + j

    fill(a)
    t = a.to_torch()
    assert tuple(t.shape) == canonical
    expected = np.zeros(canonical, dtype=np.int32)
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            expected[i, j] = i * 100 + j
    np.testing.assert_array_equal(t.contiguous().cpu().numpy(), expected)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_from_torch_canonical_round_trips(backend, layout):
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    src = torch.arange(canonical[0] * canonical[1], dtype=torch.int32).reshape(canonical)
    a.from_torch(src)
    np.testing.assert_array_equal(a.to_numpy(), src.cpu().numpy())


# ----------------------------------------------------------------------------
# to_numpy(dtype=...): both backends accept the optional dtype kwarg.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_accepts_dtype_kwarg(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    arr = a.to_numpy(dtype=np.float64)
    assert arr.dtype == np.float64
    assert arr.shape == canonical


# ----------------------------------------------------------------------------
# Pickle: pre-existing asymmetry preserved. ``Ndarray`` supports pickle
# (round-trips via ``to_numpy()``); ``Field`` doesn't, because adding it
# would require re-allocating SNodes after the runtime is materialized.
#
# The user-stated scope of the qd.tensor symmetry work is to make the
# ``qd.tensor(...)`` *behavior* symmetric, not to delete pre-existing
# functionality from the underlying types. Symmetric pickle is planned
# for the upcoming ``Tensor`` wrapper (§8.11), which will round-trip via
# ``to_numpy()`` + reconstruct, working uniformly on both backends. Until
# then this test pins the *current* (asymmetric) behavior so further
# refactors don't accidentally change it.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_pickle_ndarray_works(layout):
    import pickle  # noqa: PLC0415

    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)
    blob = pickle.dumps(a)
    restored = pickle.loads(blob)
    assert restored.shape == canonical
    assert restored.dtype == qd.f32


@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_pickle_field_raises(layout):
    import pickle  # noqa: PLC0415

    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=qd.Backend.FIELD, layout=layout)
    with pytest.raises((TypeError, pickle.PicklingError, AttributeError, NotImplementedError)):
        pickle.dumps(a)


# ----------------------------------------------------------------------------
# qd.tensor() kwarg validation is symmetric: every accepted-kwarg works
# on both backends, and every rejected-kwarg raises on both backends.
# Locks the contract so a stray ``order=`` typo (etc.) doesn't silently
# work on one path.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_qd_tensor_rejects_order_kwarg_on_both_backends(backend):
    with pytest.raises(TypeError, match="order="):
        qd.tensor(qd.f32, shape=(3, 4), backend=backend, order="ji")


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_qd_tensor_rejects_unknown_kwarg_on_both_backends(backend):
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.tensor(qd.f32, shape=(3, 4), backend=backend, definitely_not_a_kwarg=42)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_qd_tensor_accepts_needs_grad_on_both_backends(backend):
    a = qd.tensor(qd.f32, shape=(3, 4), backend=backend, needs_grad=True)
    assert a.grad is not None
    assert tuple(a.grad.shape) == (3, 4)


# ----------------------------------------------------------------------------
# tensor.shape: canonical on both backends regardless of layout.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_shape_is_canonical(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    assert tuple(a.shape) == canonical
