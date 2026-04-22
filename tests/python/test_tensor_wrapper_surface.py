"""Tensor-wrapper full-surface tests (stork-18).

Pins the behavior the upcoming factory flip (stork-19) relies on:

- Interop forwards — ``to_numpy``/``from_numpy``/``to_torch``/``from_torch``/
  ``to_dlpack``/``fill``/``copy_from`` all produce / consume canonical
  views via the wrapper, on both backends, at every layout.
- Pickle via ``__reduce__`` round-trips symmetrically on both backends
  (the pre-existing ``Field`` asymmetry is closed *at the wrapper layer*
  without touching the upstream ``Field`` type).
- ``.grad`` wraps lazily, returns a ``Tensor`` (not the bare impl), and
  is identity-stable across accesses (so autograd-tape identity checks
  hold after migration).
- ``VectorTensor`` / ``MatrixTensor`` carry ``element_shape`` and
  round-trip via pickle on both backends.
"""

import itertools
import pickle

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]
_LAYOUTS_RANK2 = [(0, 1), (1, 0)]


# ----------------------------------------------------------------------
# Interop forwards
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_to_numpy_canonical(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = np.arange(12, dtype=np.int32).reshape(canonical)
    a.from_numpy(src)
    t = qd._Tensor(a)
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_to_numpy_dtype_cast(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    t = qd._Tensor(a)
    out = t.to_numpy(dtype=np.float64)
    assert out.dtype == np.float64
    assert out.shape == canonical


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_from_numpy_roundtrip(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    t = qd._Tensor(a)
    src = np.arange(12, dtype=np.int32).reshape(canonical) * 7
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_to_torch_roundtrip(backend, layout):
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    src = np.arange(12, dtype=np.float32).reshape(canonical)
    a.from_numpy(src)
    t = qd._Tensor(a)
    out = t.to_torch()
    assert tuple(out.shape) == canonical
    np.testing.assert_array_equal(out.cpu().numpy(), src)

    # from_torch of the same tensor is a no-op in canonical values.
    t.from_torch(out)
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_to_dlpack_canonical_shape(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    t = qd._Tensor(a)
    cap = t.to_dlpack()
    # Consume via torch so we get actual shape metadata (both backends
    # produce canonical-shape DLPack after stork-16).
    torch = pytest.importorskip("torch")
    got = torch.utils.dlpack.from_dlpack(cap)
    assert tuple(got.shape) == canonical


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_fill(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    t = qd._Tensor(a)
    t.fill(7)
    np.testing.assert_array_equal(t.to_numpy(), np.full(canonical, 7, dtype=np.int32))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_copy_from_wrapper_and_bare(backend, layout):
    canonical = (3, 4)
    src = np.arange(12, dtype=np.int32).reshape(canonical) + 100

    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    a.from_numpy(src)

    b = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    tb = qd._Tensor(b)
    # Accept a wrapper on the RHS.
    tb.copy_from(qd._Tensor(a))
    np.testing.assert_array_equal(tb.to_numpy(), src)

    # And a bare impl on the RHS (convenience for migration).
    c = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    tc = qd._Tensor(c)
    tc.copy_from(a)
    np.testing.assert_array_equal(tc.to_numpy(), src)


# ----------------------------------------------------------------------
# Pickle (symmetric across backends via __reduce__ + to_numpy round-trip)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_pickle_roundtrip(backend, layout):
    canonical = (3, 4)
    src = np.arange(12, dtype=np.float32).reshape(canonical) * 3.5
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    a.from_numpy(src)
    t = qd._Tensor(a)

    blob = pickle.dumps(t)
    restored = pickle.loads(blob)

    assert isinstance(restored, qd._Tensor)
    assert restored.shape == canonical
    assert restored.dtype == qd.f32
    np.testing.assert_array_equal(restored.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_pickle_preserves_layout_at_canonical_indices(backend, layout):
    """Round-tripped tensor must give the same *canonical* values at every
    canonical coordinate, even for non-identity layouts. The underlying
    physical layout on the reconstructed side isn't required to match
    (pickle drops layout tags today), but the user-visible data must.
    """
    canonical = (3, 4)
    src = np.arange(12, dtype=np.int32).reshape(canonical)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    a.from_numpy(src)
    t = qd._Tensor(a)

    restored = pickle.loads(pickle.dumps(t))
    for ci in itertools.product(*(range(d) for d in canonical)):
        assert int(restored[ci]) == int(src[ci])


# ----------------------------------------------------------------------
# Lazy .grad wrapping
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_grad_is_wrapped(backend):
    a = qd.tensor(qd.f32, shape=(4,), backend=backend, needs_grad=True)
    t = qd._Tensor(a)
    g = t.grad
    assert isinstance(g, qd._Tensor)
    assert g.shape == (4,)
    assert g.dtype == qd.f32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_grad_is_identity_stable(backend):
    a = qd.tensor(qd.f32, shape=(4,), backend=backend, needs_grad=True)
    t = qd._Tensor(a)
    g1 = t.grad
    g2 = t.grad
    # ``cached_property`` guarantees the same wrapper object is returned
    # on repeat access; needed for autograd-tape identity checks.
    assert g1 is g2


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_grad_none_when_no_grad_int_dtype(backend):
    # Pre-existing Quadrants asymmetry: a ``Field`` with a real dtype
    # always has a zombie ``.grad`` stub created by ``create_field_member``
    # regardless of ``needs_grad=``, while an ``Ndarray`` correctly reports
    # ``grad = None``. Int dtypes skip the stub path on both backends, so
    # this test uses ``qd.i32`` to get symmetric ``grad is None`` behaviour.
    # The real-dtype asymmetry is tracked in the design doc and will be
    # cleaned up when the wrapper becomes the factory default in stork-19.
    a = qd.tensor(qd.i32, shape=(4,), backend=backend)
    t = qd._Tensor(a)
    assert t.grad is None


# ----------------------------------------------------------------------
# Vector / Matrix wrappers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_tensor_wraps_and_has_element_shape(backend):
    a = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    t = qd._VectorTensor(a)
    assert t.shape == (4,)
    assert t.element_shape == (3,)
    assert t.dtype == qd.f32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_tensor_pickle_roundtrip(backend):
    a = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(4, 3)
    a.from_numpy(src)
    t = qd._VectorTensor(a)

    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, qd._VectorTensor)
    assert restored.shape == (4,)
    assert restored.element_shape == (3,)
    np.testing.assert_array_equal(restored.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_wraps_and_has_element_shape(backend):
    a = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    t = qd._MatrixTensor(a)
    assert t.shape == (3,)
    assert t.element_shape == (2, 2)
    assert t.dtype == qd.f32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_pickle_roundtrip(backend):
    a = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    a.from_numpy(src)
    t = qd._MatrixTensor(a)

    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, qd._MatrixTensor)
    assert restored.shape == (3,)
    assert restored.element_shape == (2, 2)
    np.testing.assert_array_equal(restored.to_numpy(), src)


# ----------------------------------------------------------------------
# ``_wrap`` helper picks the right subclass
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrap_picks_scalar_tensor(backend):
    a = qd.tensor(qd.f32, shape=(4,), backend=backend)
    t = qd._wrap(a)
    assert isinstance(t, qd._Tensor)
    assert not isinstance(t, (qd._VectorTensor, qd._MatrixTensor))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrap_picks_vector_tensor(backend):
    a = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    t = qd._wrap(a)
    assert isinstance(t, qd._VectorTensor)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrap_picks_matrix_tensor(backend):
    a = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    t = qd._wrap(a)
    assert isinstance(t, qd._MatrixTensor)
