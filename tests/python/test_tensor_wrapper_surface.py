"""Tensor-wrapper full-surface tests.

Pins the wrapper behavior the user-facing factory relies on:

- Interop forwards - ``to_numpy``/``from_numpy``/``to_torch``/``from_torch``/``to_dlpack``/``fill``/``copy_from`` all
  produce / consume canonical views via the wrapper, on both backends, at every layout.
- Pickle via ``__reduce__`` round-trips symmetrically on both backends (the pre-existing ``Field`` asymmetry is closed
  *at the wrapper layer* without touching the upstream ``Field`` type).
- ``.grad`` wraps lazily, returns a ``Tensor`` (not the bare impl), and is identity-stable across accesses (so
  autograd-tape identity checks hold after migration).
- ``VectorTensor`` / ``MatrixTensor`` carry ``element_shape`` and round-trip via pickle on both backends.
- ``qd.wrap`` picks the right subclass for a bare impl.

Stork-19 flipped ``qd.tensor()`` (and the Vector/Matrix variants) to return wrappers, so most tests below can use the
factory result directly. The ``qd.wrap`` tests still allocate bare impls (via ``qd.field`` / ``qd.ndarray`` /
``qd.Vector.field`` / ``qd.Matrix.field``) and exercise the explicit wrap path.
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
    t = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = np.arange(12, dtype=np.int32).reshape(canonical)
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_to_numpy_dtype_cast(backend, layout):
    canonical = (3, 4)
    t = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    out = t.to_numpy(dtype=np.float64)
    assert out.dtype == np.float64
    assert out.shape == canonical


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_from_numpy_roundtrip(backend, layout):
    canonical = (3, 4)
    t = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = np.arange(12, dtype=np.int32).reshape(canonical) * 7
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_to_torch_roundtrip(backend, layout):
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    t = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    src = np.arange(12, dtype=np.float32).reshape(canonical)
    t.from_numpy(src)
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
    # ``field_to_dlpack`` in C++ unconditionally imports torch to check its DLPack byte-offset support. Skip early when
    # torch is absent so the test is marked "skipped" rather than crashing the worker.
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    t = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    cap = t.to_dlpack()
    got = torch.utils.dlpack.from_dlpack(cap)
    assert tuple(got.shape) == canonical


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_fill(backend, layout):
    canonical = (3, 4)
    t = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
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

    tb = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    # Accept a wrapper on the RHS.
    tb.copy_from(a)
    np.testing.assert_array_equal(tb.to_numpy(), src)

    # And a bare impl on the RHS (convenience for migration).
    tc = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    tc.copy_from(a._unwrap())
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
    t = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout)
    t.from_numpy(src)

    blob = pickle.dumps(t)
    restored = pickle.loads(blob)

    assert isinstance(restored, qd.Tensor)
    assert restored.shape == canonical
    assert restored.dtype == qd.f32
    np.testing.assert_array_equal(restored.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapper_pickle_preserves_layout_at_canonical_indices(backend, layout):
    """Round-tripped tensor must give the same *canonical* values at every canonical coordinate, even for non-identity
    layouts. The underlying physical layout on the reconstructed side isn't required to match (pickle drops layout tags
    today), but the user-visible data must.
    """
    canonical = (3, 4)
    src = np.arange(12, dtype=np.int32).reshape(canonical)
    t = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    t.from_numpy(src)

    restored = pickle.loads(pickle.dumps(t))
    for ci in itertools.product(*(range(d) for d in canonical)):
        assert int(restored[ci]) == int(src[ci])


# ----------------------------------------------------------------------
# Lazy .grad wrapping
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_grad_is_wrapped(backend):
    t = qd.tensor(qd.f32, shape=(4,), backend=backend, needs_grad=True)
    g = t.grad
    assert isinstance(g, qd.Tensor)
    assert g.shape == (4,)
    assert g.dtype == qd.f32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_grad_is_identity_stable(backend):
    t = qd.tensor(qd.f32, shape=(4,), backend=backend, needs_grad=True)
    g1 = t.grad
    g2 = t.grad
    # ``cached_property`` guarantees the same wrapper object is returned on repeat access; needed for autograd-tape
    # identity checks.
    assert g1 is g2


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_grad_none_when_no_grad_int_dtype(backend):
    # Pre-existing Quadrants asymmetry: a ``Field`` with a real dtype always has a zombie ``.grad`` stub created by
    # ``create_field_member`` regardless of ``needs_grad=``, while an ``Ndarray`` correctly reports ``grad = None``.
    # Int dtypes skip the stub path on both backends, so this test uses ``qd.i32`` to get symmetric ``grad is None``
    # behaviour. Cleaning up the real-dtype asymmetry is tracked in the design doc and deferred to a follow-up branch.
    t = qd.tensor(qd.i32, shape=(4,), backend=backend)
    assert t.grad is None


# ----------------------------------------------------------------------
# Vector / Matrix wrappers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_tensor_wraps_and_has_element_shape(backend):
    t = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    assert isinstance(t, qd.VectorTensor)
    assert t.shape == (4,)
    assert t.element_shape == (3,)
    assert t.dtype == qd.f32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_tensor_pickle_roundtrip(backend):
    t = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(4, 3)
    t.from_numpy(src)

    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, qd.VectorTensor)
    assert restored.shape == (4,)
    assert restored.element_shape == (3,)
    np.testing.assert_array_equal(restored.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_wraps_and_has_element_shape(backend):
    t = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    assert isinstance(t, qd.MatrixTensor)
    assert t.shape == (3,)
    assert t.element_shape == (2, 2)
    assert t.dtype == qd.f32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_pickle_roundtrip(backend):
    t = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    t.from_numpy(src)

    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, qd.MatrixTensor)
    assert restored.shape == (3,)
    assert restored.element_shape == (2, 2)
    np.testing.assert_array_equal(restored.to_numpy(), src)


# ----------------------------------------------------------------------
# ``qd.wrap`` helper picks the right subclass when given a bare impl
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrap_picks_scalar_tensor(backend):
    if backend is qd.Backend.FIELD:
        a = qd.field(qd.f32, shape=(4,))
    else:
        a = qd.ndarray(qd.f32, shape=(4,))
    t = qd.wrap(a)
    assert isinstance(t, qd.Tensor)
    assert not isinstance(t, (qd.VectorTensor, qd.MatrixTensor))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrap_picks_vector_tensor(backend):
    if backend is qd.Backend.FIELD:
        a = qd.Vector.field(3, qd.f32, shape=(4,))
    else:
        a = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    t = qd.wrap(a)
    assert isinstance(t, qd.VectorTensor)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrap_picks_matrix_tensor(backend):
    if backend is qd.Backend.FIELD:
        a = qd.Matrix.field(2, 2, qd.f32, shape=(3,))
    else:
        a = qd.Matrix.ndarray(2, 2, qd.f32, shape=(3,))
    t = qd.wrap(a)
    assert isinstance(t, qd.MatrixTensor)


# ----------------------------------------------------------------------
# copy= kwarg on to_numpy / to_torch
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_copy_true(backend):
    """copy=True (default) returns an independent array."""
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    arr = t.to_numpy(copy=True)
    np.testing.assert_array_equal(arr, src)
    arr[0] = 999.0
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_copy_false(backend):
    """copy=False returns a zero-copy view (CPU backend supports this).

    Marked needs_torch because Field.to_numpy(copy=False) uses DLPack which requires torch.
    """
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    arr = t.to_numpy(copy=False)
    np.testing.assert_array_equal(arr, src)
    arr[0] = 999.0
    assert t.to_numpy(copy=True)[0] == 999.0


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_copy_false_with_dtype_raises(backend):
    """copy=False combined with dtype conversion must raise."""
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    t.from_numpy(np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="copy=False"):
        t.to_numpy(dtype=np.float64, copy=False)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_torch_copy_true(backend):
    """copy=True (default) returns an independent torch tensor."""
    torch = pytest.importorskip("torch")
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    out = t.to_torch(copy=True)
    np.testing.assert_array_equal(out.numpy(), src)
    out[0] = 999.0
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_torch_copy_false(backend):
    """copy=False returns a zero-copy torch tensor (CPU backend)."""
    torch = pytest.importorskip("torch")
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    out = t.to_torch(copy=False)
    np.testing.assert_array_equal(out.numpy(), src)
    out[0] = 999.0
    assert t.to_numpy(copy=True)[0] == 999.0


# ----------------------------------------------------------------------
# copy=None (best-effort zero-copy)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_copy_none_returns_correct_data(backend):
    """copy=None never raises and returns correct data regardless of whether zero-copy is available."""
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    arr = t.to_numpy(copy=None)
    np.testing.assert_array_equal(arr, src)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_copy_none_zerocopy_when_available(backend):
    """copy=None returns a zero-copy view on CPU with torch installed and a supported dtype."""
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    arr = t.to_numpy(copy=None)
    np.testing.assert_array_equal(arr, src)
    arr[0] = 999.0
    assert t.to_numpy(copy=True)[0] == 999.0


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_copy_none_with_dtype_falls_back(backend):
    """copy=None with dtype conversion silently falls back to a copy."""
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    arr = t.to_numpy(dtype=np.float64, copy=None)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, src.astype(np.float64))
    arr[0] = 999.0
    np.testing.assert_array_equal(t.to_numpy(), src)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_to_torch_copy_none_zerocopy_when_available(backend):
    """copy=None returns a zero-copy torch tensor on CPU with a supported dtype."""
    torch = pytest.importorskip("torch")
    t = qd.tensor(qd.f32, shape=(4,), backend=backend)
    src = np.arange(4, dtype=np.float32)
    t.from_numpy(src)
    out = t.to_torch(copy=None)
    np.testing.assert_array_equal(out.numpy(), src)
    out[0] = 999.0
    assert t.to_numpy(copy=True)[0] == 999.0


# ----------------------------------------------------------------------
# copy=None on Matrix / Vector types
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_to_numpy_copy_none(backend):
    """copy=None on MatrixField / MatrixNdarray returns correct data without raising."""
    t = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    t.from_numpy(src)
    arr = t.to_numpy(copy=None)
    np.testing.assert_array_equal(arr, src)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_to_torch_copy_none(backend):
    """copy=None on MatrixField / MatrixNdarray returns correct torch tensor."""
    torch = pytest.importorskip("torch")
    t = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    t.from_numpy(src)
    out = t.to_torch(copy=None)
    np.testing.assert_array_equal(out.numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_to_numpy_copy_none_dtype_fallback(backend):
    """copy=None with dtype on MatrixField / MatrixNdarray silently falls back."""
    t = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    t.from_numpy(src)
    arr = t.to_numpy(dtype=np.float64, copy=None)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, src.astype(np.float64))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_to_numpy_copy_none(backend):
    """copy=None on VectorField / VectorNdarray returns correct data without raising."""
    t = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(4, 3)
    t.from_numpy(src)
    arr = t.to_numpy(copy=None)
    np.testing.assert_array_equal(arr, src)


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_to_torch_copy_none(backend):
    """copy=None on VectorField / VectorNdarray returns correct torch tensor."""
    torch = pytest.importorskip("torch")
    t = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(4, 3)
    t.from_numpy(src)
    out = t.to_torch(copy=None)
    np.testing.assert_array_equal(out.numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_to_numpy_copy_none_dtype_fallback(backend):
    """copy=None with dtype on VectorField / VectorNdarray silently falls back."""
    t = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    src = np.arange(12, dtype=np.float32).reshape(4, 3)
    t.from_numpy(src)
    arr = t.to_numpy(dtype=np.float64, copy=None)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, src.astype(np.float64))
