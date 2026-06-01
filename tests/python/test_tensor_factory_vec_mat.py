"""Tests for ``qd.Vector.tensor`` / ``qd.Matrix.tensor`` factory classmethods.

These are the per-tensor backend dispatchers for vector- and matrix-element tensors, parallel to scalar
``qd.tensor()``. The dispatch implementation lives in ``quadrants/_tensor.py`` as ``_tensor_vec`` / ``_tensor_mat``
(private); the public surface is the classmethods exercised here.

Each behavioural test is parametrized over both backends.
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _vec_reference(backend):
    cls = qd.Vector
    return cls.field if backend is qd.Backend.FIELD else cls.ndarray


def _mat_reference(backend):
    cls = qd.Matrix
    return cls.field if backend is qd.Backend.FIELD else cls.ndarray


# ----------------------------------------------------------------------------
# qd.Vector.tensor
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_vector_tensor_default_backend_matches_vector_ndarray():
    """Post stork-19 ``qd.Vector.tensor`` returns a ``qd.VectorTensor`` wrapper; the underlying impl must match what
    ``qd.Vector.ndarray`` returns directly."""
    a = qd.Vector.tensor(3, qd.f32, shape=(4,))
    b = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    assert isinstance(a, qd.VectorTensor)
    assert type(a._unwrap()) is type(b)
    assert a.shape == b.shape == (4,)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_tensor_explicit_backend_matches_underlying(backend):
    a = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)
    ref = _vec_reference(backend)(3, qd.f32, shape=(4,))
    assert isinstance(a, qd.VectorTensor)
    assert type(a._unwrap()) is type(ref)
    assert a.shape == ref.shape == (4,)


@test_utils.test(arch=qd.cpu)
def test_vector_tensor_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.Vector.tensor(3, qd.f32, shape=(4,), backend="oops")


@test_utils.test(arch=qd.cpu)
def test_vector_tensor_rejects_unknown_kwarg():
    """``qd.Vector.tensor`` shares the same kwarg-validation contract as ``qd.tensor``, minus ``layout=`` (layout
    semantics over an extra element axis are out of scope for now). ``order=`` raises with a dedicated layout-pointing
    message."""
    with pytest.raises(TypeError, match="layout="):
        qd.Vector.tensor(3, qd.f32, shape=(4,), order="ji")
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.Vector.tensor(3, qd.f32, shape=(4,), offset=(1,))
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.Vector.tensor(3, qd.f32, shape=(4,), nonsense=42)
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.Vector.tensor(3, qd.f32, shape=(4,), layout=(0,))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_vector_tensor_kernel_roundtrip(backend):
    v = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i in range(4):
                for j in qd.static(range(3)):
                    x[i][j] = i * 10.0 + j

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i in range(4):
                for j in qd.static(range(3)):
                    x[i][j] = i * 10.0 + j

    fill(v)
    arr = v.to_numpy()
    assert arr.shape == (4, 3)
    assert arr[2, 1] == 21.0


# ----------------------------------------------------------------------------
# qd.Matrix.tensor
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_default_backend_matches_matrix_ndarray():
    a = qd.Matrix.tensor(2, 3, qd.f32, shape=(4,))
    b = qd.Matrix.ndarray(2, 3, qd.f32, shape=(4,))
    assert isinstance(a, qd.MatrixTensor)
    assert type(a._unwrap()) is type(b)
    assert a.shape == b.shape == (4,)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_explicit_backend_matches_underlying(backend):
    a = qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), backend=backend)
    ref = _mat_reference(backend)(2, 3, qd.f32, shape=(4,))
    assert isinstance(a, qd.MatrixTensor)
    assert type(a._unwrap()) is type(ref)
    assert a.shape == ref.shape == (4,)


@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), backend=99)


@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_rejects_unknown_kwarg():
    """``qd.Matrix.tensor`` shares the same kwarg-validation contract as ``qd.tensor``, minus ``layout=`` (layout
    semantics over an extra element axis are out of scope for now). ``order=`` raises with a dedicated layout-pointing
    message."""
    with pytest.raises(TypeError, match="layout="):
        qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), order="ji")
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), offset=(1,))
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), nonsense=42)
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), layout=(0,))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_kernel_roundtrip(backend):
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i in range(3):
                for r in qd.static(range(2)):
                    for c in qd.static(range(2)):
                        x[i][r, c] = i * 100.0 + r * 10.0 + c

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i in range(3):
                for r in qd.static(range(2)):
                    for c in qd.static(range(2)):
                        x[i][r, c] = i * 100.0 + r * 10.0 + c

    fill(m)
    arr = m.to_numpy()
    assert arr.shape == (3, 2, 2)
    assert arr[1, 0, 1] == 101.0
    assert arr[2, 1, 0] == 210.0


# ----------------------------------------------------------------------------
# Privacy guard: tensor_vec / tensor_mat are no longer in the public API.
# ----------------------------------------------------------------------------


def test_tensor_vec_mat_not_public():
    """Ensure the old top-level names were dropped from the public surface."""
    assert not hasattr(qd, "tensor_vec"), "qd.tensor_vec leaked into public API"
    assert not hasattr(qd, "tensor_mat"), "qd.tensor_mat leaked into public API"
