"""Tests for ``qd.tensor`` scalar dispatch.

Scope: scalar-element tensor allocation via ``qd.tensor()`` dispatching to ``qd.field`` or ``qd.ndarray`` based on the
``backend=`` kwarg. No layout, no vec/mat.

Each behavioural test is parametrized over both backends so coverage stays symmetric. Tests that probe a single dispatch
path (default backend, unknown-kwarg rejection, error paths) keep their original shape.
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _expected_impl_type(backend):
    return qd.ScalarField if backend is qd.Backend.FIELD else qd.Ndarray


@test_utils.test(arch=qd.cpu)
def test_tensor_default_backend_is_ndarray():
    a = qd.tensor(qd.f32, shape=(4, 5))
    assert isinstance(a, qd.Tensor)
    assert isinstance(a._unwrap(), qd.Ndarray)
    assert a.shape == (4, 5)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_explicit_backend_allocates(backend):
    a = qd.tensor(qd.f32, shape=(4, 5), backend=backend)
    assert isinstance(a, qd.Tensor)
    assert isinstance(a._unwrap(), _expected_impl_type(backend))
    assert a.shape == (4, 5)


@pytest.mark.parametrize(
    "backend_int,expected",
    [(0, qd.Backend.FIELD), (1, qd.Backend.NDARRAY)],
    ids=["int0=field", "int1=ndarray"],
)
@test_utils.test(arch=qd.cpu)
def test_tensor_int_backend_value_accepted(backend_int, expected):
    """``backend=0`` and ``backend=1`` work too - IntEnum coercion."""
    a = qd.tensor(qd.f32, shape=(3,), backend=backend_int)
    assert isinstance(a, qd.Tensor)
    assert isinstance(a._unwrap(), _expected_impl_type(expected))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_dtype_propagation(backend):
    a = qd.tensor(qd.i32, shape=(4,), backend=backend)
    assert a.dtype == qd.i32


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_int_shape_normalised(backend):
    """Passing an int as shape works the same as a 1-tuple."""
    a = qd.tensor(qd.f32, shape=8, backend=backend)
    assert a.shape == (8,)


@test_utils.test(arch=qd.cpu)
def test_tensor_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.tensor(qd.f32, shape=(3,), backend=42)
    with pytest.raises(ValueError, match="backend="):
        qd.tensor(qd.f32, shape=(3,), backend="field")


@test_utils.test(arch=qd.cpu)
def test_tensor_rejects_unknown_kwarg():
    """Backend-specific knobs and typos are rejected up front; users who genuinely need a backend-specific knob must
    drop down to ``qd.field`` or ``qd.ndarray`` directly.

    ``order=`` gets its own dedicated rejection in :func:`test_tensor_layout.test_order_kwarg_rejected` because the
    factory raises a different error message pointing users at ``layout=``."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.tensor(qd.f32, shape=(4, 5), offset=(1, 1))
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.tensor(qd.f32, shape=(4, 5), nonsense=42)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_kernel_roundtrip(backend):
    """Allocate via ``qd.tensor()``, fill in a kernel, read back via numpy.

    Uses the backend-appropriate annotation (``qd.template()`` for FIELD, ``qd.types.ndarray()`` for NDARRAY); the
    polymorphic ``qd.Tensor`` annotation is not yet available on this branch.
    """
    a = qd.tensor(qd.i32, shape=(4,), backend=backend)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i in range(4):
                x[i] = i * 2

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i in range(4):
                x[i] = i * 2

    fill(a)
    assert list(a.to_numpy()) == [0, 2, 4, 6]


# ----------------------------------------------------------------------------
# qd.tensor() with compound dtype (vector / matrix types)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_compound_vector_returns_vector_tensor(backend):
    """``qd.tensor(vec3_type, shape)`` must return a ``VectorTensor``, not the base ``Tensor``, so that
    ``element_shape`` is available."""
    vec3 = qd.types.vector(3, qd.f32)
    a = qd.tensor(vec3, shape=(4,), backend=backend)
    assert isinstance(a, qd.VectorTensor), f"expected VectorTensor, got {type(a).__name__}"
    assert a.element_shape == (3,)
    assert a.shape == (4,)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_compound_matrix_returns_matrix_tensor(backend):
    """``qd.tensor(mat2x3_type, shape)`` must return a ``MatrixTensor``."""
    mat23 = qd.types.matrix(2, 3, qd.f32)
    a = qd.tensor(mat23, shape=(5,), backend=backend)
    assert isinstance(a, qd.MatrixTensor), f"expected MatrixTensor, got {type(a).__name__}"
    assert a.element_shape == (2, 3)
    assert a.shape == (5,)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_compound_vector_roundtrip(backend):
    """Allocate via ``qd.tensor(vec3, ...)`` and verify data roundtrip."""
    vec3 = qd.types.vector(3, qd.f32)
    a = qd.tensor(vec3, shape=(4,), backend=backend)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            for j in qd.static(range(3)):
                x[i][j] = i * 10.0 + j

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == (4, 3)
    assert arr[2, 1] == 21.0
