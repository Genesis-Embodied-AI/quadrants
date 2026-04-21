"""Tests for ``qd.tensor`` scalar dispatch.

Scope: scalar-element tensor allocation via ``qd.tensor()`` dispatching to
``qd.field`` or ``qd.ndarray`` based on the ``backend=`` kwarg. No layout,
no vec/mat.

Each behavioural test is parametrized over both backends so coverage stays
symmetric. Tests that probe a single dispatch path (default backend,
unknown-kwarg rejection, error paths) keep their original shape.
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _expected_type(backend):
    return qd.ScalarField if backend is qd.Backend.FIELD else qd.Ndarray


@test_utils.test(arch=qd.cpu)
def test_tensor_default_backend_is_ndarray():
    a = qd.tensor(qd.f32, shape=(4, 5))
    assert isinstance(a, qd.Ndarray)
    assert a.shape == (4, 5)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_explicit_backend_allocates(backend):
    a = qd.tensor(qd.f32, shape=(4, 5), backend=backend)
    assert isinstance(a, _expected_type(backend))
    assert a.shape == (4, 5)


@pytest.mark.parametrize(
    "backend_int,expected",
    [(0, qd.Backend.FIELD), (1, qd.Backend.NDARRAY)],
    ids=["int0=field", "int1=ndarray"],
)
@test_utils.test(arch=qd.cpu)
def test_tensor_int_backend_value_accepted(backend_int, expected):
    """``backend=0`` and ``backend=1`` work too — IntEnum coercion."""
    a = qd.tensor(qd.f32, shape=(3,), backend=backend_int)
    assert isinstance(a, _expected_type(expected))


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
    """Backend-specific knobs and typos are rejected up front; users who
    genuinely need a backend-specific knob must drop down to ``qd.field``
    or ``qd.ndarray`` directly. Only kwargs that are never going to be
    accepted are listed here — branch-specific kwargs (``needs_grad=``
    on PR 5, ``layout=`` on PR 6) get their own acceptance + rejection
    tests on the branch where they land."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.tensor(qd.f32, shape=(4, 5), order="ji")
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.tensor(qd.f32, shape=(4, 5), offset=(1, 1))
    with pytest.raises(TypeError, match="unexpected keyword"):
        qd.tensor(qd.f32, shape=(4, 5), nonsense=42)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_kernel_roundtrip(backend):
    """Allocate via ``qd.tensor()``, fill in a kernel, read back via numpy.

    Uses the backend-appropriate annotation (``qd.template()`` for FIELD,
    ``qd.types.ndarray()`` for NDARRAY); the polymorphic ``qd.tensor_t``
    annotation is not yet available on this branch.
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
