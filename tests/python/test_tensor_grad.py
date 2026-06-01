"""Tests for ``needs_grad=True`` on the tensor factories.

The factories pass ``needs_grad`` through to ``qd.field`` / ``qd.ndarray`` via ``**kwargs``; ``qd.Vector.ndarray`` /
``qd.Matrix.ndarray`` accept ``needs_grad`` and allocate a companion grad ndarray of matching shape and element type
(real-only). These tests lock that behaviour as part of the public contract on every (factory, backend) combination.

Behavioural tests are parametrized over both backends. The int-dtype rejection check is NDARRAY-only because the FIELD
path goes through the older Vector/Matrix.field code which raises a different error (out of scope for this branch's
contract).
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


# ----------------------------------------------------------------------------
# Scalar qd.tensor()
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_needs_grad_allocates_grad(backend):
    a = qd.tensor(qd.f32, shape=(4,), backend=backend, needs_grad=True)
    assert a.grad is not None
    assert a.grad.shape == a.shape


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_grad_kernel_roundtrip(backend):
    """Write to primal and grad through a kernel; read back canonically."""
    a = qd.tensor(qd.f32, shape=(4,), backend=backend, needs_grad=True)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def write_primal(x: qd.template()):
            for i in range(4):
                x[i] = i * 10.0

        @qd.kernel
        def write_grad(x: qd.template()):
            for i in range(4):
                x.grad[i] = i * 100.0

    else:

        @qd.kernel
        def write_primal(x: qd.types.ndarray()):
            for i in range(4):
                x[i] = i * 10.0

        @qd.kernel
        def write_grad(x: qd.types.ndarray()):
            for i in range(4):
                x.grad[i] = i * 100.0

    write_primal(a)
    write_grad(a)
    assert list(a.to_numpy()) == [0.0, 10.0, 20.0, 30.0]
    assert list(a.grad.to_numpy()) == [0.0, 100.0, 200.0, 300.0]


# ----------------------------------------------------------------------------
# Vector / Matrix tensor factories
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_vec_needs_grad_allocates_grad(backend):
    v = qd.Vector.tensor(3, qd.f32, shape=(2,), backend=backend, needs_grad=True)
    assert v.grad is not None
    assert tuple(v.grad.shape) == tuple(v.shape)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_mat_needs_grad_allocates_grad(backend):
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend, needs_grad=True)
    assert m.grad is not None
    assert tuple(m.grad.shape) == tuple(m.shape)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_vec_grad_kernel_roundtrip(backend):
    v = qd.Vector.tensor(3, qd.f32, shape=(2,), backend=backend, needs_grad=True)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def write_primal(x: qd.template()):
            for i in range(2):
                for j in qd.static(range(3)):
                    x[i][j] = i * 10.0 + j

        @qd.kernel
        def write_grad(x: qd.template()):
            for i in range(2):
                for j in qd.static(range(3)):
                    x.grad[i][j] = i * 100.0 + j * 10.0

    else:

        @qd.kernel
        def write_primal(x: qd.types.ndarray()):
            for i in range(2):
                for j in qd.static(range(3)):
                    x[i][j] = i * 10.0 + j

        @qd.kernel
        def write_grad(x: qd.types.ndarray()):
            for i in range(2):
                for j in qd.static(range(3)):
                    x.grad[i][j] = i * 100.0 + j * 10.0

    write_primal(v)
    write_grad(v)
    primal = v.to_numpy()
    grad = v.grad.to_numpy()
    assert primal[0, 0] == 0.0 and primal[1, 2] == 12.0
    assert grad[0, 0] == 0.0 and grad[1, 2] == 120.0


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_mat_grad_kernel_roundtrip(backend):
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=backend, needs_grad=True)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def write_primal(x: qd.template()):
            for i in range(3):
                for r in qd.static(range(2)):
                    for c in qd.static(range(2)):
                        x[i][r, c] = i * 10.0 + r * 2.0 + c

        @qd.kernel
        def write_grad(x: qd.template()):
            for i in range(3):
                for r in qd.static(range(2)):
                    for c in qd.static(range(2)):
                        x.grad[i][r, c] = i * 100.0 + r * 20.0 + c * 10.0

    else:

        @qd.kernel
        def write_primal(x: qd.types.ndarray()):
            for i in range(3):
                for r in qd.static(range(2)):
                    for c in qd.static(range(2)):
                        x[i][r, c] = i * 10.0 + r * 2.0 + c

        @qd.kernel
        def write_grad(x: qd.types.ndarray()):
            for i in range(3):
                for r in qd.static(range(2)):
                    for c in qd.static(range(2)):
                        x.grad[i][r, c] = i * 100.0 + r * 20.0 + c * 10.0

    write_primal(m)
    write_grad(m)
    primal = m.to_numpy()
    grad = m.grad.to_numpy()
    assert primal[2, 1, 1] == 23.0
    assert grad[2, 1, 1] == 230.0


# ----------------------------------------------------------------------------
# Negative path: int dtype + needs_grad on the NDARRAY backend. Kept NDARRAY-only because the FIELD path raises
# through the legacy create_field machinery with a different error class / message; the Vector.ndarray / Matrix.ndarray
# rejection added in this branch is the focused contract.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_ndarray_needs_grad_rejects_int_dtype():
    with pytest.raises(qd.QuadrantsRuntimeError, match="needs_grad"):
        qd.Vector.tensor(3, qd.i32, shape=(2,), backend=qd.Backend.NDARRAY, needs_grad=True)


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_ndarray_needs_grad_rejects_int_dtype():
    with pytest.raises(qd.QuadrantsRuntimeError, match="needs_grad"):
        qd.Matrix.tensor(2, 2, qd.i32, shape=(3,), backend=qd.Backend.NDARRAY, needs_grad=True)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test()
def test_has_grad_has_dual_reflect_storage_allocation(backend):
    """``Tensor.has_grad()`` / ``has_dual()`` report whether the adjoint / dual storage is actually allocated.

    Internal details: on the FIELD backend ``self.grad`` / ``self.dual`` are non-``None`` for every real-dtype field
    (the wrapper is allocated up-front so ``qd.root.lazy_grad()`` / ``qd.root.lazy_dual()`` can populate it later), so
    callers cannot rely on a plain ``is not None`` check to distinguish "allocated and writable" from "wrapper present
    but un-placed". Pin that ``has_grad`` / ``has_dual`` mirror the SNode-level truth on FIELD and the storage-presence
    truth on NDARRAY (where the grad ndarray is only allocated when ``needs_grad=True``).
    """
    x = qd.tensor(dtype=qd.f32, shape=(4,), backend=backend)
    assert not x.has_grad()
    assert not x.has_dual()
    y = qd.tensor(dtype=qd.f32, shape=(4,), backend=backend, needs_grad=True)
    assert y.has_grad()
    assert not y.has_dual()
