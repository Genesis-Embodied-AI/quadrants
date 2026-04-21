"""Tests for ``needs_grad=True`` on the tensor factories.

The factories pass ``needs_grad`` through to ``qd.field`` / ``qd.ndarray``
via ``**kwargs``; ``qd.Vector.ndarray`` / ``qd.Matrix.ndarray`` accept
``needs_grad`` and allocate a companion grad ndarray of matching shape and
element type (real-only). These tests lock that behaviour as part of the
public contract on every (factory, backend) combination.
"""

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_tensor_field_needs_grad_allocates_grad():
    a = qd.tensor(qd.f32, shape=(4,), needs_grad=True)
    assert a.grad is not None
    assert a.grad.shape == a.shape


@test_utils.test(arch=qd.cpu)
def test_tensor_ndarray_needs_grad_allocates_grad():
    a = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.NDARRAY, needs_grad=True)
    assert a.grad is not None
    assert a.grad.shape == a.shape


@test_utils.test(arch=qd.cpu)
def test_tensor_field_grad_kernel_roundtrip():
    """Write to primal and grad through a kernel; read back canonically."""
    a = qd.tensor(qd.f32, shape=(4,), needs_grad=True)

    @qd.kernel
    def write_primal(x: qd.template()):
        for i in range(4):
            x[i] = i * 10.0

    @qd.kernel
    def write_grad(x: qd.template()):
        for i in range(4):
            x.grad[i] = i * 100.0

    write_primal(a)
    write_grad(a)
    assert list(a.to_numpy()) == [0.0, 10.0, 20.0, 30.0]
    assert list(a.grad.to_numpy()) == [0.0, 100.0, 200.0, 300.0]


@test_utils.test(arch=qd.cpu)
def test_tensor_ndarray_grad_kernel_roundtrip():
    a = qd.tensor(qd.f32, shape=(4,), backend=qd.Backend.NDARRAY, needs_grad=True)

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


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_field_needs_grad():
    v = qd.Vector.tensor(3, qd.f32, shape=(2,), needs_grad=True)
    assert v.grad is not None
    assert v.grad.shape == v.shape


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_field_needs_grad():
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), needs_grad=True)
    assert m.grad is not None
    assert m.grad.shape == m.shape


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_ndarray_needs_grad_allocates_grad():
    v = qd.Vector.tensor(3, qd.f32, shape=(2,), backend=qd.Backend.NDARRAY, needs_grad=True)
    assert v.grad is not None
    assert tuple(v.grad.shape) == tuple(v.shape)
    assert v.grad.element_shape == v.element_shape


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_ndarray_needs_grad_allocates_grad():
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=qd.Backend.NDARRAY, needs_grad=True)
    assert m.grad is not None
    assert tuple(m.grad.shape) == tuple(m.shape)
    assert m.grad.element_shape == m.element_shape


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_ndarray_grad_kernel_roundtrip():
    v = qd.Vector.tensor(3, qd.f32, shape=(2,), backend=qd.Backend.NDARRAY, needs_grad=True)

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


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_ndarray_grad_kernel_roundtrip():
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=qd.Backend.NDARRAY, needs_grad=True)

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


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_ndarray_needs_grad_rejects_int_dtype():
    """needs_grad=True requires a real (floating-point) element dtype."""
    import pytest

    with pytest.raises(qd.QuadrantsRuntimeError, match="needs_grad"):
        qd.Vector.tensor(3, qd.i32, shape=(2,), backend=qd.Backend.NDARRAY, needs_grad=True)


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_ndarray_needs_grad_rejects_int_dtype():
    import pytest

    with pytest.raises(qd.QuadrantsRuntimeError, match="needs_grad"):
        qd.Matrix.tensor(2, 2, qd.i32, shape=(3,), backend=qd.Backend.NDARRAY, needs_grad=True)
