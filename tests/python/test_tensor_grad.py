"""Tests for ``needs_grad=True`` on the tensor factories.

The factories already pass ``needs_grad`` through to ``qd.field`` /
``qd.ndarray`` via ``**kwargs`` (PRs 2-3); this PR adds explicit coverage
to lock that behaviour as part of the public contract.
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
    v = qd.tensor_vec(3, qd.f32, shape=(2,), needs_grad=True)
    assert v.grad is not None
    assert v.grad.shape == v.shape


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_field_needs_grad():
    m = qd.tensor_mat(2, 2, qd.f32, shape=(3,), needs_grad=True)
    assert m.grad is not None
    assert m.grad.shape == m.shape


# Note: qd.Vector.ndarray and qd.Matrix.ndarray do not currently accept
# needs_grad=True (see quadrants/lang/matrix.py). qd.tensor_vec/_mat with
# backend=NDARRAY inherits that limitation. The scalar ndarray backend
# (qd.tensor + Backend.NDARRAY) does support needs_grad — covered above.
