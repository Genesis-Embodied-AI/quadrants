"""Differentiation through ``dataclasses.dataclass`` containers.

These tests pin that gradients flow correctly when kernel arguments are wrapped in plain Python dataclasses, across the
tensor types Quadrants exposes:

* ``qd.ndarray`` — typed-dataclass annotation + ``qd.template()`` path; gradient via ``kernel.grad()``.
* ``qd.field`` — ``qd.template()`` path; gradient via ``qd.ad.Tape``.
* ``qd.tensor(backend=NDARRAY)`` — same path as ``qd.ndarray``; the dispatcher returns a wrapper
  whose ndarray ``_impl`` is unwrapped by the dataclass-annotation infrastructure.
* ``qd.tensor(backend=FIELD)`` — works when the dataclass member is annotated ``qd.Tensor``
  (or ``qd.template()``). With ``object`` / no annotation the wrapper survives into kernel scope
  and host-side ``__getitem__`` asserts.
* mixed — single dataclass holding both a ``qd.ndarray`` and a ``qd.field`` member.

Pattern mirrors ``test_ad_ndarray.py`` (ndarray) and ``test_ad_basics.py`` (field). See
``docs/source/user_guide/compound_types.md`` overview table — column "supports differentiation?"
for ``dataclasses.dataclass``.
"""

import dataclasses

import numpy as np

import quadrants as qd

from tests import test_utils

archs_support_ndarray_ad = [qd.cpu, qd.cuda, qd.amdgpu]


# ----------------------------------------------------------------------------
# qd.ndarray members
# ----------------------------------------------------------------------------


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_dataclass_ndarray_typed_annotation():
    """dataclass holding qd.ndarrays, passed via typed-dataclass kernel-arg annotation."""
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.f64, 1]
        b: qd.types.NDArray[qd.f64, 1]
        p: qd.types.NDArray[qd.f64, 1]

    @qd.kernel
    def compute(s: State):
        for i in range(N):
            s.p[i] = s.a[i] * s.b[i] + 1.0

    a = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f64, shape=N)
    p = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    for i in range(N):
        a[i] = 3.0
        b[i] = float(i + 1)

    state = State(a=a, b=b, p=p)
    compute(state)
    np.testing.assert_allclose(p.to_numpy(), 3.0 * np.arange(1, N + 1) + 1.0)

    for i in range(N):
        p.grad[i] = 1.0

    compute.grad(state)
    np.testing.assert_allclose(a.grad.to_numpy(), b.to_numpy())


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_dataclass_ndarray_template():
    """dataclass holding qd.ndarrays, passed via qd.template()."""
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.f64, 1]
        b: qd.types.NDArray[qd.f64, 1]
        p: qd.types.NDArray[qd.f64, 1]

    @qd.kernel
    def compute(s: qd.template()):
        for i in range(N):
            s.p[i] = s.a[i] * s.b[i] + 1.0

    a = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f64, shape=N)
    p = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    for i in range(N):
        a[i] = 3.0
        b[i] = float(i + 1)

    state = State(a=a, b=b, p=p)
    compute(state)
    np.testing.assert_allclose(p.to_numpy(), 3.0 * np.arange(1, N + 1) + 1.0)

    for i in range(N):
        p.grad[i] = 1.0

    compute.grad(state)
    np.testing.assert_allclose(a.grad.to_numpy(), b.to_numpy())


# ----------------------------------------------------------------------------
# qd.field members
# ----------------------------------------------------------------------------


@test_utils.test(default_fp=qd.f64, require=[qd.extension.adstack, qd.extension.data64])
def test_ad_dataclass_field_template_tape():
    """dataclass holding qd.fields, passed via qd.template(), gradient via qd.ad.Tape."""
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a: object
        b: object
        loss: object

    a = qd.field(qd.f64, shape=(N,), needs_grad=True)
    b = qd.field(qd.f64, shape=(N,))
    loss = qd.field(qd.f64, shape=(), needs_grad=True)
    for i in range(N):
        a[i] = 3.0
        b[i] = float(i + 1)

    state = State(a=a, b=b, loss=loss)

    @qd.kernel
    def compute(s: qd.template()):
        for i in range(N):
            s.loss[None] += s.a[i] * s.b[i]

    with qd.ad.Tape(loss):
        compute(state)

    # loss = sum_i a[i] * b[i]; dloss/da[i] = b[i]
    np.testing.assert_allclose(a.grad.to_numpy(), b.to_numpy())
    expected_loss = float((3.0 * np.arange(1, N + 1)).sum())
    np.testing.assert_allclose(loss[None], expected_loss)


# ----------------------------------------------------------------------------
# qd.tensor dispatcher
# ----------------------------------------------------------------------------


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_dataclass_tensor_ndarray_backend():
    """dataclass holding qd.tensor(..., backend=NDARRAY) members; ndarray-AD via kernel.grad()."""
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.f64, 1]
        b: qd.types.NDArray[qd.f64, 1]
        p: qd.types.NDArray[qd.f64, 1]

    @qd.kernel
    def compute(s: State):
        for i in range(N):
            s.p[i] = s.a[i] * s.b[i] + 1.0

    a = qd.tensor(qd.f64, shape=(N,), backend=qd.Backend.NDARRAY, needs_grad=True)
    b = qd.tensor(qd.f64, shape=(N,), backend=qd.Backend.NDARRAY)
    p = qd.tensor(qd.f64, shape=(N,), backend=qd.Backend.NDARRAY, needs_grad=True)
    for i in range(N):
        a[i] = 3.0
        b[i] = float(i + 1)

    state = State(a=a, b=b, p=p)
    compute(state)
    np.testing.assert_allclose(p.to_numpy(), 3.0 * np.arange(1, N + 1) + 1.0)

    for i in range(N):
        p.grad[i] = 1.0

    compute.grad(state)
    np.testing.assert_allclose(a.grad.to_numpy(), b.to_numpy())


@test_utils.test(default_fp=qd.f64, require=[qd.extension.adstack, qd.extension.data64])
def test_ad_dataclass_tensor_field_backend_tape():
    """dataclass holding qd.tensor(..., backend=FIELD) members; field-AD via qd.ad.Tape.

    Note: members must be annotated as ``qd.Tensor`` (not ``object``) when the value is a ``qd.tensor(...)`` wrapper.
    The typed-dataclass / template machinery uses the member annotation to decide whether to unwrap the wrapper into
    its underlying impl before the kernel sees ``s.x[i]``. With ``object`` annotation the wrapper survives into kernel
    scope and its host-side ``__getitem__`` asserts.
    """
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.Tensor
        b: qd.Tensor
        loss: qd.Tensor

    a = qd.tensor(qd.f64, shape=(N,), backend=qd.Backend.FIELD, needs_grad=True)
    b = qd.tensor(qd.f64, shape=(N,), backend=qd.Backend.FIELD)
    loss = qd.tensor(qd.f64, shape=(), backend=qd.Backend.FIELD, needs_grad=True)
    for i in range(N):
        a[i] = 3.0
        b[i] = float(i + 1)

    state = State(a=a, b=b, loss=loss)

    @qd.kernel
    def compute(s: qd.template()):
        for i in range(N):
            s.loss[None] += s.a[i] * s.b[i]

    with qd.ad.Tape(loss._unwrap()):
        compute(state)

    np.testing.assert_allclose(a.grad.to_numpy(), b.to_numpy())


# ----------------------------------------------------------------------------
# Mixed: ndarray + field + tensor in the same dataclass
# ----------------------------------------------------------------------------


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_dataclass_mixed_ndarray_and_tensor_ndarray_backend():
    """Single dataclass holds one qd.ndarray member and one qd.tensor(NDARRAY) member; verify the
    kernel can read/write both and that gradients flow through both."""
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a_nd: qd.types.NDArray[qd.f64, 1]
        a_tens: qd.types.NDArray[qd.f64, 1]
        b: qd.types.NDArray[qd.f64, 1]
        p: qd.types.NDArray[qd.f64, 1]

    @qd.kernel
    def compute(s: State):
        for i in range(N):
            s.p[i] = (s.a_nd[i] + s.a_tens[i]) * s.b[i]

    a_nd = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    a_tens = qd.tensor(qd.f64, shape=(N,), backend=qd.Backend.NDARRAY, needs_grad=True)
    b = qd.ndarray(qd.f64, shape=N)
    p = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    for i in range(N):
        a_nd[i] = 2.0
        a_tens[i] = 5.0
        b[i] = float(i + 1)

    state = State(a_nd=a_nd, a_tens=a_tens, b=b, p=p)
    compute(state)
    np.testing.assert_allclose(p.to_numpy(), 7.0 * np.arange(1, N + 1))

    for i in range(N):
        p.grad[i] = 1.0

    compute.grad(state)
    # dp/da_nd[i] = b[i] ; dp/da_tens[i] = b[i]
    np.testing.assert_allclose(a_nd.grad.to_numpy(), b.to_numpy())
    np.testing.assert_allclose(a_tens.grad.to_numpy(), b.to_numpy())


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_dataclass_mixed_ndarray_and_field_in_same_class():
    """Single dataclass holds both a qd.ndarray member and a qd.field member. The kernel reads
    and writes both. Differentiation is checked through the ndarray path via ``kernel.grad()``
    (the field is along for the ride; its grad allocation must coexist with ndarray grads)."""
    N = 5

    @dataclasses.dataclass(frozen=True)
    class State:
        a_nd: qd.types.NDArray[qd.f64, 1]
        out_field: object
        b: qd.types.NDArray[qd.f64, 1]
        p: qd.types.NDArray[qd.f64, 1]

    @qd.kernel
    def compute(s: qd.template()):
        for i in range(N):
            s.p[i] = s.a_nd[i] * s.b[i]
            s.out_field[i] = s.a_nd[i] + s.b[i]

    a_nd = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    out_field = qd.field(qd.f64, shape=(N,))
    b = qd.ndarray(qd.f64, shape=N)
    p = qd.ndarray(qd.f64, shape=N, needs_grad=True)
    for i in range(N):
        a_nd[i] = 3.0
        b[i] = float(i + 1)

    state = State(a_nd=a_nd, out_field=out_field, b=b, p=p)
    compute(state)
    np.testing.assert_allclose(p.to_numpy(), 3.0 * np.arange(1, N + 1))
    np.testing.assert_allclose(out_field.to_numpy(), 3.0 + np.arange(1, N + 1))

    for i in range(N):
        p.grad[i] = 1.0

    compute.grad(state)
    np.testing.assert_allclose(a_nd.grad.to_numpy(), b.to_numpy())
