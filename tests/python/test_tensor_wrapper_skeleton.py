"""Skeleton tests for the ``qd.Tensor`` wrapper introspection surface.

Pins the wrapper's basic introspection (``shape``, ``dtype``, ``layout``, ``_unwrap()``, ``repr``). Layout-aware host
indexing and surface-method forwards live in ``test_tensor_layout_host_indexing.py`` and
``test_tensor_wrapper_surface.py`` respectively.

Stork-19 flipped ``qd.tensor()`` to return wrappers, so we can use it directly here. To exercise the explicit-
construction path (``qd.Tensor(impl)``), we drop down to ``qd.field`` / ``qd.ndarray`` (which still return bare impls).
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_construction_forwards_basic_attrs(backend):
    """``qd.tensor()`` already returns a wrapper post stork-19."""
    t = qd.tensor(qd.f32, shape=(3, 4), backend=backend)
    assert isinstance(t, qd.Tensor)
    assert t.shape == (3, 4)
    assert t.dtype == qd.f32
    assert t.layout is None
    impl = t._unwrap()
    assert impl is not t
    # Wrapping the bare impl explicitly returns an equivalent wrapper.
    t2 = qd.Tensor(impl)
    assert t2.shape == (3, 4)
    assert t2._unwrap() is impl


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", [(0, 1), (1, 0)])
@test_utils.test(arch=qd.cpu)
def test_wrapper_layout_is_canonical_and_introspectable(backend, layout):
    t = qd.tensor(qd.f32, shape=(3, 4), backend=backend, layout=layout)
    # ``shape`` is always canonical; ``layout`` reflects the user-supplied permutation (or ``None`` for identity,
    # normalised at the impl layer).
    assert t.shape == (3, 4)
    if layout == (0, 1):
        assert t.layout is None
    else:
        assert t.layout == layout


def test_wrapper_rejects_non_tensor():
    with pytest.raises(TypeError, match="Tensor.*requires"):
        qd.Tensor(42)
    with pytest.raises(TypeError, match="Tensor.*requires"):
        qd.Tensor("not a tensor")


def test_wrapper_rejects_double_wrap():
    """Wrapping an existing wrapper would silently confuse identity; the constructor rejects it with a clear
    TypeError."""
    qd.init(arch=qd.x64)
    t = qd.tensor(qd.i32, shape=(4,))
    with pytest.raises(TypeError, match="Tensor.*requires"):
        qd.Tensor(t)


@test_utils.test(arch=qd.cpu)
def test_wrapper_repr_includes_backend_and_layout():
    t = qd.tensor(qd.i32, shape=(2, 3), backend=qd.Backend.NDARRAY, layout=(1, 0))
    r = repr(t)
    assert "NDARRAY" in r
    assert "(2, 3)" in r
    assert "(1, 0)" in r
