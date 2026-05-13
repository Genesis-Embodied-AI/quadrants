"""Pin host-side ``t[i, j]`` semantics on layout-tagged tensors.

Background
----------
The canonical -> physical index permutation for layout-tagged tensors is implemented in ``ast_transformer.py``
(``build_Subscript`` and ``build_struct_for``), which only fires inside ``@qd.kernel`` bodies. Python-scope (host)
``__getitem__`` / ``__setitem__`` on the *bare* ``Ndarray`` is **not** layout-aware - the index hits the host accessor
directly. Field is layout-aware for free because its host accessor walks the SNode tree (which applies ``order=``).

Per the user-stated rule that ``layout=`` must be invisible to users ("anything that doesn't work with non-identity
layout is also useless"), host-side indexing has to return the *canonical* element, identical to ``a.to_numpy()[i, j]``.

The ``qd.Tensor`` wrapper owns this fix: on a layout-tagged ndarray it translates the canonical user key to physical
coords before hitting the impl accessor; on a field it simply delegates. Both paths give the user the canonical view,
symmetric across backends.

Stork-19 flips ``qd.tensor()`` (and the Vector/Matrix variants) to return ``qd.Tensor`` wrappers, so the natural
user path here goes through the wrapper and the previous gotcha-B xfails turn into unconditional passes.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]

_LAYOUTS_RANK2 = [(0, 1), (1, 0)]
_LAYOUTS_RANK3 = [(0, 1, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0)]


def _is_identity(layout):
    return layout == tuple(range(len(layout)))


def _fill_via_from_numpy(a, canonical_shape):
    """Populate ``a`` with distinct, position-encoding values via the layout-aware ``from_numpy`` path. Returns the
    source numpy array so the test can use it as the canonical reference."""
    src = np.arange(int(np.prod(canonical_shape)), dtype=np.int32).reshape(canonical_shape)
    a.from_numpy(src)
    return src


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_host_getitem_canonical_rank2(backend, layout):
    """``a[i, j]`` at host scope must equal ``a.to_numpy()[i, j]``."""
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _fill_via_from_numpy(a, canonical)

    np_view = a.to_numpy()
    np.testing.assert_array_equal(np_view, src)  # to_numpy is canonical (already pinned)

    for ci in itertools.product(*(range(d) for d in canonical)):
        host = a[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"host indexing leaked physical layout: a{list(ci)} = {int(host)}, "
            f"canonical (to_numpy/source) = {canon}, layout={layout}"
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_host_getitem_canonical_rank3(backend, layout):
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _fill_via_from_numpy(a, canonical)

    np_view = a.to_numpy()
    np.testing.assert_array_equal(np_view, src)

    for ci in itertools.product(*(range(d) for d in canonical)):
        host = a[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"host indexing leaked physical layout: a{list(ci)} = {int(host)}, "
            f"canonical (to_numpy/source) = {canon}, layout={layout}"
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_host_setitem_canonical_rank2(backend, layout):
    """``a[i, j] = v`` at host scope must place ``v`` at canonical coordinate ``(i, j)``, i.e.
    ``a.to_numpy()[i, j] == v``."""
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    expected = np.zeros(canonical, dtype=np.int32)
    for n, ci in enumerate(itertools.product(*(range(d) for d in canonical))):
        v = 1000 + n
        a[ci] = v
        expected[ci] = v

    np.testing.assert_array_equal(a.to_numpy(), expected)


# ----------------------------------------------------------------------
# Bare-impl path: same canonical-view contract must hold when the user constructs the wrapper explicitly from a bare
# impl (i.e. allocates via ``qd.field`` / ``qd.ndarray`` and wraps with ``qd.Tensor(impl)``). Sanity check that the
# wrapper's host-side fix isn't tied to the factory codepath.
# ----------------------------------------------------------------------


def _alloc_bare(backend, dtype, canonical, layout):
    """Allocate a bare impl with the given layout, mirroring the factory logic. Used by the wrapper-from-bare tests
    below."""
    if backend is qd.Backend.FIELD:
        if _is_identity(layout):
            return qd.field(dtype, canonical)
        # qd.field expects ``order=`` as an axis-char string ("ji", "kij", ...)
        order = "".join(chr(ord("i") + ax) for ax in layout)
        f = qd.field(dtype, canonical, order=order)
        # ``_field`` now tags ``_qd_layout`` automatically; no extra attribute needed on the returned object.
        return f
    # NDARRAY
    if _is_identity(layout):
        return qd.ndarray(dtype, canonical)
    physical = tuple(canonical[ax] for ax in layout)
    arr = qd.ndarray(dtype, physical)
    arr._qd_layout = tuple(layout)
    return arr


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapped_from_bare_host_getitem_rank2(backend, layout):
    canonical = (3, 4)
    a = _alloc_bare(backend, qd.i32, canonical, layout)
    t = qd.Tensor(a)
    src = _fill_via_from_numpy(t, canonical)

    assert t.shape == canonical
    assert t.layout == (None if _is_identity(layout) else layout)

    for ci in itertools.product(*(range(d) for d in canonical)):
        host = t[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"wrapped-from-bare host indexing broke canonical view: "
            f"t{list(ci)} = {int(host)}, canonical = {canon}, "
            f"backend={backend!r}, layout={layout}"
        )
