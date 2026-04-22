"""Layout-aware interop with NumPy / PyTorch / DLPack.

Pins the canonical-view contract end-to-end on layout-tagged tensors,
across both backends. The contract under test:

- ``a.shape``, ``a.to_numpy().shape``, ``a.to_torch().shape``, and the
  shape carried by ``a.to_dlpack()`` all report the **canonical** shape
  (the same shape the user passed to ``qd.tensor(..., shape=)``),
  regardless of ``_qd_layout``.
- Element values match canonical indexing in every accessor.
- ``from_numpy(canonical_arr)`` round-trips: ``to_numpy()`` of the
  loaded ndarray equals the input.
- ``a.grad`` accessors mirror the primal: same canonical view, same
  element values written by the kernel.
- DLPack carries non-trivial strides on layout-tagged ndarrays (the
  layout shows up as a stride pattern, not a shape permutation).

A small Genesis-shaped smoke test at the bottom exercises the
``(n_dofs, _B)`` + ``layout=(1, 0)`` combination that motivates this
work, ensuring it round-trips through both numpy and torch.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


# ----------------------------------------------------------------------------
# Helpers: canonical reference values for a kernel-fillable tensor.
# ----------------------------------------------------------------------------


def _expected_canonical(shape):
    """Return a numpy array whose entry at canonical index ``ci`` is a
    distinct integer encoding ``ci`` (large enough to catch single-axis
    swaps). Matches the kernel-side fill below."""
    out = np.zeros(shape, dtype=np.int32)
    base = 1
    for dim in reversed(shape):
        base *= max(dim, 1) * 10
    coeffs = []
    rolling = 1
    for dim in reversed(shape):
        coeffs.append(rolling)
        rolling *= max(dim, 1) * 10
    coeffs = list(reversed(coeffs))
    for ci in itertools.product(*[range(d) for d in shape]):
        v = sum(c * k for c, k in zip(ci, coeffs))
        out[ci] = v
    return out


def _make_fill_kernel(shape, backend):
    """Build a kernel that fills its argument with the same canonical
    values as :func:`_expected_canonical`.

    Uses explicit per-axis ``x[i, j, ...]`` indexing (one kernel per rank)
    rather than ``x[I]`` with a Vector index from ``grouped(...)``. The
    canonical→physical AST rewrite at :func:`build_Subscript` only fires
    when the subscript arity matches ``_qd_layout`` length, which means
    ``x[I]`` (single Vector) bypasses the rewrite and writes at the
    physical positions of canonical indices — silently OOB on permuted
    layouts.
    """
    coeffs = []
    rolling = 1
    for dim in reversed(shape):
        coeffs.append(rolling)
        rolling *= max(dim, 1) * 10
    coeffs = list(reversed(coeffs))

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    if len(shape) == 2:
        c0, c1 = coeffs
        d0, d1 = shape

        @qd.kernel
        def fill(x: annotation):
            for i, j in qd.ndrange(d0, d1):
                x[i, j] = i * c0 + j * c1

    elif len(shape) == 3:
        c0, c1, c2 = coeffs
        d0, d1, d2 = shape

        @qd.kernel
        def fill(x: annotation):
            for i, j, k in qd.ndrange(d0, d1, d2):
                x[i, j, k] = i * c0 + j * c1 + k * c2

    else:
        raise NotImplementedError(f"_make_fill_kernel: rank {len(shape)} not supported")

    return fill


# Representative layouts: identity, full reverse, an inner-axis swap, and
# a non-trivial cyclic shift. Keeping the set small so the cross-product
# (backend × layout × accessor) stays cheap.
_LAYOUTS_RANK2 = [(0, 1), (1, 0)]
_LAYOUTS_RANK3 = [(0, 1, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0)]


# ----------------------------------------------------------------------------
# to_numpy() returns the canonical view on every layout × backend.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_canonical_view_rank2(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    fill = _make_fill_kernel(canonical, backend)
    fill(a)

    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, _expected_canonical(canonical))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_to_numpy_canonical_view_rank3(backend, layout):
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    fill = _make_fill_kernel(canonical, backend)
    fill(a)

    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, _expected_canonical(canonical))


# ----------------------------------------------------------------------------
# from_numpy(canonical) → to_numpy() round-trip.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_from_numpy_to_numpy_roundtrip_rank2(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _expected_canonical(canonical)
    a.from_numpy(src)
    np.testing.assert_array_equal(a.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_from_numpy_to_numpy_roundtrip_rank3(backend, layout):
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _expected_canonical(canonical)
    a.from_numpy(src)
    np.testing.assert_array_equal(a.to_numpy(), src)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_from_numpy_rejects_physical_shape(backend, layout):
    """Passing the *physical* shape (instead of canonical) on a
    layout-tagged ndarray must raise a clear shape-mismatch error,
    not silently scramble the data."""
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    physical = tuple(canonical[axis] for axis in layout)
    if physical == canonical:
        pytest.skip("identity layout: canonical and physical coincide")
    bad = np.zeros(physical, dtype=np.int32)
    with pytest.raises((ValueError, RuntimeError)):
        a.from_numpy(bad)


# ----------------------------------------------------------------------------
# to_dlpack() / to_torch() also return canonical views.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_to_dlpack_canonical_shape_rank2(backend, layout):
    """``from_dlpack(t.to_dlpack())`` must report the canonical shape and
    canonical-indexed values, regardless of ``_qd_layout``."""
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    fill = _make_fill_kernel(canonical, backend)
    fill(a)

    t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    assert tuple(t.shape) == canonical
    np.testing.assert_array_equal(t.contiguous().cpu().numpy(), _expected_canonical(canonical))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_to_dlpack_canonical_shape_rank3(backend, layout):
    torch = pytest.importorskip("torch")
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    fill = _make_fill_kernel(canonical, backend)
    fill(a)

    t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    assert tuple(t.shape) == canonical
    np.testing.assert_array_equal(t.contiguous().cpu().numpy(), _expected_canonical(canonical))


@test_utils.test(arch=qd.cpu)
def test_to_dlpack_layout_shows_up_as_strides_not_shape():
    """The DLPack export must keep the canonical *shape* but reflect the
    layout via non-contiguous *strides*; i.e. the resulting torch tensor
    is logically transposed but physically still the same buffer."""
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=(1, 0))
    fill = _make_fill_kernel(canonical, qd.Backend.NDARRAY)
    fill(a)

    t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    assert tuple(t.shape) == canonical
    # On a non-identity layout the dlpack view is by construction not
    # contiguous in the canonical sense — calling ``.contiguous()`` is
    # how the canonical layout would be materialised.
    assert not t.is_contiguous()
    np.testing.assert_array_equal(t.contiguous().cpu().numpy(), _expected_canonical(canonical))


# ----------------------------------------------------------------------------
# .grad accessor mirrors the primal under the same accessor contract.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_grad_to_numpy_canonical_view_rank2(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout, needs_grad=True)
    assert a.grad is not None
    assert tuple(a.grad.shape) == canonical

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i, j in qd.ndrange(*canonical):
                x[i, j] = float(i * 10 + j)
                x.grad[i, j] = float(i * 100 + j * 10)

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i, j in qd.ndrange(*canonical):
                x[i, j] = float(i * 10 + j)
                x.grad[i, j] = float(i * 100 + j * 10)

    fill(a)

    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == canonical
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            assert primal[i, j] == float(i * 10 + j)
            assert grad[i, j] == float(i * 100 + j * 10)


# ----------------------------------------------------------------------------
# Identity-layout / no-layout paths must remain byte-identical to legacy
# (no extra allocation, no transpose, no behavioural drift).
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_no_layout_to_numpy_unchanged():
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY)
    fill = _make_fill_kernel(canonical, qd.Backend.NDARRAY)
    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, _expected_canonical(canonical))


@test_utils.test(arch=qd.cpu)
def test_identity_layout_to_numpy_unchanged():
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=(0, 1))
    fill = _make_fill_kernel(canonical, qd.Backend.NDARRAY)
    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, _expected_canonical(canonical))


# ----------------------------------------------------------------------------
# Genesis-shaped smoke test: (n_dofs, _B) with layout=(1, 0).
#
# This is the shape pattern used by linesearch / Mgrad / aref-style
# tensors in the rigid solver. Migrating those into ``layout=(1, 0)``
# is the immediate motivation for canonicalising every Python accessor;
# this test exercises the round-trip end-to-end on both backends.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_genesis_shaped_dofs_batch_layout(backend):
    n_dofs, batch = 5, 7
    canonical = (n_dofs, batch)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=(1, 0))
    assert tuple(a.shape) == canonical

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for d, b in qd.ndrange(n_dofs, batch):
                x[d, b] = float(d * 1000 + b)

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for d, b in qd.ndrange(n_dofs, batch):
                x[d, b] = float(d * 1000 + b)

    fill(a)

    arr = a.to_numpy()
    assert arr.shape == canonical
    for d in range(n_dofs):
        for b in range(batch):
            assert arr[d, b] == float(d * 1000 + b)

    torch = pytest.importorskip("torch")
    t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    assert tuple(t.shape) == canonical
    np.testing.assert_array_equal(t.contiguous().cpu().numpy(), arr)
