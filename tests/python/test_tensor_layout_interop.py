"""Layout-aware interop with NumPy / PyTorch / DLPack.

Pins the canonical-view contract end-to-end on layout-tagged tensors, across both backends. The contract under test:

- ``a.shape``, ``a.to_numpy().shape``, ``a.to_torch().shape``, and the shape carried by ``a.to_dlpack()`` all report
  the **canonical** shape (the same shape the user passed to ``qd.tensor(..., shape=)``), regardless of ``_qd_layout``.
- Element values match canonical indexing in every accessor.
- ``from_numpy(canonical_arr)`` round-trips: ``to_numpy()`` of the loaded ndarray equals the input.
- ``a.grad`` accessors mirror the primal: same canonical view, same element values written by the kernel.
- DLPack carries non-trivial strides on layout-tagged ndarrays (the layout shows up as a stride pattern, not a shape
  permutation).

A small Genesis-shaped smoke test at the bottom exercises the ``(n_dofs, _B)`` + ``layout=(1, 0)`` combination that
motivates this work, ensuring it round-trips through both numpy and torch.
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
    """Return a numpy array whose entry at canonical index ``ci`` is a distinct integer encoding ``ci`` (large enough
    to catch single-axis swaps). Matches the kernel-side fill below."""
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
    """Build a kernel that fills its argument with the same canonical values as :func:`_expected_canonical`, using
    explicit per-axis ``x[i, j, ...]`` indexing (one kernel per rank).

    The complementary single-Vector form ``x[I]`` (with ``I`` from ``qd.grouped(...)``) is exercised separately in
    :func:`test_grouped_vector_subscript_canonical_view_*` below; both forms must agree on canonical-view semantics
    for layout-tagged ndarrays.
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


# Representative layouts: identity, full reverse, an inner-axis swap, and a non-trivial cyclic shift. Keeping the set
# small so the cross-product (backend × layout × accessor) stays cheap.
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
    """Passing the *physical* shape (instead of canonical) on a layout-tagged ndarray must raise a clear shape-mismatch
    error, not silently scramble the data."""
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
    """``from_dlpack(t.to_dlpack())`` must report the canonical shape and canonical-indexed values, regardless of
    ``_qd_layout`` / SNode ``order=``. Both backends are required to behave identically."""
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
    """The DLPack export must keep the canonical *shape* but reflect the layout via non-contiguous *strides*; i.e. the
    resulting torch tensor is logically transposed but physically still the same buffer."""
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=(1, 0))
    fill = _make_fill_kernel(canonical, qd.Backend.NDARRAY)
    fill(a)

    t = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    assert tuple(t.shape) == canonical
    # On a non-identity layout the dlpack view is by construction not contiguous in the canonical sense — calling
    # ``.contiguous()`` is how the canonical layout would be materialised.
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


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_grad_to_numpy_canonical_view_rank3(backend, layout):
    """Rank-3 coverage for ``.grad`` under the canonical-view contract. Guards against rank-2 symmetries hiding a
    permutation bug in the grad-tag propagation from :func:`_with_layout`."""
    canonical = (2, 3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=backend, layout=layout, needs_grad=True)
    assert a.grad is not None
    assert tuple(a.grad.shape) == canonical

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i, j, k in qd.ndrange(*canonical):
                x[i, j, k] = float(i * 1000 + j * 10 + k)
                x.grad[i, j, k] = float(i * 10000 + j * 100 + k * 10)

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i, j, k in qd.ndrange(*canonical):
                x[i, j, k] = float(i * 1000 + j * 10 + k)
                x.grad[i, j, k] = float(i * 10000 + j * 100 + k * 10)

    fill(a)

    primal = a.to_numpy()
    grad = a.grad.to_numpy()
    assert primal.shape == grad.shape == canonical
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            for k in range(canonical[2]):
                assert primal[i, j, k] == float(i * 1000 + j * 10 + k)
                assert grad[i, j, k] == float(i * 10000 + j * 100 + k * 10)


# ----------------------------------------------------------------------------
# Identity-layout / no-layout paths must remain byte-identical to legacy (no extra allocation, no transpose, no
# behavioural drift).
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
# Single-Vector subscript ``x[I]`` (from ``qd.grouped(...)``) must obey the same canonical-view contract as
# multi-arg ``x[i, j, ...]`` on layout-tagged ndarrays.
#
# Regression for the AST rewrite in :func:`build_Subscript`: prior to the fix, ``x[I]`` skipped the
# canonical→physical permutation because the subscript arity (1) didn't match ``len(_qd_layout)`` (N). On a permuted
# layout this silently wrote at canonical indices into a differently-shaped physical buffer — out-of-bounds for any
# non-square canonical shape.
# ----------------------------------------------------------------------------


def _make_grouped_ndrange_fill_kernel(shape, backend):
    """Companion to :func:`_make_fill_kernel` that uses ``x[I]`` with ``I`` coming from
    ``qd.grouped(qd.ndrange(...))``."""
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
            for I in qd.grouped(qd.ndrange(d0, d1)):
                x[I] = I[0] * c0 + I[1] * c1

    elif len(shape) == 3:
        c0, c1, c2 = coeffs
        d0, d1, d2 = shape

        @qd.kernel
        def fill(x: annotation):
            for I in qd.grouped(qd.ndrange(d0, d1, d2)):
                x[I] = I[0] * c0 + I[1] * c1 + I[2] * c2

    else:
        raise NotImplementedError(f"_make_grouped_ndrange_fill_kernel: rank {len(shape)} not supported")

    return fill


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_grouped_vector_subscript_canonical_view_rank2(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    fill = _make_grouped_ndrange_fill_kernel(canonical, backend)
    fill(a)

    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, _expected_canonical(canonical))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_grouped_vector_subscript_canonical_view_rank3(backend, layout):
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    fill = _make_grouped_ndrange_fill_kernel(canonical, backend)
    fill(a)

    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, _expected_canonical(canonical))


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_grouped_struct_for_vector_subscript_rank2(backend, layout):
    """``for I in qd.grouped(x)`` is the other source of single-Vector subscripts in real kernels — the loop var ``I``
    has rank ``len(x.shape)`` and is used as ``x[I]``. Pin canonical-view behaviour here too."""
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def fill(x: annotation):
        for I in qd.grouped(x):
            x[I] = I[0] * 100 + I[1]

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    expected = np.zeros(canonical, dtype=np.int32)
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            expected[i, j] = i * 100 + j
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", list(itertools.permutations(range(3))))
@test_utils.test(arch=qd.cpu)
def test_grouped_struct_for_vector_subscript_rank3_all_permutations(backend, layout):
    """Extend the ``for I in qd.grouped(x)`` coverage to every rank-3 permutation: this is where
    ``build_struct_for``'s canonical-reorder-of-physical-indices fix can regress silently on symmetric rank-2 layouts
    (``(0,1)`` / ``(1,0)`` are self-inverse, so confusing ``layout`` with ``invperm(layout)`` still passes). Rank 3
    has permutations that are **not** self-inverse (e.g. ``(1, 2, 0)``), which catch that class of bug."""
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def fill(x: annotation):
        for I in qd.grouped(x):
            x[I] = I[0] * 10000 + I[1] * 100 + I[2]

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    expected = np.zeros(canonical, dtype=np.int32)
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            for k in range(canonical[2]):
                expected[i, j, k] = i * 10000 + j * 100 + k
    np.testing.assert_array_equal(arr, expected)


# ----------------------------------------------------------------------------
# Multi-target ``for i, j in x`` on a layout-tagged tensor: the runtime delivers *physical* loop indices, but
# ``build_struct_for`` rebinds the user names to canonical positions via the inverse permutation, so the user's ``i``
# is always canonical-axis-0 regardless of layout. Mirrors the canonical->physical translation in
# :func:`build_Subscript`. Verified on both backends so ``GS_ENABLE_NDARRAY``-style switching is transparent.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_multi_target_struct_for_on_layout_tagged_tensor(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill(x: qd.template()):
            for i, j in x:
                x[i, j] = i * 100 + j

    else:

        @qd.kernel
        def fill(x: qd.types.ndarray()):
            for i, j in x:
                x[i, j] = i * 100 + j

    fill(a)
    arr = a.to_numpy()
    assert arr.shape == canonical
    expected = np.zeros(canonical, dtype=np.int32)
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            expected[i, j] = i * 100 + j
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_grouped_vector_subscript_matches_per_axis(backend):
    """Cross-check: ``x[I]`` and ``x[i, j]`` must produce byte-identical results on the same layout-tagged tensor."""
    canonical = (3, 5)
    layout = (1, 0)
    via_grouped = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    via_per_axis = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    _make_grouped_ndrange_fill_kernel(canonical, backend)(via_grouped)
    _make_fill_kernel(canonical, backend)(via_per_axis)

    np.testing.assert_array_equal(via_grouped.to_numpy(), via_per_axis.to_numpy())


# ----------------------------------------------------------------------------
# Genesis-shaped smoke test: (n_dofs, _B) with layout=(1, 0).
#
# This is the shape pattern used by linesearch / Mgrad / aref-style tensors in the rigid solver. Migrating those
# into ``layout=(1, 0)`` is the immediate motivation for canonicalising every Python accessor; this test exercises
# the round-trip end-to-end on both backends.
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


# ----------------------------------------------------------------------------
# Pickle: as of stork-19 ``qd.tensor()`` returns a ``qd.Tensor`` wrapper, whose ``__reduce__`` round-trips via
# ``to_numpy()`` (the canonical view) and rebuilds a fresh wrapper through the factory. The factory is given the
# original ``layout=`` kwarg, so the restored tensor preserves the layout *and* its canonical-indexed values match
# the original. See 8.7 in ``perso_hugh/doc/quadrants-tensor.md`` for rationale.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_pickle_layout_tagged_ndarray_roundtrip_preserves_layout(layout):
    import pickle  # noqa: PLC0415 — local import keeps test self-contained

    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)
    fill = _make_fill_kernel(canonical, qd.Backend.NDARRAY)
    fill(a)

    restored = pickle.loads(pickle.dumps(a))

    assert tuple(restored.shape) == canonical
    # Identity layouts normalize to ``None`` at the wrapper layer.
    expected_layout = tuple(layout) if tuple(layout) != tuple(range(len(layout))) else None
    assert restored.layout == expected_layout
    np.testing.assert_array_equal(restored.to_numpy(), _expected_canonical(canonical))


# ----------------------------------------------------------------------------
# .fill(val) and copy_from(other): the two non-kernel Python entry points that mutate an ndarray via quadrants-internal
# kernels. fill(val) uses a C++ bulk-fill on x64/cuda so it's layout-agnostic by construction, but pin the observable
# behaviour so a future switch to the kernel path doesn't regress silently. copy_from expects the physical shapes to
# match (enforced in ``Ndarray.copy_from``), so we test the natural use case: two ndarrays with the same layout.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_fill_scalar_on_layout_tagged_ndarray(layout):
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)
    a.fill(7)
    arr = a.to_numpy()
    assert arr.shape == canonical
    np.testing.assert_array_equal(arr, np.full(canonical, 7, dtype=np.int32))


@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_copy_from_matching_layout(layout):
    canonical = (2, 3, 4)
    src = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)
    dst = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)

    _make_fill_kernel(canonical, qd.Backend.NDARRAY)(src)
    dst.copy_from(src)

    np.testing.assert_array_equal(dst.to_numpy(), _expected_canonical(canonical))
    np.testing.assert_array_equal(dst.to_numpy(), src.to_numpy())


# ----------------------------------------------------------------------------
# .grad.to_dlpack() on a layout-tagged ndarray must carry the canonical shape too (grad-tag propagation + dlpack
# layout path working together). Hardening: ensures the permuted-strides code path is exercised on the grad buffer,
# not only the primal.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_grad_to_dlpack_canonical_view_rank2(layout):
    torch = pytest.importorskip("torch")
    canonical = (3, 4)
    a = qd.tensor(qd.f32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout, needs_grad=True)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i, j in qd.ndrange(*canonical):
            x[i, j] = float(i * 10 + j)
            x.grad[i, j] = float(i * 100 + j * 10)

    fill(a)

    t = torch.utils.dlpack.from_dlpack(a.grad.to_dlpack())
    assert tuple(t.shape) == canonical
    expected = np.zeros(canonical, dtype=np.float32)
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            expected[i, j] = float(i * 100 + j * 10)
    np.testing.assert_array_equal(t.contiguous().cpu().numpy(), expected)


# ----------------------------------------------------------------------------
# Mixed kernel arguments: a layout-tagged ndarray and an untagged ndarray of the same canonical shape, written in the
# same kernel. This is the Genesis-migration pattern ("this one solver tensor is layout=..., the rest stay default")
# and must Just Work: canonical indices on either side produce canonical-view results through ``to_numpy()``.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_kernel_mixed_tagged_and_untagged_ndarray(layout):
    canonical = (3, 4)
    tagged = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY, layout=layout)
    untagged = qd.tensor(qd.i32, shape=canonical, backend=qd.Backend.NDARRAY)

    @qd.kernel
    def run(tagged_arr: qd.types.ndarray(), untagged_arr: qd.types.ndarray()):
        for i, j in qd.ndrange(*canonical):
            tagged_arr[i, j] = i * 10 + j
            untagged_arr[i, j] = tagged_arr[i, j] * 2 + 1

    run(tagged, untagged)

    expected = np.zeros(canonical, dtype=np.int32)
    for i in range(canonical[0]):
        for j in range(canonical[1]):
            expected[i, j] = i * 10 + j
    np.testing.assert_array_equal(tagged.to_numpy(), expected)
    np.testing.assert_array_equal(untagged.to_numpy(), expected * 2 + 1)
