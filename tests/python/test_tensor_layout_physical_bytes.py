"""Pin the **physical-byte equivalence** contract for ``layout=``.

Motivation:

The ``layout=`` kwarg on ``qd.tensor`` is documented as a *physical memory permutation*: ``shape`` is the canonical
(user-indexed) shape, and ``layout`` describes the order in which canonical axes are materialised in memory, outermost
first. The promise to downstream users (in particular Genesis kernels) is that:

    qd.tensor(dtype, shape=S_natural, backend=X)

and

    qd.tensor(dtype, shape=permute(S_natural, layout), backend=X,
              layout=invert(layout))

allocate **byte-identical** physical buffers. Equivalently: indexing the second tensor with a permuted canonical key
writes to the *same* raw memory address as indexing the first tensor with the natural key.

That promise drives the canonical-shape rewrites Genesis is doing for its constraint-state tensors, where shape
``(n_constraints, B)`` is being rotated to ``(B, n_constraints)`` with ``layout=(1, 0)`` so that batch-major access
patterns at the call sites read more naturally without changing the underlying memory order. If the rewrite quietly
produced a different physical buffer, every kernel access would land on a different cache line and the rewrite would
silently regress performance.

Existing layout tests in ``test_tensor_layout.py`` / ``test_tensor_factory_layout_ndarray.py`` /
``test_tensor_layout_interop.py`` cover the canonical-view contract (``shape``, ``to_numpy``, ``to_torch``,
``to_dlpack``, ``.grad``), and ``test_to_dlpack_layout_shows_up_as_strides_not_shape`` confirms the DLPack view uses
non-contiguous strides on layout-tagged tensors. None of them assert that the *raw bytes* of two equivalent
(canonical, layout) configurations are bit-identical, which is the property this file pins.

Method: write a single sentinel via host setitem at canonical-equivalent positions on both tensors, dump the raw 1-D
byte view via DLPack (which, unlike ``to_torch()``, returns a faithful non-copy view of the underlying allocation),
and assert the raw arrays compare equal.
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _raw_view(tensor):
    """Return a 1-D numpy view of the underlying physical allocation.

    Goes through DLPack to bypass the canonical-view permutation that ``to_torch()`` materialises (see
    ``test_to_dlpack_layout_shows_up_as_strides_not_shape``); the untyped storage of the resulting torch tensor *is*
    the underlying physical buffer.
    """
    torch = pytest.importorskip("torch")
    t = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
    flat = torch.empty(t.numel(), dtype=t.dtype)
    flat.set_(t.untyped_storage(), 0, (t.numel(),), (1,))
    return flat.cpu().numpy()


def _permute(shape, layout):
    return tuple(shape[axis] for axis in layout)


def _invert(layout):
    inv = [0] * len(layout)
    for src, dst in enumerate(layout):
        inv[dst] = src
    return tuple(inv)


# ---------------------------------------------------------------------------
# Rank-2 transpose: the case Genesis is actually using.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_layout_rank2_transpose_is_byte_identical_to_default(backend):
    """``qd.tensor(shape=(N, B))`` and ``qd.tensor(shape=(B, N), layout=(1, 0))`` must have byte-identical physical
    allocations.

    Sentinel write at canonical-equivalent positions:
      - parent      A[c=2, b=4] = 777
      - layout-test B[b=4, c=2] = 777
    """
    pytest.importorskip("torch")
    qd.init(arch=qd.x64)
    N, B = 7, 11

    parent = qd.tensor(qd.f32, shape=(N, B), backend=backend)
    rotated = qd.tensor(qd.f32, shape=(B, N), backend=backend, layout=(1, 0))

    parent.fill(0.0)
    rotated.fill(0.0)
    parent[2, 4] = 777.0
    rotated[4, 2] = 777.0

    parent_raw = _raw_view(parent)
    rotated_raw = _raw_view(rotated)

    assert parent_raw.shape == rotated_raw.shape == (N * B,)
    np.testing.assert_array_equal(parent_raw, rotated_raw)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_layout_rank2_transpose_full_pattern_byte_identical(backend):
    """Same as the sentinel test, but writes a unique value at every canonical position so the assertion catches any
    single-element permutation bug, not just one corner."""
    pytest.importorskip("torch")
    qd.init(arch=qd.x64)
    N, B = 5, 7

    parent = qd.tensor(qd.i32, shape=(N, B), backend=backend)
    rotated = qd.tensor(qd.i32, shape=(B, N), backend=backend, layout=(1, 0))

    for c in range(N):
        for b in range(B):
            v = c * 1000 + b
            parent[c, b] = v
            rotated[b, c] = v

    np.testing.assert_array_equal(_raw_view(parent), _raw_view(rotated))


# ---------------------------------------------------------------------------
# Rank-3: parameterise over every non-identity permutation. The contract generalises: for any ``layout`` permutation,
# allocating with ``shape=permute(canonical, layout)`` and ``layout=invert(layout)`` must match the natural-canonical
# default.
# ---------------------------------------------------------------------------


_RANK3_NON_IDENTITY = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _RANK3_NON_IDENTITY)
def test_layout_rank3_arbitrary_permutation_byte_identical(backend, layout):
    pytest.importorskip("torch")
    qd.init(arch=qd.x64)
    canonical = (3, 4, 5)

    parent = qd.tensor(qd.i32, shape=canonical, backend=backend)
    rotated_shape = _permute(canonical, layout)
    rotated_layout = _invert(layout)
    rotated = qd.tensor(qd.i32, shape=rotated_shape, backend=backend, layout=rotated_layout)

    for i, j, k in itertools.product(*(range(d) for d in canonical)):
        v = i * 100 + j * 10 + k
        parent[i, j, k] = v
        # Convert canonical (i, j, k) into rotated tensor's canonical key by applying the layout permutation in reverse.
        rotated_key = tuple((i, j, k)[axis] for axis in layout)
        rotated[rotated_key] = v

    np.testing.assert_array_equal(_raw_view(parent), _raw_view(rotated))


# ---------------------------------------------------------------------------
# Kernel-write equivalence: the physical-byte-identical guarantee must hold for kernel-driven writes too, not just
# host setitem (since Genesis fills state from kernels).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_layout_rank2_kernel_writes_byte_identical(backend):
    """Same canonical fill kernel run on parent and rotated tensors must produce byte-identical physical buffers."""
    pytest.importorskip("torch")
    qd.init(arch=qd.x64)
    N, B = 5, 7

    parent = qd.tensor(qd.i32, shape=(N, B), backend=backend)
    rotated = qd.tensor(qd.i32, shape=(B, N), backend=backend, layout=(1, 0))

    if backend is qd.Backend.FIELD:

        @qd.kernel
        def fill_parent(x: qd.template()):
            for c, b in qd.ndrange(N, B):
                x[c, b] = c * 1000 + b

        @qd.kernel
        def fill_rotated(x: qd.template()):
            for b, c in qd.ndrange(B, N):
                x[b, c] = c * 1000 + b

    else:

        @qd.kernel
        def fill_parent(x: qd.types.ndarray()):
            for c, b in qd.ndrange(N, B):
                x[c, b] = c * 1000 + b

        @qd.kernel
        def fill_rotated(x: qd.types.ndarray()):
            for b, c in qd.ndrange(B, N):
                x[b, c] = c * 1000 + b

    fill_parent(parent)
    fill_rotated(rotated)

    np.testing.assert_array_equal(_raw_view(parent), _raw_view(rotated))


# ---------------------------------------------------------------------------
# Sanity: identity layout must NOT change the physical bytes vs. no-layout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_layout_identity_is_byte_identical_to_no_layout(backend):
    pytest.importorskip("torch")
    qd.init(arch=qd.x64)
    N, B = 5, 7

    no_layout = qd.tensor(qd.f32, shape=(N, B), backend=backend)
    identity_layout = qd.tensor(qd.f32, shape=(N, B), backend=backend, layout=(0, 1))

    for c in range(N):
        for b in range(B):
            v = float(c * 1000 + b)
            no_layout[c, b] = v
            identity_layout[c, b] = v

    np.testing.assert_array_equal(_raw_view(no_layout), _raw_view(identity_layout))


# ---------------------------------------------------------------------------
# Negative control on NDARRAY (whose physical allocation is not subject to the field-side global SNode root-tree
# fusion that can collapse distinct-shape ``qd.field`` allocations into surprising layouts on the CPU x64 backend).
# Without ``layout=(1, 0)``, simply swapping the canonical shape must produce a *different* physical buffer.
# ---------------------------------------------------------------------------


def test_layout_negative_control_ndarray_swapped_shape_no_layout_differs():
    pytest.importorskip("torch")
    qd.init(arch=qd.x64)
    N, B = 5, 7

    parent = qd.tensor(qd.i32, shape=(N, B), backend=qd.Backend.NDARRAY)
    swapped_no_layout = qd.tensor(qd.i32, shape=(B, N), backend=qd.Backend.NDARRAY)

    for c in range(N):
        for b in range(B):
            v = c * 1000 + b
            parent[c, b] = v
            swapped_no_layout[b, c] = v  # canonical-equivalent key

    parent_raw = _raw_view(parent)
    swapped_raw = _raw_view(swapped_no_layout)
    assert parent_raw.shape == swapped_raw.shape == (N * B,)
    # The two buffers must NOT match: without ``layout=(1, 0)``, shape ``(B, N)`` row-major lays ``b`` out as the slow
    # axis, not ``c``.
    assert not np.array_equal(parent_raw, swapped_raw), (
        "negative control failed: shape-swap-without-layout produced a "
        "byte-identical buffer; the byte-identity tests above could be "
        "passing by accident"
    )
