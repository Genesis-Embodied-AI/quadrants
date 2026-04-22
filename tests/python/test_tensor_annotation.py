"""Tests for ``qd.Tensor``: value-dispatch kernel-argument annotation.

A kernel parameter annotated with ``qd.Tensor`` accepts either a Field
(handled like ``qd.template()``) or an Ndarray (handled like
``qd.types.ndarray()``). The same kernel object compiles distinct cache
entries for each branch.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

# ----------------------------------------------------------------------------
# Class identity
# ----------------------------------------------------------------------------


def test_tensor_is_a_class():
    """As of stork-19, ``qd.Tensor`` is the wrapper *class* (not a
    Template singleton). Used both as kernel-arg annotation and as a
    constructor: ``qd.Tensor(impl)`` produces a wrapper. The annotation
    branch in ``_func_base.py`` recognises ``annotation is qd.Tensor``
    explicitly.
    """
    assert isinstance(qd.Tensor, type)


@test_utils.test(arch=qd.cpu)
def test_tensor_factory_returns_wrapper():
    """Post stork-19, ``qd.tensor(...)`` returns ``qd.Tensor`` instances."""
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    b = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.FIELD)
    assert isinstance(a, qd.Tensor)
    assert isinstance(b, qd.Tensor)


@test_utils.test(arch=qd.cpu)
def test_tensor_double_wrap_rejected():
    """``qd.Tensor`` requires an Ndarray or Field impl; rejects wrapping
    another wrapper to avoid silent identity confusion."""
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    with pytest.raises(TypeError):
        qd.Tensor(a)


# ----------------------------------------------------------------------------
# Ndarray branch
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_ndarray():
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(x.shape[0]):
            x[i] = i * 10

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), [0, 10, 20, 30])


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_ndarray_with_layout():
    """Layout-tagged ndarrays must dispatch correctly through qd.Tensor."""
    M, N = 3, 4
    a = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY, layout=(1, 0))

    @qd.kernel
    def fill(x: qd.Tensor):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 100 + j

    fill(a)
    arr = a.to_numpy()
    # to_numpy() returns the canonical view on layout-tagged ndarrays.
    assert arr.shape == (M, N)
    assert arr[2, 3] == 203


# ----------------------------------------------------------------------------
# Field branch
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_accepts_field():
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.FIELD)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = i * 10

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), [0, 10, 20, 30])


# ----------------------------------------------------------------------------
# Cross-call dispatch: same kernel object, both backends, separate cache entries
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_dispatch_same_kernel_both_backends():
    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = i + 1

    a_field = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.FIELD)
    a_nd = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    fill(a_field)
    fill(a_nd)

    np.testing.assert_array_equal(a_field.to_numpy(), [1, 2, 3, 4])
    np.testing.assert_array_equal(a_nd.to_numpy(), [1, 2, 3, 4])

    # Two cache entries (one per backend branch).
    assert len(fill._primal.mapper.mapping) == 2


@test_utils.test(arch=qd.cpu)
def test_tensor_repeat_same_backend_reuses_cache():
    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i] = 7

    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    b = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    fill(a)
    fill(b)

    assert len(fill._primal.mapper.mapping) == 1


@test_utils.test(arch=qd.cpu)
def test_tensor_layouts_keep_separate_cache_entries():
    @qd.kernel
    def k(x: qd.Tensor):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    a_id = qd.tensor(qd.i32, shape=(2, 3), backend=qd.Backend.NDARRAY)
    a_swap = qd.tensor(qd.i32, shape=(2, 3), backend=qd.Backend.NDARRAY, layout=(1, 0))

    k(a_id)
    assert len(k._primal.mapper.mapping) == 1
    k(a_swap)
    assert len(k._primal.mapper.mapping) == 2


# ----------------------------------------------------------------------------
# Vector / matrix element types: qd.Tensor must dispatch the compound-element
# tensors built by qd.Vector.tensor / qd.Matrix.tensor on both backends.
# ----------------------------------------------------------------------------


BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_tensor_dispatch_vector_element(backend):
    """qd.Tensor must accept Vector-element tensors on both backends and
    let kernels write each component through canonical indexing."""
    qd.init(arch=qd.x64)
    a = qd.Vector.tensor(3, qd.f32, shape=(4,), backend=backend)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(4):
            x[i][0] = float(i)
            x[i][1] = float(i) + 0.5
            x[i][2] = float(i) + 0.25

    fill(a)
    arr = a.to_numpy()
    assert arr.shape[0] == 4
    np.testing.assert_allclose(arr[2, 0], 2.0)
    np.testing.assert_allclose(arr[2, 1], 2.5)
    np.testing.assert_allclose(arr[2, 2], 2.25)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_tensor_dispatch_matrix_element(backend):
    """qd.Tensor must accept Matrix-element tensors on both backends."""
    qd.init(arch=qd.x64)
    a = qd.Matrix.tensor(2, 3, qd.f32, shape=(2,), backend=backend)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i in range(2):
            for r in range(2):
                for c in range(3):
                    x[i][r, c] = float(i * 100 + r * 10 + c)

    fill(a)
    arr = a.to_numpy()
    assert arr.shape[0] == 2
    np.testing.assert_allclose(arr[1, 1, 2], 112.0)
    np.testing.assert_allclose(arr[0, 0, 0], 0.0)


# ----------------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------------


def test_tensor_is_in_qd_namespace():
    assert qd.Tensor is not None
    # also reachable via direct import
    from quadrants._tensor import Tensor as direct

    assert direct is qd.Tensor


# ----------------------------------------------------------------------------
# Module-scope kernel decl with qd.Tensor annotation.
#
# This is the *Genesis* pattern: every Genesis kernel is a module-level
# global, decorated with ``@qd.kernel`` at import time — long before
# ``qd.init()`` runs and long before any tensor is allocated. The tests
# above all decorate inside the test body (after ``@test_utils.test``
# has called ``qd.init()``), so they don't exercise this code path.
#
# Pinning here that:
# - The decorator is happy with ``qd.Tensor`` evaluated at module load
#   time (i.e. before any qd.init).
# - First call lazily compiles for whatever backend / layout the arg
#   actually has.
# - The four (backend × layout) combinations called against the *same*
#   module-level kernel object produce four distinct cache entries with
#   no fragmentation, and each writes the right canonical values.
# Runs on whatever archs the test runner targets (no ``arch=`` filter)
# so cpu and gpu codegen are both covered.
# ----------------------------------------------------------------------------


_MOD_M, _MOD_N = 3, 4
_MOD_LAYOUTS = [(0, 1), (1, 0)]
_MOD_LAYOUT_IDS = ["identity", "transposed"]


@qd.kernel
def _module_level_fill_2d(x: qd.Tensor):
    # Canonical indexing on both axes; the AST rewrite (ndarray) /
    # SNode order (field) handles non-identity layouts so this same
    # kernel body is correct under any permutation.
    for i, j in qd.ndrange(_MOD_M, _MOD_N):
        x[i, j] = i * 100 + j


def _expected_canonical():
    out = np.zeros((_MOD_M, _MOD_N), dtype=np.int32)
    for i in range(_MOD_M):
        for j in range(_MOD_N):
            out[i, j] = i * 100 + j
    return out


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _MOD_LAYOUTS, ids=_MOD_LAYOUT_IDS)
@test_utils.test()
def test_module_level_qd_tensor_kernel(backend, layout):
    a = qd.tensor(qd.i32, shape=(_MOD_M, _MOD_N), backend=backend, layout=layout)
    _module_level_fill_2d(a)
    np.testing.assert_array_equal(a.to_numpy(), _expected_canonical())


@test_utils.test()
def test_module_level_qd_tensor_kernel_all_combos_share_decl():
    """The same module-level kernel object, called against all four
    (backend × layout) combos, must produce four distinct cache entries
    (one per combo) and write correct canonical values for each.

    Mirrors the Genesis pattern after the stork-20 ``set_gravity``
    collapse: one decl, multiple backend/layout instances at runtime.
    """
    expected = _expected_canonical()
    n_before = len(_module_level_fill_2d._primal.mapper.mapping)

    tensors = []
    for backend in BACKENDS:
        for layout in _MOD_LAYOUTS:
            t = qd.tensor(qd.i32, shape=(_MOD_M, _MOD_N), backend=backend, layout=layout)
            _module_level_fill_2d(t)
            tensors.append((backend, layout, t))

    for backend, layout, t in tensors:
        np.testing.assert_array_equal(
            t.to_numpy(),
            expected,
            err_msg=f"backend={backend} layout={layout}",
        )

    n_after = len(_module_level_fill_2d._primal.mapper.mapping)
    # Exactly four new entries — one per (backend, layout) combo.
    # Catches both wrapper-unwrap-hook fragmentation (would push count
    # higher) and accidental cache collision between layouts (would
    # push it lower and silently reuse the wrong compiled code).
    assert n_after - n_before == 4, (
        f"expected 4 new cache entries (one per backend×layout combo), "
        f"got {n_after - n_before}"
    )
