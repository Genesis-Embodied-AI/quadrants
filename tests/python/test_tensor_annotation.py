"""Tests for ``qd.Tensor``: value-dispatch kernel-argument annotation.

A kernel parameter annotated with ``qd.Tensor`` accepts either a Field
(handled like ``qd.template()``) or an Ndarray (handled like
``qd.types.ndarray()``). The same kernel object compiles distinct cache
entries for each branch.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants._tensor import _TensorAnnotation

from tests import test_utils

# ----------------------------------------------------------------------------
# Singleton + identity
# ----------------------------------------------------------------------------


def test_tensor_is_singleton_instance():
    assert isinstance(qd.Tensor, _TensorAnnotation)


def test_tensor_is_a_template_subclass():
    """qd.Tensor must inherit from Template so the upfront slot detection
    in _func_base.py registers it as a template slot."""
    from quadrants.types.annotations import Template

    assert isinstance(qd.Tensor, Template)


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
    assert arr.shape == (N, M)
    # canonical (i=2, j=3) -> physical (3, 2)
    assert arr[3, 2] == 203


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
