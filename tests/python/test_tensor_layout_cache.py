"""Layout participates in the kernel cache key.

an earlier change plumbed the ``_qd_layout`` slot through ``TemplateMapper`` features, making it part of the kernel
cache key automatically. This file pins that contract down via direct ``Kernel.mapper.mapping`` inspection so a
future refactor that drops the slot would surface immediately.

Why this matters: if two different layouts shared a single compiled kernel, the AST subscript rewrite would happen
exactly once (for the layout chosen at first compile time), and subsequent calls with a different layout would either
silently mis-index or crash.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import quadrants as qd
from quadrants._tensor import _with_layout
from quadrants.lang.matrix import MatrixType

from tests.helpers import test_utils
from tests.helpers.runtime import qd_init_same_arch


def _allocate(M, N, layout=None, dtype=qd.i32):
    """Allocate at the physical shape for ``layout`` and tag if non-None."""
    if layout is None:
        a = qd.tensor(dtype, shape=(M, N), backend=qd.Backend.NDARRAY)
        return a
    physical_shape = tuple((M, N)[axis] for axis in layout)
    a = qd.tensor(dtype, shape=physical_shape, backend=qd.Backend.NDARRAY)
    _with_layout(a, layout)
    return a


# ----------------------------------------------------------------------------
# Different layouts compile to different cache entries
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_two_layouts_produce_two_cache_entries():
    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    M, N = 2, 3
    a_id = _allocate(M, N, layout=(0, 1))
    a_swap = _allocate(M, N, layout=(1, 0))

    k(a_id)
    assert len(k._primal.mapper.mapping) == 1

    k(a_swap)
    assert len(k._primal.mapper.mapping) == 2


@test_utils.test(arch=qd.cpu)
def test_untagged_vs_identity_tagged_are_different_cache_entries():
    """layout=None and layout=(0, 1) are distinct cache entries.

    They produce byte-identical IR (the AST hook short-circuits on identity), but they're keyed differently because
    ``None != (0, 1)``. Documenting this so a future "normalise identity to None" refactor is a deliberate decision
    rather than an accident.
    """

    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    untagged = _allocate(2, 3)
    identity = _allocate(2, 3, layout=(0, 1))

    k(untagged)
    k(identity)
    keys = list(k._primal.mapper.mapping.keys())
    assert len(keys) == 2
    # Inner tuple shape: (element_type, ndim, needs_grad, boundary, layout)
    layouts = sorted(
        (key[0][-1] for key in keys),
        key=lambda x: (x is not None, x),
    )
    assert layouts == [None, (0, 1)]


# ----------------------------------------------------------------------------
# Re-using the same layout reuses the cache entry
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_repeat_same_layout_reuses_cache():
    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    a1 = _allocate(2, 3, layout=(1, 0))
    a2 = _allocate(2, 3, layout=(1, 0))

    k(a1)
    assert len(k._primal.mapper.mapping) == 1
    k(a2)
    assert len(k._primal.mapper.mapping) == 1


# ----------------------------------------------------------------------------
# Switching back and forth: each layout produces exactly one cache entry
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_switching_layouts_does_not_pollute_cache():
    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    a_id = _allocate(2, 3, layout=(0, 1))
    a_swap = _allocate(2, 3, layout=(1, 0))

    for _ in range(3):
        k(a_id)
        k(a_swap)
    assert len(k._primal.mapper.mapping) == 2


# ----------------------------------------------------------------------------
# Layout slot is the trailing element of the per-arg feature tuple
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_is_trailing_feature_slot():
    @qd.kernel
    def k(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i * 10 + j

    a = _allocate(2, 3, layout=(1, 0))
    k(a)
    keys = list(k._primal.mapper.mapping.keys())
    assert len(keys) == 1
    inner = keys[0][0]
    # (element_type_or_id, ndim, needs_grad, boundary, layout)
    assert len(inner) == 5
    assert inner[1] == 2  # ndim
    assert inner[-1] == (1, 0)


# ----------------------------------------------------------------------------
# Two different ndim with same layout-shape don't collide
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_ndim_distinct_in_cache_key():
    """Distinct kernel objects keep their mappers separate; verify a rank-1 and rank-2 (layout-tagged) ndarray each get
    exactly one cache entry on their own kernel — no cross-pollution."""

    @qd.kernel
    def k1d(x: qd.types.ndarray()):
        for i in range(x.shape[0]):
            x[i] = i

    @qd.kernel
    def k2d(x: qd.types.ndarray()):
        for i, j in qd.ndrange(2, 3):
            x[i, j] = i + j

    a1 = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    a2 = _allocate(2, 3, layout=(1, 0))

    k1d(a1)
    k2d(a2)

    assert len(k1d._primal.mapper.mapping) == 1
    assert len(k2d._primal.mapper.mapping) == 1


@test_utils.test(arch=qd.cpu)
def test_fastcache_layout_distinct_keys():
    """Two ndarrays of the same dtype/rank but different layouts must NOT share a fastcache entry, otherwise the second
    call reuses IR compiled for the first layout and silently mis-indexes.
    """
    from quadrants.lang._fast_caching.args_hasher import stringify_obj_type

    a_plain = qd.ndarray(qd.f32, shape=(4, 3))
    a_layout = qd.ndarray(qd.f32, shape=(3, 4))
    a_layout._qd_layout = (1, 0)

    key_plain = stringify_obj_type(False, ("x",), a_plain, None)
    key_layout = stringify_obj_type(False, ("x",), a_layout, None)

    assert key_plain is not None
    assert key_layout is not None
    assert key_plain != key_layout, (
        f"Plain and layout-tagged ndarrays must produce different cache keys, " f"got {key_plain!r} for both"
    )


@test_utils.test(arch=qd.cpu)
def test_fastcache_vector_ndarray_layout_distinct():
    from quadrants.lang._fast_caching.args_hasher import stringify_obj_type

    v_plain = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    v_layout = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    v_layout._qd_layout = (0,)

    key_plain = stringify_obj_type(False, ("x",), v_plain, None)
    key_layout = stringify_obj_type(False, ("x",), v_layout, None)

    assert key_plain is not None
    assert key_layout is not None
    # Identity layout tagged is technically distinct from None
    assert key_plain != key_layout


@test_utils.test(arch=qd.cpu)
def test_fastcache_matrix_ndarray_layout_distinct():
    from quadrants.lang._fast_caching.args_hasher import stringify_obj_type

    m_plain = qd.Matrix.ndarray(2, 2, qd.f32, shape=(4,))
    m_layout = qd.Matrix.ndarray(2, 2, qd.f32, shape=(4,))
    m_layout._qd_layout = (0,)

    key_plain = stringify_obj_type(False, ("x",), m_plain, None)
    key_layout = stringify_obj_type(False, ("x",), m_layout, None)

    assert key_plain is not None
    assert key_layout is not None
    assert key_plain != key_layout


@pytest.mark.parametrize(
    "backend,dtype",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), id="field-vector"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), id="field-matrix"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), id="ndarray-vector"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), id="ndarray-matrix"),
    ],
)
@pytest.mark.skipif(
    os.environ.get("QD_KERNEL_COVERAGE") == "1",
    reason="Coverage probes change the kernel AST, preventing offline cache hits after reinit",
)
@test_utils.test()
def test_compound_layout_offline_cache(backend: qd.Backend, dtype: MatrixType, tmp_path: Path) -> None:
    """Reload each layout from the offline cache and check every scalar value, with fastcache disabled."""

    @qd.kernel
    def k1(x: qd.Tensor) -> None:
        test_utils.fill_tensor_with_index_codes(x)

    shape = (2, 4, 5)
    layouts = (None, (1, 0, 2), (2, 0, 1))
    expected = test_utils.new_numpy_tensor_containing_index_codes(shape, dtype.get_shape())

    for expect_hit in (False, True):
        # Reinitialization clears compiled kernels from memory and flushes the preceding runtime's disk cache.
        qd_init_same_arch(offline_cache=True, src_ll_cache=False, offline_cache_file_path=str(tmp_path))
        assert not k1._primal.compiled_kernel_data_by_key
        assert not k1._primal.materialized_kernels

        for layout in layouts:
            x = qd.tensor(dtype, shape, backend=backend, layout=layout)
            # The observation is cumulative, so reset it to check this particular layout's launch.
            k1._primal.fe_ll_cache_observations.cache_hit = False
            k1(x)
            assert k1._primal.fe_ll_cache_observations.cache_hit == expect_hit, layout
            assert not k1._primal.src_ll_cache_observations.cache_loaded
            np.testing.assert_array_equal(x.to_numpy(), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(qd.types.vector(3, qd.f32), id="ndarray-vector"),
        pytest.param(qd.types.matrix(2, 3, qd.f32), id="ndarray-matrix"),
    ],
)
@test_utils.test()
def test_compound_layout_fastcache(dtype: MatrixType, tmp_path: Path) -> None:
    """Reload each ndarray layout through fastcache and check that the restored kernel indexes it correctly."""

    @qd.kernel(fastcache=True)
    def write(x: qd.Tensor) -> None:
        test_utils.fill_tensor_with_index_codes(x)

    shape = (2, 4, 5)
    layouts = (None, (1, 0, 2), (2, 0, 1))

    for expect_hit in (False, True):
        # Fastcache loads its compiled artifact from the disk cache; a fastcache hit must bypass frontend compilation.
        qd_init_same_arch(offline_cache=True, src_ll_cache=True, offline_cache_file_path=str(tmp_path))
        assert not write._primal.compiled_kernel_data_by_key
        assert not write._primal.materialized_kernels

        for layout in layouts:
            x = qd.tensor(dtype, shape, backend=qd.Backend.NDARRAY, layout=layout)
            expected = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
            write._primal.src_ll_cache_observations.cache_loaded = False
            write._primal.fe_ll_cache_observations.cache_hit = False
            write(x)
            assert write._primal.src_ll_cache_observations.cache_loaded == expect_hit, layout
            assert not write._primal.fe_ll_cache_observations.cache_hit
            np.testing.assert_array_equal(x.to_numpy(), expected)
