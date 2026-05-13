"""Layout participates in the kernel cache key.

an earlier change plumbed the ``_qd_layout`` slot through ``TemplateMapper`` features, making it part of the kernel
cache key automatically. This file pins that contract down via direct ``Kernel.mapper.mapping`` inspection so a
future refactor that drops the slot would surface immediately.

Why this matters: if two different layouts shared a single compiled kernel, the AST subscript rewrite would happen
exactly once (for the layout chosen at first compile time), and subsequent calls with a different layout would either
silently mis-index or crash.
"""

import quadrants as qd
from quadrants._tensor import _with_layout

from tests import test_utils


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
    exactly one cache entry on their own kernel - no cross-pollution."""

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
