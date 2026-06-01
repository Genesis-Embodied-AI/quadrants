"""Kernel-side tests for the ``qd.Tensor`` wrapper.

Pins two contracts:

1. **Cache stability under wrapping.** Calling the same kernel with a ``Tensor(impl)`` wrapper and with the bare
   ``impl`` must produce exactly *one* compiled-kernel cache entry, not two. This is gotcha A from the design doc
   (§8.11): the unwrap hook must run *before* ``TemplateMapper.lookup`` computes ``id``-based hashes.

2. **Functional equivalence.** A kernel called with a wrapper must read and write the same memory as the same kernel
   called with the bare impl, on both backends. No data corruption, no shape confusion.

Stork-19 flipped ``qd.tensor()`` to return wrappers, so to construct a *bare* impl for these tests we drop down to
``qd.field`` / ``qd.ndarray`` directly. Wrapping then goes through ``qd.Tensor(impl)``.
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


def _alloc_bare(backend, dtype, shape):
    if backend is qd.Backend.FIELD:
        return qd.field(dtype, shape)
    return qd.ndarray(dtype, shape)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_kernel_accepts_wrapper_and_writes_correctly(backend):
    """Wrapper-as-arg writes match bare-impl writes (functional equivalence)."""

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def fill(x: annotation):
        for i in range(4):
            x[i] = i + 1

    a = _alloc_bare(backend, qd.i32, (4,))
    fill(qd.Tensor(a))

    expected = [1, 2, 3, 4]
    assert list(a.to_numpy()) == expected


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_kernel_cache_no_fragmentation_under_wrapping(backend):
    """Calling with bare impl and with ``Tensor(impl)`` must hit the same cache entry: gotcha A regression test."""

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def noop(x: annotation):
        for i in range(2):
            x[i] = x[i] + 0

    a = _alloc_bare(backend, qd.i32, (2,))
    # ``noop`` is a ``QuadrantsCallable``; the actual ``Kernel`` (which owns the JIT cache) lives at ``._primal``.
    cache = noop._primal.materialized_kernels

    noop(a)
    n_after_bare = len(cache)

    noop(qd.Tensor(a))
    n_after_wrapped = len(cache)

    assert n_after_wrapped == n_after_bare, (
        f"cache fragmented: {n_after_bare} entries after bare-impl call, "
        f"{n_after_wrapped} after wrapper call. The unwrap hook in "
        f"@qd.kernel.__call__ must run before TemplateMapper.lookup so "
        f"id(arg) matches across (impl) and (Tensor(impl)) call shapes."
    )

    # And a fresh wrapper around the same impl must also hit the same entry.
    noop(qd.Tensor(a))
    assert len(cache) == n_after_bare


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_kernel_wrapper_round_trips_through_to_numpy(backend):
    """End-to-end: write via wrapper, read via wrapper.to_numpy() (which forwards to impl.to_numpy()), values match."""

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def fill(x: annotation):
        for i, j in qd.ndrange(3, 4):
            x[i, j] = i * 10 + j

    a = _alloc_bare(backend, qd.i32, (3, 4))
    t = qd.Tensor(a)
    fill(t)

    arr = t._impl.to_numpy()
    expected = [[i * 10 + j for j in range(4)] for i in range(3)]
    for i in range(3):
        assert list(arr[i]) == expected[i]
