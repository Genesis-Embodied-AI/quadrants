"""Kernel-side tests for the opt-in ``qd._Tensor`` wrapper (stork-17 POC).

Pins two contracts that the upcoming full migration depends on:

1. **Cache stability under wrapping.** Calling the same kernel with a
   ``_Tensor(impl)`` wrapper and with the bare ``impl`` must produce
   exactly *one* compiled-kernel cache entry, not two. This is gotcha A
   from the design doc (§8.11): the unwrap hook must run *before*
   ``TemplateMapper.lookup`` computes ``id``-based hashes.

2. **Functional equivalence.** A kernel called with a wrapper must read
   and write the same memory as the same kernel called with the bare
   impl, on both backends. No data corruption, no shape confusion.
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_kernel_accepts_wrapper_and_writes_correctly(backend):
    """Wrapper-as-arg writes match bare-impl writes (functional equivalence)."""

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def fill(x: annotation):
        for i in range(4):
            x[i] = i + 1

    a = qd.tensor(qd.i32, shape=(4,), backend=backend)
    fill(qd._Tensor(a))

    expected = [1, 2, 3, 4]
    assert list(a.to_numpy()) == expected


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_kernel_cache_no_fragmentation_under_wrapping(backend):
    """Calling with bare impl and with ``_Tensor(impl)`` must hit the same
    cache entry: gotcha A regression test.
    """

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def noop(x: annotation):
        for i in range(2):
            x[i] = x[i] + 0

    a = qd.tensor(qd.i32, shape=(2,), backend=backend)

    noop(a)
    n_after_bare = len(noop.materialized_kernels)

    noop(qd._Tensor(a))
    n_after_wrapped = len(noop.materialized_kernels)

    assert n_after_wrapped == n_after_bare, (
        f"cache fragmented: {n_after_bare} entries after bare-impl call, "
        f"{n_after_wrapped} after wrapper call. The unwrap hook in "
        f"@qd.kernel.__call__ must run before TemplateMapper.lookup so "
        f"id(arg) matches across (impl) and (Tensor(impl)) call shapes."
    )

    # And a fresh wrapper around the same impl must also hit the same entry.
    noop(qd._Tensor(a))
    assert len(noop.materialized_kernels) == n_after_bare


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_kernel_wrapper_round_trips_through_to_numpy(backend):
    """End-to-end: write via wrapper, read via wrapper.to_numpy() (which
    forwards to impl.to_numpy()), values match.
    """

    annotation = qd.template() if backend is qd.Backend.FIELD else qd.types.ndarray()

    @qd.kernel
    def fill(x: annotation):
        for i, j in qd.ndrange(3, 4):
            x[i, j] = i * 10 + j

    a = qd.tensor(qd.i32, shape=(3, 4), backend=backend)
    t = qd._Tensor(a)
    fill(t)

    arr = t._impl.to_numpy()
    expected = [[i * 10 + j for j in range(4)] for i in range(3)]
    for i in range(3):
        assert list(arr[i]) == expected[i]
