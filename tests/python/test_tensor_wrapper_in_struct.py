"""Wrapper-in-struct unwrap behaviour (`hp/tensor-stork-21`).

Pins the contract that a ``qd.Tensor`` wrapper stored as a *field* of a
struct (plain frozen dataclass or ``@qd.data_oriented``) and passed to a
kernel via ``qd.template()`` is unwrapped to its bare impl before the
kernel-side type resolution / template-mapper / fastcache validator
run.

Without this contract the Genesis Tier-1 migration crashes with
``QuadrantsRuntimeTypeError: Argument of type Tensor cannot be
converted into required type NdarrayType(...)`` and (when no crash)
silently falls off the fastcache fast-path with
``[FASTCACHE][PARAM_INVALID] Parameter with path (...) and type
Tensor not allowed by fast cache``.

See ``perso_hugh/doc/quadrants-tensor.md`` §8.14 and
``perso_hugh/doc/genesis_tensor_migration.md`` Phase-1 findings.

The stork-19 unwrap hook in ``Kernel.__call__`` only covers top-level
positional / keyword args; the in-struct cases are covered here.
"""

import dataclasses
import sys

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]
LAYOUTS = [(0, 1), (1, 0)]
LAYOUT_IDS = ["identity", "transposed"]

_M, _N = 3, 4


def _expected_canonical():
    out = np.zeros((_M, _N), dtype=np.int32)
    for i in range(_M):
        for j in range(_N):
            out[i, j] = i * 100 + j
    return out


# ----------------------------------------------------------------------------
# Repro 1: plain frozen dataclass struct holding a qd.Tensor wrapper field.
#
# Mirrors the Genesis ndarray-backend struct shape (``DATA_ORIENTED =
# partial(dataclasses.dataclass, frozen=True)`` in ``array_class.py``).
# Without the in-struct unwrap fix the kernel call raises
# ``QuadrantsRuntimeTypeError`` from ``_recursive_set_args``.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@test_utils.test()
def test_dataclass_struct_with_wrapper_field_kernel_runs(backend, layout):
    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 2]

    @qd.kernel
    def fill(state: State):
        for i, j in qd.ndrange(_M, _N):
            state.a[i, j] = i * 100 + j

    a = qd.tensor(qd.i32, shape=(_M, _N), backend=backend, layout=layout)
    assert isinstance(a, qd.Tensor)

    state = State(a=a)
    fill(state)

    np.testing.assert_array_equal(a.to_numpy(), _expected_canonical())


# ----------------------------------------------------------------------------
# Repro 2: @qd.data_oriented struct holding a qd.Tensor wrapper field.
#
# Mirrors the Genesis field-backend struct shape (``DATA_ORIENTED =
# qd.data_oriented`` in ``array_class.py``). The kernel takes the
# struct as ``qd.template()``; field access happens at AST-rewrite time.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@test_utils.test()
def test_data_oriented_struct_with_wrapper_field_kernel_runs(backend, layout):
    @qd.data_oriented
    class State:
        def __init__(self, a):
            self.a = a

    @qd.kernel
    def fill(state: qd.template()):
        for i, j in qd.ndrange(_M, _N):
            state.a[i, j] = i * 100 + j

    a = qd.tensor(qd.i32, shape=(_M, _N), backend=backend, layout=layout)
    state = State(a=a)
    fill(state)

    np.testing.assert_array_equal(a.to_numpy(), _expected_canonical())


# ----------------------------------------------------------------------------
# Mixed bare-impl + wrapper struct fields. Confirms the in-struct
# unwrap is applied selectively per field — bare impls pass straight
# through (no double-unwrap), wrappers are see-through.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test()
def test_mixed_bare_and_wrapper_struct_fields(backend):
    @dataclasses.dataclass(frozen=True)
    class State:
        wrapped: qd.types.NDArray[qd.i32, 1]
        bare: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State):
        for i in range(4):
            state.wrapped[i] = i + 10
            state.bare[i] = i + 20

    wrapped = qd.tensor(qd.i32, shape=(4,), backend=backend)
    if backend is qd.Backend.NDARRAY:
        bare = qd.ndarray(qd.i32, (4,))
    else:
        bare = qd.field(qd.i32, (4,))

    state = State(wrapped=wrapped, bare=bare)
    fill(state)

    np.testing.assert_array_equal(
        wrapped.to_numpy(), np.array([10, 11, 12, 13], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        bare.to_numpy(), np.array([20, 21, 22, 23], dtype=np.int32)
    )


# ----------------------------------------------------------------------------
# Fastcache: no PARAM_INVALID warning, fast path stays hot.
#
# Pins the soft-failure side of the §8.14 bug: even when the kernel
# eventually runs (e.g. via the slow path), the wrapper-as-struct-field
# would have tripped the fastcache's per-parameter validator
# (``args_hasher.stringify_obj_type``) and emitted
# ``[FASTCACHE][PARAM_INVALID]``, killing the recompile-avoidance for
# the whole call. Following the harness used by
# ``test_src_ll_cache_arg_warnings``: capfd + ``@qd.pure`` makes the
# warning observable end-to-end.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="capfd flaky for stderr on Windows"
)
def test_struct_wrapper_field_does_not_invalidate_fastcache(backend, capfd):
    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.pure
    @qd.kernel
    def k(state: State):
        for i in range(2):
            state.a[i] = state.a[i] + 1

    a = qd.tensor(qd.i32, shape=(2,), backend=backend)
    state = State(a=a)

    # Drain any prior warnings unrelated to this kernel.
    capfd.readouterr()

    k(state)
    _out, err = capfd.readouterr()

    assert "[FASTCACHE][PARAM_INVALID]" not in err, (
        "Wrapper-in-struct must not be rejected by the fastcache validator. "
        "If you see this, check args_hasher.stringify_obj_type unwraps "
        "qd.Tensor wrappers before its type whitelist."
    )
    # ``[FASTCACHE][INVALID_FUNC]`` rides on PARAM_INVALID for @qd.pure
    # kernels; assert it independently for a stronger signal.
    assert "[FASTCACHE][INVALID_FUNC]" not in err


# ----------------------------------------------------------------------------
# JIT cache: wrapper-in-struct shares a cache entry with bare-in-struct.
#
# A wrapper passed as a struct field must dispatch the same compiled
# kernel as the bare impl in the same struct slot — otherwise we'd
# fragment the JIT cache per call, which is the kernel-arg version of
# the same regression covered for top-level args by
# ``test_kernel_cache_no_fragmentation_under_wrapping``.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test()
def test_struct_wrapper_field_shares_cache_with_bare(backend):
    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def k(state: State):
        for i in range(2):
            state.a[i] = state.a[i] + 0

    if backend is qd.Backend.NDARRAY:
        bare = qd.ndarray(qd.i32, (2,))
    else:
        bare = qd.field(qd.i32, (2,))
    wrapped = qd.Tensor(bare)

    cache = k._primal.materialized_kernels

    k(State(a=bare))
    n_after_bare = len(cache)
    k(State(a=wrapped))
    n_after_wrapped = len(cache)

    assert n_after_wrapped == n_after_bare, (
        f"struct-field cache fragmented: {n_after_bare} entries with bare "
        f"impl, {n_after_wrapped} with wrapper. The in-struct unwrap in "
        f"_template_mapper_hotpath._extract_arg must run before id-based "
        f"keying."
    )
