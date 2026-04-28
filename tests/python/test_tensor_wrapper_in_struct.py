"""Wrapper-in-struct unwrap behaviour (`hp/tensor-stork-20`).

Pins the contract that a ``qd.Tensor`` wrapper stored as a *field* of a struct passed to a kernel is unwrapped to its
bare impl before the kernel-side type resolution / template-mapper / fastcache validator / AST-build-time attribute
access run.

Without this contract the Genesis Tier-1 migration crashes with ``QuadrantsRuntimeTypeError: Argument of type Tensor
cannot be converted into required type NdarrayType(...)`` and (when no crash) silently falls off the fastcache
fast-path with ``[FASTCACHE][PARAM_INVALID] Parameter with path (...) and type Tensor not allowed by fast cache``.

Genesis uses two different struct shapes depending on the global ``gs.use_ndarray`` flag (see
``genesis/utils/array_class.py``):

- ``gs.use_ndarray=True``:  ``@dataclass(frozen=True)`` struct, kernel takes the struct as a single arg and the
  per-field annotation ``qd.types.NDArray[...]`` makes Quadrants recurse and bind each field as a separate kernel arg.
  Tests that mirror this shape are parametrised over the **ndarray** backend only.
- ``gs.use_ndarray=False``: ``@qd.data_oriented`` struct, kernel takes the struct as ``qd.template()`` and accesses
  fields via attribute lookup at AST-build time. Tests that mirror this shape are parametrised over the **field**
  backend only. (The other direction is forbidden by an existing pre-stork-20 check: "Ndarray shouldn't be passed in
  via ``qd.template()``".)

Both shapes had a wrapper-unwrap gap before this branch. See ``perso_hugh/doc/quadrants-tensor.md`` §8.14 and
``perso_hugh/doc/genesis_tensor_migration.md`` Phase-1 findings.
"""

import dataclasses
import sys

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

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
# Mirrors the Genesis ndarray-backend struct shape (``DATA_ORIENTED = partial(dataclasses.dataclass, frozen=True)``
# with ``qd.types.NDArray`` field annotations). Without the in-struct unwrap fix in
# ``_func_base._recursive_set_args`` the kernel call raises ``QuadrantsRuntimeTypeError`` from the dataclass-field
# recursion. Ndarray-backend only because the dataclass-of-NDArray pattern only makes sense for ndarray-backed values.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@test_utils.test()
def test_dataclass_struct_with_wrapper_field_kernel_runs(layout):
    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 2]

    @qd.kernel
    def fill(state: State):
        for i, j in qd.ndrange(_M, _N):
            state.a[i, j] = i * 100 + j

    a = qd.tensor(qd.i32, shape=(_M, _N), backend=qd.Backend.NDARRAY, layout=layout)
    assert isinstance(a, qd.Tensor)

    state = State(a=a)
    fill(state)

    np.testing.assert_array_equal(a.to_numpy(), _expected_canonical())


# ----------------------------------------------------------------------------
# Repro 2: @qd.data_oriented struct holding a qd.Tensor wrapper field.
#
# Mirrors the Genesis field-backend struct shape (``DATA_ORIENTED = qd.data_oriented`` with kernel arg
# ``state: qd.template()``). The kernel body resolves ``state.a`` at AST-build time via ``build_Attribute``; without
# the wrapper-unwrap there ``state.a[i, j]`` would land in ``impl.subscript`` with a Python wrapper that the IR layer
# doesn't understand (``__getitem__ cannot be called in Quadrants-scope``). Field-backend only — see module docstring.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@test_utils.test()
def test_data_oriented_struct_with_wrapper_field_kernel_runs(layout):
    @qd.data_oriented
    class State:
        def __init__(self, a):
            self.a = a

    @qd.kernel
    def fill(state: qd.template()):
        for i, j in qd.ndrange(_M, _N):
            state.a[i, j] = i * 100 + j

    a = qd.tensor(qd.i32, shape=(_M, _N), backend=qd.Backend.FIELD, layout=layout)
    state = State(a=a)
    fill(state)

    np.testing.assert_array_equal(a.to_numpy(), _expected_canonical())


# ----------------------------------------------------------------------------
# Mixed bare-impl + wrapper struct fields. Confirms the in-struct unwrap is applied selectively per field — bare impls
# pass straight through (no double-unwrap), wrappers are see-through. Genesis Phase 1 only migrates a subset (Tier-1)
# of struct fields; the rest stay on bare ``V``, so the mixed-shape case is the live one. Ndarray-only: same
# constraint as repro 1.
# ----------------------------------------------------------------------------


@test_utils.test()
def test_mixed_bare_and_wrapper_struct_fields():
    @dataclasses.dataclass(frozen=True)
    class State:
        wrapped: qd.types.NDArray[qd.i32, 1]
        bare: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State):
        for i in range(4):
            state.wrapped[i] = i + 10
            state.bare[i] = i + 20

    wrapped = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    bare = qd.ndarray(qd.i32, (4,))

    state = State(wrapped=wrapped, bare=bare)
    fill(state)

    np.testing.assert_array_equal(wrapped.to_numpy(), np.array([10, 11, 12, 13], dtype=np.int32))
    np.testing.assert_array_equal(bare.to_numpy(), np.array([20, 21, 22, 23], dtype=np.int32))


# ----------------------------------------------------------------------------
# Fastcache: no PARAM_INVALID warning, fast path stays hot.
#
# Pins the soft-failure side of the §8.14 bug: even when the kernel eventually runs (via the slow path), the
# wrapper-as-struct-field would have tripped the fastcache's per-parameter validator
# (``args_hasher.stringify_obj_type``) and emitted ``[FASTCACHE][PARAM_INVALID]``, killing the recompile-avoidance
# for the whole call. Following the harness used by ``test_src_ll_cache_arg_warnings``: capfd + ``@qd.pure`` makes
# the warning observable end-to-end.
#
# Ndarray-only — fastcache exercises the dataclass walker (``dataclass_to_repr``) for this struct shape; the
# data_oriented walker has its own per-field code path which is exercised transitively by the dataclass test via the
# same ``stringify_obj_type`` recursion.
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="capfd flaky for stderr on Windows")
def test_struct_wrapper_field_does_not_invalidate_fastcache(capfd):
    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.pure
    @qd.kernel
    def k(state: State):
        for i in range(2):
            state.a[i] = state.a[i] + 1

    a = qd.tensor(qd.i32, shape=(2,), backend=qd.Backend.NDARRAY)
    state = State(a=a)

    capfd.readouterr()  # drain prior unrelated warnings

    k(state)
    _out, err = capfd.readouterr()

    assert "[FASTCACHE][PARAM_INVALID]" not in err, (
        "Wrapper-in-struct must not be rejected by the fastcache validator. "
        "If you see this, check args_hasher.stringify_obj_type unwraps "
        "qd.Tensor wrappers before its type whitelist."
    )
    assert "[FASTCACHE][INVALID_FUNC]" not in err


# ----------------------------------------------------------------------------
# JIT cache: wrapper-in-struct shares a cache entry with bare-in-struct.
#
# A wrapper passed as a struct field must dispatch the same compiled kernel as the bare impl in the same struct slot —
# otherwise we'd fragment the JIT cache per call. Struct-field analog of
# ``test_kernel_cache_no_fragmentation_under_wrapping`` from ``test_tensor_wrapper_kernel.py``. Ndarray-only: same
# reason as repro 1.
# ----------------------------------------------------------------------------


@test_utils.test()
def test_struct_wrapper_field_shares_cache_with_bare():
    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def k(state: State):
        for i in range(2):
            state.a[i] = state.a[i] + 0

    bare = qd.ndarray(qd.i32, (2,))
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
