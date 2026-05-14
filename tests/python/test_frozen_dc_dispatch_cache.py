"""Tests for frozen-dataclass dispatch caching (``hp/tensor-stork-25``).

Verifies that the per-launch field plan cache (``_frozen_dc_plans``) and the per-instance unwrapped-value cache
(``_qd_dc_unwrapped``) in ``_func_base._recursive_set_args`` work correctly for frozen dataclasses with ndarray
fields. The caching must be transparent: calling a kernel multiple times with the same frozen struct instance must
produce identical results, and the cached values must not interfere when the same struct type is used with different
kernel instances or when a new struct instance is created.
"""

import dataclasses

import numpy as np

import quadrants as qd
from quadrants.lang._func_base import _frozen_dc_plans, _get_frozen_dc_plan

from tests import test_utils

_M, _N = 3, 4


# ---------------------------------------------------------------------------
# Basic: frozen dataclass with ndarray fields, called multiple times.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_basic_ndarray():
    """Calling a kernel with a frozen-dataclass struct arg multiple times produces correct results each time."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 2]

    @qd.kernel
    def fill(state: State):
        for i, j in qd.ndrange(_M, _N):
            state.a[i, j] = i * 100 + j

    a = qd.ndarray(qd.i32, shape=(_M, _N))
    state = State(a=a)

    fill(state)
    expected = np.array([[i * 100 + j for j in range(_N)] for i in range(_M)], dtype=np.int32)
    np.testing.assert_array_equal(a.to_numpy(), expected)

    fill(state)
    np.testing.assert_array_equal(a.to_numpy(), expected)

    fill(state)
    np.testing.assert_array_equal(a.to_numpy(), expected)


# ---------------------------------------------------------------------------
# Multi-field: struct with many ndarray fields (mirrors Genesis pattern).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_multi_field():
    """A frozen dataclass with multiple ndarray fields, all used in the same kernel."""

    @dataclasses.dataclass(frozen=True)
    class State:
        pos: qd.types.NDArray[qd.f32, 1]
        vel: qd.types.NDArray[qd.f32, 1]
        acc: qd.types.NDArray[qd.f32, 1]

    @qd.kernel
    def step(state: State, dt: float):
        for i in range(4):
            state.vel[i] += state.acc[i] * dt
            state.pos[i] += state.vel[i] * dt

    pos = qd.ndarray(qd.f32, shape=(4,))
    vel = qd.ndarray(qd.f32, shape=(4,))
    acc = qd.ndarray(qd.f32, shape=(4,))

    acc.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    state = State(pos=pos, vel=vel, acc=acc)
    dt = 0.1

    step(state, dt)
    np.testing.assert_allclose(vel.to_numpy(), [0.1, 0.2, 0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(pos.to_numpy(), [0.01, 0.02, 0.03, 0.04], atol=1e-6)

    step(state, dt)
    np.testing.assert_allclose(vel.to_numpy(), [0.2, 0.4, 0.6, 0.8], atol=1e-6)


# ---------------------------------------------------------------------------
# Separate struct instances: new instance must NOT use stale cached values.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_separate_instances():
    """Two different instances of the same frozen dataclass type must produce independent results."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State, val: qd.i32):
        for i in range(4):
            state.a[i] = val

    a1 = qd.ndarray(qd.i32, shape=(4,))
    a2 = qd.ndarray(qd.i32, shape=(4,))

    s1 = State(a=a1)
    s2 = State(a=a2)

    fill(s1, 10)
    fill(s2, 20)

    np.testing.assert_array_equal(a1.to_numpy(), [10, 10, 10, 10])
    np.testing.assert_array_equal(a2.to_numpy(), [20, 20, 20, 20])


# ---------------------------------------------------------------------------
# Multiple kernels sharing the same struct: field plan cache keyed correctly.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_multiple_kernels():
    """Two kernels using different subsets of a frozen struct's fields must not interfere via the plan cache."""

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.types.NDArray[qd.i32, 1]
        y: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def write_x(state: State):
        for i in range(4):
            state.x[i] = 42

    @qd.kernel
    def write_y(state: State):
        for i in range(4):
            state.y[i] = 99

    x = qd.ndarray(qd.i32, shape=(4,))
    y = qd.ndarray(qd.i32, shape=(4,))
    state = State(x=x, y=y)

    write_x(state)
    np.testing.assert_array_equal(x.to_numpy(), [42, 42, 42, 42])
    np.testing.assert_array_equal(y.to_numpy(), [0, 0, 0, 0])

    write_y(state)
    np.testing.assert_array_equal(x.to_numpy(), [42, 42, 42, 42])
    np.testing.assert_array_equal(y.to_numpy(), [99, 99, 99, 99])


# ---------------------------------------------------------------------------
# Unwrapped-value cache: verify _qd_dc_unwrapped is populated on frozen instances.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_unwrapped_cache_populated():
    """After the first kernel call, the frozen instance should have ``_qd_dc_unwrapped`` cached."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def noop(state: State):
        for i in range(4):
            state.a[i] = state.a[i]

    a = qd.ndarray(qd.i32, shape=(4,))
    state = State(a=a)

    assert not hasattr(state, "_qd_dc_unwrapped")
    noop(state)
    assert hasattr(state, "_qd_dc_unwrapped")
    cached = state._qd_dc_unwrapped
    assert "a" in cached
    assert cached["a"] is a


# ---------------------------------------------------------------------------
# Many fields: exercise with a larger struct to stress the plan cache.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_many_fields():
    """A struct with 10 fields: only the 2 used by the kernel should be in the plan, all 10 in the unwrap cache."""

    @dataclasses.dataclass(frozen=True)
    class BigState:
        f0: qd.types.NDArray[qd.f32, 1]
        f1: qd.types.NDArray[qd.f32, 1]
        f2: qd.types.NDArray[qd.f32, 1]
        f3: qd.types.NDArray[qd.f32, 1]
        f4: qd.types.NDArray[qd.f32, 1]
        f5: qd.types.NDArray[qd.f32, 1]
        f6: qd.types.NDArray[qd.f32, 1]
        f7: qd.types.NDArray[qd.f32, 1]
        f8: qd.types.NDArray[qd.f32, 1]
        f9: qd.types.NDArray[qd.f32, 1]

    @qd.kernel
    def use_two(bs: BigState):
        for i in range(4):
            bs.f0[i] = 1.0
            bs.f9[i] = 2.0

    arrays = [qd.ndarray(qd.f32, shape=(4,)) for _ in range(10)]
    state = BigState(*arrays)

    use_two(state)
    np.testing.assert_allclose(arrays[0].to_numpy(), [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(arrays[9].to_numpy(), [2.0, 2.0, 2.0, 2.0])
    for idx in range(1, 9):
        np.testing.assert_allclose(arrays[idx].to_numpy(), [0.0, 0.0, 0.0, 0.0])

    use_two(state)
    np.testing.assert_allclose(arrays[0].to_numpy(), [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(arrays[9].to_numpy(), [2.0, 2.0, 2.0, 2.0])


# -----------------------------------------------------------------------------------------------------------------------
# id() reuse guard: _get_frozen_dc_plan must not return a stale plan when a new used_params set is allocated at the same
# address as a garbage-collected one.
# -----------------------------------------------------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_plan_id_reuse_guard():
    """Simulate id() reuse: after the original used_params set is deleted, a new set at the same address must not
    receive the stale cached plan.  CPython 3.13+ (mimalloc) makes this scenario likely in practice.
    """
    from quadrants.lang._dataclass_util import create_flat_name

    @dataclasses.dataclass(frozen=True)
    class S:
        x: qd.types.NDArray[qd.i32, 1]

    fields_dict = {f.name: f for f in dataclasses.fields(S)}
    flat_x = create_flat_name("S", "x")

    params_a = {flat_x, "__qd_S__qd_y"}
    plan_a = _get_frozen_dc_plan(params_a, S, "S", fields_dict)
    assert len(plan_a) == 1
    assert plan_a[0][0] == "x"
    assert plan_a[0][1] == flat_x

    saved_id = id(params_a)
    del params_a

    params_b = {"__qd_S__qd_z"}
    if id(params_b) == saved_id:
        plan_b = _get_frozen_dc_plan(params_b, S, "S", fields_dict)
        assert plan_b == (), (
            f"id() reuse guard failed: got plan {plan_b!r} from a stale cache entry "
            f"instead of empty tuple for params_b={params_b!r}"
        )


# -----------------------------------------------------------------------------------------------------------------------
# qd.reset() must clear _frozen_dc_plans so stale entries from a previous qd.init() cycle don't leak into a fresh
# session.
# -----------------------------------------------------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_plans_cleared_on_reset():
    """After qd.reset(), _frozen_dc_plans must be empty."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State):
        for i in range(4):
            state.a[i] = 7

    a = qd.ndarray(qd.i32, shape=(4,))
    fill(State(a=a))
    np.testing.assert_array_equal(a.to_numpy(), [7, 7, 7, 7])

    assert len(_frozen_dc_plans) > 0, "Plan cache should be populated after kernel call"

    qd.reset()
    assert len(_frozen_dc_plans) == 0, "qd.reset() must clear _frozen_dc_plans"


# ---------------------------------------------------------------------------
# End-to-end: kernel correctness survives qd.reset() with frozen dataclass args.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_kernel_correct_after_reset():
    """A kernel using a frozen dataclass must produce correct results after qd.reset() re-initializes the runtime,
    even though the plan cache was cleared.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State, val: qd.i32):
        for i in range(4):
            state.a[i] = val

    a1 = qd.ndarray(qd.i32, shape=(4,))
    fill(State(a=a1), 10)
    np.testing.assert_array_equal(a1.to_numpy(), [10, 10, 10, 10])

    qd.reset()
    qd.init(arch=qd.cpu)

    a2 = qd.ndarray(qd.i32, shape=(4,))
    fill(State(a=a2), 20)
    np.testing.assert_array_equal(a2.to_numpy(), [20, 20, 20, 20])


# ---------------------------------------------------------------------------
# Regression: qd.grouped() on flattened struct ndarray field after field→ndarray switch.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_grouped_on_struct_ndarray_after_field_to_ndarray_switch():
    """Frozen-dataclass struct fields resolved via the flattened-name path (build_Name) must be promoted to AnyArray
    just like the build_Attribute path does.  Without the fix in build_Name, qd.grouped() on a struct ndarray field
    raises TypeError after a field→ndarray backend switch because the raw ScalarNdarray is not recognized by
    begin_frontend_struct_for.
    """

    @dataclasses.dataclass(frozen=True)
    class Info:
        weights: qd.Tensor

    @qd.kernel
    def sum_via_grouped(info: qd.template(), out: qd.types.ndarray()):
        for I in qd.grouped(info.weights):
            out[0] += info.weights[I]

    # Cycle 1: field backend
    w1 = qd.field(qd.f32, shape=(4,))
    w1.fill(1.0)
    o1 = qd.ndarray(qd.f32, shape=(1,))
    sum_via_grouped(Info(weights=qd.Tensor(w1)), o1)
    np.testing.assert_allclose(o1.to_numpy(), [4.0])

    qd.reset()
    qd.init(arch=qd.cpu)

    # Cycle 2: ndarray backend — this is the case that failed before the fix.
    w2 = qd.ndarray(qd.f32, shape=(4,))
    w2.fill(2.0)
    o2 = qd.ndarray(qd.f32, shape=(1,))
    sum_via_grouped(Info(weights=qd.Tensor(w2)), o2)
    np.testing.assert_allclose(o2.to_numpy(), [8.0])
