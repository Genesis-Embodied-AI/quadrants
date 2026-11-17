"""Frozen dataclass arguments preserve values across repeated launches and resets."""

import dataclasses

import numpy as np

import quadrants as qd

from tests.helpers import test_utils

_M, _N = 3, 4


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


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_multiple_kernels():
    """Two kernels can use different subsets of the same frozen dataclass."""

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


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_template_key_cache_populated():
    """The template key should depend on the way we use it"""

    @dataclasses.dataclass(frozen=True)
    class Base1:
        x1: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass(frozen=True)
    class Base2:
        x2: qd.types.NDArray[qd.f32, 2]

    @dataclasses.dataclass(frozen=True)
    class Sub(Base1, Base2):
        pass

    @qd.kernel
    def use1(dat: Base1):
        for i in range(4):
            dat.x1[i] = 1

    @qd.kernel
    def use2(dat: Base2):
        for i, j in qd.ndrange(2, 2):
            dat.x2[i, j] = 2.0

    sub = Sub(x1=qd.ndarray(qd.i32, shape=(4,)), x2=qd.ndarray(qd.f32, shape=(2, 2)))

    assert not hasattr(sub, "_qd_spec_key")
    use1(sub)
    assert id(Base1) in sub._qd_spec_key
    use2(sub)
    assert id(Base1) in sub._qd_spec_key and id(Base2) in sub._qd_spec_key
    assert sub._qd_spec_key[id(Base1)][0] is Base1 and sub._qd_spec_key[id(Base2)][0] is Base2

    # Distinct field sets (i32/1D vs f32/2D) must yield distinct keys, i.e. neither annotation reused the other's.
    assert sub._qd_spec_key[id(Base1)][1] != sub._qd_spec_key[id(Base2)][1]


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_field_only_ancestor_views():
    """Each ancestor annotation selects its own field members."""

    @dataclasses.dataclass(frozen=True)
    class Base1:
        a: qd.Template

    @dataclasses.dataclass(frozen=True)
    class Base2:
        b: qd.Template

    @dataclasses.dataclass(frozen=True)
    class Sub(Base1, Base2):
        pass

    @qd.kernel
    def use1(dat: Base1):
        for i in range(4):
            dat.a[i] = 1

    @qd.kernel
    def use2(dat: Base2):
        for i in range(4):
            dat.b[i] = 2

    a = qd.field(qd.i32, shape=(4,))
    b = qd.field(qd.i32, shape=(4,))
    sub = Sub(a=a, b=b)

    use1(sub)
    use2(sub)

    np.testing.assert_array_equal(a.to_numpy(), [1, 1, 1, 1])
    np.testing.assert_array_equal(b.to_numpy(), [2, 2, 2, 2])


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_cache_many_fields():
    """Only the two used members of a ten-field dataclass affect the result."""

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


@test_utils.test(arch=qd.cpu)
def test_frozen_dc_kernel_correct_after_reset():
    """Frozen dataclass arguments still work after resetting and reinitializing the runtime."""

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
    def sum_via_grouped(info: qd.Template, out: qd.types.ndarray()):
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
