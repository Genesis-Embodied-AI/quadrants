"""Field-only and mixed dataclass arguments preserve their launch behavior."""

import dataclasses

import numpy as np

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(arch=qd.cpu)
def test_all_field_struct_tensor_basic():
    """A frozen dataclass can contain only Tensor-wrapped fields."""

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.Tensor
        y: qd.Tensor

    @qd.kernel
    def write(state: State):
        for i in range(4):
            state.x[i] = 10
            state.y[i] = 20

    x = qd.field(qd.i32, shape=(4,))
    y = qd.field(qd.i32, shape=(4,))
    state = State(x=qd.Tensor(x), y=qd.Tensor(y))

    write(state)
    np.testing.assert_array_equal(x.to_numpy(), [10, 10, 10, 10])
    np.testing.assert_array_equal(y.to_numpy(), [20, 20, 20, 20])

    x.fill(0)
    y.fill(0)
    write(state)
    np.testing.assert_array_equal(x.to_numpy(), [10, 10, 10, 10])
    np.testing.assert_array_equal(y.to_numpy(), [20, 20, 20, 20])


@test_utils.test(arch=qd.cpu)
def test_ndarray_struct_not_skipped():
    """Ndarray members still populate kernel parameters."""

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State, val: qd.i32):
        for i in range(4):
            state.a[i] = val

    a = qd.ndarray(qd.i32, shape=(4,))
    state = State(a=a)

    fill(state, 42)
    np.testing.assert_array_equal(a.to_numpy(), [42, 42, 42, 42])

    fill(state, 99)
    np.testing.assert_array_equal(a.to_numpy(), [99, 99, 99, 99])


@test_utils.test(arch=qd.cpu)
def test_mixed_field_and_scalar_not_skipped():
    """A frozen struct mixing a qd.Tensor(Field) with a plain float must NOT be skipped — the float consumes a slot."""

    @dataclasses.dataclass(frozen=True)
    class Config:
        data: qd.Tensor
        scale: float

    @qd.kernel
    def apply_scale(cfg: Config, out: qd.types.ndarray()):
        for i in range(4):
            out[i] = cfg.data[i] * cfg.scale

    data = qd.field(qd.f32, shape=(4,))
    data.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    out = qd.ndarray(qd.f32, shape=(4,))

    cfg = Config(data=qd.Tensor(data), scale=2.0)
    apply_scale(cfg, out)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 4.0, 6.0, 8.0])

    out2 = qd.ndarray(qd.f32, shape=(4,))
    apply_scale(cfg, out2)
    np.testing.assert_allclose(out2.to_numpy(), [2.0, 4.0, 6.0, 8.0])


@test_utils.test(arch=qd.cpu)
def test_mixed_field_and_ndarray_not_skipped():
    """A frozen struct with one Field tensor and one Ndarray must NOT be skipped."""

    @dataclasses.dataclass(frozen=True)
    class State:
        field_data: qd.Tensor
        ndarray_data: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def combine(state: State, out: qd.types.ndarray()):
        for i in range(4):
            out[i] = state.field_data[i] + state.ndarray_data[i]

    f = qd.field(qd.i32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    nd = qd.ndarray(qd.i32, shape=(4,))
    nd.from_numpy(np.array([10, 20, 30, 40], dtype=np.int32))
    out = qd.ndarray(qd.i32, shape=(4,))

    state = State(field_data=qd.Tensor(f), ndarray_data=nd)
    combine(state, out)
    np.testing.assert_array_equal(out.to_numpy(), [11, 22, 33, 44])


@test_utils.test(arch=qd.cpu)
def test_nested_frozen_dc_all_field():
    """Nested frozen dataclass where both levels are all-Field."""

    @dataclasses.dataclass(frozen=True)
    class Inner:
        x: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Inner

    @qd.kernel
    def write(outer: Outer):
        for i in range(4):
            outer.inner.x[i] = 77

    x = qd.field(qd.i32, shape=(4,))
    inner = Inner(x=qd.Tensor(x))
    outer = Outer(inner=inner)

    write(outer)
    np.testing.assert_array_equal(x.to_numpy(), [77, 77, 77, 77])

    write(outer)
    np.testing.assert_array_equal(x.to_numpy(), [77, 77, 77, 77])


@test_utils.test(arch=qd.cpu)
def test_all_field_struct_multiple_kernels():
    """Same all-Field struct instance passed to different kernels using different subsets."""

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.Tensor
        y: qd.Tensor

    @qd.kernel
    def write_x(state: State):
        for i in range(4):
            state.x[i] = 42

    @qd.kernel
    def write_y(state: State):
        for i in range(4):
            state.y[i] = 99

    x = qd.field(qd.i32, shape=(4,))
    y = qd.field(qd.i32, shape=(4,))
    state = State(x=qd.Tensor(x), y=qd.Tensor(y))

    write_x(state)
    np.testing.assert_array_equal(x.to_numpy(), [42, 42, 42, 42])
    np.testing.assert_array_equal(y.to_numpy(), [0, 0, 0, 0])

    write_y(state)
    np.testing.assert_array_equal(x.to_numpy(), [42, 42, 42, 42])
    np.testing.assert_array_equal(y.to_numpy(), [99, 99, 99, 99])


@test_utils.test(arch=qd.cpu)
def test_tensor_wrapping_ndarray_not_skipped():
    """Tensor-wrapped ndarray members still populate kernel parameters."""

    @dataclasses.dataclass(frozen=True)
    class State:
        data: qd.Tensor

    @qd.kernel
    def fill(state: State):
        for i in range(4):
            state.data[i] = 55

    nd = qd.ndarray(qd.i32, shape=(4,))
    state = State(data=qd.Tensor(nd))

    fill(state)
    np.testing.assert_array_equal(nd.to_numpy(), [55, 55, 55, 55])

    # Second call
    nd.fill(0)
    fill(state)
    np.testing.assert_array_equal(nd.to_numpy(), [55, 55, 55, 55])


@test_utils.test(arch=qd.cpu)
def test_field_only_dataclass_survives_reset():
    """After qd.reset() + qd.init(), a new all-Field struct must work correctly."""

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.Tensor

    @qd.kernel
    def fill(state: State):
        for i in range(4):
            state.x[i] = 11

    x1 = qd.field(qd.i32, shape=(4,))
    s1 = State(x=qd.Tensor(x1))
    fill(s1)
    np.testing.assert_array_equal(x1.to_numpy(), [11, 11, 11, 11])

    qd.reset()
    qd.init(arch=qd.cpu)

    x2 = qd.field(qd.i32, shape=(4,))
    s2 = State(x=qd.Tensor(x2))
    fill(s2)
    np.testing.assert_array_equal(x2.to_numpy(), [11, 11, 11, 11])


@test_utils.test(arch=qd.cpu)
def test_struct_with_int_field_not_skipped():
    """A frozen struct with an int field (consumes a slot) must NOT be skipped."""

    @dataclasses.dataclass(frozen=True)
    class Params:
        data: qd.Tensor
        count: int

    @qd.kernel
    def fill_n(params: Params, out: qd.types.ndarray()):
        for i in range(params.count):
            out[i] = params.data[i] * 2

    data = qd.field(qd.i32, shape=(4,))
    data.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    out = qd.ndarray(qd.i32, shape=(4,))

    params = Params(data=qd.Tensor(data), count=4)
    fill_n(params, out)
    np.testing.assert_array_equal(out.to_numpy(), [2, 4, 6, 8])

    # Second call
    out2 = qd.ndarray(qd.i32, shape=(4,))
    fill_n(params, out2)
    np.testing.assert_array_equal(out2.to_numpy(), [2, 4, 6, 8])


@test_utils.test(arch=qd.cpu)
def test_all_field_struct_wrong_annotation_raises():
    """A field-only dataclass cannot be passed to a scalar parameter."""

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.Tensor

    @qd.kernel
    def correct_kernel(state: State):
        for i in range(4):
            state.x[i] = 1

    @qd.kernel
    def wrong_kernel(val: qd.f32):
        pass

    x = qd.field(qd.i32, shape=(4,))
    state = State(x=qd.Tensor(x))

    correct_kernel(state)

    # Passing the struct to a kernel expecting a float must raise, not silently skip
    import pytest

    with pytest.raises(Exception):
        wrong_kernel(state)


@test_utils.test(arch=qd.cpu)
def test_all_field_struct_passed_as_ndarray_param_raises():
    """If an all-Field struct is passed where an ndarray is expected, it must error."""

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.Tensor

    @qd.kernel
    def correct_kernel(state: State):
        for i in range(4):
            state.x[i] = 5

    @qd.kernel
    def ndarray_kernel(arr: qd.types.ndarray()):
        for i in range(4):
            arr[i] = 99

    x = qd.field(qd.i32, shape=(4,))
    state = State(x=qd.Tensor(x))

    correct_kernel(state)

    import pytest

    with pytest.raises(Exception):
        ndarray_kernel(state)
