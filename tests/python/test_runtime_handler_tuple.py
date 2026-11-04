import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test()
def test_runtime_handler_tuple_example():
    @dataclasses.dataclass(frozen=True)
    class GridState:
        vel: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class CouplingData:
        vel_delta: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class CPICData:
        is_cell_separated: qd.Tensor
        cells_link_idx: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class RigidState:
        links_vel: qd.Tensor

    class Handler(Protocol):
        def __call__(self, i_c: int, vel: float) -> float: ...

    @qd.data_oriented
    class Bound[*Ts]:
        def __init__(self, func: Callable[[int, float, *Ts], float], *data: *Ts) -> None:
            self.func = func
            self.data = data

        @qd.func
        def __call__(self, i_c: int, vel: float) -> float:
            return self.func(i_c, vel, *self.data)

    @qd.func
    def func_add_coupling_vel(i_c: int, vel: float, coupling_data: CouplingData) -> float:
        return vel + coupling_data.vel_delta[i_c]

    @qd.func
    def func_replace_cpic_vel(i_c: int, vel: float, cpic_data: CPICData, rigid_state: RigidState) -> float:
        i_l = cpic_data.cells_link_idx[i_c]
        return rigid_state.links_vel[i_l] if cpic_data.is_cell_separated[i_c] else vel

    @qd.kernel
    def kernel_g2p(grid_state: GridState, vel_particles: qd.Tensor, handlers: tuple[Handler, ...]) -> None:
        for i_c in range(vel_particles.shape[0]):
            vel = grid_state.vel[i_c]
            for handler in handlers:
                vel = handler(i_c, vel)
            vel_particles[i_c] = vel

    def tensor(values, dtype):
        result = qd.tensor(dtype, shape=len(values), backend=qd.Backend.NDARRAY)
        result.from_numpy(np.asarray(values, dtype=np.float32 if dtype == qd.f32 else np.int32))
        return result

    grid_state = GridState(tensor([1.0] * 4, qd.f32))
    coupling_data = CouplingData(tensor([0.5] * 4, qd.f32))
    cpic_data = CPICData(tensor([False, True, False, True], qd.i32), tensor([0, 0, 1, 1], qd.i32))
    rigid_state = RigidState(tensor([-2.0, -3.0], qd.f32))
    vel_particles = tensor([0.0] * 4, qd.f32)
    handlers: tuple[Handler, ...] = (
        Bound(func_add_coupling_vel, coupling_data),
        Bound(func_replace_cpic_vel, cpic_data, rigid_state),
    )
    kernel_g2p(grid_state, vel_particles, handlers)
    np.testing.assert_array_equal(vel_particles.to_numpy(), [1.5, -2.0, 1.5, -3.0])


@test_utils.test()
def test_runtime_handler_tuple_mutation_and_specialization():
    @dataclasses.dataclass
    class Data:
        values: str
        scale: str
        unused: object

    @qd.data_oriented
    class Bound:
        def __init__(self, func, data):
            self.func = func
            self.data = (data,)

        @qd.func
        def __call__(self, i, value):
            return self.func(i, value, *self.data)

    @qd.func
    def add(i, value, data: qd.RuntimeTyped):
        return value + data.values[i] * data.scale

    @qd.func
    def multiply(i, value, data: qd.RuntimeTyped):
        return value * data.values[i] * data.scale

    # The actual tuple overrides this unrelated annotation.
    @qd.kernel(fastcache=True)
    def run(handlers: qd.f32, output: qd.Tensor):
        value = 1.0
        for handler in handlers:
            value = handler(0, value)
        output[0] = value

    def array(value, dtype=qd.f32):
        result = qd.ndarray(dtype, shape=1)
        result[0] = value
        return result

    output = array(0)
    data = Data(array(2), 1.0, array(100))
    handler = Bound(add, data)
    handlers = (handler,)
    run(handlers, output)
    assert output[0] == 3
    count = len(run._primal.materialized_kernels)
    data.scale = 2.0
    run(handlers, output)
    assert output[0] == 5
    assert len(run._primal.materialized_kernels) == count
    data.values = array(3)
    run(handlers, output)
    assert output[0] == 7
    assert len(run._primal.materialized_kernels) == count
    handler.data = (Data(array(4), 3.0, array(999)),)
    run(handlers, output)
    assert output[0] == 13
    assert len(run._primal.materialized_kernels) == count
    run((Bound(add, Data(array(5), 2.0, array(0))),), output)
    assert output[0] == 11
    assert len(run._primal.materialized_kernels) == count
    handler.func = multiply
    run(handlers, output)
    assert output[0] == 12
    assert len(run._primal.materialized_kernels) == count + 1
    handler.data[0].values = array(7, qd.i32)
    run(handlers, output)
    assert output[0] == 21
    assert len(run._primal.materialized_kernels) == count + 2
    run((), output)
    assert output[0] == 1
    assert run._primal.fast_checksum is None


@test_utils.test()
def test_runtime_handler_tuple_order_nested_and_aliases():
    @dataclasses.dataclass
    class Data:
        values: object

    @qd.data_oriented
    class Handler:
        def __init__(self, data, bias):
            self.data = (data,)
            self.bias = bias

        @qd.func
        def __call__(self, value):
            return value * self.data[0].values[0] + self.bias

    @qd.kernel
    def run(groups: tuple, output: qd.Tensor):
        value = 1.0
        for handlers in groups:
            for handler in handlers:
                value = handler(value)
        output[0] = value

    def data(value):
        result = qd.ndarray(qd.f32, shape=1)
        result[0] = value
        return Data(result)

    output = qd.ndarray(qd.f32, shape=1)
    first = Handler(data(2), 1)
    second = Handler(data(3), 3)
    run(((first, second), ()), output)
    assert output[0] == 12
    run(((second, first),), output)
    assert output[0] == 13
    second.bias = 4
    run(((first, second),), output)
    assert output[0] == 13
    first.data = second.data
    run(((first, second),), output)
    assert output[0] == 16
    second.data = (data(5),)
    run(((first, second),), output)
    assert output[0] == 24


@test_utils.test()
def test_runtime_handler_tuple_keyword_and_optional_capture():
    @dataclasses.dataclass
    class Data:
        value: object
        optional: object

    @qd.func
    def read(data: qd.RuntimeTyped):
        result = data.value
        if qd.static(data.optional is not None):
            result += data.optional[0]
        return result

    @qd.data_oriented
    class Handler:
        def __init__(self, data):
            self.data = data

        @qd.func
        def __call__(self):
            return read(data=self.data)

    @qd.kernel
    def run(handlers: qd.RuntimeTyped) -> qd.f32:
        result = 0.0
        for handler in handlers:
            result += handler()
        return result

    data = Data(2.5, None)
    handlers = (Handler(data),)
    assert run(handlers) == 2.5
    data.value = 3.5
    assert run(handlers) == 3.5
    data.optional = qd.ndarray(qd.f32, shape=1)
    data.optional[0] = 4
    assert run(handlers) == 7.5
    data.optional = None
    data.value = 6
    assert run(handlers) == 6


@test_utils.test()
def test_runtime_tuple_unrolling_and_calls():
    @qd.func
    def add(value):
        return value + 1

    @qd.func
    def double(value):
        return value * 2

    @qd.kernel
    def run(handlers: tuple) -> qd.i32:
        value = 1
        for handler in handlers:
            value = handler(value)
        return value

    assert run((add, double)) == 4
    assert run((double, add)) == 3
    assert run(()) == 1

    @qd.kernel
    def sum_nested(values: tuple) -> qd.i32:
        result = 0
        for group in values:
            for value in group:
                if qd.static(value == 2):
                    continue
                result += value
        return result

    assert sum_nested(((1, 2), (), (3,))) == 4


@test_utils.test()
def test_runtime_handler_tuple_rejects_cycles_and_unsupported_captures():
    import pytest

    @qd.data_oriented
    class Handler:
        def __init__(self, data):
            self.data = data

        @qd.func
        def __call__(self):
            return 1

    @qd.kernel
    def run(handlers: tuple) -> qd.i32:
        result = 0
        for handler in handlers:
            result += handler()
        return result

    handler = Handler(None)
    handler.data = (handler,)
    with pytest.raises(Exception, match="contains a cycle"):
        run((handler,))
    handler.data = object()
    with pytest.raises(Exception, match="unsupported runtime type"):
        run((handler,))
