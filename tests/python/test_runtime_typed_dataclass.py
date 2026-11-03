from dataclasses import dataclass
from typing import Final

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("concrete_annotation", [False, True])
@test_utils.test()
def test_runtime_members_ignore_annotations(concrete_annotation):
    @dataclass
    class State:
        values: str
        increment: str

    annotation = State if concrete_annotation else qd.RuntimeTyped

    @qd.kernel
    def advance(state: annotation):
        state.values[0] += state.increment

    values = qd.ndarray(qd.f32, shape=1)
    state = State(values, 1.25)
    advance(state)
    assert values[0] == 1.25
    compiled = len(advance._primal.materialized_kernels)
    state.increment = 2.5
    advance(state)
    assert values[0] == 3.75
    assert len(advance._primal.materialized_kernels) == compiled
    state.increment = 3
    advance(state)
    assert values[0] == 6.75
    assert len(advance._primal.materialized_kernels) == compiled + 1
    state.increment = 1.25
    advance(state)
    assert values[0] == 8
    assert len(advance._primal.materialized_kernels) == compiled + 1


@test_utils.test()
def test_runtime_nested_func_calls_and_pruning():
    @dataclass
    class Inner:
        value: object
        unused: object

    @dataclass(frozen=True)
    class Outer:
        inner: object

    @qd.func
    def read(inner: qd.RuntimeTyped):
        return inner.value

    @qd.func
    def forward(outer: qd.RuntimeTyped):
        return read(inner=outer.inner) + read(outer.inner)

    @qd.kernel
    def run(outer: qd.RuntimeTyped) -> qd.f32:
        return forward(outer)

    inner = Inner(1.25, qd.ndarray(qd.i32, shape=1))
    outer = Outer(inner)
    assert run(outer) == 2.5
    inner.value = 3.5
    assert run(outer) == 7
    inner.value = 4
    assert run(outer) == 8


@test_utils.test()
def test_runtime_unrelated_dataclass_classes():
    @dataclass
    class First:
        value: object

    @dataclass(slots=True)
    class Second:
        value: float

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.f32:
        return state.value

    assert read(First(1.5)) == 1.5
    assert read(Second(2.5)) == 2.5
    assert read(First(3.5)) == 3.5


@test_utils.test()
def test_runtime_array_dtype_and_rank_changes():
    @dataclass
    class State:
        array: object

    @qd.kernel
    def fill(state: qd.RuntimeTyped):
        for i in qd.grouped(state.array):
            state.array[i] = 7

    state = State(qd.ndarray(qd.i32, shape=2))
    first = state.array
    fill(state)
    np.testing.assert_array_equal(first.to_numpy(), [7, 7])
    state.array = qd.ndarray(qd.f32, shape=(2, 3))
    fill(state)
    np.testing.assert_array_equal(state.array.to_numpy(), np.full((2, 3), 7))
    first.fill(0)
    state.array = first
    fill(state)
    np.testing.assert_array_equal(first.to_numpy(), [7, 7])


@test_utils.test()
def test_runtime_external_array_and_field():
    @dataclass
    class State:
        array: object

    @qd.kernel
    def write(state: qd.RuntimeTyped):
        state.array[0] = 9

    array = np.zeros(2, dtype=np.float32)
    write(State(array))
    assert array[0] == 9
    field = qd.field(qd.i32, shape=2)
    write(State(field))
    assert field[0] == 9


@test_utils.test()
def test_runtime_final_through_nested_func():
    @dataclass(frozen=True)
    class Config:
        enabled: Final[bool]
        scale: object

    @dataclass(frozen=True)
    class State:
        config: object

    @qd.func
    def apply(config: qd.RuntimeTyped):
        result = 0.0
        if qd.static(config.enabled):
            result = config.scale
        return result

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.f32:
        return apply(state.config)

    assert run(State(Config(True, 2.5))) == 2.5
    assert run(State(Config(False, 2.5))) == 0
    assert run(State(Config(True, 3.5))) == 3.5


@test_utils.test(require=qd.extension.data64, default_ip=qd.i64, default_fp=qd.f64)
def test_runtime_scalar_precision():
    @dataclass
    class State:
        value: qd.i32

    @qd.kernel
    def integer(state: qd.RuntimeTyped) -> qd.i64:
        return state.value

    @qd.kernel
    def real(state: qd.RuntimeTyped) -> qd.f64:
        return state.value

    assert integer(State(2**40)) == 2**40
    assert integer(State(np.int64(2**40))) == 2**40
    assert real(State(1.0 + 2**-40)) == 1.0 + 2**-40
    assert real(State(np.float64(1.0 + 2**-40))) == 1.0 + 2**-40


@test_utils.test()
def test_runtime_bad_inputs():
    @dataclass
    class State:
        value: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return state.value

    with pytest.raises(Exception, match="requires a Python dataclass instance"):
        read(1)
    with pytest.raises(Exception, match="state.value.*unsupported runtime type"):
        read(State(object()))
    cyclic = State(None)
    cyclic.value = cyclic
    with pytest.raises(Exception, match="state.value.*cycle"):
        read(cyclic)
