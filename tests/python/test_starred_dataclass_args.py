"""Dataclass arguments forwarded through a compile-time tuple or list (issue #1018)."""

import dataclasses

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("templated", [False, True, "class"])
@test_utils.test()
def test_starred_dataclass_args(container, templated):
    @dataclasses.dataclass(frozen=True)
    class State:
        values: qd.Tensor

    annotation = qd.Template if templated == "class" else qd.template() if templated else State

    @qd.func
    def add(state: annotation, other: annotation):
        for i in range(state.values.shape[0]):
            state.values[i] += other.values[i]

    @qd.kernel
    def forward(args: qd.template()):
        add(*qd.static(container(args)))

    state = State(qd.field(qd.i32, shape=4))
    other = State(qd.field(qd.i32, shape=4))
    other.values.fill(3)
    forward((state, other))
    np.testing.assert_array_equal(state.values.to_numpy(), [3] * 4)


@pytest.mark.parametrize("bound", [False, True])
@test_utils.test()
def test_starred_dataclass_args_nested_pruning(bound):
    @dataclasses.dataclass(frozen=True)
    class Inner:
        unused: qd.types.NDArray[qd.i32, 1]
        values: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass(frozen=True)
    class State:
        unused: qd.types.NDArray[qd.i32, 1]
        inner: Inner
        extra: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write(offset: qd.i32, state: State, other: State, scale: qd.i32):
        state.inner.values[0] = other.extra[0] * scale + offset

    @qd.data_oriented
    class Action:
        def __init__(self, args):
            self.args = args
            self.func = self.write if bound else write

        @qd.func
        def write(self, offset: qd.i32, state: State, other: State, scale: qd.i32):
            state.inner.values[0] = other.extra[0] * scale + offset

        @qd.kernel
        def run(self):
            self.func(5, *self.args)

    def make_state():
        return State(
            qd.ndarray(qd.i32, shape=1),
            Inner(qd.ndarray(qd.i32, shape=1), qd.ndarray(qd.i32, shape=1)),
            qd.ndarray(qd.i32, shape=1),
        )

    state, other = make_state(), make_state()
    other.extra[0] = 7
    action = Action((state, other, 3))
    action.run()
    assert state.inner.values[0] == 26
    other.extra[0] = 11
    action.run()
    assert state.inner.values[0] == 38


@test_utils.test()
def test_starred_dataclass_args_mixed_template():
    @dataclasses.dataclass(frozen=True)
    class State:
        values: qd.Tensor
        unused: qd.Tensor

    @qd.func
    def write(state: State, other: qd.template(), scale: qd.i32):
        state.values[0] = other.values[0] * scale

    @qd.kernel
    def run(args: qd.template()):
        write(*args)

    state = State(qd.field(qd.i32, shape=1), qd.field(qd.i32, shape=1))
    other = State(qd.field(qd.i32, shape=1), qd.field(qd.i32, shape=1))
    other.values[0] = 7
    run((state, other, 3))
    assert state.values[0] == 21


@test_utils.test()
def test_starred_dataclass_args_typed_literal():
    @dataclasses.dataclass(frozen=True)
    class State:
        unused: qd.types.NDArray[qd.i32, 1]
        values: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def write(state: State, value: qd.i32):
        state.values[0] = value

    @qd.kernel
    def run(state: State):
        write(*(state, 17))

    state = State(qd.ndarray(qd.i32, shape=1), qd.ndarray(qd.i32, shape=1))
    run(state)
    assert state.values[0] == 17


@test_utils.test()
def test_starred_dataclass_args_evaluated_once_per_pass():
    @dataclasses.dataclass(frozen=True)
    class State:
        first: qd.Tensor
        second: qd.Tensor

    state = State(qd.field(qd.i32, shape=1), qd.field(qd.i32, shape=1))
    calls = []

    def make_args():
        calls.append(None)
        return (state,)

    @qd.func
    def write(state: State):
        state.first[0] += 1
        state.second[0] += 2

    @qd.kernel
    def baseline():
        args = qd.static(make_args())
        write(args[0])

    @qd.kernel
    def starred():
        write(*qd.static(make_args()))

    baseline()
    baseline_calls = len(calls)
    calls.clear()
    starred()
    assert len(calls) == baseline_calls
    assert state.first[0] == 2
    assert state.second[0] == 4
