import dataclasses
import typing

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsRuntimeTypeError

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@pytest.mark.parametrize("present_first", [False, True])
@test_utils.test(arch=qd.cpu)
def test_optional_dataclass_specialization(fastcache, present_first):
    @dataclasses.dataclass(frozen=True)
    class State:
        value: qd.Tensor
        scale: float

    @qd.kernel(fastcache=fastcache)
    def add(first: float, state: State | None, out: qd.Tensor, last: float):
        for i in range(out.shape[0]):
            out[i] = first + last
            if qd.static(state is not None):
                out[i] += state.value[i] * state.scale
            if qd.static(None is state):
                out[i] += 1.0

    out = qd.ndarray(qd.f32, shape=4)
    first = qd.ndarray(qd.f32, shape=4)
    second = qd.ndarray(qd.f32, shape=4)
    first.fill(3)
    second.fill(5)
    states = [None, State(first, 2.0), None, State(second, 4.0), None]
    if present_first:
        states = states[1:] + states[:1]
    for state in states:
        add(2.0, state, out, 7.0)
        expected = 10 if state is None else 9 + state.value[0] * state.scale
        np.testing.assert_array_equal(out.to_numpy(), np.full(4, expected))
    assert len(add._primal.mapper.mapping) == 2


@test_utils.test()
def test_optional_dataclass_absent_branch_is_not_traced():
    @dataclasses.dataclass(frozen=True)
    class State:
        value: qd.Tensor

    @qd.kernel
    def absent(state: typing.Optional[State], out: qd.Tensor):
        if qd.static(state is not None):
            out[0] = state.does_not_exist[0]
        else:
            out[0] = 42

    out = qd.ndarray(qd.i32, shape=1)
    absent(None, out)
    assert out[0] == 42


@test_utils.test()
def test_required_dataclass_still_rejects_none():
    @dataclasses.dataclass(frozen=True)
    class State:
        value: qd.Tensor

    @qd.kernel
    def required(state: State):
        pass

    with pytest.raises(QuadrantsRuntimeTypeError, match="requires a Python dataclass instance"):
        required(None)
