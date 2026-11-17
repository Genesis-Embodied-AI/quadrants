"""Direct dataclass arguments use identity-based member bindings without changing runtime values."""

from dataclasses import dataclass

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test(arch=qd.cpu, offline_cache=True)
def test_dataclass_alias_layout_and_live_values(fastcache: bool) -> None:
    """Shared and independent dataclass arguments select compatible bindings, including through functions."""

    @dataclass
    class State:
        value: int
        unused: int

    @qd.func
    def read(state: State) -> int:
        return state.value

    @qd.kernel(fastcache=fastcache)
    def add(first: State, second: State) -> int:
        return read(first) + read(state=second)

    shared = State(2, 0)
    assert add(shared, shared) == 4
    first, second = State(2, 1), State(9, 2)
    assert add(first, second) == 11
    second.value = 13
    assert add(first, second) == 15
    assert add(shared, shared) == 4
    assert add(State(5, 3), State(6, 4)) == 11
