from dataclasses import dataclass

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test(arch=qd.cpu, offline_cache=True)
def test_fixed_aliases(fastcache: bool) -> None:
    """Read live scalars through stored aliases and nested tuple aliases."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class DataOriented:
        def __init__(self, shared: bool) -> None:
            self.first = State(3)
            self.second = self.first if shared else State(7)
            self.states = ((self.second,),)

    @qd.kernel(fastcache=fastcache)
    def read(data_oriented_obj: qd.Template) -> qd.i32:
        return data_oriented_obj.first.value + data_oriented_obj.second.value + data_oriented_obj.states[0][0].value

    distinct = DataOriented(False)
    assert read(distinct) == 17
    shared = DataOriented(True)
    assert read(shared) == 9
    shared.first.value = 5
    assert read(shared) == 15
    fresh = DataOriented(True)
    assert read(fresh) == 9


@test_utils.test(arch=qd.cpu)
def test_alias_key_tracks_whole_argument_combination() -> None:
    """Switch between combinations sharing a dataclass and combinations with independent dataclasses."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.kernel
    def read(first: qd.Template, second: qd.Template) -> qd.i32:
        return first.state.value + second.state.value

    first = DataOriented(State(3))
    shared = DataOriented(first.state)
    independent = DataOriented(State(7))
    assert read(first, shared) == 6
    assert read(first, independent) == 10
    first.state.value = 5
    assert read(first, shared) == 10
    assert read(first, independent) == 12
