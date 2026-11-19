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
    class Owner:
        def __init__(self, shared: bool) -> None:
            self.first = State(3)
            self.second = self.first if shared else State(7)
            self.states = ((self.second,),)

    @qd.kernel(fastcache=fastcache)
    def read(owner: qd.Template) -> qd.i32:
        return owner.first.value + owner.second.value + owner.states[0][0].value

    distinct = Owner(False)
    assert read(distinct) == 17
    shared = Owner(True)
    assert read(shared) == 9
    shared.first.value = 5
    assert read(shared) == 15
    fresh = Owner(True)
    assert read(fresh) == 9


@test_utils.test(arch=qd.cpu)
def test_alias_key_tracks_whole_argument_combination() -> None:
    """Switch between combinations sharing a dataclass and combinations with independent dataclasses."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.kernel
    def read(first: qd.Template, second: qd.Template) -> qd.i32:
        return first.state.value + second.state.value

    first = Owner(State(3))
    shared = Owner(first.state)
    independent = Owner(State(7))
    assert read(first, shared) == 6
    assert read(first, independent) == 10
    first.state.value = 5
    assert read(first, shared) == 10
    assert read(first, independent) == 12
