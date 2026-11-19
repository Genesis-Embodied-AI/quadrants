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
    with pytest.raises(Exception, match="fixed after first kernel use"):
        shared.second = State(9)
    with pytest.raises(Exception, match="fixed after first kernel use"):
        del shared.states
    with pytest.raises(Exception, match="Cannot add container reference"):
        shared.new_state = State(9)


@pytest.mark.parametrize("nested", [False, True])
@test_utils.test(arch=qd.cpu)
def test_mutable_container_rejected(nested: bool) -> None:
    """Reject lists even when hidden inside tuples."""

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.items = (([],),) if nested else []

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return 3

    with pytest.raises(Exception, match="Mutable Python containers"):
        read(Owner())


@test_utils.test(arch=qd.cpu)
def test_python_call_returning_dataclass_rejected() -> None:
    """Python methods cannot supply a computed runtime dataclass route."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(3)

        def selected(self) -> State:
            return self.state

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.selected().value

    with pytest.raises(Exception, match="Python calls returning runtime dataclasses"):
        read(Owner())


@test_utils.test(arch=qd.cpu)
def test_nested_reference_guard_preserves_custom_setattr() -> None:
    """Lock nested dataclass references while preserving ordinary setters and setup on fresh instances."""

    @dataclass
    class State:
        value: int

    @dataclass
    class Pair:
        state: State

        def __setattr__(self, name: str, value: object) -> None:
            object.__setattr__(self, name, value)

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.pair = Pair(State(3))

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.pair.state.value

    owner = Owner()
    owner.pair = Pair(State(7))
    assert read(owner) == 7
    with pytest.raises(Exception, match="fixed after first kernel use"):
        owner.pair.state = State(9)
    fresh = Owner()
    fresh.pair = Pair(State(11))
    assert read(fresh) == 11


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


@test_utils.test(arch=qd.cpu)
def test_tuple_only_dataclass() -> None:
    """A tuple can be the only route to a runtime dataclass."""

    @dataclass
    class State:
        value: int
        array: qd.Tensor

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.states = (State(3, qd.ndarray(qd.i32, shape=1)),)

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.states[0].value + owner.states[0].array[0]

    owner = Owner()
    owner.states[0].array[0] = 4
    assert read(owner) == 7
    owner.states[0].value = 6
    owner.states[0].array[0] = 8
    assert read(owner) == 14
