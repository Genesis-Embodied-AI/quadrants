from dataclasses import dataclass

import pytest

import quadrants as qd
from tests.helpers import test_utils


@pytest.mark.parametrize("route", ["list", "method", "attributes", "nested_attributes"])
@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test()
def test_nested_dataclass_alias_rejected(route: str, fastcache: bool) -> None:
    """An alias must not silently bind a scalar through a different attribute path."""

    @dataclass
    class State:
        value: int

    @dataclass
    class Pair:
        first: State
        second: State

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            """Create two access routes to one dataclass."""
            state = State(3)
            self.state = state
            self.states = [state]
            if route == "attributes":
                self.second = state
            elif route == "nested_attributes":
                self.pair = Pair(state, state)

        def get_alias(self) -> State:
            """Return the dataclass through the stored list."""
            return self.states[0]

    @qd.kernel(fastcache=fastcache)
    def read(owner: qd.Template, mode: qd.Template) -> qd.i32:
        """Read through the route whose object identity hides its binding path."""
        if qd.static(mode == "list"):
            return owner.states[0].value
        elif qd.static(mode == "method"):
            return owner.get_alias().value
        elif qd.static(mode == "attributes"):
            return owner.second.value
        else:
            return owner.pair.second.value

    with pytest.raises(Exception, match="dataclass.*alias|alias.*dataclass"):
        read(Owner(), route)


@test_utils.test()
def test_unused_nested_dataclass_alias_is_harmless() -> None:
    """Unaccessed aliased scalar members must not prevent compilation."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            """Store an unused alias beside an independent dataclass."""
            self.first = State(3)
            self.second = self.first
            self.used = State(9)

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        """Read only the independent dataclass."""
        return owner.used.value

    owner = Owner()
    assert read(owner) == 9
    owner.used.value = 11
    assert read(owner) == 11
