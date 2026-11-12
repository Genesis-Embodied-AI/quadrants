import dataclasses

import pytest

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(arch=qd.cpu)
def test_data_oriented_decorator_rejects_dataclass() -> None:
    """Applying data_oriented to a dataclass fails immediately."""

    @dataclasses.dataclass
    class State:
        count: int = 3

    with pytest.raises(TypeError, match="both a Python dataclass and data-oriented"):
        qd.data_oriented(State)

    class Child(State):
        pass

    with pytest.raises(TypeError, match="both a Python dataclass and data-oriented"):
        qd.data_oriented(Child)


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("fastcache", [False, True])
@pytest.mark.parametrize("route", ["typed", "template", "data_oriented", "dataclass"])
@test_utils.test(arch=qd.cpu)
def test_dataclass_on_data_oriented_rejected(inherited: bool, fastcache: bool, route: str) -> None:
    """Detect the reverse decorator order and inherited data-oriented membership at compilation."""

    @qd.data_oriented
    class Base:
        count: int = 3

    if inherited:

        class State(Base):
            count: int = 3

    else:
        State = Base
    State = dataclasses.dataclass(frozen=True)(State)

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State()

    @dataclasses.dataclass(frozen=True)
    class Container:
        state: State

    @qd.kernel(fastcache=fastcache)
    def typed(state: State) -> qd.i32:
        return state.count

    @qd.kernel(fastcache=fastcache)
    def templated(state: qd.Template) -> qd.i32:
        return qd.static(state.count)

    @qd.kernel(fastcache=fastcache)
    def nested(owner: qd.Template) -> qd.i32:
        return qd.static(owner.state.count)

    @qd.kernel(fastcache=fastcache)
    def contained(owner: Container) -> qd.i32:
        return owner.state.count

    kernel, arg = {
        "typed": (typed, State()),
        "template": (templated, State()),
        "data_oriented": (nested, Owner()),
        "dataclass": (contained, Container(State())),
    }[route]
    with pytest.raises(Exception, match="both a Python dataclass and data-oriented"):
        kernel(arg)
