import dataclasses

import pytest

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(arch=qd.cpu)
def test_data_oriented_dataclass_rejected_at_compilation() -> None:
    """Dataclass handling rejects data_oriented applied after dataclass."""

    @dataclasses.dataclass
    class State:
        count: int = 3

    State = qd.data_oriented(State)

    @qd.kernel
    def read(state: State) -> qd.i32:
        return state.count

    with pytest.raises(Exception, match="both a Python dataclass and data-oriented"):
        read(State())


@test_utils.test(arch=qd.cpu)
def test_dataclass_on_data_oriented_rejected() -> None:
    """Detect the reverse decorator order when the dataclass is stored on an owner."""

    @dataclasses.dataclass(frozen=True)
    @qd.data_oriented
    class State:
        count: int = 3

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State()

    @qd.kernel(fastcache=True)
    def read(owner: qd.Template) -> qd.i32:
        return qd.static(owner.state.count)

    with pytest.raises(Exception, match="both a Python dataclass and data-oriented"):
        read(Owner())
