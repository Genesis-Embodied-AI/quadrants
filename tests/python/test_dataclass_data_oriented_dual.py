import dataclasses

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize(
    "dataclass_first", [True, False], ids=["dataclass_then_data_oriented", "data_oriented_then_dataclass"]
)
@test_utils.test(arch=qd.cpu)
def test_dual_dataclass_data_oriented_member_rejected(dataclass_first: bool) -> None:
    """Reject dual dataclass/data-oriented objects through a data-oriented owner, in either decorator order."""

    class State:
        count: int = 3

    if dataclass_first:
        State = qd.data_oriented(dataclasses.dataclass(frozen=True)(State))
    else:
        State = dataclasses.dataclass(frozen=True)(qd.data_oriented(State))

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State()

    @qd.kernel(fastcache=True)
    def read(owner: qd.Template) -> qd.i32:
        return qd.static(owner.state.count)

    with pytest.raises(Exception, match="both a Python dataclass and data-oriented"):
        read(Owner())
