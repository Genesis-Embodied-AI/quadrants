import dataclasses

import pytest

import quadrants as qd
from quadrants.lang._fast_caching.args_hasher import _FAIL_FASTCACHE, dataclass_to_repr
from quadrants.lang._runtime_typing.dataclass_schemas import (
    cached_infer_dataclass_schema,
)

from tests.helpers import test_utils


@pytest.mark.parametrize("via_func", [False, True])
@test_utils.test(arch=qd.cpu)
def test_data_oriented_dataclass_member_rejected(via_func: bool) -> None:
    """Reject access to a data-oriented object stored inside a dataclass, from either a kernel or qd.func.

    data_oriented_obj.state is allowed: a dataclass inside a data-oriented object. Accessing
    data_oriented_obj.state.inner is forbidden: inner is data-oriented inside a dataclass, a nesting that can cause
    incorrect cached-kernel reuse.
    """

    @qd.data_oriented
    class Inner:
        def __init__(self) -> None:
            self.count = 3

    @dataclasses.dataclass
    class State:
        inner: qd.Template

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.func
    def read_member(state: State) -> qd.i32:
        return qd.static(state.inner.count)

    @qd.kernel(fastcache=True)
    def read(data_oriented_obj: qd.Template, use_func: qd.Template) -> qd.i32:
        if qd.static(use_func):
            return read_member(data_oriented_obj.state)
        else:
            return qd.static(data_oriented_obj.state.inner.count)

    with pytest.raises(Exception, match="Data-oriented objects inside Python dataclass"):
        read(DataOriented(State(Inner())), via_func)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_dataclass_member_disables_artifact_lookup() -> None:
    """An unsupported child cannot produce a disk-cache key."""

    @qd.data_oriented
    class Inner:
        pass

    @dataclasses.dataclass(frozen=True)
    class State:
        inner: qd.Template

    state = State(Inner())
    schema = cached_infer_dataclass_schema(state, "state")
    assert dataclass_to_repr(False, ("0",), state, schema) is _FAIL_FASTCACHE


@pytest.mark.parametrize("in_data_oriented", [False, True])
@test_utils.test(arch=qd.cpu)
def test_function_accepts_unused_unsupported_dataclass_member(in_data_oriented: bool) -> None:
    """A function may receive a dataclass whose unsupported member appears only in an inactive static branch."""

    @qd.data_oriented
    class Inner:
        def __init__(self) -> None:
            self.count = 7

    @dataclasses.dataclass(frozen=True)
    class State:
        inner: qd.Template

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.func
    def unused(state: State) -> qd.i32:
        if qd.static(False):
            return state.inner.count
        return 3

    if in_data_oriented:

        @qd.kernel(fastcache=True)
        def run(data_oriented_obj: qd.Template) -> qd.i32:
            return unused(data_oriented_obj.state)

    else:

        @qd.kernel(fastcache=True)
        def run(state: State) -> qd.i32:
            return unused(state)

    state = State(Inner())
    arg = DataOriented(state) if in_data_oriented else state
    assert run(arg) == 3
    assert run(arg) == 3
