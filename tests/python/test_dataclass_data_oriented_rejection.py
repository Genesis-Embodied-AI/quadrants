import dataclasses

import pytest

import quadrants as qd
from quadrants.lang._fast_caching.args_hasher import _FAIL_FASTCACHE, dataclass_to_repr
from quadrants.lang._runtime_typing.runtime_dataclass import cached_infer_dataclass_type

from tests.helpers import test_utils


@pytest.mark.parametrize("via_func", [False, True])
@test_utils.test(arch=qd.cpu)
def test_data_oriented_dataclass_member_rejected(via_func: bool) -> None:
    """Reject access to a data-oriented object stored inside a dataclass, from either a kernel or qd.func.

    owner.state is allowed: a dataclass inside a data-oriented object. Accessing owner.state.inner is forbidden: inner
    is data-oriented inside a dataclass, a nesting that can cause incorrect cached-kernel reuse.
    """

    @qd.data_oriented
    class Inner:
        def __init__(self) -> None:
            self.count = 3

    @dataclasses.dataclass
    class State:
        inner: qd.Template

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.func
    def read_member(state: State) -> qd.i32:
        return qd.static(state.inner.count)

    @qd.kernel(fastcache=True)
    def read(owner: qd.Template, use_func: qd.Template) -> qd.i32:
        if qd.static(use_func):
            return read_member(owner.state)
        else:
            return qd.static(owner.state.inner.count)

    with pytest.raises(Exception, match="Data-oriented objects inside Python dataclass"):
        read(Owner(State(Inner())), via_func)


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
    schema = cached_infer_dataclass_type(state, "state")
    assert dataclass_to_repr(False, ("0",), state, schema) is _FAIL_FASTCACHE


@test_utils.test(arch=qd.cpu)
def test_function_accepts_unused_unsupported_dataclass_member() -> None:
    """A function may receive a dataclass whose unsupported member appears only in an inactive static branch."""

    @qd.data_oriented
    class Inner:
        def __init__(self) -> None:
            self.count = 7

    @dataclasses.dataclass(frozen=True)
    class State:
        inner: qd.Template

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.func
    def unused(state: State) -> qd.i32:
        if qd.static(False):
            return state.inner.count
        return 3

    @qd.kernel(fastcache=True)
    def run(owner: qd.Template) -> qd.i32:
        return unused(owner.state)

    owner = Owner(State(Inner()))
    assert run(owner) == 3
    assert run(owner) == 3
