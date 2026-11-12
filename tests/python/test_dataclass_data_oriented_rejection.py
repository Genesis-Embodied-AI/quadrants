import dataclasses

import pytest

import quadrants as qd
from quadrants.lang._fast_caching.args_hasher import _FAIL_FASTCACHE, dataclass_to_repr

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("inside_owner", [False, True])
@pytest.mark.parametrize("via_func", [False, True])
@test_utils.test(arch=qd.cpu)
def test_data_oriented_dataclass_member_rejected(
    frozen: bool, inside_owner: bool, via_func: bool, fastcache: bool
) -> None:
    """Reject used data-oriented members through direct, owner, and inlined-function routes."""

    @qd.data_oriented
    class Inner:
        def __init__(self) -> None:
            self.count = 3

    @dataclasses.dataclass(frozen=frozen, unsafe_hash=True)
    class State:
        inner: Inner

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.func
    def read_member(state: qd.Template) -> qd.i32:
        return qd.static(state.inner.count)

    @qd.kernel(fastcache=fastcache)
    def read(arg: qd.Template, nested: qd.Template, use_func: qd.Template) -> qd.i32:
        if qd.static(nested):
            if qd.static(use_func):
                return read_member(arg.state)
            else:
                return qd.static(arg.state.inner.count)
        else:
            if qd.static(use_func):
                return read_member(arg)
            else:
                return qd.static(arg.inner.count)

    state = State(Inner())
    arg = Owner(state) if inside_owner else state
    with pytest.raises(Exception, match="Data-oriented objects inside Python dataclass"):
        read(arg, inside_owner, via_func)


@test_utils.test(arch=qd.cpu)
def test_data_oriented_dataclass_member_disables_artifact_lookup() -> None:
    """An unsupported child cannot produce a disk-cache key."""

    @qd.data_oriented
    class Inner:
        pass

    @dataclasses.dataclass(frozen=True)
    class State:
        inner: Inner

    assert dataclass_to_repr(False, ("0",), State(Inner())) is _FAIL_FASTCACHE


@test_utils.test(arch=qd.cpu)
def test_unused_data_oriented_dataclass_member() -> None:
    """A template dataclass may carry an unused data-oriented child."""

    @qd.data_oriented
    class Inner:
        pass

    @dataclasses.dataclass(frozen=True)
    class State:
        inner: Inner

    @qd.kernel(fastcache=True)
    def unused(state: qd.Template) -> qd.i32:
        return 3

    assert unused(State(Inner())) == 3


@pytest.mark.parametrize("fastcache", [False, True])
@pytest.mark.parametrize("inside_owner", [False, True])
@test_utils.test(arch=qd.cpu)
def test_forwarded_unused_data_oriented_dataclass_member(fastcache: bool, inside_owner: bool) -> None:
    """Forwarding a dataclass and skipping a static branch leaves its unsupported member unused."""

    @qd.data_oriented
    class Inner:
        def __init__(self) -> None:
            self.count = 7

    @dataclasses.dataclass(frozen=True)
    class State:
        inner: Inner

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.func
    def unused(state: qd.Template) -> qd.i32:
        if qd.static(False):
            return state.inner.count
        return 3

    @qd.kernel(fastcache=fastcache)
    def run(arg: qd.Template, nested: qd.Template) -> qd.i32:
        if qd.static(nested):
            return unused(arg.state)
        else:
            return unused(arg)

    state = State(Inner())
    arg = Owner(state) if inside_owner else state
    assert run(arg, inside_owner) == 3
    assert run(arg, inside_owner) == 3
