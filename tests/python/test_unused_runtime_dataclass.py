from dataclasses import dataclass

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("unused", ["application notes", {1, 2}, object(), [1, 2]])
@test_utils.test()
def test_unused_runtime_member(unused):
    @dataclass
    class State:
        value: str
        unused: int

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return state.value

    assert read(State(7, unused)) == 7


@pytest.mark.parametrize("keyword", [False, True])
@test_utils.test()
def test_unused_runtime_member_through_nested_func(keyword):
    @dataclass
    class Inner:
        value: object
        unused: object

    @dataclass(frozen=True)
    class Outer:
        inner: object

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return renamed.value

    @qd.func
    def forward(container: qd.RuntimeTyped):
        if qd.static(keyword):
            return read(renamed=container.inner)
        else:
            return read(container.inner)

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return forward(state)

    child = Inner(7, object())
    state = Outer(child)
    assert run(state) == 7
    child.value = 9
    child.unused = {1, 2}
    assert run(state) == 9


@test_utils.test()
def test_used_unsupported_runtime_member_reports_original_path():
    @dataclass
    class State:
        value: object

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return renamed.value

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return read(renamed=state)

    with pytest.raises(Exception, match=r"state.value.*unsupported runtime type set"):
        run(State({1, 2}))
