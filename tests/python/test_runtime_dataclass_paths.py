from dataclasses import dataclass

import pytest

import quadrants as qd
from quadrants.lang import _runtime_dataclass

from tests.helpers import test_utils


@test_utils.test()
def test_diagnostic_path_cycle_shared_child_and_recovery():
    @dataclass
    class Node:
        child: object

    @dataclass(frozen=True)
    class Pair:
        left: object
        right: object

    shared = Node(1)
    state = Pair(shared, shared)
    resolver = _runtime_dataclass.resolve_dataclass
    schema = resolver(state, "root.argument")
    assert schema.__dataclass_fields__["left"].type is schema.__dataclass_fields__["right"].type

    shared.child = shared
    with pytest.raises(Exception, match=r"root.argument.left.child.*cycle"):
        resolver(state, "root.argument")
    shared.child = 1
    resolver(state, "fresh")
    broken = Node(None)
    broken.child = broken
    with pytest.raises(Exception, match=r"next.right.child.*cycle"):
        resolver(Pair(shared, broken), "next")
    resolver(state, "recovered")


@pytest.mark.skipif(not hasattr(_runtime_dataclass, "DeferredMemberType"), reason="Deferred errors postdate #1067 snapshot")
@test_utils.test()
def test_diagnostic_path_nested_use_and_inlined_forwarding():
    @dataclass
    class Inner:
        value: object
        unsupported: object

    @dataclass(frozen=True)
    class Outer:
        nested: object

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return renamed.unsupported

    @qd.kernel
    def used(state: qd.RuntimeTyped) -> qd.i32:
        return read(state.nested)

    @qd.kernel
    def unused(state: qd.RuntimeTyped) -> qd.i32:
        return state.nested.value

    state = Outer(Inner(7, object()))
    assert unused(state) == 7
    with pytest.raises(Exception, match=r"state.nested.unsupported.*unsupported runtime type object"):
        used(state)
    assert unused(Outer(Inner(9, set()))) == 9
