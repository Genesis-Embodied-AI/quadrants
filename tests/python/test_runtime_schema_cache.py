import gc
import weakref
from dataclasses import dataclass

import pytest

import quadrants as qd
from quadrants.lang._runtime_typing.dataclass_schemas import (
    _schema_by_dataclass_instance_and_context,
    cached_infer_dataclass_schema,
)

from tests.helpers import test_utils


@pytest.mark.parametrize("slots", [False, True])
@test_utils.test(arch=qd.cpu)
def test_schema_cache_does_not_modify_instances(slots: bool) -> None:
    @dataclass(frozen=True, slots=slots)
    class State:
        value: object

    state = State(1)
    before = dict(vars(state)) if not slots else None
    first = cached_infer_dataclass_schema(state, "state")
    assert cached_infer_dataclass_schema(state, "state") is first
    if slots:
        assert _schema_by_dataclass_instance_and_context.get(state) is None
    else:
        assert vars(state) == before
        assert _schema_by_dataclass_instance_and_context.get(state) is not None
    assert not hasattr(state, "_qd_runtime_schemas")
    assert not hasattr(state, "_qd_runtime_guarded_schemas")


@test_utils.test(arch=qd.cpu)
def test_guard_cache_does_not_retain_parent_through_child() -> None:
    @dataclass
    class Child:
        value: int = 0

    @dataclass(frozen=True)
    class State:
        child: object

    child = Child()
    state = State(child)
    child.parent = state
    before = dict(vars(state))
    schema = cached_infer_dataclass_schema(state, "state")
    assert cached_infer_dataclass_schema(state, "state") is schema
    assert vars(state) == before
    assert _schema_by_dataclass_instance_and_context.get(state) is not None
    key_id = id(state)
    ref = weakref.ref(state)
    del state, child, before
    gc.collect()
    assert ref() is None
    assert key_id not in _schema_by_dataclass_instance_and_context._entries
