import gc
import weakref
from dataclasses import dataclass

import pytest

import quadrants as qd
from quadrants.lang._runtime_dataclass import _instance_schemas, infer_dataclass_type

from tests.helpers import test_utils


@pytest.mark.parametrize("slots", [False, True])
@test_utils.test(arch=qd.cpu)
def test_schema_cache_does_not_modify_instances(slots):
    @dataclass(frozen=True, slots=slots)
    class State:
        value: object

    state = State(1)
    before = dict(vars(state)) if not slots else None
    first = infer_dataclass_type(state, "state")
    assert infer_dataclass_type(state, "state") is first
    if slots:
        assert _instance_schemas.get(state) is None
    else:
        assert vars(state) == before
        assert _instance_schemas.get(state) is not None
    assert not hasattr(state, "_qd_runtime_schemas")
    assert not hasattr(state, "_qd_runtime_guarded_schemas")


@test_utils.test(arch=qd.cpu)
def test_guard_cache_does_not_retain_parent_through_child():
    @qd.data_oriented
    class Child:
        pass

    @dataclass(frozen=True)
    class State:
        child: object

    child = Child()
    state = State(child)
    child.parent = state
    before = dict(vars(state))
    schema = infer_dataclass_type(state, "state")
    assert infer_dataclass_type(state, "state") is schema
    assert vars(state) == before
    assert _instance_schemas.get(state) is not None
    key_id = id(state)
    ref = weakref.ref(state)
    del state, child
    gc.collect()
    assert ref() is None
    assert key_id not in _instance_schemas._entries
