import gc
import weakref
from dataclasses import dataclass

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang._runtime_typing.dataclass_schemas import (
    _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path,
    cached_infer_dataclass_schema,
    infer_member_type,
)
from quadrants.lang._runtime_typing.runtime_dataclass_guards import (
    SchemaNeedingMemberChecks,
)

from tests.helpers import test_utils


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("unsafe_hash", [False, True])
@test_utils.test(arch=qd.cpu)
def test_schema_cache_does_not_modify_instances(slots: bool, unsafe_hash: bool) -> None:
    @dataclass(frozen=not unsafe_hash, unsafe_hash=unsafe_hash, slots=slots)
    class State:
        value: object

    state = State(1)
    before = dict(vars(state)) if not slots else None
    first = cached_infer_dataclass_schema(state, "state")
    assert first.__dataclass_params__.frozen is True
    assert first.__dataclass_params__.unsafe_hash is False
    assert cached_infer_dataclass_schema(state, "state") is first
    if slots:
        assert _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path.get(state) is None
    else:
        assert vars(state) == before
        assert _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path.get(state) is not None
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
    assert _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path.get(state) is not None
    key_id = id(state)
    ref = weakref.ref(state)
    del state, child, before
    gc.collect()
    assert ref() is None
    assert key_id not in _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path._entries


@test_utils.test(arch=qd.cpu)
def test_inferred_member_check_requirements() -> None:
    """Inference distinguishes immutable values, mutable contents, and fixed Tensor backing objects."""

    @dataclass(frozen=True)
    class Frozen:
        value: object

    @dataclass
    class Mutable:
        value: object

    @dataclass(unsafe_hash=True)
    class UnsafeHash:
        value: object

    tensor = qd.tensor(qd.f32, shape=2, backend=qd.Backend.NDARRAY)
    for py_obj, expected in (
        (None, False),
        (1, False),
        (1.0, False),
        (np.int32(1), False),
        (np.zeros(2), False),
        (tensor, False),
        (tensor._unwrap(), False),
        (qd.Vector([1, 2]), False),
        (qd.Struct(value=1), True),
        (Frozen(1), False),
        (UnsafeHash(1), False),
        (UnsafeHash(Mutable(1)), True),
        (Mutable(1), True),
        (Frozen(Mutable(1)), True),
        ([], True),
    ):
        _, needs_member_check = infer_member_type(py_obj, "state.value", frozenset())
        assert needs_member_check is expected, type(py_obj)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@test_utils.test(arch=qd.cpu)
def test_array_members_do_not_need_schema_checks(backend: str) -> None:
    """Array metadata changes preserve its schema type, including inside guarded containers."""

    @dataclass
    class Mutable:
        value: object

    @dataclass(frozen=True)
    class State:
        array: object
        other: object

    if backend == "torch":
        torch = pytest.importorskip("torch")
        array = torch.zeros(4)
    else:
        array = np.zeros(4, dtype=np.float32)
    _, needs_member_check = infer_member_type(array, "state.array", frozenset())
    assert not needs_member_check
    state = State(array, None)
    schema = cached_infer_dataclass_schema(state, "state")
    assert not schema._qd_needs_member_check
    if backend == "torch":
        array.resize_(2, 2)
    else:
        array.shape = (2, 2)
        array.dtype = np.int32
    assert cached_infer_dataclass_schema(state, "state") is schema

    guarded = State(array, Mutable(1))
    cached_infer_dataclass_schema(guarded, "state")
    entries = _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path.get(guarded)
    assert entries is not None
    entry = next(iter(entries.values()))
    assert isinstance(entry, SchemaNeedingMemberChecks)
    assert [name for name, _ in entry.member_guards] == ["other"]


@test_utils.test(arch=qd.cpu)
def test_matrix_schema_uses_recorded_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matrix inference avoids value reads, and schema reuse checks only mutable dataclass children."""

    @dataclass
    class Mutable:
        value: object

    @dataclass(frozen=True)
    class State:
        matrix: object
        other: object

    def reject_numpy_conversion(self: qd.Matrix) -> None:
        raise AssertionError("Schema inference must not read matrix values")

    monkeypatch.setattr(qd.Matrix, "to_numpy", reject_numpy_conversion)
    for matrix in (qd.Matrix([[1, 2]], dt=qd.i32), qd.Vector([1, 2], dt=qd.i32)):
        state = State(matrix, None)
        schema = cached_infer_dataclass_schema(state, "state")
        assert schema.__dataclass_fields__["matrix"].type.dtype == qd.i32
        assert not schema._qd_needs_member_check
        matrix.fill(3)
        assert cached_infer_dataclass_schema(state, "state") is schema

        guarded = State(matrix, Mutable(1))
        cached_infer_dataclass_schema(guarded, "state")
        entries = _schema_by_dataclass_instance_then_numeric_defaults_and_ancestor_path.get(guarded)
        assert entries is not None
        entry = next(iter(entries.values()))
        assert isinstance(entry, SchemaNeedingMemberChecks)
        assert [name for name, _ in entry.member_guards] == ["other"]
