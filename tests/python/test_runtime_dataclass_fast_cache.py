import typing
from dataclasses import dataclass

import pytest

import quadrants as qd
from quadrants.lang import _runtime_dataclass

from tests.helpers import test_utils


@test_utils.test()
def test_frozen_tensor_rebinding_invalidates_both_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass(frozen=True)
    class State:
        tensor: object

    @qd.kernel
    def read(state: State) -> qd.i32:
        return state.tensor[0]

    @qd.kernel
    def twice(state: State) -> qd.i32:
        return state.tensor[0] * 2

    first = qd.ndarray(qd.i32, shape=1)
    first[0] = 3
    wrapped = qd.Tensor(first)
    state = State(wrapped)
    assert read(state) == 3
    assert twice(state) == 6
    infer = _runtime_dataclass.infer_kernel_dataclass_arg_types

    def reject_inference(*args: object, **kwargs: object) -> typing.NoReturn:
        raise AssertionError("unchanged arguments should reuse inferred metadata")

    monkeypatch.setattr(_runtime_dataclass, "infer_kernel_dataclass_arg_types", reject_inference)
    assert read(state) == 3
    assert twice(state) == 6
    monkeypatch.setattr(_runtime_dataclass, "infer_kernel_dataclass_arg_types", infer)

    replacement = qd.ndarray(qd.i32, shape=1)
    replacement[0] = 9
    wrapped._impl = replacement
    assert read(state) == 9
    assert twice(state) == 18
    assert len(read._primal.materialized_kernels) == 1
    assert len(twice._primal.materialized_kernels) == 1

    field = qd.field(qd.i32, shape=1)
    field[0] = 5
    wrapped._impl = field
    assert read(state) == 5
    assert twice(state) == 10
    wrapped._impl = first
    assert read(state) == 3
    assert twice(state) == 6


@test_utils.test()
def test_non_weakrefable_dataclass_with_dict_rechecks_tensor() -> None:
    @dataclass(frozen=True)
    class State:
        __slots__ = ("tensor", "__dict__")
        tensor: object

    @qd.kernel
    def read(state: State) -> qd.i32:
        return state.tensor[0]

    first = qd.ndarray(qd.i32, shape=1)
    first[0] = 4
    wrapped = qd.Tensor(first)
    state = State(wrapped)
    assert read(state) == 4
    replacement = qd.ndarray(qd.i32, shape=1)
    replacement[0] = 7
    wrapped._impl = replacement
    assert read(state) == 7


@test_utils.test(arch=qd.cpu, require=qd.extension.data64)
def test_cached_inference_rechecks_numeric_defaults() -> None:
    @dataclass(frozen=True)
    class State:
        value: object

    @qd.kernel
    def read(state: State) -> qd.f64:
        return state.value

    state = State(1.0 + 2**-30)
    assert read(state) == 1.0
    qd.lang.impl.get_runtime().default_fp = qd.f64
    assert read(state) == state.value
    assert len(read._primal.materialized_kernels) == 2


@test_utils.test(require=qd.extension.data64)
def test_nested_frozen_tensor_rebinding_changes_inferred_dtype() -> None:
    @dataclass(frozen=True)
    class Inner:
        tensor: object

    @dataclass(frozen=True)
    class Outer:
        inner: object

    @qd.kernel
    def read(state: Outer) -> qd.f64:
        return state.inner.tensor[0]

    first = qd.ndarray(qd.i32, shape=1)
    first[0] = 3
    wrapped = qd.Tensor(first)
    inner = Inner(wrapped)
    state = Outer(inner)
    assert read(state) == 3
    assert read(state) == 3
    replacement = qd.ndarray(qd.f64, shape=1)
    replacement[0] = 1.25
    wrapped._impl = replacement
    assert read(state) == 1.25
    assert len(read._primal.materialized_kernels) == 2
    wrapped._impl = first
    assert read(state) == 3
    assert len(read._primal.materialized_kernels) == 2
