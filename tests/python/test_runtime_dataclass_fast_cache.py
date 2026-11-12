import typing
from dataclasses import dataclass

import pytest

import quadrants as qd
from quadrants.lang._runtime_typing import runtime_dataclass as _runtime_dataclass

from tests.helpers import test_utils


@test_utils.test()
def test_frozen_tensor_reuses_metadata_in_both_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def reject_inference(*args: object, **kwargs: object) -> typing.NoReturn:
        raise AssertionError("unchanged arguments should reuse inferred metadata")

    monkeypatch.setattr(_runtime_dataclass, "infer_kernel_dataclass_arg_types", reject_inference)
    first[0] = 9
    assert read(state) == 9
    assert twice(state) == 18


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
