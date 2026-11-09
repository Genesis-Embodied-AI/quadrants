import dataclasses
import enum
import sys
import types
import typing

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


class Mode(enum.IntEnum):
    A = 1
    B = 2


def encode(value: typing.Any) -> int:
    """Expose the actual constant's type and value through a kernel result."""
    return len(type(value).__name__) * 100 + (len(value) if isinstance(value, str) else int(value))


@pytest.mark.parametrize(
    "annotation",
    [
        typing.Final,
        typing.Final[int],
        typing.Final[list],
        typing.Final[object],
        typing.Final["Missing"],
        "Final[Missing]",
    ],
)
@test_utils.test()
def test_final_uses_actual_type_and_value(annotation: typing.Any) -> None:
    @dataclasses.dataclass(frozen=True)
    class State:
        value: annotation

    @qd.func
    def inner(state: qd.RuntimeTyped):
        return qd.static(encode(state.value))

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return inner(state)

    values = [True, 1, 1.0, "one", Mode.A, np.int32(1), np.float32(1)]
    for value in values:
        assert run(State(value)) == encode(value)
    assert len(run._primal.materialized_kernels) == len(values)
    for value in values:
        assert run(State(value)) == encode(value)
    assert len(run._primal.materialized_kernels) == len(values)
    assert run(State(2)) == encode(2)
    assert len(run._primal.materialized_kernels) == len(values) + 1


@test_utils.test()
def test_final_unsupported_arriving_value_is_deferred() -> None:
    @dataclasses.dataclass(frozen=True)
    class State:
        value: typing.Final[int]
        good: typing.Final

    @qd.kernel
    def unused(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(state.good)

    @qd.kernel
    def used(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(state.value[0])

    state = State([7], 3)
    assert unused(state) == 3
    with pytest.raises(Exception, match="state.value.*not a supported compile-time constant"):
        used(state)


@test_utils.test()
def test_bare_final_requires_frozen_container_path() -> None:
    @dataclasses.dataclass(frozen=True)
    class Inner:
        value: typing.Final

    @dataclasses.dataclass
    class Outer:
        inner: object

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(state.inner.value)

    with pytest.raises(Exception, match="not frozen but reaches"):
        run(Outer(Inner(3)))


@test_utils.test(raise_on_templated_floats=True)
def test_final_float_policy_uses_arriving_value() -> None:
    @dataclasses.dataclass(frozen=True)
    class State:
        value: typing.Final[int]

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(int(state.value))

    assert run(State(3)) == 3
    with pytest.raises(ValueError, match="Floats not allowed"):
        run(State(1.5))


@test_utils.test()
def test_string_final_resolves_only_marker_in_defining_scope() -> None:
    module = types.ModuleType("_qd_runtime_final_annotation_test")
    sys.modules[module.__name__] = module
    try:
        # The unresolved inner annotation and the expression must never be evaluated. The child intentionally
        # shadows F so an inherited field must use its defining class's namespace.
        exec(
            "from __future__ import annotations\n"
            "import dataclasses\n"
            "from typing import Final as F\n"
            "def explode():\n    raise AssertionError('inner annotation evaluated')\n"
            "@dataclasses.dataclass(frozen=True)\n"
            "class Parent:\n    value: F[explode()]\n    other: OnlyUnderTypeChecking\n"
            "@dataclasses.dataclass(frozen=True)\n"
            "class Child(Parent):\n    F = int\n",
            module.__dict__,
        )

        @qd.kernel
        def run(state: qd.RuntimeTyped) -> qd.i32:
            return qd.static(state.value) + state.other

        assert run(module.Child(7, 2)) == 9
        assert run(module.Child(11, 3)) == 14
    finally:
        sys.modules.pop(module.__name__, None)


@test_utils.test()
def test_nonfinal_string_annotation_is_not_a_marker() -> None:
    @dataclasses.dataclass(frozen=True)
    class State:
        Final = int
        a: "Final"
        b: "NotFinal"
        c: "list[typing.Final[int]]"

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return state.a + state.b + state.c

    assert run(State(1, 2, 3)) == 6
    assert run(State(2, 3, 4)) == 9
    assert len(run._primal.materialized_kernels) == 1


@test_utils.test()
def test_string_alias_to_subscripted_final() -> None:
    @dataclasses.dataclass(frozen=True)
    class State:
        Constant = typing.Final[list]
        value: "Constant"

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(state.value)

    assert run(State(3)) == 3
    assert run(State(5)) == 5
    assert run(State(3)) == 3
    assert len(run._primal.materialized_kernels) == 2
