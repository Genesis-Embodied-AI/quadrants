"""Inlined functions use arriving values regardless of parameter and return annotations."""

from dataclasses import dataclass
from typing import Protocol

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("annotation_kind", ["base", "protocol", "template", "untyped", "unrelated", "string"])
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("storage", ["field", "ndarray"])
@test_utils.test()
def test_dataclass_forwarding(annotation_kind, keyword, storage):
    @dataclass(frozen=True)
    class Particles:
        vel: qd.Tensor

    @dataclass(frozen=True)
    class FluidParticles(Particles):
        rho: qd.Tensor

    class HasVel(Protocol):
        vel: qd.Tensor

    def push(particles):
        particles.vel[0] += particles.rho[0]

    annotations = {
        "base": Particles,
        "protocol": HasVel,
        "template": qd.Template,
        "unrelated": int,
        "string": "TypeOnlyAvailableToTheTypeChecker",
    }
    if annotation_kind != "untyped":
        push.__annotations__["particles"] = annotations[annotation_kind]
    push = qd.func(push)

    @qd.func
    def forward(other: HasVel):
        if qd.static(keyword):
            push(particles=other)
        else:
            push(other)

    @qd.kernel
    def run(particles: FluidParticles):
        forward(particles)

    create = qd.field if storage == "field" else qd.ndarray
    vel = create(qd.f32, shape=1)
    rho = create(qd.f32, shape=1)
    rho[0] = 2
    run(FluidParticles(vel, rho))
    run(FluidParticles(vel, rho))
    assert vel[0] == 4


@test_utils.test()
def test_nested_dataclasses_and_bound_methods():
    @dataclass
    class State:
        value: str

    @dataclass
    class Wrapper:
        child: object

    @qd.func
    def read(state: "Ignored", bias: str = 3):
        return state.value + bias

    @qd.data_oriented
    class Reader:
        @qd.func
        def read(self: "IgnoredSelf", state: int):
            return read(state=state.child)

    reader = Reader()

    @qd.kernel
    def run(state: Wrapper) -> qd.i32:
        return reader.read(state)

    assert run(Wrapper(State(7))) == 10


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test()
def test_runtime_dataclass_specialization_and_pruning(fastcache):
    @dataclass
    class State:
        value: str
        unused: object

    @dataclass
    class OtherState:
        unused: object
        value: bool

    @qd.func
    def read(value: State) -> int:
        return value.value

    @qd.kernel(fastcache=fastcache)
    def run(value: qd.Template) -> qd.f32:
        return read(value)

    state = State(2, object())
    assert run(state) == 2
    count = len(run._primal.materialized_kernels)
    state.value = 5
    assert run(state) == 5
    assert len(run._primal.materialized_kernels) == count
    state.value = 2.5
    assert run(state) == 2.5
    assert len(run._primal.materialized_kernels) == count + 1
    assert run(OtherState(object(), 4.5)) == 4.5
    assert run(State(8, object())) == 8


@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc])
@test_utils.test()
def test_annotations_do_not_cast_or_require_returns(decorate):
    @decorate
    def identity(value: qd.i32) -> qd.i32:
        unused: "UnknownLocal"
        copied: qd.i32 = value
        result: "UnknownLocal" = copied
        return result

    @decorate
    def discard(value: "Missing") -> "AlsoMissing":
        pass

    @decorate
    def add(value: "Missing[", increment=0.5) -> tuple[int, ...]:
        return identity(value) + increment

    @qd.kernel
    def run(value: qd.f32) -> qd.f32:
        discard(value)
        return add(value)

    assert run(1.25) == 1.75


@test_utils.test()
def test_one_func_accepts_dataclasses_and_scalars():
    @dataclass
    class State:
        value: str

    @qd.func
    def read(value: State, is_container: bool):
        if qd.static(is_container):
            return value.value
        else:
            return value

    @qd.kernel
    def run(state: State, scalar: qd.f32) -> qd.f32:
        return read(state, True) + read(scalar, False) + read(state, True)

    assert run(State(2), 1.25) == 5.25
    assert run(State(3.5), 2.25) == 9.25


@pytest.mark.parametrize("annotation", [qd.Template, qd.i32, object, qd.ref(qd.i32)])
@test_utils.test()
def test_expression_parameters_are_local_values(annotation):
    def update(value, enabled=True):
        if enabled:
            value += 4
        return value

    update.__annotations__["value"] = annotation
    update = qd.func(update)

    @qd.kernel
    def run() -> qd.i32:
        value = 3
        result = update(value)
        return value * 100 + result * 10 + update(2)

    assert run() == 376

    @qd.kernel
    def conditional(enabled: qd.i32) -> qd.i32:
        return update(2, enabled)

    assert conditional(0) == 2
    assert conditional(1) == 6


@test_utils.test()
def test_compile_time_values_without_template_annotations():
    @qd.func
    def add(value):
        return value + 1

    @qd.func
    def apply(operation: int, count: str, config: float):
        count: "UnknownLocal"
        result = 0
        for i in qd.static(range(count)):
            result += operation(config[i])
        return result

    @qd.kernel
    def run() -> qd.i32:
        return apply(add, 2, (3, 5))

    assert run() == 10


@test_utils.test()
def test_missing_member_is_reported_at_use():
    @dataclass
    class State:
        present: int

    @qd.func
    def read(state: "HasMissing"):
        return state.missing

    @qd.kernel
    def run(state: State) -> qd.i32:
        return read(state)

    with pytest.raises(qd.QuadrantsCompilationError, match="missing"):
        run(State(1))


@pytest.mark.parametrize("call", ["duplicate", "missing", "unexpected"])
@test_utils.test()
def test_argument_binding_errors(call):
    @qd.func
    def add(left: "Ignored", right: "Ignored"):
        return left + right

    @qd.kernel
    def run() -> qd.i32:
        result = 0
        if qd.static(call == "duplicate"):
            result = add(1, left=2, right=3)
        elif qd.static(call == "missing"):
            result = add(1)
        else:
            result = add(left=1, right=2, extra=3)
        return result

    message = {"duplicate": "Multiple values", "missing": "Missing argument", "unexpected": "Unexpected argument"}
    with pytest.raises(qd.QuadrantsSyntaxError, match=message[call]):
        run()
