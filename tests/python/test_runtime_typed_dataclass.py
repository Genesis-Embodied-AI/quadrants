from dataclasses import dataclass
from typing import Final

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("annotation_kind", ["marker", "concrete", "unrelated", "scalar", "template", "ndarray"])
@test_utils.test()
def test_runtime_members_ignore_annotations(annotation_kind):
    @dataclass
    class State:
        values: str
        increment: str

    @dataclass
    class Unrelated:
        missing: int

    annotation = {
        "marker": qd.RuntimeTyped,
        "concrete": State,
        "unrelated": Unrelated,
        "scalar": qd.f32,
        "template": qd.Template,
        "ndarray": qd.types.ndarray(),
    }[annotation_kind]

    @qd.func
    def amount(state: annotation):
        return state.increment

    @qd.kernel
    def advance(state: annotation):
        state.values[0] += amount(state=state)

    values = qd.ndarray(qd.f32, shape=1)
    state = State(values, 1.25)
    advance(state)
    assert values[0] == 1.25
    compiled = len(advance._primal.materialized_kernels)
    state.increment = 2.5
    advance(state)
    assert values[0] == 3.75
    assert len(advance._primal.materialized_kernels) == compiled
    state.increment = 3
    advance(state)
    assert values[0] == 6.75
    assert len(advance._primal.materialized_kernels) == compiled + 1
    state.increment = 1.25
    advance(state)
    assert values[0] == 8
    assert len(advance._primal.materialized_kernels) == compiled + 1


@test_utils.test()
def test_runtime_nested_func_calls_and_pruning():
    @dataclass
    class Inner:
        value: object
        unused: object

    @dataclass(frozen=True)
    class Outer:
        inner: object

    @qd.func
    def read(inner: qd.RuntimeTyped):
        return inner.value

    @qd.func
    def forward(outer: qd.RuntimeTyped):
        return read(inner=outer.inner) + read(outer.inner)

    @qd.kernel
    def run(outer: qd.RuntimeTyped) -> qd.f32:
        return forward(outer)

    inner = Inner(1.25, qd.ndarray(qd.i32, shape=1))
    outer = Outer(inner)
    assert run(outer) == 2.5
    inner.value = 3.5
    assert run(outer) == 7
    inner.value = 4
    assert run(outer) == 8


@test_utils.test()
def test_runtime_unrelated_dataclass_classes():
    @dataclass
    class First:
        value: object

    @dataclass(slots=True)
    class Second:
        value: float

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.f32:
        return state.value

    assert read(First(1.5)) == 1.5
    assert read(Second(2.5)) == 2.5
    assert read(First(3.5)) == 3.5


@test_utils.test()
def test_runtime_array_dtype_and_rank_changes():
    @dataclass
    class State:
        array: object

    @qd.kernel
    def fill(state: qd.RuntimeTyped):
        for i in qd.grouped(state.array):
            state.array[i] = 7

    state = State(qd.ndarray(qd.i32, shape=2))
    first = state.array
    fill(state)
    np.testing.assert_array_equal(first.to_numpy(), [7, 7])
    state.array = qd.ndarray(qd.f32, shape=(2, 3))
    fill(state)
    np.testing.assert_array_equal(state.array.to_numpy(), np.full((2, 3), 7))
    first.fill(0)
    state.array = first
    fill(state)
    np.testing.assert_array_equal(first.to_numpy(), [7, 7])


@test_utils.test()
def test_runtime_external_array_and_field():
    @dataclass
    class State:
        array: object

    @qd.kernel
    def write(state: qd.RuntimeTyped):
        state.array[0] = 9

    array = np.zeros(2, dtype=np.float32)
    write(State(array))
    assert array[0] == 9
    field = qd.field(qd.i32, shape=2)
    write(State(field))
    assert field[0] == 9


@test_utils.test()
def test_runtime_final_through_nested_func():
    @dataclass(frozen=True)
    class Config:
        enabled: Final[bool]
        scale: object

    @dataclass(frozen=True)
    class State:
        config: object

    @qd.func
    def apply(config: qd.RuntimeTyped):
        result = 0.0
        if qd.static(config.enabled):
            result = config.scale
        return result

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.f32:
        return apply(state.config)

    assert run(State(Config(True, 2.5))) == 2.5
    assert run(State(Config(False, 2.5))) == 0
    assert run(State(Config(True, 3.5))) == 3.5


@test_utils.test(require=qd.extension.data64, default_ip=qd.i64, default_fp=qd.f64)
def test_runtime_scalar_precision():
    @dataclass
    class State:
        value: qd.i32

    @qd.kernel
    def integer(state: qd.RuntimeTyped) -> qd.i64:
        return state.value

    @qd.kernel
    def real(state: qd.RuntimeTyped) -> qd.f64:
        return state.value

    assert integer(State(2**40)) == 2**40
    assert integer(State(np.int64(2**40))) == 2**40
    assert real(State(1.0 + 2**-40)) == 1.0 + 2**-40
    assert real(State(np.float64(1.0 + 2**-40))) == 1.0 + 2**-40


@test_utils.test()
def test_runtime_bad_inputs():
    @dataclass
    class State:
        value: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return state.value

    with pytest.raises(Exception, match="requires a Python dataclass instance"):
        read(1)
    with pytest.raises(Exception, match="state.value.*unsupported runtime type"):
        read(State(object()))
    cyclic = State(None)
    cyclic.value = cyclic
    with pytest.raises(Exception, match="state.value.*cycle"):
        read(cyclic)


@test_utils.test(require=qd.extension.data64)
def test_runtime_matrix_and_quadrants_struct_members():
    @qd.dataclass
    class Pair:
        count: qd.i64
        weight: qd.f64

    @dataclass
    class State:
        vector: object
        pair: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.f64:
        return state.vector[1] + state.pair.weight + state.pair.count

    state = State(qd.Vector([1, 2], dt=qd.i32), Pair(count=2**40, weight=0.5))
    assert read(state) == 2**40 + 2.5

    @qd.kernel
    def make_pair() -> Pair:
        return Pair(count=qd.i64(1) << 40, weight=qd.f64(0.5))

    state.pair = make_pair()
    assert read(state) == 2**40 + 2.5


@test_utils.test()
def test_runtime_bool_members():
    @dataclass
    class State:
        enabled: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return qd.cast(state.enabled, qd.i32)

    for value in (True, False, np.bool_(True), np.bool_(False)):
        assert read(State(value)) == int(value)


@test_utils.test()
def test_runtime_offline_cache(tmp_path):
    @dataclass(frozen=True)
    class Config:
        bias: Final[int]
        value: object

    @dataclass(frozen=True)
    class State:
        config: object
        output: object
        unused: object

    arch = qd.lang.impl.current_cfg().arch
    for bias, value in ((3, 1.25), (3, 2.5), (7, 2.5), (7, 2), (3, 1.25)):
        qd.init(arch=arch, offline_cache=True, offline_cache_file_path=str(tmp_path))

        @qd.func
        def calculate(config: qd.RuntimeTyped):
            return qd.static(config.bias) + config.value

        @qd.kernel(fastcache=True)
        def run(state: qd.RuntimeTyped):
            state.output[0] = calculate(state.config)

        output = qd.ndarray(qd.f32, shape=1)
        run(State(Config(bias, value), output, qd.ndarray(qd.i32, shape=2)))
        assert output[0] == bias + value


@test_utils.test()
def test_runtime_unresolved_member_annotations():
    @dataclass
    class Inner:
        value: "NotImported"

    @dataclass
    class Outer:
        inner: "AlsoNotImported"

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.f32:
        return state.inner.value

    assert read(Outer(Inner(2.5))) == 2.5


@test_utils.test()
def test_runtime_optional_member():
    @dataclass
    class State:
        value: object

    @qd.func
    def from_func(state: qd.RuntimeTyped):
        result = 7
        if qd.static(state.value is not None):
            result = state.value[0]
        return result

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        result = 7
        if qd.static(state.value is not None):
            result = state.value[0]
        return result + from_func(state)

    state = State(None)
    assert read(state) == 14
    state.value = qd.ndarray(qd.i32, shape=1)
    state.value[0] = 9
    assert read(state) == 18
    state.value = None
    assert read(state) == 14


@test_utils.test()
def test_runtime_dataclass_then_scalar_restores_annotation():
    @dataclass
    class State:
        value: int

    @qd.kernel
    def run(value: qd.Template, is_container: qd.Template) -> qd.f32:
        if qd.static(is_container):
            return value.value
        else:
            return value

    assert run(2, False) == 2
    assert run(State(3.5), True) == 3.5
    assert run(4, False) == 4
    assert run(State(5.5), True) == 5.5


@test_utils.test()
def test_runtime_subclass_fields_visible_through_base_annotation():
    @dataclass
    class Base:
        value: int

    @dataclass
    class Derived(Base):
        extra: str

    @qd.func
    def read(state: Base):
        return state.value + state.extra

    @qd.kernel
    def run(state: Base) -> qd.f32:
        return read(state)

    assert run(Derived(2, 3.5)) == 5.5


@test_utils.test()
def test_runtime_frozen_wrapper_backing_and_nested_scalar_changes():
    @dataclass
    class Inner:
        increment: object

    @dataclass(frozen=True)
    class State:
        array: object
        inner: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.f32:
        return state.array[0] + state.inner.increment

    array = qd.ndarray(qd.f32, shape=1)
    array[0] = 2
    field = qd.field(qd.i32, shape=1)
    field[0] = 7
    replacement_field = qd.field(qd.i32, shape=1)
    replacement_field[0] = 11
    wrapped = qd.Tensor(array)
    inner = Inner(1.5)
    state = State(wrapped, inner)
    assert read(state) == 3.5
    compiled = len(read._primal.materialized_kernels)
    inner.increment = 2.5
    assert read(state) == 4.5
    assert len(read._primal.materialized_kernels) == compiled
    wrapped._impl = field
    assert read(state) == 9.5
    wrapped._impl = replacement_field
    assert read(state) == 13.5
    inner.increment = 3
    assert read(state) == 14
    wrapped._impl = array
    inner.increment = 1.5
    assert read(state) == 3.5
    assert len(read._primal.materialized_kernels) == compiled + 3


@pytest.mark.parametrize("slots", [False, True])
@test_utils.test(require=qd.extension.data64)
def test_runtime_frozen_nested_wrapper_dtype_rank_and_replacement(slots):
    @dataclass(frozen=True, slots=slots)
    class Inner:
        array: object

    @dataclass(frozen=True, slots=slots)
    class Outer:
        inner: object

    @qd.kernel
    def fill(state: qd.RuntimeTyped):
        for i in qd.grouped(state.inner.array):
            state.inner.array[i] = 3

    first = qd.ndarray(qd.i32, shape=2)
    wrapped = qd.Tensor(first)
    state = Outer(Inner(wrapped))
    fill(state)
    np.testing.assert_array_equal(first.to_numpy(), [3, 3])
    compiled = len(fill._primal.materialized_kernels)
    compatible = qd.ndarray(qd.i32, shape=4)
    wrapped._impl = compatible
    fill(state)
    np.testing.assert_array_equal(compatible.to_numpy(), [3] * 4)
    assert len(fill._primal.materialized_kernels) == compiled
    wrapped._impl = qd.ndarray(qd.f64, shape=(2, 3))
    fill(state)
    np.testing.assert_array_equal(wrapped.to_numpy(), np.full((2, 3), 3))
    assert len(fill._primal.materialized_kernels) == compiled + 1
    first.fill(0)
    wrapped._impl = first
    fill(state)
    np.testing.assert_array_equal(first.to_numpy(), [3, 3])
    assert len(fill._primal.materialized_kernels) == compiled + 1


@test_utils.test()
def test_runtime_frozen_guard_detects_mutable_child_cycle_and_bad_type():
    @dataclass
    class Inner:
        value: object

    @dataclass(frozen=True)
    class Outer:
        inner: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.f32:
        return state.inner.value

    child = Inner(1.5)
    state = Outer(child)
    assert read(state) == 1.5
    child.value = state
    with pytest.raises(qd.QuadrantsRuntimeTypeError, match="cycle"):
        read(state)
    child.value = []
    with pytest.raises(qd.QuadrantsRuntimeTypeError, match="state.inner.value.*unsupported runtime type list"):
        read(state)
    child.value = 2
    assert read(state) == 2


@test_utils.test()
def test_runtime_frozen_guards_keep_final_value_validation():
    @dataclass(frozen=True)
    class State:
        array: object
        bias: Final[int]

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return state.array[0] + state.bias

    array = qd.ndarray(qd.i32, shape=1)
    array[0] = 2
    state = State(qd.Tensor(array), 3)
    assert read(state) == 5
    compiled = len(read._primal.materialized_kernels)
    object.__setattr__(state, "bias", 5)
    assert read(state) == 7
    assert len(read._primal.materialized_kernels) == compiled + 1
