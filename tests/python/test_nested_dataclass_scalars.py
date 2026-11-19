"""Dataclass scalar parameters retain live values and compatible alias bindings inside data-oriented objects."""

from dataclasses import dataclass
from typing import Final

import pytest

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(offline_cache=False)
def test_nested_scalar_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared and separate containers use compatible kernels; warmed calls reuse the alias key."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class DataOriented:
        def __init__(self, shared: bool) -> None:
            self.first = State(3)
            self.second = self.first if shared else State(7)
            self.aliases = (self.second,)

    @qd.kernel
    def read(data_oriented_obj: qd.Template) -> int:
        return data_oriented_obj.first.value + data_oriented_obj.aliases[0].value

    for shared in (False, True, False, True):
        data_oriented_obj = DataOriented(shared)
        assert read(data_oriented_obj) == (6 if shared else 3 + 7)
        data_oriented_obj.second.value = 11
        count = len(read._primal.materialized_kernels)
        assert read(data_oriented_obj) == (22 if shared else 3 + 11)
        assert len(read._primal.materialized_kernels) == count

    def unexpected_walk(args: tuple[object, ...]) -> None:
        raise AssertionError("A warmed call must reuse its cached alias key")

    monkeypatch.setattr("quadrants.lang._container_aliases.get_container_alias_key", unexpected_walk)
    assert read(data_oriented_obj) == 22


@pytest.mark.parametrize("in_data_oriented", [False, True])
@test_utils.test(require=qd.extension.data64)
def test_dataclass_preserves_i64(in_data_oriented: bool) -> None:
    """A dataclass scalar keeps its declared 64-bit width through either argument route."""

    @dataclass
    class State:
        value: qd.i64

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state

    argument_type = qd.Template if in_data_oriented else State

    @qd.kernel
    def read(arg: argument_type) -> qd.i64:
        if qd.static(in_data_oriented):
            return arg.state.value
        return arg.value

    state = State(2**33)
    assert read(DataOriented(state) if in_data_oriented else state) == 2**33


@pytest.mark.parametrize("in_data_oriented", [False, True])
@test_utils.test()
def test_dataclass_scalar_static_constants(in_data_oriented: bool) -> None:
    """Final and Template members remain compile-time constants through either argument route."""

    @dataclass(frozen=True)
    class State:
        count: Final[int]
        enabled: qd.Template
        value: float

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state
            self.offset = 4

    argument_type = qd.Template if in_data_oriented else State

    @qd.kernel
    def read(arg: argument_type) -> qd.f32:
        result = 0.0
        if qd.static(in_data_oriented):
            if qd.static(arg.state.enabled):
                for _ in qd.static(range(arg.state.count)):
                    result += arg.state.value
            return result + qd.static(arg.offset)
        if qd.static(arg.enabled):
            for _ in qd.static(range(arg.count)):
                result += arg.value
        return result + 4

    state = State(3, True, 2.5)
    assert read(DataOriented(state) if in_data_oriented else state) == 11.5


@pytest.mark.parametrize("in_data_oriented", [False, True])
@test_utils.test()
def test_dataclass_runtime_scalar_rejects_static(in_data_oriented: bool) -> None:
    """A runtime dataclass scalar cannot control a static expression."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state

    argument_type = qd.Template if in_data_oriented else State

    @qd.kernel
    def read(arg: argument_type) -> qd.i32:
        if qd.static(in_data_oriented):
            return qd.static(arg.state.value)
        return qd.static(arg.value)

    state = State(3)
    expected_error = qd.QuadrantsSyntaxError if in_data_oriented else ValueError
    with pytest.raises(expected_error, match=r"qd\.static"):
        read(DataOriented(state) if in_data_oriented else state)


@test_utils.test()
def test_data_oriented_shared_child_keeps_runtime_primitive_opt_in() -> None:
    """Data-oriented objects sharing a child must each respect their template_primitives setting."""

    @qd.data_oriented
    class Child:
        def __init__(self) -> None:
            self.value = 3

    @qd.data_oriented
    class DefaultDataOriented:
        def __init__(self, child: Child) -> None:
            self.child = child

    @qd.data_oriented(template_primitives=False)
    class RuntimeDataOriented:
        def __init__(self, child: Child) -> None:
            self.child = child

    @qd.kernel
    def read(default: qd.Template, runtime: qd.Template) -> qd.i32:
        return runtime.child.value

    child = Child()
    default, runtime = DefaultDataOriented(child), RuntimeDataOriented(child)
    assert read(default, runtime) == 3
    child.value = 7
    assert read(default, runtime) == 7


def test_alias_cache_reuses_graphs_and_releases_data_oriented_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache hits skip discovery; new argument combinations reuse graphs without retaining the data-oriented objects."""
    import gc
    import weakref

    from quadrants.lang import _container_aliases as aliases

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state
            self.cycle = self

    state = State(3)
    first, second = DataOriented(state), DataOriented(state)
    args = (first, second, 1)
    key = aliases.cached_get_container_alias_key(args)

    def unexpected_walk(py_obj: object) -> None:
        raise AssertionError("A cached container graph must not be walked again")

    monkeypatch.setattr(aliases, "_collect_container_ids", unexpected_walk)
    assert aliases.cached_get_container_alias_key(args) == key
    state.value = 9
    for scalar in range(140):
        changed_args = (first, second, scalar)
        assert aliases.cached_get_container_alias_key(changed_args) == key
    first_ref = weakref.ref(first)
    del args, changed_args, first, second
    gc.collect()
    assert first_ref() is None


@pytest.mark.parametrize("shared", [False, True])
@test_utils.test(offline_cache=False, require=qd.extension.data64)
def test_direct_dataclasses_and_dataclasses_in_data_oriented_share_binding(shared: bool) -> None:
    """Direct dataclass arguments and dataclasses inside data-oriented objects share scalar and array bindings."""

    @dataclass
    class Child:
        count: qd.i64
        array: qd.Tensor

    @dataclass
    class State:
        unused: int
        child: Child

    @qd.data_oriented(template_primitives=False)
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state
            self.offset = 2

    @qd.kernel
    def read(first: State, data_oriented_obj: qd.Template, last: qd.i64) -> qd.i64:
        return (
            first.child.count
            + data_oriented_obj.state.child.count
            + data_oriented_obj.state.child.array[0]
            + data_oriented_obj.offset
            + last
        )

    array = qd.ndarray(qd.i64, shape=1)
    array[0] = 5
    first = State(0, Child(2**33, array))
    data_oriented_obj = DataOriented(State(1, first.child if shared else Child(7, array)))
    for count in (9, 11):
        data_oriented_obj.state.child.count = count
        assert read(first, data_oriented_obj, 13) == first.child.count + count + 20
        data_oriented_obj.offset += 1
        array[0] -= 1


@pytest.mark.parametrize("in_data_oriented", [False, True])
@pytest.mark.parametrize("frozen,unsafe_hash", [(False, False), (True, False), (False, True)])
@test_utils.test()
def test_dataclass_launch_cache(in_data_oriented: bool, frozen: bool, unsafe_hash: bool) -> None:
    """Mutable dataclasses read live scalars; frozen and unsafe_hash dataclasses reuse cached launches."""

    @dataclass(frozen=frozen, unsafe_hash=unsafe_hash)
    class State:
        array: qd.Tensor
        count: int

    @qd.data_oriented
    class DataOriented:
        def __init__(self, state: State) -> None:
            self.state = state

    argument_type = qd.Template if in_data_oriented else State

    @qd.kernel
    def read(arg: argument_type) -> qd.i32:
        if qd.static(in_data_oriented):
            return arg.state.array[0] + arg.state.count
        return arg.array[0] + arg.count

    state = State(qd.ndarray(qd.i32, shape=1), 3)
    arg = DataOriented(state) if in_data_oriented else state
    state.array[0] = 2
    assert read(arg) == 5
    immutable = frozen or unsafe_hash
    assert bool(read._primal.launch_context_buffer_cache._launch_ctx_cache) is immutable
    state.array[0] = 7
    if not immutable:
        state.count = 9
    assert read(arg) == (10 if immutable else 16)
