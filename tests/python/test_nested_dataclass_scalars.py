"""Dataclass scalar parameters retain live values and compatible alias bindings beneath data-oriented owners."""

from dataclasses import dataclass
from typing import Final

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test(offline_cache=True, require=qd.extension.data64)
def test_nested_scalar_aliases(fastcache: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared and separate containers use compatible kernels; warmed calls reuse the alias key."""

    @dataclass
    class State:
        value: qd.i64

    @qd.data_oriented
    class Owner:
        def __init__(self, shared: bool) -> None:
            self.first = State(2**33)
            self.second = self.first if shared else State(7)
            self.aliases = (self.second,)

    @qd.kernel(fastcache=fastcache)
    def read(owner: qd.Template) -> qd.i64:
        return owner.first.value + owner.aliases[0].value

    for shared in (False, True, False, True):
        owner = Owner(shared)
        assert read(owner) == (2**34 if shared else 2**33 + 7)
        owner.second.value = 11
        count = len(read._primal.materialized_kernels)
        assert read(owner) == (22 if shared else 2**33 + 11)
        assert len(read._primal.materialized_kernels) == count

    def unexpected_walk(args: tuple[object, ...]) -> None:
        raise AssertionError("A warmed call must reuse its cached alias key")

    monkeypatch.setattr("quadrants.lang._container_aliases.get_container_alias_key", unexpected_walk)
    assert read(owner) == 22


@test_utils.test(offline_cache=True)
def test_nested_scalar_disk_cache() -> None:
    """Disk reuse distinguishes alias layouts but accepts new scalar values in the same layout."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self, shared: bool, value: int) -> None:
            self.first = State(value)
            self.second = self.first if shared else State(value + 1)

    @qd.kernel(fastcache=True)
    def read(owner: qd.Template) -> qd.i32:
        return owner.second.value

    assert read(Owner(False, 2)) == 3
    assert read(Owner(True, 2)) == 2
    assert not read._primal.src_ll_cache_observations.cache_loaded
    assert read(Owner(True, 8)) == 8
    assert read._primal.src_ll_cache_observations.cache_loaded


@test_utils.test(offline_cache=True)
def test_nested_scalar_cross_argument_alias() -> None:
    """Member bindings shared between two owners survive disk reuse and live scalar updates."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state

    @qd.kernel(fastcache=True)
    def read(first: qd.Template, second: qd.Template) -> qd.i32:
        return first.state.value + second.state.value

    shared = State(3)
    first, second = Owner(shared), Owner(shared)
    assert read(first, second) == 6
    shared.value = 9
    assert read(first, second) == 18
    assert read(Owner(State(2)), Owner(State(7))) == 9
    assert not read._primal.src_ll_cache_observations.cache_loaded


@test_utils.test()
def test_nested_scalar_static_constants() -> None:
    """Final and Template members remain constants; ordinary owner scalars keep their existing policy."""

    @dataclass(frozen=True)
    class State:
        count: Final[int]
        enabled: qd.Template
        value: float

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(3, True, 2.5)
            self.offset = 4

    @qd.kernel
    def read(owner: qd.Template) -> qd.f32:
        result = 0.0
        if qd.static(owner.state.enabled):
            for _ in qd.static(range(owner.state.count)):
                result += owner.state.value
        return result + qd.static(owner.offset)

    assert read(Owner()) == 11.5


@test_utils.test()
def test_nested_runtime_scalar_rejects_static() -> None:
    """A runtime dataclass scalar cannot control a static expression."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(3)

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return qd.static(owner.state.value)

    with pytest.raises(qd.QuadrantsSyntaxError, match="runtime kernel argument"):
        read(Owner())


@test_utils.test()
def test_shared_owner_keeps_runtime_primitive_opt_in() -> None:
    """A default owner visited first must not hide the runtime opt-in of another owner sharing its child."""

    @qd.data_oriented
    class Child:
        def __init__(self) -> None:
            self.value = 3

    @qd.data_oriented
    class DefaultOwner:
        def __init__(self, child: Child) -> None:
            self.child = child

    @qd.data_oriented(template_primitives=False)
    class RuntimeOwner:
        def __init__(self, child: Child) -> None:
            self.child = child

    @qd.kernel
    def read(default: qd.Template, runtime: qd.Template) -> qd.i32:
        return runtime.child.value

    child = Child()
    default, runtime = DefaultOwner(child), RuntimeOwner(child)
    assert read(default, runtime) == 3
    child.value = 7
    assert read(default, runtime) == 7


def test_alias_cache_reuses_graphs_and_releases_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache hits skip discovery; new argument combinations reuse graphs without retaining their owners."""
    import gc
    import weakref

    from quadrants.lang import _container_aliases as aliases

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self, state: State) -> None:
            self.state = state
            self.cycle = self

    state = State(3)
    first, second = Owner(state), Owner(state)
    args = (first, second, 1)
    cache: aliases.AliasKeyCache = {}
    key = aliases.get_cached_container_alias_key(args, tuple(map(id, args)), cache)

    def unexpected_walk(py_obj: object) -> None:
        raise AssertionError("A cached container graph must not be walked again")

    monkeypatch.setattr(aliases, "_collect_container_ids", unexpected_walk)
    assert aliases.get_cached_container_alias_key(args, tuple(map(id, args)), cache) == key
    state.value = 9
    for scalar in range(140):
        changed_args = (first, second, scalar)
        assert aliases.get_cached_container_alias_key(changed_args, tuple(map(id, changed_args)), cache) == key
    assert len(cache) == 128
    first_ref = weakref.ref(first)
    del args, changed_args, first, second
    gc.collect()
    assert first_ref() is None
    assert not cache
