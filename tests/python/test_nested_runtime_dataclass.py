from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

import quadrants as qd
from quadrants.lang._fast_caching import args_hasher
from quadrants.lang.kernel_arguments import ArgMetadata

from tests.helpers import test_utils


@test_utils.test()
def test_nested_dataclass_hash_ignores_declared_child_type() -> None:
    """An unused annotation must not invent a nested Final constraint in the disk-cache key."""

    @dataclass(frozen=True)
    class Declared:
        value: Final = 3

    @dataclass
    class State:
        value: Declared

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(7)

    result = args_hasher.hash_args(False, [Owner()], [ArgMetadata(qd.Template, "owner")])
    assert isinstance(result, str)


@pytest.mark.parametrize("template_primitives", [True, False])
@test_utils.test()
def test_nested_dataclass_scalar_changes_are_runtime(template_primitives: bool) -> None:
    """Dataclass scalar values remain runtime inputs regardless of their containing object's policy."""

    @dataclass
    class State:
        value: str

    @qd.data_oriented(template_primitives=template_primitives)
    class Owner:
        def __init__(self) -> None:
            self.state = State(3)

    @qd.kernel
    def read(owner: qd.Template) -> qd.f32:
        return owner.state.value

    owner = Owner()
    assert read(owner) == 3
    compiled = len(read._primal.materialized_kernels)
    owner.state.value = 8
    assert read(owner) == 8
    assert len(read._primal.materialized_kernels) == compiled
    owner.state.value = 2.5
    assert read(owner) == 2.5
    assert len(read._primal.materialized_kernels) == compiled + 1


@test_utils.test()
def test_nested_dataclass_unused_invalid_final() -> None:
    """Unused invalid Final members are deferred even beneath a data-oriented object."""

    @dataclass
    class State:
        value: str
        unused: Final

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(4, {1, 2})

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.state.value

    assert read(Owner()) == 4


@test_utils.test()
def test_nested_dataclass_final_replacement_changes_specialization() -> None:
    """Replacing a frozen dataclass updates baked Final values inside its data-oriented owner."""

    @dataclass(frozen=True)
    class State:
        bias: Final[str]
        value: str

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(2, 3)

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return qd.static(owner.state.bias) + owner.state.value

    owner = Owner()
    assert read(owner) == 5
    owner.state = State(7, 4)
    assert read(owner) == 11


@test_utils.test()
def test_nested_dataclass_used_invalid_member_reports_path() -> None:
    """Used unsupported members fail with their kernel argument path."""

    @dataclass(frozen=True)
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State({1, 2})

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return len(owner.state.value)

    with pytest.raises(Exception, match=r"owner\.state\.value"):
        read(Owner())


@test_utils.test()
def test_nested_dataclass_offline_cache(tmp_path: Path) -> None:
    """Disk-cache reuse must bind current nested scalar values and retain Final specialization."""

    @dataclass(frozen=True)
    class State:
        bias: Final[str]
        value: str

    @qd.data_oriented
    class Owner:
        def __init__(self, bias: int, value: float) -> None:
            self.state = State(bias, value)

    arch = qd.lang.impl.current_cfg().arch
    for bias, value in ((2, 1.5), (2, 4.5), (7, 4.5), (2, 8.5)):
        qd.init(arch=arch, offline_cache=True, offline_cache_file_path=str(tmp_path))

        @qd.kernel(fastcache=True)
        def run(owner: qd.Template, output: qd.types.ndarray()):
            output[0] = qd.static(owner.state.bias) + owner.state.value

        output = qd.ndarray(qd.f32, shape=1)
        run(Owner(bias, value), output)
        assert output[0] == bias + value
        if value in (4.5, 8.5) and bias == 2:
            assert run._primal.src_ll_cache_observations.cache_loaded


@test_utils.test()
def test_nested_dataclass_final_keeps_mutable_ancestor_constraint() -> None:
    """A frozen child's Final field remains invalid beneath a mutable dataclass."""

    @dataclass(frozen=True)
    class Child:
        value: Final = 3

    @dataclass
    class Parent:
        child: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = Parent(Child())

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.state.child.value

    with pytest.raises(Exception, match=r"owner\.state\.child\.value"):
        read(Owner())


@test_utils.test()
def test_nested_dataclass_forwarded_to_func() -> None:
    """An inlined function receives live members from a nested dataclass."""

    @dataclass
    class State:
        value: str

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(3)

    @qd.func
    def get_value(state: qd.RuntimeTyped):
        return state.value

    @qd.kernel
    def read(owner: qd.Template) -> qd.f32:
        return get_value(owner.state)

    owner = Owner()
    assert read(owner) == 3
    owner.state.value = 7
    assert read(owner) == 7
    owner.state.value = 2.5
    assert read(owner) == 2.5


@test_utils.test()
def test_nested_dataclass_tensor_replacement() -> None:
    """Replacing a nested tensor with a different dtype changes the specialization."""

    @dataclass
    class State:
        value: str

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(qd.tensor(qd.i32, shape=1, backend=qd.Backend.NDARRAY))

    @qd.kernel
    def read(owner: qd.Template) -> qd.f32:
        return owner.state.value[0]

    owner = Owner()
    owner.state.value[0] = 3
    assert read(owner) == 3
    owner.state.value = qd.tensor(qd.f32, shape=1, backend=qd.Backend.NDARRAY)
    owner.state.value[0] = 2.5
    assert read(owner) == 2.5


@test_utils.test()
def test_nested_dataclass_preserves_data_oriented_child_policy() -> None:
    """A dataclass's data-oriented child keeps its own compile-time primitive policy."""

    @qd.data_oriented
    class Options:
        def __init__(self) -> None:
            self.count = 3

    @dataclass
    class State:
        options: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(Options())

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return qd.static(owner.state.options.count)

    assert read(Owner()) == 3


@pytest.mark.parametrize("dataclass_root", [True, False])
@test_utils.test()
def test_alternating_dataclass_and_data_oriented_nesting(dataclass_root: bool) -> None:
    """Dataclass inference continues through intervening data-oriented objects."""

    @dataclass
    class Leaf:
        value: str

    @qd.data_oriented
    class Middle:
        def __init__(self) -> None:
            self.leaf = Leaf(3)

    @dataclass
    class State:
        middle: str

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.state = State(Middle())

    @qd.kernel
    def read_owner(owner: qd.Template) -> qd.f32:
        return owner.state.middle.leaf.value

    @qd.kernel
    def read_state(state: qd.RuntimeTyped) -> qd.f32:
        return state.middle.leaf.value

    owner = Owner()
    read, argument = (read_state, owner.state) if dataclass_root else (read_owner, owner)
    assert read(argument) == 3
    owner.state.middle.leaf.value = 7
    assert read(argument) == 7
    owner.state.middle.leaf.value = 2.5
    assert read(argument) == 2.5


@pytest.mark.parametrize("alias_stored_member", [False, True])
@test_utils.test()
def test_nested_dataclass_property_is_rejected(alias_stored_member: bool) -> None:
    """A property is unsupported even when it aliases a discovered stored dataclass."""

    @dataclass
    class State:
        value: int = 3

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.stored = State()

        @property
        def state(self) -> State:
            return self.stored if alias_stored_member else State()

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.state.value

    with pytest.raises(Exception, match=r"Owner\.state.*not discovered"):
        read(Owner())


@pytest.mark.parametrize("template_primitives", [True, False])
@test_utils.test()
def test_dataclass_added_after_discovery_is_rejected(template_primitives: bool) -> None:
    """Compiling a new kernel must not silently bake a dataclass absent from cached discovery paths."""

    @dataclass
    class State:
        value: int = 3

    @qd.data_oriented(template_primitives=template_primitives)
    class Owner:
        def __init__(self) -> None:
            self.value = 1

    @qd.kernel
    def initial(owner: qd.Template) -> qd.i32:
        return owner.value

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.added.value

    owner = Owner()
    assert initial(owner) == 1
    owner.added = State()
    with pytest.raises(Exception, match=r"Owner\.added.*not discovered"):
        read(owner)


@test_utils.test()
def test_unused_dataclass_property_is_not_evaluated() -> None:
    """Property detection must not execute unused getters during discovery."""

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.value = 3

        @property
        def state(self) -> object:
            raise AssertionError("Unused getter executed")

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.value

    assert read(Owner()) == 3


@test_utils.test()
def test_dataclass_discovery_guard_allows_data_oriented_dataclass_fields() -> None:
    """A class with both decorators still exposes its declared dataclass fields."""

    @dataclass
    class Leaf:
        value: int

    @qd.data_oriented
    @dataclass
    class Middle:
        child: Leaf

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            self.middle = Middle(Leaf(3))

    @qd.kernel
    def read(owner: qd.Template) -> qd.i32:
        return owner.middle.child.value

    owner = Owner()
    assert read(owner) == 3
    owner.middle.child.value = 7
    assert read(owner) == 7
