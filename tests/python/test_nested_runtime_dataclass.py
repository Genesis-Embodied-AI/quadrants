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
