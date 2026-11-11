from dataclasses import dataclass
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
