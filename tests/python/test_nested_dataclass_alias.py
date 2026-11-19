from dataclasses import dataclass

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(offline_cache=True)
def test_unused_nested_dataclass_alias_is_harmless() -> None:
    """Unaccessed aliased scalar members must not prevent compilation."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self) -> None:
            """Store an unused alias beside an independent dataclass."""
            self.first = State(3)
            self.second = self.first
            self.used = State(9)

    @qd.kernel(fastcache=True)
    def read(owner: qd.Template) -> qd.i32:
        """Read only the independent dataclass."""
        return owner.used.value

    owner = Owner()
    assert read(owner) == 9
    new_owner = Owner()
    new_owner.used.value = 15
    assert read(new_owner) == 15
    assert read._primal.src_ll_cache_observations.cache_loaded
    owner.used.value = 11
    assert read(owner) == 11


@test_utils.test(offline_cache=True)
def test_nested_dataclass_alias_uses_distinct_disk_cache_entry() -> None:
    """A cache entry compiled with distinct objects must not omit an aliased scalar argument."""

    @dataclass
    class State:
        value: int

    @qd.data_oriented
    class Owner:
        def __init__(self, aliased: bool) -> None:
            """Choose distinct or shared dataclasses without changing their inferred types."""
            self.first = State(3)
            self.second = self.first if aliased else State(7)

    @qd.kernel(fastcache=True)
    def read(owner: qd.Template, output: qd.types.ndarray()) -> None:
        """Read the path that an identity-only lookup can confuse with first."""
        output[0] = owner.second.value

    output = qd.ndarray(qd.i32, shape=1)
    original = Owner(False)
    read(original, output)
    assert output[0] == 7
    aliased = Owner(True)
    read(aliased, output)
    assert output[0] == 3
    assert not read._primal.src_ll_cache_observations.cache_loaded
