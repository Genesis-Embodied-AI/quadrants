"""Mixed parameter sources retain their native slots and live values across launches."""

from dataclasses import dataclass

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test(arch=qd.cpu, offline_cache=True)
def test_mixed_parameter_sources(fastcache: bool) -> None:
    """Pruned fields and buffer-view components must not shift other parameters' launch values."""

    @dataclass
    class State:
        unused: qd.i32
        value: qd.i32

    @qd.data_oriented(template_primitives=False)
    class Owner:
        def __init__(self) -> None:
            self.value = 5
            self.array = qd.ndarray(qd.i32, shape=1)
            self.array[0] = 6

    @qd.kernel(fastcache=fastcache)
    def read(
        direct: qd.i32,
        state: State,
        view: qd.BufferView,
        vector: qd.types.vector(2, qd.i32),
        owner: qd.Template,
    ) -> qd.i32:
        return direct + state.value + view[0] + vector[1] + owner.value + owner.array[0]

    data = qd.ndarray(qd.i32, shape=3)
    data.from_numpy(np.array([90, 3, 8], dtype=np.int32))
    state = State(100, 2)
    owner = Owner()
    assert read(1, state, data[1:2], qd.Vector([90, 4]), owner) == 21
    state.value = 20
    owner.value = 50
    owner.array[0] = 60
    assert read(10, state, data[2:3], qd.Vector([90, 40]), owner) == 188
