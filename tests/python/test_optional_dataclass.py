import dataclasses
import typing

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsRuntimeTypeError

from tests.helpers import test_utils


@pytest.mark.parametrize("fastcache", [False, True])
@pytest.mark.parametrize("present_first", [False, True])
@test_utils.test(arch=qd.cpu)
def test_optional_dataclass_specialization(fastcache, present_first):
    @dataclasses.dataclass(frozen=True)
    class State:
        value: qd.Tensor
        scale: float

    @qd.kernel(fastcache=fastcache)
    def add(first: float, state: State | None, out: qd.Tensor, last: float):
        for i in range(out.shape[0]):
            out[i] = first + last
            if qd.static(state is not None):
                out[i] += state.value[i] * state.scale
            if qd.static(None is state):
                out[i] += 1.0

    out = qd.ndarray(qd.f32, shape=4)
    first = qd.ndarray(qd.f32, shape=4)
    second = qd.ndarray(qd.f32, shape=4)
    first.fill(3)
    second.fill(5)
    states = [None, State(first, 2.0), None, State(second, 4.0), None]
    if present_first:
        states = states[1:] + states[:1]
    for state in states:
        add(2.0, state, out, 7.0)
        expected = 10 if state is None else 9 + state.value[0] * state.scale
        np.testing.assert_array_equal(out.to_numpy(), np.full(4, expected))
    assert len(add._primal.mapper.mapping) == 2


@test_utils.test()
def test_optional_dataclass_absent_branch_is_not_traced():
    @dataclasses.dataclass(frozen=True)
    class State:
        value: qd.Tensor

    @qd.kernel
    def absent(state: typing.Optional[State], out: qd.Tensor):
        if qd.static(state is not None):
            out[0] = state.does_not_exist[0]
        else:
            out[0] = 42

    out = qd.ndarray(qd.i32, shape=1)
    absent(None, out)
    assert out[0] == 42


@test_utils.test()
def test_required_dataclass_still_rejects_none():
    @dataclasses.dataclass(frozen=True)
    class State:
        value: qd.Tensor

    @qd.kernel
    def required(state: State):
        pass

    with pytest.raises(QuadrantsRuntimeTypeError, match="requires a Python dataclass instance"):
        required(None)


@test_utils.test(arch=qd.cpu)
def test_optional_dataclass_cache_across_processes(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "optional_dataclass_cache.py"
    script.write_text(
        textwrap.dedent(
            """
            import dataclasses
            import sys
            import quadrants as qd

            qd.init(arch=qd.cpu, offline_cache=True, offline_cache_file_path=sys.argv[1])

            @dataclasses.dataclass(frozen=True)
            class State:
                value: object

            @qd.kernel(fastcache=True)
            def read(state: State | None, out: qd.Tensor, offset: int):
                out[0] = offset
                if qd.static(state is not None):
                    out[0] += state.value[0]

            out = qd.ndarray(qd.i32, shape=1)
            value = qd.ndarray(qd.i32, shape=1)
            value[0] = 12
            states = [None, State(value)]
            if sys.argv[2] == "reverse":
                states.reverse()
            for state in states * 2:
                read(state, out, 5)
                assert out[0] == (5 if state is None else 17)
            """
        )
    )
    for order in ("forward", "reverse"):
        result = subprocess.run(
            [sys.executable, str(script), str(tmp_path / "cache"), order],
            env={**os.environ, "QD_OFFLINE_CACHE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
