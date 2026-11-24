import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import quadrants as qd

from tests.helpers import test_utils

_SEEDS = (1, 2, 1, 2)
_CASES = (
    "quadrants_default",
    "python_default",
    "attribute_default",
    "static_seed",
    "constant",
    "vector_norm",
)


def _run_kernel(cache_dir, seed, case, cache_enabled):
    qd.init(
        arch=qd.cpu,
        random_seed=seed,
        cpu_max_num_threads=1,
        num_compile_threads=1,
        offline_cache_file_path=cache_dir,
        offline_cache=cache_enabled,
        src_ll_cache=cache_enabled,
    )
    if case == "quadrants_default":

        @qd.func
        def helper(seed=qd.cfg.random_seed):
            return seed

        @qd.kernel(fastcache=True)
        def read() -> qd.i32:
            return helper()

    elif case == "python_default":

        def read_seed(value=qd.cfg.random_seed):
            return value

        @qd.kernel(fastcache=True)
        def read() -> qd.i32:
            return qd.static(read_seed())

    elif case == "attribute_default":
        if seed == 1:

            @qd.func
            def selected():
                return 1

        else:

            @qd.func
            def selected(seed=qd.cfg.random_seed):
                return seed

        lib = SimpleNamespace(helper=selected)

        @qd.kernel(fastcache=True)
        def read() -> qd.i32:
            return lib.helper()

    elif case == "static_seed":

        @qd.kernel(fastcache=True)
        def read() -> qd.i32:
            return qd.static(qd.cfg.random_seed)

    elif case == "constant":

        @qd.kernel(fastcache=True)
        def read() -> qd.i32:
            return 7

    elif case == "vector_norm":

        @qd.kernel(fastcache=True)
        def read() -> qd.f32:
            return qd.Vector([3.0, 4.0]).norm()

    elif case == "random":

        @qd.kernel(fastcache=True)
        def read() -> qd.u32:
            return qd.random(qd.u32)

    value = [read() for _ in range(4)] if case == "random" else read()
    observations = read._primal.src_ll_cache_observations
    print("RESULT=" + json.dumps([value, observations.cache_loaded]))
    qd.reset()


def _run_process(tmp_path, seed, case, cache_enabled=True):
    proc = subprocess.run(
        [sys.executable, __file__, str(tmp_path), str(seed), case, str(int(cache_enabled))],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(
        next(line.removeprefix("RESULT=") for line in proc.stdout.splitlines() if line.startswith("RESULT="))
    )


@test_utils.test(arch=qd.cpu)
@pytest.mark.parametrize("case", _CASES)
def test_seed_cache_across_processes(tmp_path, case):
    results = [_run_process(tmp_path, seed, case) for seed in _SEEDS]
    print(f"{case}: {results}")
    if case in ("constant", "vector_norm"):
        value = 7 if case == "constant" else 5.0
        assert results == [[value, index >= 1] for index in range(len(_SEEDS))]
    else:
        assert results == [[seed, index >= 2] for index, seed in enumerate(_SEEDS)]


@test_utils.test(arch=qd.cpu)
def test_runtime_random_seed_across_processes(tmp_path):
    first = _run_process(tmp_path, 1, "random")
    second = _run_process(tmp_path, 2, "random")
    assert first == _run_process(tmp_path, 1, "random", False)
    assert second == [_run_process(tmp_path, 2, "random", False)[0], True]
    assert first[0] != second[0]


if __name__ == "__main__":
    _run_kernel(sys.argv[1], int(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])))
