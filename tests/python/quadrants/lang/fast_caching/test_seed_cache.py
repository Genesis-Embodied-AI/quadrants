import json
import os
import subprocess
import sys

import quadrants as qd

from tests.helpers import test_utils


def _run_kernels(cache_dir, seed, stack_limit, cache_enabled):
    qd.init(
        arch=qd.cpu,
        random_seed=seed,
        cuda_stack_limit=stack_limit,
        cpu_max_num_threads=1,
        num_compile_threads=1,
        offline_cache_file_path=cache_dir,
        offline_cache=cache_enabled,
        src_ll_cache=cache_enabled,
    )
    config_alias = qd.cfg

    @qd.kernel(fastcache=True)
    def draw() -> qd.u32:
        return qd.random(qd.u32)

    @qd.kernel(fastcache=True)
    def seed_constant() -> qd.i32:
        return qd.static(qd.cfg.random_seed)

    @qd.func
    def read_seed() -> qd.i32:
        return qd.static(config_alias.random_seed)

    @qd.kernel(fastcache=True)
    def helper_seed() -> qd.i32:
        return read_seed()

    def read_limit():
        return getattr(config_alias, "cuda_stack_limit")

    @qd.kernel(fastcache=True)
    def optional_limit(enabled: qd.Template) -> qd.i32:
        if qd.static(enabled):
            return qd.static(read_limit())
        else:
            return 0

    @qd.real_func
    def real_seed() -> qd.i32:
        return qd.static(qd.cfg.random_seed)

    @qd.kernel(fastcache=True)
    def first_caller() -> qd.i32:
        return real_seed()

    @qd.kernel(fastcache=True)
    def second_caller() -> qd.i32:
        return real_seed() + 1

    draws = [draw() for _ in range(4)]
    # The first specialization does not depend on the limit. The second adds it to L1.
    assert optional_limit(False) == 0
    # The second caller cannot observe the config reads of a body compiled by the first caller.
    first_value = first_caller()
    second_value = second_caller()
    assert (first_value, second_value) == (seed, seed + 1)
    first_real_cache = first_caller._primal.src_ll_cache_observations
    if first_real_cache.cache_stored:
        assert not second_caller._primal.src_ll_cache_observations.cache_stored
    results = [
        [draws, draw._primal.src_ll_cache_observations.cache_loaded],
        [seed_constant(), seed_constant._primal.src_ll_cache_observations.cache_loaded],
        [helper_seed(), helper_seed._primal.src_ll_cache_observations.cache_loaded],
        [optional_limit(True), optional_limit._primal.src_ll_cache_observations.cache_loaded],
        [first_value, first_real_cache.cache_loaded],
    ]
    print("RESULT=" + json.dumps(results))
    qd.reset()


@test_utils.test(arch=qd.cpu)
def test_seed_cache_across_processes(tmp_path):
    def run(seed, stack_limit, cache_enabled=True):
        proc = subprocess.run(
            [sys.executable, __file__, str(tmp_path), str(seed), str(stack_limit), str(int(cache_enabled))],
            env={**os.environ, "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return json.loads(
            next(line.removeprefix("RESULT=") for line in proc.stdout.splitlines() if line.startswith("RESULT="))
        )

    first = run(1, 1024)
    second = run(2, 2048)
    assert first[0] == run(1, 1024, False)[0]
    assert second[0] == [run(2, 2048, False)[0][0], True]
    assert first[0][0] != second[0][0]
    assert first[1:] == [[1, False], [1, False], [1024, False], [1, False]]
    assert second[1:] == [[2, False], [2, False], [2048, False], [2, False]]
    # Change one field at a time. Each kernel reuses its own specialization, not its sibling's.
    assert run(1, 2048) == [[first[0][0], True], [1, True], [1, True], [2048, True], [1, True]]


if __name__ == "__main__":
    _run_kernels(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])))
