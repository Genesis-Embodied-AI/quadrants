import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.fixture(autouse=True)
def clean_scheduling_environment(monkeypatch):
    for name in (
        "CPU_WORK_SCHEDULING",
        "MAKE_CPU_MULTITHREADING_LOOP",
        "CPU_FIXED_BLOCK_DIM",
        "DEFAULT_CPU_BLOCK_DIM",
        "CPU_PER_WORKER_MIN_BLOCK_DIM",
    ):
        monkeypatch.delenv("QD_" + name, raising=False)


@pytest.mark.parametrize(
    "init_options,env_vars,option_to_check,expected_value,expects_deprecation_warning,expected_error",
    [
        # Defaults.
        ({}, {}, "cpu_work_scheduling", qd.CPUWorkScheduling.PER_WORKER, False, None),
        ({}, {}, "cpu_fixed_block_dim", 32, False, None),
        ({}, {}, "cpu_per_worker_min_block_dim", 512, False, None),
        # Current qd.init() options.
        (
            {"cpu_work_scheduling": qd.CPUWorkScheduling.PER_WORKER},
            {},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.PER_WORKER,
            False,
            None,
        ),
        (
            {"cpu_work_scheduling": qd.CPUWorkScheduling.FIXED_SIZE},
            {},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.FIXED_SIZE,
            False,
            None,
        ),
        ({"cpu_fixed_block_dim": 17}, {}, "cpu_fixed_block_dim", 17, False, None),
        ({"cpu_per_worker_min_block_dim": 16}, {}, "cpu_per_worker_min_block_dim", 16, False, None),
        # Current environment variables.
        (
            {},
            {"QD_CPU_WORK_SCHEDULING": "PER_WORKER"},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.PER_WORKER,
            False,
            None,
        ),
        (
            {},
            {"QD_CPU_WORK_SCHEDULING": "FIXED_SIZE"},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.FIXED_SIZE,
            False,
            None,
        ),
        ({}, {"QD_CPU_FIXED_BLOCK_DIM": "17"}, "cpu_fixed_block_dim", 17, False, None),
        ({}, {"QD_CPU_PER_WORKER_MIN_BLOCK_DIM": "16"}, "cpu_per_worker_min_block_dim", 16, False, None),
        # Deprecated qd.init() options.
        (
            {"make_cpu_multithreading_loop": True},
            {},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.PER_WORKER,
            True,
            None,
        ),
        (
            {"make_cpu_multithreading_loop": False},
            {},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.FIXED_SIZE,
            True,
            None,
        ),
        ({"default_cpu_block_dim": 17}, {}, "cpu_fixed_block_dim", 17, True, None),
        # Deprecated environment variables.
        (
            {},
            {"QD_MAKE_CPU_MULTITHREADING_LOOP": "1"},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.PER_WORKER,
            True,
            None,
        ),
        (
            {},
            {"QD_MAKE_CPU_MULTITHREADING_LOOP": "0"},
            "cpu_work_scheduling",
            qd.CPUWorkScheduling.FIXED_SIZE,
            True,
            None,
        ),
        ({}, {"QD_DEFAULT_CPU_BLOCK_DIM": "17"}, "cpu_fixed_block_dim", 17, True, None),
        # Supplying both names is an error, even when their values agree.
        (
            {"make_cpu_multithreading_loop": True, "cpu_work_scheduling": qd.CPUWorkScheduling.PER_WORKER},
            {},
            None,
            None,
            False,
            "Cannot specify both make_cpu_multithreading_loop and cpu_work_scheduling",
        ),
        (
            {"cpu_work_scheduling": qd.CPUWorkScheduling.PER_WORKER},
            {"QD_MAKE_CPU_MULTITHREADING_LOOP": "1"},
            None,
            None,
            False,
            "Cannot specify both make_cpu_multithreading_loop and cpu_work_scheduling",
        ),
        (
            {"make_cpu_multithreading_loop": True},
            {"QD_CPU_WORK_SCHEDULING": "PER_WORKER"},
            None,
            None,
            False,
            "Cannot specify both make_cpu_multithreading_loop and cpu_work_scheduling",
        ),
        (
            {},
            {"QD_MAKE_CPU_MULTITHREADING_LOOP": "1", "QD_CPU_WORK_SCHEDULING": "PER_WORKER"},
            None,
            None,
            False,
            "Cannot specify both make_cpu_multithreading_loop and cpu_work_scheduling",
        ),
        (
            {"default_cpu_block_dim": 17, "cpu_fixed_block_dim": 17},
            {},
            None,
            None,
            False,
            "Cannot specify both default_cpu_block_dim and cpu_fixed_block_dim",
        ),
        (
            {"cpu_fixed_block_dim": 17},
            {"QD_DEFAULT_CPU_BLOCK_DIM": "17"},
            None,
            None,
            False,
            "Cannot specify both default_cpu_block_dim and cpu_fixed_block_dim",
        ),
        (
            {"default_cpu_block_dim": 17},
            {"QD_CPU_FIXED_BLOCK_DIM": "17"},
            None,
            None,
            False,
            "Cannot specify both default_cpu_block_dim and cpu_fixed_block_dim",
        ),
        (
            {},
            {"QD_DEFAULT_CPU_BLOCK_DIM": "17", "QD_CPU_FIXED_BLOCK_DIM": "17"},
            None,
            None,
            False,
            "Cannot specify both default_cpu_block_dim and cpu_fixed_block_dim",
        ),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_options(
    init_options, env_vars, option_to_check, expected_value, expects_deprecation_warning, expected_error, monkeypatch
):
    for name, value in env_vars.items():
        monkeypatch.setenv(name, value)
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            qd.init(arch=qd.cpu, **init_options)
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        qd.init(arch=qd.cpu, **init_options)
    assert getattr(qd.lang.impl.current_cfg(), option_to_check) == expected_value
    deprecation_warnings = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
    assert bool(deprecation_warnings) == expects_deprecation_warning
    for warning in deprecation_warnings:
        assert f"is deprecated; use {option_to_check}" in str(warning.message)


@pytest.mark.parametrize("value", [0, -1])
@test_utils.test(arch=qd.cpu)
def test_cpu_fixed_block_dim_invalid(value):
    with pytest.raises(RuntimeError, match="cpu_fixed_block_dim must be >= 1"):
        qd.init(arch=qd.cpu, cpu_fixed_block_dim=value)


@pytest.mark.parametrize(
    "init_cpu_fixed_block_dim,loop_block_dim_override,expected_block_indices",
    [
        # expected_block_indices gives the block index returned by qd.block_idx() at each iteration.
        # For a block size of 3, iterations 0, 1, and 2 run in block 0; iterations 3, 4, and 5 run in block 1.
        # Iteration index:      0, 1, 2, 3, 4, 5, 6, 7
        # Expected block:      0, 0, 0, 1, 1, 1, 2, 2
        # loop_block_dim_override replaces init_cpu_fixed_block_dim for this loop; None means no override.
        (1, None, [0, 1, 2, 3, 4, 5, 6, 7]),
        (3, None, [0, 0, 0, 1, 1, 1, 2, 2]),
        (16, None, [0, 0, 0, 0, 0, 0, 0, 0]),
        (1, 4, [0, 0, 0, 0, 1, 1, 1, 1]),
        (16, 4, [0, 0, 0, 0, 1, 1, 1, 1]),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_fixed_block_dim_execution(init_cpu_fixed_block_dim, loop_block_dim_override, expected_block_indices):
    qd.init(
        arch=qd.cpu,
        cpu_work_scheduling=qd.CPUWorkScheduling.FIXED_SIZE,
        cpu_fixed_block_dim=init_cpu_fixed_block_dim,
        cpu_max_num_threads=4,
    )
    block_indices = qd.ndarray(qd.i32, shape=8)

    @qd.kernel
    def k_record_blocks(block_indices: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        if qd.static(loop_block_dim_override is not None):
            qd.loop_config(block_dim=loop_block_dim_override)
        for i in range(8):
            block_indices[i] = qd.block_idx()

    k_record_blocks(block_indices)
    np.testing.assert_array_equal(block_indices.to_numpy(), expected_block_indices)


def _cpu_scheduling_cache_child(args: list[str]) -> None:
    cache_directory, options_json, fastcache_json, expected_json, warm_json = args
    options = json.loads(options_json)
    options["cpu_work_scheduling"] = getattr(qd.CPUWorkScheduling, options["cpu_work_scheduling"])
    fastcache = json.loads(fastcache_json)
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        offline_cache=True,
        offline_cache_file_path=cache_directory,
        **options,
    )

    @qd.kernel(fastcache=fastcache)
    def k_record_blocks(block_indices: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(8):
            block_indices[i] = qd.block_idx()

    block_indices = qd.ndarray(qd.i32, shape=8)
    k_record_blocks(block_indices)
    np.testing.assert_array_equal(block_indices.to_numpy(), json.loads(expected_json))
    if fastcache:
        assert k_record_blocks._primal.src_ll_cache_observations.cache_loaded == json.loads(warm_json)
    qd.reset()


@pytest.mark.parametrize("fastcache", [False, True])
@pytest.mark.parametrize(
    "mode,key,value,expected_base,expected_changed",
    [
        ("PER_WORKER", "cpu_per_worker_min_block_dim", 3, [0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 1, 1, 1, 2, 2]),
        ("FIXED_SIZE", "cpu_fixed_block_dim", 2, [0, 0, 0, 1, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2, 3, 3]),
        ("PER_WORKER", "cpu_work_scheduling", "FIXED_SIZE", [0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 1, 1, 1, 2, 2]),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_cache_across_processes(tmp_path, fastcache, mode, key, value, expected_base, expected_changed):
    base = {"cpu_work_scheduling": mode, "cpu_per_worker_min_block_dim": 2, "cpu_fixed_block_dim": 3}
    changed = {**base, key: value}
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    # Each process shares the cache directory. Returning to the first settings should reuse its cached kernel.
    for options, expected, warm in [
        (base, expected_base, False),
        (changed, expected_changed, False),
        (base, expected_base, True),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                _cpu_scheduling_cache_child.__name__,
                str(tmp_path / "cache"),
                json.dumps(options),
                json.dumps(fastcache),
                json.dumps(expected),
                json.dumps(warm),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
