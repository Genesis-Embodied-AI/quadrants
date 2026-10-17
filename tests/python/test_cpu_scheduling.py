import json
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


@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_defaults():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        qd.init(arch=qd.cpu)
    assert qd.lang.impl.current_cfg().cpu_work_scheduling == qd.CPUWorkScheduling.PER_WORKER


@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_environment(monkeypatch):
    monkeypatch.setenv("QD_CPU_WORK_SCHEDULING", "FIXED_SIZE")
    qd.init(arch=qd.cpu)
    assert qd.lang.impl.current_cfg().cpu_work_scheduling == qd.CPUWorkScheduling.FIXED_SIZE


@pytest.mark.parametrize(
    "old, value, new, expected, env_value",
    [
        ("make_cpu_multithreading_loop", True, "cpu_work_scheduling", qd.CPUWorkScheduling.PER_WORKER, "1"),
        ("make_cpu_multithreading_loop", False, "cpu_work_scheduling", qd.CPUWorkScheduling.FIXED_SIZE, "0"),
        ("default_cpu_block_dim", 17, "cpu_fixed_block_dim", 17, "17"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_deprecated_alias(old, value, new, expected, env_value, monkeypatch):
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated; use {new}"):
        qd.init(arch=qd.cpu, **{old: value})
    assert getattr(qd.lang.impl.current_cfg(), new) == expected
    monkeypatch.setenv("QD_" + old.upper(), env_value)
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated; use {new}"):
        qd.init(arch=qd.cpu)
    assert getattr(qd.lang.impl.current_cfg(), new) == expected


@pytest.mark.parametrize("old_env,new_env", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize(
    "old, old_value, new, new_value, new_env_value",
    [
        ("make_cpu_multithreading_loop", True, "cpu_work_scheduling", qd.CPUWorkScheduling.PER_WORKER, "PER_WORKER"),
        ("default_cpu_block_dim", 17, "cpu_fixed_block_dim", 17, "17"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_conflicting_aliases(
    old_env, new_env, old, old_value, new, new_value, new_env_value, monkeypatch
):
    kwargs = {}
    if old_env:
        monkeypatch.setenv("QD_" + old.upper(), str(int(old_value)))
    else:
        kwargs[old] = old_value
    if new_env:
        monkeypatch.setenv("QD_" + new.upper(), new_env_value)
    else:
        kwargs[new] = new_value
    # Reject both names even when they request identical settings.
    with pytest.raises(ValueError, match=f"Cannot specify both {old} and {new}"):
        qd.init(arch=qd.cpu, **kwargs)


@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_empty_environment(monkeypatch):
    monkeypatch.setenv("QD_MAKE_CPU_MULTITHREADING_LOOP", "")
    monkeypatch.setenv("QD_DEFAULT_CPU_BLOCK_DIM", "")
    qd.init(arch=qd.cpu, cpu_work_scheduling=qd.CPUWorkScheduling.FIXED_SIZE, cpu_fixed_block_dim=17)
    assert qd.lang.impl.current_cfg().cpu_fixed_block_dim == 17


@pytest.mark.parametrize("value", [0, -1])
@test_utils.test(arch=qd.cpu)
def test_cpu_fixed_block_dim_invalid(value):
    with pytest.raises(RuntimeError, match="cpu_fixed_block_dim must be >= 1"):
        qd.init(arch=qd.cpu, cpu_fixed_block_dim=value)


@pytest.mark.parametrize(
    "cpu_fixed_block_dim,loop_block_dim,expected",
    [
        (1, None, [0, 1, 2, 3, 4, 5, 6, 7]),
        (3, None, [0, 0, 0, 1, 1, 1, 2, 2]),
        (16, None, [0, 0, 0, 0, 0, 0, 0, 0]),
        (1, 4, [0, 0, 0, 0, 1, 1, 1, 1]),
        (16, 4, [0, 0, 0, 0, 1, 1, 1, 1]),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_fixed_block_dim_execution(cpu_fixed_block_dim, loop_block_dim, expected):
    qd.init(
        arch=qd.cpu,
        cpu_work_scheduling=qd.CPUWorkScheduling.FIXED_SIZE,
        cpu_fixed_block_dim=cpu_fixed_block_dim,
        cpu_max_num_threads=4,
    )
    out = qd.ndarray(qd.i32, shape=8)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        if qd.static(loop_block_dim is not None):
            qd.loop_config(block_dim=loop_block_dim)
        for i in range(8):
            out[i] = qd.block_idx()

    k_record_blocks(out)
    np.testing.assert_array_equal(out.to_numpy(), expected)


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
    script = tmp_path / "record_blocks.py"
    script.write_text(
        """
import json
import sys
import numpy as np
import quadrants as qd

options = json.loads(sys.argv[2])
options["cpu_work_scheduling"] = getattr(qd.CPUWorkScheduling, options["cpu_work_scheduling"])
fastcache = bool(int(sys.argv[3]))
qd.init(arch=qd.cpu, cpu_max_num_threads=4, offline_cache=True,
        offline_cache_file_path=sys.argv[1], **options)

@qd.kernel(fastcache=fastcache)
def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    for i in range(8):
        out[i] = qd.block_idx()

out = qd.ndarray(qd.i32, shape=8)
k_record_blocks(out)
np.testing.assert_array_equal(out.to_numpy(), json.loads(sys.argv[4]))
if fastcache:
    assert k_record_blocks._primal.src_ll_cache_observations.cache_loaded == bool(int(sys.argv[5]))
qd.reset()
"""
    )
    # Each process shares the cache directory. Returning to the first settings should reuse its cached kernel.
    for options, expected, warm in [
        (base, expected_base, False),
        (changed, expected_changed, False),
        (base, expected_base, True),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(tmp_path / "cache"),
                json.dumps(options),
                str(int(fastcache)),
                json.dumps(expected),
                str(int(warm)),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
