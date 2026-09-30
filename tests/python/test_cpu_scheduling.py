import warnings
from enum import Enum

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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        qd.init(arch=qd.cpu)
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
    cfg = qd.lang.impl.current_cfg()
    assert cfg.cpu_work_scheduling == qd.CPUWorkScheduling.PER_WORKER
    assert cfg.cpu_fixed_block_dim == 32
    assert cfg.cpu_per_worker_min_block_dim == 512


@pytest.mark.parametrize("mode", [qd.CPUWorkScheduling.PER_WORKER, qd.CPUWorkScheduling.FIXED_SIZE])
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_enum(mode, monkeypatch):
    qd.init(arch=qd.cpu, cpu_work_scheduling=mode)
    assert qd.lang.impl.current_cfg().cpu_work_scheduling == mode
    monkeypatch.setenv("QD_CPU_WORK_SCHEDULING", mode.name)
    qd.init(arch=qd.cpu)
    assert qd.lang.impl.current_cfg().cpu_work_scheduling == mode
    # A keyword overrides the environment for the same spelling.
    monkeypatch.setenv("QD_CPU_WORK_SCHEDULING", "invalid")
    qd.init(arch=qd.cpu, cpu_work_scheduling=mode)
    assert qd.lang.impl.current_cfg().cpu_work_scheduling == mode


@pytest.mark.parametrize("value", ["PER_WORKER", 0, 1, True, False, None, Enum("Other", "PER_WORKER").PER_WORKER])
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_invalid_enum(value):
    with pytest.raises(TypeError, match="cpu_work_scheduling must be a qd.CPUWorkScheduling member"):
        qd.init(arch=qd.cpu, cpu_work_scheduling=value)


@pytest.mark.parametrize("value", ["0", "1", "True", "per_worker", "qd.CPUWorkScheduling.PER_WORKER"])
@test_utils.test(arch=qd.cpu)
def test_cpu_scheduling_invalid_environment(value, monkeypatch):
    monkeypatch.setenv("QD_CPU_WORK_SCHEDULING", value)
    with pytest.raises(ValueError, match="QD_CPU_WORK_SCHEDULING must be PER_WORKER or FIXED_SIZE"):
        qd.init(arch=qd.cpu)


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
def test_cpu_scheduling_conflicting_aliases(old_env, new_env, old, old_value, new, new_value, new_env_value, monkeypatch):
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


@pytest.mark.parametrize("cpu_fixed_block_dim", [1, 17, 64, 512])
@pytest.mark.parametrize("loop_block_dim", [None, 32])
@test_utils.test(arch=qd.cpu)
def test_cpu_fixed_block_dim_execution(cpu_fixed_block_dim, loop_block_dim):
    qd.init(
        arch=qd.cpu,
        cpu_work_scheduling=qd.CPUWorkScheduling.FIXED_SIZE,
        cpu_fixed_block_dim=cpu_fixed_block_dim,
        cpu_max_num_threads=4,
    )
    out = qd.ndarray(qd.i32, shape=200)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        if qd.static(loop_block_dim is not None):
            qd.loop_config(block_dim=loop_block_dim)
        for i in range(200):
            out[i] = qd.block_idx()

    k_record_blocks(out)
    width = cpu_fixed_block_dim if loop_block_dim is None else loop_block_dim
    np.testing.assert_array_equal(out.to_numpy(), np.arange(200, dtype=np.int32) // width)
