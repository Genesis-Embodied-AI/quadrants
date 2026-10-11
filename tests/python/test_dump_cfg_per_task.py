import pathlib
import re

import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(offline_cache=False)
def test_dump_cfg_is_per_task(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    # QD_DUMP_CFG must be observation-only: once a kernel is offloaded, cfg_optimization runs per offloaded task,
    # and the dump must follow that granularity (one graph per task, "_task<N>" suffix) rather than forcing the
    # CFG back onto the whole-kernel path. Two independent top-level loops become two offloaded tasks, so we
    # expect per-task dump files for task0 and task1.
    monkeypatch.setenv("QD_DUMP_CFG", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    n = 128
    a = qd.ndarray(qd.i32, shape=(n,))
    b = qd.ndarray(qd.i32, shape=(n,))

    @qd.kernel
    def two_loops(
        a: qd.types.ndarray(dtype=qd.i32, ndim=1),
        b: qd.types.ndarray(dtype=qd.i32, ndim=1),
    ):
        for i in range(n):
            a[i] = i
        for i in range(n):
            b[i] = i * 2

    two_loops(a, b)
    qd.sync()

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    for i in range(n):
        assert a_np[i] == i
        assert b_np[i] == i * 2

    cfg_files = [f.name for f in tmp_path.glob("*_CFG_*")]
    assert cfg_files, f"No CFG dumps written under {tmp_path}"

    task0 = [f for f in cfg_files if "_task0_" in f]
    task1 = [f for f in cfg_files if "_task1_" in f]
    assert task0, f"No per-task (task0) CFG dumps found; got {cfg_files}"
    assert task1, f"No per-task (task1) CFG dumps found; got {cfg_files}"

    # Each optimized task is dumped both before and after its own per-task optimization.
    assert any("_task0_before_cfg_opt" in f for f in cfg_files), cfg_files
    assert any("_task0_post_cfg_opt" in f for f in cfg_files), cfg_files

    # The per-task codegen phases (before_lower_access, simplify_IV, ...) lower each task in isolation on its own
    # worker thread, where the local task index is always 0. If the dump used that local index, only the
    # whole-kernel post-offload phase would emit a task1 file and every codegen phase would collide on task0.
    # Requiring task1 dumps from more than one distinct phase pins that the kernel-wide task id is used instead.
    phases_with_task1 = {m.group(1) for f in cfg_files if (m := re.search(r"_CFG_(.+)_task1_", f))}
    assert len(phases_with_task1) >= 2, f"Expected task1 dumps from multiple phases, got {sorted(phases_with_task1)}"
