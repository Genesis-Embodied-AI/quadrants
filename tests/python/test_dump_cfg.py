import pathlib

import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(offline_cache=False)
def test_dump_cfg_is_per_task_and_does_not_change_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    # QD_DUMP_CFG=1 must be a pure debugging side effect: once the kernel is offloaded it writes ONE CFG file per
    # offloaded task (the exact per-task graph the compiler optimizes), and must NOT fall back to dumping a single
    # whole-kernel graph. Regression test for QD_DUMP_CFG previously forcing the whole-kernel cfg path.
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

    # Behavior is unchanged: results are still correct.
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    for i in range(n):
        assert a_np[i] == i
        assert b_np[i] == i * 2

    cfg_files = list(tmp_path.glob("*_CFG_*.txt"))
    assert len(cfg_files) > 0, f"No CFG dumps found under {tmp_path}"

    # Every post-offload CFG dump is tagged with the task it belongs to; none is an un-tagged whole-kernel graph
    # (which is what the old "force whole-kernel path" behavior produced).
    assert all("_task" in f.name for f in cfg_files), [f.name for f in cfg_files]

    # The two independent top-level loops become at least two offloaded tasks, so we get at least task0 and task1.
    task_tags = {tag for f in cfg_files for tag in f.name.split("_") if tag.startswith("task")}
    assert {"task0", "task1"}.issubset(task_tags), sorted(task_tags)
