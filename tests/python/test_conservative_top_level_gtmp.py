"""Checks specific to the always-on conservative-gtmp benchmark branch."""

import re

import numpy as np

import quadrants as qd
from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_unused_top_level_values_are_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    @qd.kernel
    def snapshot_probe(out: qd.types.ndarray(qd.i32, ndim=1), n: qd.i32):
        unused = n * 13 + 7
        for i in range(4):
            out[i] = i + 1

    out = qd.ndarray(qd.i32, shape=4)
    snapshot_probe(out, 5)
    np.testing.assert_array_equal(out.to_numpy(), [1, 2, 3, 4])
    # This kernel otherwise needs no shared slots. The unused calculation and
    # stores must survive post-offload simplification, not just initial insertion.
    paths = list(tmp_path.glob("*snapshot_probe*after_simplify_III.ll"))
    assert len(paths) == 1
    ir = paths[0].read_text()
    assert "global tmp var" in ir
    assert re.search(r"= mul ", ir)
    temp_ids = re.findall(r"(\$\d+) = global tmp var", ir)
    assert any(f"global store [{name} <-" in ir for name in temp_ids)
    # Snapshot stores must be in serial tasks, never inside the parallel body.
    parallel = ir.split("offloaded range_for", 1)[1]
    assert "global tmp var" not in parallel


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_existing_shared_slots_do_not_overlap_snapshots():
    @qd.kernel
    def shared_probe(out: qd.types.ndarray(qd.f64, ndim=1), n: qd.i32):
        total = qd.cast(n, qd.f64) + 0.5
        for i in range(n):
            total += qd.cast(i, qd.f64)
        for i in range(n):
            out[i] = total + i

    out = qd.ndarray(qd.f64, shape=5)
    shared_probe(out, 5)
    np.testing.assert_array_equal(out.to_numpy(), [15.5, 16.5, 17.5, 18.5, 19.5])
