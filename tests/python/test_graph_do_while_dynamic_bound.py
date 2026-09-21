"""Regression for https://github.com/Genesis-Embodied-AI/quadrants/issues/899."""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _check_dynamic_bound():
    @qd.kernel(graph=True)
    def k(
        n: qd.types.ndarray(qd.i32, 0),
        c: qd.types.ndarray(qd.i32, 0),
        out: qd.types.ndarray(qd.i32, 0),
    ):
        for _ in range(n[()]):
            qd.atomic_add(out[()], 10)
        while qd.graph.do_while(c):
            for _ in range(n[()]):
                qd.atomic_add(out[()], 1)
            for _ in range(1):
                n[()] = 1
                c[()] -= 1

    n, c, out = (qd.ndarray(qd.i32, shape=()) for _ in range(3))
    n.from_numpy(np.array(0, np.int32))
    c.from_numpy(np.array(2, np.int32))
    out.from_numpy(np.array(0, np.int32))
    k(n, c, out)

    actual_n, actual_c, actual_out = (int(a.to_numpy()) for a in (n, c, out))
    print(f"issue899: n={actual_n}, c={actual_c}, out={actual_out}; expected n=1, c=0, out=1", flush=True)
    assert actual_n == 1
    assert actual_c == 0
    assert actual_out == 1, "The second graph iteration must reload n after the first iteration writes n=1"


@pytest.mark.parametrize("whole_kernel", [False, True], ids=["default-frontend", "whole-kernel"])
@test_utils.test(arch=qd.cuda, offline_cache=False, advanced_optimization=True)
def test_graph_do_while_dynamic_bound(monkeypatch, whole_kernel):
    if whole_kernel:
        monkeypatch.setenv("QD_SPLIT_MAX_COST_RATIO", "0")
    else:
        monkeypatch.delenv("QD_SPLIT_MAX_COST_RATIO", raising=False)
    _check_dynamic_bound()


@test_utils.test(arch=qd.cuda, offline_cache=False, advanced_optimization=False)
def test_graph_do_while_dynamic_bound_without_advanced_optimization(monkeypatch):
    monkeypatch.setenv("QD_SPLIT_MAX_COST_RATIO", "0")
    _check_dynamic_bound()
