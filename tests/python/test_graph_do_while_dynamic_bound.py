"""Regressions for stale graph do-while range bounds (issue #899)."""

import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test(offline_cache=False, advanced_optimization=True)
def test_graph_do_while_dynamic_bound():
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

    assert int(n.to_numpy()) == 1
    assert int(c.to_numpy()) == 0
    assert int(out.to_numpy()) == 1, "The second graph iteration must reload n after the first writes n=1"


@test_utils.test(offline_cache=False, advanced_optimization=True)
def test_graph_do_while_nested_dynamic_bound():
    @qd.kernel(graph=True)
    def k(
        n: qd.types.ndarray(qd.i32, 0),
        outer: qd.types.ndarray(qd.i32, 0),
        inner: qd.types.ndarray(qd.i32, 0),
        out: qd.types.ndarray(qd.i32, 0),
    ):
        for _ in range(n[()]):
            qd.atomic_add(out[()], 1000)
        while qd.graph.do_while(outer):
            for _ in range(n[()]):
                qd.atomic_add(out[()], 10)
            while qd.graph.do_while(inner):
                for _ in range(n[()]):
                    qd.atomic_add(out[()], 1)
                for _ in range(1):
                    n[()] += 1
                    inner[()] -= 1
            for _ in range(n[()]):
                qd.atomic_add(out[()], 100)
            for _ in range(1):
                inner[()] = 2
                outer[()] -= 1

    n, outer, inner, out = (qd.ndarray(qd.i32, shape=()) for _ in range(4))
    for arr, value in ((n, 0), (outer, 2), (inner, 2), (out, 0)):
        arr.from_numpy(np.array(value, np.int32))
    k(n, outer, inner, out)

    assert int(n.to_numpy()) == 4
    assert int(outer.to_numpy()) == 0
    assert int(inner.to_numpy()) == 2
    # Outer iteration 1: 0 + (0 + 1) + 200; iteration 2: 20 + (2 + 3) + 400.
    assert int(out.to_numpy()) == 626


@test_utils.test(offline_cache=False, advanced_optimization=True)
def test_graph_do_while_sibling_dynamic_bound():
    @qd.kernel(graph=True)
    def k(
        n: qd.types.ndarray(qd.i32, 0),
        first: qd.types.ndarray(qd.i32, 0),
        second: qd.types.ndarray(qd.i32, 0),
        out: qd.types.ndarray(qd.i32, 0),
    ):
        for _ in range(n[()]):
            qd.atomic_add(out[()], 10000)
        while qd.graph.do_while(first):
            for _ in range(n[()]):
                qd.atomic_add(out[()], 1)
            for _ in range(1):
                n[()] += 1
                first[()] -= 1
        for _ in range(n[()]):
            qd.atomic_add(out[()], 10)
        while qd.graph.do_while(second):
            for _ in range(n[()]):
                qd.atomic_add(out[()], 100)
            for _ in range(1):
                n[()] += 1
                second[()] -= 1
        for _ in range(n[()]):
            qd.atomic_add(out[()], 1000)

    n, first, second, out = (qd.ndarray(qd.i32, shape=()) for _ in range(4))
    for arr, value in ((n, 0), (first, 2), (second, 2), (out, 0)):
        arr.from_numpy(np.array(value, np.int32))
    k(n, first, second, out)

    assert int(n.to_numpy()) == 4
    assert int(first.to_numpy()) == 0
    assert int(second.to_numpy()) == 0
    assert int(out.to_numpy()) == 4521  # (0 + 1) + 20 + (200 + 300) + 4000.
