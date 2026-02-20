import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ifexpr_vector():
    n_grids = 10

    g_v = qd.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = qd.field(float, (n_grids, n_grids, n_grids))

    @qd.kernel
    def func():
        for I in qd.grouped(g_m):
            cond = (I < 3) & (g_v[I] < 0) | (I > n_grids - 3) & (g_v[I] > 0)
            g_v[I] = 0 if cond else g_v[I]

    with pytest.raises(qd.QuadrantsSyntaxError, match='Please use "qd.select" instead.'):
        func()


@test_utils.test()
def test_ifexpr_scalar():
    n_grids = 10

    g_v = qd.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = qd.field(float, (n_grids, n_grids, n_grids))

    @qd.kernel
    def func():
        for I in qd.grouped(g_m):
            cond = (I[0] < 3) and (g_v[I][0] < 0) or (I[0] > n_grids - 3) and (g_v[I][0] > 0)
            g_v[I] = 0 if cond else g_v[I]

    func()
