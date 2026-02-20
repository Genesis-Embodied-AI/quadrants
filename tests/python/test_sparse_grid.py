import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse)
def test_sparse_grid():
    # create a 2D sparse grid
    grid = qd.sparse.grid(
        {
            "pos": qd.math.vec2,
            "mass": qd.f32,
            "grid2particles": qd.types.vector(20, qd.i32),
        },
        shape=(10, 10),
    )

    # access
    grid[0, 0].pos = qd.math.vec2(1, 2)
    grid[0, 0].mass = 1.0
    grid[0, 0].grid2particles[2] = 123

    # print the usage of the sparse grid, which is in [0,1]
    assert qd.sparse.usage(grid) == test_utils.approx(0.01)
