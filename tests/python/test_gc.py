import quadrants as qd

from tests import test_utils


def _test_block_gc():
    N = 100000

    dx = 1 / 128
    inv_dx = 1.0 / dx

    x = qd.Vector.field(2, dtype=qd.f32)

    indices = qd.ij

    grid_m = qd.field(dtype=qd.i32)

    grid = qd.root.pointer(indices, 64)
    grid.pointer(indices, 32).dense(indices, 8).place(grid_m)

    qd.root.dense(qd.i, N).place(x)

    @qd.kernel
    def init():
        for i in x:
            x[i] = qd.Vector([qd.random() * 0.1 + 0.5, qd.random() * 0.1 + 0.5])

    init()

    @qd.kernel
    def build_grid():
        for p in x:
            base = int(qd.floor(x[p] * inv_dx - 0.5))
            grid_m[base] += 1

    @qd.kernel
    def move():
        for p in x:
            x[p] += qd.Vector([0.0, 0.1])

    assert grid._num_dynamically_allocated == 0
    for _ in range(100):
        grid.deactivate_all()
        # Scatter the particles to the sparse grid
        build_grid()
        # Move the block of particles
        move()

    qd.sync()
    # The block of particles can occupy at most two blocks on the sparse grid.
    # It's fine to run 100 times and do just one final check, because
    # num_dynamically_allocated stores the number of slots *ever* allocated.
    assert 1 <= grid._num_dynamically_allocated <= 2, grid._num_dynamically_allocated


@test_utils.test(require=qd.extension.sparse)
def test_block():
    _test_block_gc()


@test_utils.test(require=qd.extension.sparse, exclude=qd.metal)
def test_dynamic_gc():
    x = qd.field(dtype=qd.i32)

    L = qd.root.dynamic(qd.i, 1024 * 1024, chunk_size=1024)
    L.place(x)

    assert L._num_dynamically_allocated == 0

    for i in range(100):
        x[1024] = 1
        L.deactivate_all()
        assert L._num_dynamically_allocated <= 2


@test_utils.test(require=qd.extension.sparse)
def test_pointer_gc():
    x = qd.field(dtype=qd.i32)

    L = qd.root.pointer(qd.ij, 32)
    L.pointer(qd.ij, 32).dense(qd.ij, 8).place(x)

    assert L._num_dynamically_allocated == 0

    for i in range(1024):
        x[i * 8, i * 8] = 1
        assert L._num_dynamically_allocated == 1
        L.deactivate_all()

        # Note that being inactive doesn't mean it's not allocated.
        assert L._num_dynamically_allocated == 1
