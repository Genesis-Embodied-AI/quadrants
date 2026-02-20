import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_parallel_range_for():
    n = 1024 * 1024
    val = qd.field(qd.i32, shape=(n))

    @qd.kernel
    def fill():
        qd.loop_config(parallelize=8, block_dim=8)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i


@test_utils.test()
def test_serial_for():
    @qd.kernel
    def foo() -> qd.i32:
        a = 0
        qd.loop_config(serialize=True)
        for i in range(100):
            a = a + 1
            if a == 50:
                break

        return a

    assert foo() == 50


@test_utils.test()
def test_loop_config_parallel_range_for():
    n = 1024 * 1024
    val = qd.field(qd.i32, shape=(n))

    @qd.kernel
    def fill():
        qd.loop_config(parallelize=8, block_dim=8)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i


@test_utils.test()
def test_loop_config_serial_for():
    @qd.kernel
    def foo() -> qd.i32:
        a = 0
        qd.loop_config(serialize=True)
        for i in range(100):
            a = a + 1
            if a == 50:
                break

        return a

    assert foo() == 50


@test_utils.test(arch=[qd.cpu])
def test_loop_config_block_dim_adaptive():
    n = 4096
    val = qd.field(qd.i32, shape=(n))

    @qd.kernel
    def fill():
        qd.loop_config(block_dim_adaptive=False)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i
