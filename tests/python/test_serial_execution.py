import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu, cpu_max_num_threads=1)
def test_serial_range_for():
    n = 1024 * 32
    s = qd.field(dtype=qd.i32, shape=n)
    counter = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def fill_range():
        counter[None] = 0
        for i in range(n):
            s[qd.atomic_add(counter[None], 1)] = i

    fill_range()

    for i in range(n):
        assert s[i] == i


@test_utils.test(arch=qd.cpu, cpu_max_num_threads=1)
def test_serial_struct_for():
    n = 1024 * 32
    s = qd.field(dtype=qd.i32, shape=n)
    counter = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def fill_struct():
        counter[None] = 0
        for i in s:
            s[qd.atomic_add(counter[None], 1)] = i

    fill_struct()

    for i in range(n):
        assert s[i] == i
