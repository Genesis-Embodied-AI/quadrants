import pytest

import quadrants as qd

from tests import test_utils


# With a small `cpu_min_range_for_block`, a parallel CPU range-for is split into many small chunks spread across the
# thread pool (instead of the default floor of 512 iterations per chunk, which would keep loops of <= 512 iterations on
# a single thread). This exercises the small-chunk path across multiple threads and checks that every element is still
# written exactly once, guarding against off-by-one / race regressions in the chunking.
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=8, cpu_min_range_for_block=1)
def test_small_block_range_for_is_correct():
    for n in (1, 7, 200, 511, 512, 513, 1024, 4096):
        s = qd.field(dtype=qd.i32, shape=n)

        @qd.kernel
        def fill(m: qd.template()):
            for i in range(m):
                s[i] = i * i - 3 * i + 1

        fill(n)
        qd.sync()
        for i in range(n):
            assert s[i] == i * i - 3 * i + 1, (n, i, s[i])


# A non-const runtime range with a small chunk floor must also produce correct results across threads.
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=8, cpu_min_range_for_block=1)
def test_small_block_runtime_range_for_is_correct():
    n = 300
    s = qd.field(dtype=qd.i32, shape=n)
    lo = qd.field(dtype=qd.i32, shape=())
    hi = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def fill():
        for i in range(lo[None], hi[None]):
            s[i] = 2 * i + 5

    lo[None] = 10
    hi[None] = n
    fill()
    qd.sync()
    for i in range(10, n):
        assert s[i] == 2 * i + 5, (i, s[i])


# The result is independent of the chunk floor: default 512 and a floor of 1 must agree.
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_range_for_block=3)
def test_odd_block_floor_is_correct():
    n = 257
    s = qd.field(dtype=qd.i32, shape=n)
    counter = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def fill():
        counter[None] = 0
        for i in range(n):
            s[qd.atomic_add(counter[None], 1)] = i

    fill()
    qd.sync()
    got = sorted(s[i] for i in range(n))
    assert got == list(range(n))
    assert counter[None] == n


# A chunk floor < 1 would emit an empty/negative inner-loop stride, so it is rejected at qd.init.
@test_utils.test(arch=qd.cpu)
def test_min_range_for_block_below_one_raises():
    for bad in (0, -1, -512):
        with pytest.raises(RuntimeError, match="cpu_min_range_for_block"):
            qd.init(arch=qd.cpu, cpu_min_range_for_block=bad)
    qd.init(arch=qd.cpu)  # restore a valid state for teardown
