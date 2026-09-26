import numpy as np
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


# A non-const runtime range with a small chunk floor must also produce correct results across threads, including the
# awkward bounds: empty ranges, reversed (begin > end) ranges, and negative begins. Each in-range element must be
# written exactly once and nothing outside the range touched.
@pytest.mark.parametrize("threads", [1, 4])
@test_utils.test(arch=qd.cpu, cpu_min_range_for_block=1)
def test_small_block_runtime_range_for_is_correct(threads):
    qd.init(arch=qd.cpu, cpu_max_num_threads=threads, cpu_min_range_for_block=1)
    offset = 20
    size = 1100
    out = qd.field(qd.i32, shape=size)

    @qd.kernel
    def fill(begin: qd.i32, end: qd.i32) -> qd.i32:
        total = 0
        for i in range(begin, end):
            out[i + offset] += 1
            total += i
        return total

    for begin, end in [(-7, -7), (9, 3), (-7, -6), (-7, 6), (0, 200), (0, 512), (3, 516), (3, 1028)]:
        out.fill(0)
        assert fill(begin, end) == sum(range(begin, end)), (begin, end)
        expected = np.zeros(size, dtype=np.int32)
        if end > begin:
            expected[begin + offset : end + offset] = 1
        np.testing.assert_array_equal(out.to_numpy(), expected)


# A small floor must not parallelize an explicitly serialized loop: a serial recurrence still computes correctly.
@pytest.mark.parametrize("serialize", [False, True])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_range_for_block=1)
def test_serialized_loop_still_correct_with_small_floor(serialize):
    @qd.kernel
    def recurrence() -> qd.i32:
        result = 0
        if qd.static(serialize):
            qd.loop_config(serialize=True)
        else:
            qd.loop_config(parallelize=1)
        for i in range(200):
            result = (result * 3 + i) % 10007
        return result

    expected = 0
    for i in range(200):
        expected = (expected * 3 + i) % 10007
    assert recurrence() == expected


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


# cpu_min_range_for_block changes the generated code, so it is part of the offline-cache key: a different floor must
# produce a distinct on-disk cache entry, and the same floor must reuse it.
@test_utils.test(arch=qd.cpu)
def test_min_range_for_block_offline_cache_key(tmp_path):
    def cached_files():
        return {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}

    def run(minimum):
        qd.init(
            arch=qd.cpu,
            cpu_max_num_threads=4,
            cpu_min_range_for_block=minimum,
            offline_cache=True,
            offline_cache_file_path=str(tmp_path),
            offline_cache_cleaning_policy="never",
        )

        @qd.kernel
        def total(n: qd.i32) -> qd.i32:
            result = 0
            for i in range(n):
                result += i
            return result

        assert total(200) == 19900
        qd.reset()  # flush the in-process cache to disk
        return cached_files()

    default_files = run(512)
    assert default_files, "default floor produced no cache entry"
    small_files = run(1)
    assert default_files < small_files, "a different floor must add a distinct cache entry"
    assert run(512) == small_files, "re-running the default floor must reuse its cache entry"
    assert run(1) == small_files, "re-running the small floor must reuse its cache entry"
