import pathlib
import re

import pytest

import quadrants as qd

from tests import test_utils

# GlobalTemporaryStmt prints as e.g. "<*i64> $7 = global tmp var (offset = 8 B)".
_GTMP_RE = re.compile(r"<\*(\w+)> .* = global tmp var \(offset = (\d+) B\)")


def _gtmp_offsets_by_type(dump_dir: pathlib.Path, kernel_name: str):
    files = sorted(dump_dir.glob(f"*{kernel_name}*after_offload*"))
    assert files, f"no after_offload IR dump for {kernel_name} in {dump_dir}"
    per_type: dict[str, set[int]] = {}
    for f in files:
        for line in f.read_text().splitlines():
            m = _GTMP_RE.search(line)
            if m:
                per_type.setdefault(m.group(1), set()).add(int(m.group(2)))
    return per_type


@test_utils.test(arch=[qd.cpu], offline_cache=False)
def test_stable_gtmp_offsets_are_content_keyed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Cross-offload global-temp slots are assigned in a content-keyed order.

    A value's global-temp offset depends on WHAT it is (statement kind, type, constant/arg identity,
    operand structure), not on the order in which offloads happen to be traversed. Two kernels that
    produce the same set of cross-offload values in a different source order must therefore get the
    same value->offset mapping. With the previous traversal-order allocation, swapping the two
    producing loops swapped the slots (i32@0/i64@8 vs i64@0/i32@8), which perturbed the IR -- and
    hence the per-task cache key -- of tasks that were not edited.
    """
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    n = 64

    @qd.kernel
    def order_ab(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        a = 0  # i32 cross-offload value, produced first
        for i in range(3):
            a += i
        b = qd.cast(0, qd.i64)  # i64 cross-offload value, produced second
        for i in range(5):
            b += qd.cast(i * i, qd.i64)
        for i in range(n):
            out[i] = a + qd.cast(b, qd.i32) + i

    @qd.kernel
    def order_ba(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        b = qd.cast(0, qd.i64)  # same two values, i64 produced first
        for i in range(5):
            b += qd.cast(i * i, qd.i64)
        a = 0
        for i in range(3):
            a += i
        for i in range(n):
            out[i] = a + qd.cast(b, qd.i32) + i

    out = qd.ndarray(qd.i32, shape=(n,))
    order_ab(out)
    qd.sync()
    res_ab = out.to_numpy().copy()

    order_ba(out)
    qd.sync()
    res_ba = out.to_numpy().copy()

    # Both orderings compute the same result.
    assert (res_ab == res_ba).all()

    off_ab = _gtmp_offsets_by_type(tmp_path, "order_ab")
    off_ba = _gtmp_offsets_by_type(tmp_path, "order_ba")

    # Both a cross-offload i32 slot and a cross-offload i64 slot are present.
    assert "i32" in off_ab and "i64" in off_ab, off_ab
    # Each value gets exactly one slot, and the two values occupy disjoint (non-overlapping) slots.
    assert len(off_ab["i32"]) == 1 and len(off_ab["i64"]) == 1, off_ab
    assert off_ab["i32"].isdisjoint(off_ab["i64"]), off_ab
    # The value->offset mapping is invariant to the source/traversal order of the producing offloads.
    assert off_ab == off_ba, (off_ab, off_ba)


@pytest.mark.xfail(
    reason="Known limitation flagged in PR #864 review (r3776617702): global-temp slots are bump-allocated in "
    "stable_key order, so a slot's offset is the cumulative size of every lower-keyed value rather than a function "
    "of the value alone. Inserting an unrelated cross-offload value whose key sorts earlier therefore still shifts "
    "an unchanged value's offset, re-keying tasks that were not edited. This costs some per-task cache reuse but "
    "never correctness (within a build all tasks agree on the layout). Remove this xfail when offsets are made "
    "fully content-addressed. Non-strict: stable_key incorporates typeid(*s).name(), whose ordering is "
    "implementation-defined, so whether the inserted i32 sorts before the i64 (and thus whether this property "
    "happens to hold on a given toolchain) is not portable -- a strict xfail would flip to a suite failure as an "
    "XPASS on a toolchain that orders them the other way, even though the limitation still exists.",
    strict=False,
)
@test_utils.test(arch=[qd.cpu], offline_cache=False)
def test_gtmp_offset_stable_under_unrelated_insertion(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A cross-offload value's global-temp slot should depend only on the value itself.

    Adding an *unrelated* cross-offload value elsewhere in the kernel should not move an existing
    value's offset (otherwise the untouched value's task re-keys and misses the per-task cache).
    Today it can still move: when the inserted i32 value sorts before the i64 value by stable_key,
    the dense bump allocator pushes the i64 slot to a higher offset. Whether that ordering happens
    is toolchain-dependent (stable_key uses typeid(*s).name()), so this is a non-strict xfail: on
    toolchains where the i32 sorts after the i64 the offset is unchanged and the test XPASSes, which
    is fine -- the limitation is that the offset is *not guaranteed* to be stable, not that it always
    moves.
    """
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    n = 64

    @qd.kernel
    def gtmp_only_big(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        b = qd.cast(0, qd.i64)  # a single i64 cross-offload value
        for i in range(5):
            b += qd.cast(i * i, qd.i64)
        for i in range(n):
            out[i] = qd.cast(b, qd.i32) + i

    @qd.kernel
    def gtmp_big_and_small(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        b = qd.cast(0, qd.i64)  # the identical i64 value...
        for i in range(5):
            b += qd.cast(i * i, qd.i64)
        a = 0  # ...plus an unrelated i32 value inserted elsewhere in the kernel
        for i in range(3):
            a += i
        for i in range(n):
            out[i] = qd.cast(b, qd.i32) + a + i

    out = qd.ndarray(qd.i32, shape=(n,))
    gtmp_only_big(out)
    qd.sync()
    gtmp_big_and_small(out)
    qd.sync()

    off_only = _gtmp_offsets_by_type(tmp_path, "gtmp_only_big")
    off_both = _gtmp_offsets_by_type(tmp_path, "gtmp_big_and_small")

    # Scenario sanity: the i64 value is present in both, and the extra i32 value only in the second.
    assert "i64" in off_only and "i64" in off_both, (off_only, off_both)
    assert "i32" in off_both, off_both
    # The desired (not-yet-achieved) property: the i64 value's offset is unaffected by the unrelated
    # i32 value. This is not guaranteed today (the i32 may sort before the i64 and bump its offset),
    # which is why the test is a non-strict xfail rather than a hard assertion.
    assert off_only["i64"] == off_both["i64"], (off_only, off_both)
