"""Guards the memory ordering of AMDGPU kernel atomics. The accumulating RMWs (add / sub / min / max, int and float)
must be emitted as agent-scope `monotonic`: an agent-scope `seq_cst` RMW brackets every atomic with an L2 writeback and
an L2 invalidate on gfx942 / gfx950. The bitwise RMWs, exchange and compare-and-swap keep `seq_cst`. A functional test
would still pass if the ordering silently went back to `seq_cst` (it would only run slower), so this checks the IR shape
too.
"""

import glob
import re

import numpy as np

import quadrants as qd

from tests.helpers import test_utils

_DEFINE = re.compile(r"^define .*@([\w.$]+)\(")


def _kernel_ir_lines(tmp_path, kernel_name):
    # Lines of the functions generated for `kernel_name` (its task functions and their `function_body*` loop bodies),
    # taken from the AMDGPU IR dumps. Runtime helpers linked into the same module are skipped.
    files = sorted(glob.glob(str(tmp_path / "quadrants_kernel_amdgpu_llvm_ir_*.ll")))
    assert files, "expected print_kernel_llvm_ir to emit a .ll dump, but none was found"
    lines, current = [], None
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _DEFINE.match(line)
                if m:
                    current = m.group(1)
                elif line.startswith("}"):
                    current = None
                elif current and (kernel_name in current or current.startswith("function_body")):
                    lines.append(line)
    return lines


def _rmw(lines, op):
    pattern = re.compile(r"\batomicrmw (volatile )?" + op + r" ")
    found = [line for line in lines if pattern.search(line)]
    assert found, f"expected an `atomicrmw {op}` in the kernel IR"
    return found


@test_utils.test(arch=qd.amdgpu, print_kernel_llvm_ir=True, offline_cache=False)
def test_amdgpu_accumulating_atomics_are_monotonic(monkeypatch, tmp_path):
    # print_kernel_llvm_ir writes .ll dumps to the process CWD; isolate them.
    monkeypatch.chdir(tmp_path)

    n = 4096
    bins = 7
    hist_f = qd.ndarray(qd.f32, shape=bins)
    hist_i = qd.ndarray(qd.i32, shape=bins)
    lo_i = qd.ndarray(qd.i32, shape=2)
    hi_i = qd.ndarray(qd.i32, shape=2)
    lo_f = qd.ndarray(qd.f32, shape=2)
    hi_f = qd.ndarray(qd.f32, shape=2)

    @qd.kernel
    def ordering_accumulate(
        hist_f: qd.types.ndarray(dtype=qd.f32, ndim=1),
        hist_i: qd.types.ndarray(dtype=qd.i32, ndim=1),
        lo_i: qd.types.ndarray(dtype=qd.i32, ndim=1),
        hi_i: qd.types.ndarray(dtype=qd.i32, ndim=1),
        lo_f: qd.types.ndarray(dtype=qd.f32, ndim=1),
        hi_f: qd.types.ndarray(dtype=qd.f32, ndim=1),
    ):
        for i in range(n):
            hist_f[i % bins] += 1.0
            hist_i[i % bins] -= 2
            qd.atomic_min(lo_i[i % 2], i)
            qd.atomic_max(hi_i[i % 2], i)
            qd.atomic_min(lo_f[i % 2], qd.cast(i, qd.f32))
            qd.atomic_max(hi_f[i % 2], qd.cast(i, qd.f32))

    lo_i.from_numpy(np.array([n, n], dtype=np.int32))
    hi_i.from_numpy(np.array([-1, -1], dtype=np.int32))
    lo_f.from_numpy(np.array([n, n], dtype=np.float32))
    hi_f.from_numpy(np.array([-1.0, -1.0], dtype=np.float32))
    ordering_accumulate(hist_f, hist_i, lo_i, hi_i, lo_f, hi_f)

    expected = np.array([len(range(b, n, bins)) for b in range(bins)])
    np.testing.assert_array_equal(hist_f.to_numpy(), expected.astype(np.float32))
    np.testing.assert_array_equal(hist_i.to_numpy(), (-2 * expected).astype(np.int32))
    np.testing.assert_array_equal(lo_i.to_numpy(), [0, 1])
    np.testing.assert_array_equal(hi_i.to_numpy(), [n - 2, n - 1])
    np.testing.assert_array_equal(lo_f.to_numpy(), [0.0, 1.0])
    np.testing.assert_array_equal(hi_f.to_numpy(), [n - 2, n - 1])

    lines = _kernel_ir_lines(tmp_path, "ordering_accumulate")
    for op in ("fadd", "add", "min", "max", "fmin", "fmax"):
        for line in _rmw(lines, op):
            assert 'syncscope("agent") monotonic' in line, f"expected agent-scope monotonic `atomicrmw {op}`:\n{line}"


@test_utils.test(arch=qd.amdgpu, print_kernel_llvm_ir=True, offline_cache=False)
def test_amdgpu_exchange_cas_and_bitwise_atomics_stay_seq_cst(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    flags = qd.ndarray(qd.i32, shape=3)
    old = qd.ndarray(qd.i32, shape=2)

    @qd.kernel
    def ordering_publish(flags: qd.types.ndarray(dtype=qd.i32, ndim=1), old: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(64):
            qd.atomic_or(flags[0], 1 << (i % 31))
            if i == 0:
                old[0] = qd.atomic_exchange(flags[1], 5)
                old[1] = qd.atomic_cas(flags[2], 0, 7)

    ordering_publish(flags, old)
    assert list(flags.to_numpy()[1:]) == [5, 7]
    assert list(old.to_numpy()) == [0, 0]

    lines = _kernel_ir_lines(tmp_path, "ordering_publish")
    for op in ("or", "xchg"):
        for line in _rmw(lines, op):
            assert " seq_cst" in line, f"expected `atomicrmw {op}` to stay seq_cst:\n{line}"
    cas = [line for line in lines if re.search(r"\bcmpxchg ", line)]
    assert cas and all("seq_cst seq_cst" in line for line in cas), "expected cmpxchg to stay seq_cst:\n" + "".join(cas)


@test_utils.test(arch=qd.amdgpu)
def test_amdgpu_contended_atomic_add_is_exact():
    # Many threads hitting few addresses: every update must land exactly once with relaxed ordering.
    n = 1 << 20
    counters = qd.ndarray(qd.i32, shape=3)
    total = qd.ndarray(qd.f64, shape=1)

    @qd.kernel
    def bump(counters: qd.types.ndarray(dtype=qd.i32, ndim=1), total: qd.types.ndarray(dtype=qd.f64, ndim=1)):
        for i in range(n):
            counters[i % 3] += 1
            total[0] += 0.5

    bump(counters, total)
    np.testing.assert_array_equal(counters.to_numpy(), [len(range(b, n, 3)) for b in range(3)])
    assert total.to_numpy()[0] == 0.5 * n
