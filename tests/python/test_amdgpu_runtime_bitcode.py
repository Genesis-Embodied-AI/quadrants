"""Prepared runtime bytes survive reset; mutable LLVM modules and field layouts do not."""

import os
import subprocess
import sys

import pytest

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_bitcode_fresh_structures(capfd):
    for cycle, size in enumerate((9, 31, 6, 17, 9, 31)):
        qd.reset()
        capfd.readouterr()
        qd.init(
            arch=qd.amdgpu,
            offline_cache=True,
            fast_math=bool(cycle % 2),
            log_level=qd.TRACE,
        )
        output = "".join(capfd.readouterr())
        if cycle:
            assert "AMDGPU runtime bitcode cache hit" in output
        values = qd.Vector.field(2 if cycle % 2 else 3, qd.f32, shape=size)

        @qd.kernel
        def fill():
            for i in values:
                for j in qd.static(range(values.n)):
                    values[i][j] = i * 10 + j

        fill()
        assert values[size - 1][values.n - 1] == (size - 1) * 10 + values.n - 1
    qd.reset()


@pytest.mark.parametrize(
    "diagnostic",
    [
        None,
        "print_kernel_llvm_ir",
        "print_kernel_llvm_ir_optimized",
        "print_kernel_amdgcn",
        "QD_DUMP_IR",
    ],
)
@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_bitcode_bypass(diagnostic, capfd, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if diagnostic == "QD_DUMP_IR":
        monkeypatch.setenv(diagnostic, "1")
        monkeypatch.setenv("QD_DEBUG_DUMP_PATH", str(tmp_path))
    options = {diagnostic: True} if diagnostic and diagnostic != "QD_DUMP_IR" else {}
    for _ in range(2):
        qd.reset()
        capfd.readouterr()
        qd.init(
            arch=qd.amdgpu,
            offline_cache=diagnostic is not None,
            log_level=qd.TRACE,
            **options,
        )
        output = "".join(capfd.readouterr())
        assert "AMDGPU runtime bitcode cache bypass" in output
        assert "AMDGPU runtime bitcode cache hit" not in output
    qd.reset()


@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_bitcode_process_isolation():
    script = """
import quadrants as qd
for fast_math in (True, False, True):
    qd.init(arch=qd.amdgpu, offline_cache=True, fast_math=fast_math, log_level=qd.TRACE)
    qd.reset()
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("QD_OFFLINE_CACHE", "QD_DUMP_IR")
    }
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert output.count("AMDGPU runtime bitcode cache miss") == 1, output
        assert output.count("AMDGPU runtime bitcode cache hit") == 2, output
