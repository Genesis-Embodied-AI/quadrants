"""The runtime cache survives reset, but GPU modules and runtime state belong to each initialization."""

import os
import subprocess
import sys

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize(
    "diagnostic", [None, "print_kernel_llvm_ir", "print_kernel_llvm_ir_optimized", "print_kernel_amdgcn", "QD_DUMP_IR"]
)
@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_cache_bypass(diagnostic, capfd, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if diagnostic == "QD_DUMP_IR":
        monkeypatch.setenv(diagnostic, "1")
        monkeypatch.setenv("QD_DEBUG_DUMP_PATH", str(tmp_path))
    options = {diagnostic: True} if diagnostic and diagnostic != "QD_DUMP_IR" else {}
    for _ in range(2):
        qd.reset()
        capfd.readouterr()
        qd.init(arch=qd.amdgpu, offline_cache=diagnostic is not None, log_level=qd.TRACE, **options)
        output = "".join(capfd.readouterr())
        assert "AMDGPU runtime HSACO cache bypass" in output
        assert "AMDGPU runtime HSACO cache hit" not in output
        assert "AMDGPU runtime bitcode cache bypass" in output
        assert "AMDGPU runtime bitcode cache hit" not in output
    qd.reset()


@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_cache_resets_and_process_isolation(tmp_path):
    qd.reset()
    script = tmp_path / "runtime_cache_check.py"
    script.write_text(
        """
import quadrants as qd
for cycle, fast_math in enumerate((True, True, False, False, True)):
    qd.init(arch=qd.amdgpu, offline_cache=True, fast_math=fast_math, random_seed=cycle, log_level=qd.TRACE)
    field = qd.field(qd.i32, shape=8 + cycle)

    @qd.kernel
    def fill(value: qd.i32):
        for i in field:
            field[i] = value + i

    fill(cycle * 100)
    assert field[7] == cycle * 100 + 7
    qd.reset()
"""
    )
    env = {key: value for key, value in os.environ.items() if key not in ("QD_OFFLINE_CACHE", "QD_DUMP_IR")}
    for _ in range(2):
        result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert output.count("AMDGPU runtime HSACO cache miss") == 2, output
        assert output.count("AMDGPU runtime HSACO cache hit") == 3, output
        assert output.count("AMDGPU runtime bitcode cache miss") == 1, output
        assert output.count("AMDGPU runtime bitcode cache hit") >= 4, output
