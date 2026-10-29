"""The runtime cache survives reset, but GPU modules and runtime state belong to each initialization."""

import os
import subprocess
import sys

import pytest

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_cache_survives_reset(capfd):
    for fast_math in (True, False, True):
        for cycle in range(3):
            qd.reset()
            capfd.readouterr()
            qd.init(
                arch=qd.amdgpu,
                offline_cache=True,
                fast_math=fast_math,
                random_seed=cycle,
                log_level=qd.TRACE,
            )
            output = "".join(capfd.readouterr())
            if cycle:
                assert "AMDGPU runtime HSACO cache hit" in output
            field = qd.field(qd.i32, shape=32)

            @qd.kernel
            def fill(value: qd.i32):
                for i in field:
                    field[i] = value + i

            fill(cycle * 100)
            assert field[17] == cycle * 100 + 17
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
def test_runtime_cache_bypass(diagnostic, capfd, monkeypatch, tmp_path):
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
        assert "AMDGPU runtime HSACO cache bypass" in output
        assert "AMDGPU runtime HSACO cache hit" not in output
        assert "AMDGPU runtime bitcode cache bypass" in output
        assert "AMDGPU runtime bitcode cache hit" not in output
    qd.reset()


@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_cache_process_isolation():
    script = """
import quadrants as qd
for fast_math in (True, True, False, False, True):
    qd.init(arch=qd.amdgpu, offline_cache=True, fast_math=fast_math, log_level=qd.TRACE)
    qd.reset()
"""
    env = {key: value for key, value in os.environ.items() if key not in ("QD_OFFLINE_CACHE", "QD_DUMP_IR")}
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
        assert output.count("AMDGPU runtime HSACO cache miss") == 2, output
        assert output.count("AMDGPU runtime HSACO cache hit") == 3, output
        assert output.count("AMDGPU runtime bitcode cache miss") == 1, output
        assert output.count("AMDGPU runtime bitcode cache hit") == 4, output
