import ctypes.util
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

import quadrants as qd

from tests.helpers import test_utils

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux dynamic loader regression tests")

KERNEL = """
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, src_ll_cache=False, cpu_max_num_threads=2)

@qd.kernel
def quadrants_square(x: qd.i32) -> qd.i32:
    return x * x

assert quadrants_square(7) == 49
"""

NUMBA_KERNEL = """
import numba
import torch

@numba.njit
def numba_square(x):
    return x * x

assert numba_square(7) == 49
assert torch.arange(4).square().sum().item() == 14
"""


def require_dependency(available, name):
    if not available:
        message = f"Linux loader tests require {name}"
        if os.environ.get("QD_REQUIRE_LOADER_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)


def run_child(tmp_path, source, preload):
    env = os.environ.copy()
    env["QD_OFFLINE_CACHE"] = "0"
    if preload:
        jemalloc = ctypes.util.find_library("jemalloc")
        require_dependency(jemalloc, "libjemalloc")
        env["LD_PRELOAD"] = " ".join(filter(None, [jemalloc, env.get("LD_PRELOAD")]))
        # A missing preload is otherwise only a loader warning and could produce a false pass.
        source = "from pathlib import Path\nassert 'libjemalloc' in Path('/proc/self/maps').read_text()\n" + source
    # A real file lets the Quadrants frontend retrieve the kernel source. Wait for interpreter shutdown too: the
    # allocator mismatch can abort after every assertion has passed.
    script = tmp_path / "loader_child.py"
    script.write_text(textwrap.dedent(source))
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Child exited {result.returncode}\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("compile_kernel", [False, True], ids=["import", "kernel"])
@test_utils.test(arch=qd.cpu)
def test_jemalloc_shutdown(tmp_path, compile_kernel):
    run_child(tmp_path, KERNEL if compile_kernel else "import quadrants\n", preload=True)


@pytest.mark.parametrize("preload", [False, True], ids=["default-allocator", "jemalloc"])
@pytest.mark.parametrize("quadrants_first", [False, True], ids=["numba-first", "quadrants-first"])
@test_utils.test(arch=qd.cpu)
def test_llvm_coexistence(tmp_path, preload, quadrants_first):
    for module in ("numba", "torch"):
        require_dependency(importlib.util.find_spec(module), module)
    source = KERNEL + NUMBA_KERNEL if quadrants_first else NUMBA_KERNEL + KERNEL
    # Exercise both runtimes again after the second library has initialized.
    source += "\nassert quadrants_square(9) == numba_square(9) == 81\n"
    run_child(tmp_path, source, preload)


@pytest.mark.skipif(os.environ.get("QD_TEST_MESA_LOADER") != "1", reason="Requires Mesa software EGL/LLVM")
@pytest.mark.parametrize("quadrants_first", [False, True], ids=["mesa-first", "quadrants-first"])
@test_utils.test(arch=qd.cpu)
def test_mesa_llvm_coexistence(tmp_path, quadrants_first):
    # Mesa loading a second LLVM was the original reason for DEEPBIND (taichi-dev/taichi#1355). Use software EGL
    # so this path can run on CPU CI without a display server or GPU. Keep the display alive through process exit.
    mesa = """
import ctypes
import os
from pathlib import Path

os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
egl = ctypes.CDLL('libEGL.so.1')
egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
egl.eglGetDisplay.restype = ctypes.c_void_p
egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
egl.eglInitialize.restype = ctypes.c_uint
display = egl.eglGetDisplay(None)
major, minor = ctypes.c_int(), ctypes.c_int()
assert egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)) == 1
maps = Path('/proc/self/maps').read_text()
assert 'libEGL_mesa' in maps and 'libLLVM' in maps, maps
"""
    source = KERNEL + mesa if quadrants_first else mesa + KERNEL
    source += "\nassert quadrants_square(9) == 81\n"
    run_child(tmp_path, source, preload=True)
