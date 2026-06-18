# -*- coding: utf-8 -*-

import argparse
import glob
import os
import platform
import sys

import psutil

from . import misc
from .alter import handle_alternate_actions
from .cmake import cmake_args
from .compiler import setup_clang, setup_msvc
from .llvm import setup_llvm
from .misc import banner
from .ospkg import setup_os_pkgs
from .sccache import setup_sccache
from .tinysh import Command, CommandFailed, nice, sh


# -- code --
@banner("Build Quadrants Wheel")
def build_wheel(python: Command) -> None:
    cmake_args.writeback()
    # scikit-build-core reads CMake args from CMAKE_ARGS (space-separated); bridge it from the QUADRANTS_CMAKE_ARGS
    # env that cmake_args.writeback() populates.
    os.environ["CMAKE_ARGS"] = os.environ.get("QUADRANTS_CMAKE_ARGS", "")

    plat = None
    u = platform.uname()
    match (u.system, u.machine):
        case ("Linux", "x86_64"):
            plat = "manylinux_2_27_x86_64"
        case ("Linux", "arm64") | ("Linux", "aarch64"):
            plat = "manylinux_2_27_aarch64"
        case ("Darwin", _):
            plat = "macosx_11_0_arm64"

    # Clear stale wheels so the tag-stamping step below is unambiguous.
    for whl in glob.glob("dist/quadrants-*.whl"):
        os.remove(whl)

    # Build via scikit-build-core. Use `pip wheel` (not `python -m build`) because the repo ships a top-level build.py
    # that would shadow the `build` module under `-m`. --no-build-isolation: the LLVM/clang toolchain is not
    # pip-installable (it is provisioned by setup_basic_build_env above), so build deps come from this env.
    with nice():
        python("-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", "dist", ".")

    if plat:
        wheels = glob.glob("dist/quadrants-*.whl")
        assert len(wheels) == 1, f"expected exactly one freshly built wheel, got {wheels}"
        # scikit-build-core emits a bare linux_x86_64 / macosx_* tag; stamp the distribution platform tag the project
        # ships under (manylinux / macOS).
        python("-m", "wheel", "tags", "--platform-tag", plat, "--remove", wheels[0])


def setup_basic_build_env():
    u = platform.uname()
    setup_clang(as_compiler=False)
    if (u.system, u.machine) == ("Windows", "AMD64"):
        # Use MSVC on Windows
        setup_msvc()

    setup_llvm()
    if u.system in ("Linux", "Darwin"):
        # Linux: validation layers + SPIR-V tools (shader debug printf support).
        # macOS: the SDK bundles a current MoltenVK that advertises `VK_KHR_buffer_device_address`, which
        # the adstack sizer shader needs for `ExternalTensorRead` via Physical Storage Buffer addressing.
        # The Vulkan-Taichi-assets pin at `quadrants/rhi/CMakeLists.txt:40` is too old for PSB; wiring
        # `setup_vulkan()` here lets the CMake glue pick up `$VULKAN_SDK/lib/libMoltenVK.dylib` (the flat
        # layout LunarG's macOS SDK uses - see the layout note in `vulkan.py::setup_vulkan`) and ship
        # that in the wheel instead.
        from .vulkan import setup_vulkan

        setup_vulkan()

    sccache = setup_sccache()
    python = sh.bake(sys.executable)
    return sccache, python


def _is_sccache_running():
    for proc in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            if proc.info["cmdline"] and "sccache" in proc.info["cmdline"][0]:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def action_wheel():
    setup_os_pkgs()
    sccache, python = setup_basic_build_env()

    # Explicitly start sccache server before the build
    if _is_sccache_running():
        print("sccache already appears to be running")
    else:
        sccache("--start-server")

    handle_alternate_actions()
    build_wheel(python)
    try:
        sccache("-s")
    except CommandFailed:
        pass


def parse_args():
    parser = argparse.ArgumentParser()

    # Possible actions:
    #   wheel: build the wheel
    help = 'Action, may be build target "wheel" for opening the cache directory.'
    parser.add_argument("action", type=str, nargs="?", default="wheel", help=help)

    help = "Do not build, write environment variables to file instead"
    parser.add_argument("-w", "--write-env", type=str, default=None, help=help)

    help = "Do not build, start a shell with environment variables set instead"
    parser.add_argument("-s", "--shell", action="store_true", help=help)

    help = (
        "Python version to use, e.g. '3.7', '3.11', or 'native' to not use an isolated python environment. "
        "Defaults to the same version of the current python interpreter."
    )
    parser.add_argument("--python", default=None, help=help)

    options = parser.parse_args()
    return options


def main() -> int:
    options = parse_args()
    misc.options = options

    def action_notimpl():
        raise RuntimeError(f"Unknown action: {options.action}")

    dispatch = {
        "wheel": action_wheel,
    }

    dispatch.get(options.action, action_notimpl)()

    return 0
