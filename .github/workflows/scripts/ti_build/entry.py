# -*- coding: utf-8 -*-

import argparse
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
def _get_version_defines() -> list[str]:
    """
    Resolve QD_VERSION_{MAJOR,MINOR,PATCH} the same way setup.py does, so
    that quadrants/common/version.h.in (consumed via configure_file in
    CMakeLists.txt) gets non-empty values. Without these, the generated
    quadrants/common/version.h has empty `#define QD_VERSION_MAJOR` macros
    and any source that compares against them fails to parse.
    """
    try:
        from setuptools_scm import get_version as scm_get_version

        version = scm_get_version()
        parts = version.split("+")[0].split(".dev")[0].split("rc")[0].split("b")[0].split(".")
        major = parts[0] if len(parts) > 0 else "0"
        minor = parts[1] if len(parts) > 1 else "0"
        patch = parts[2] if len(parts) > 2 else "0"
    except Exception:
        major, minor, patch = "0", "0", "0"
    return [
        f"-DQD_VERSION_MAJOR={major}",
        f"-DQD_VERSION_MINOR={minor}",
        f"-DQD_VERSION_PATCH={patch}",
    ]


@banner("Configure Quadrants (cmake only, no build)")
def configure_only() -> None:
    """
    Run cmake configure into ./build with the user-supplied
    QUADRANTS_CMAKE_ARGS, *without* building. This is used by linter-style
    workflows (e.g. clang-tidy) that only need compile_commands.json. It
    reuses the same toolchain bootstrap as `action_wheel` (clang, prebuilt
    LLVM, sccache, OS packages) but skips skbuild / setup.py entirely.
    """
    cmake_args.writeback()
    extra_args = os.environ.get("QUADRANTS_CMAKE_ARGS", "").split()
    build_dir = os.environ.get("QUADRANTS_BUILD_DIR", "build")
    cmake = sh.bake("cmake")
    with nice():
        cmake(
            "-S",
            ".",
            "-B",
            build_dir,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            *_get_version_defines(),
            *extra_args,
        )


@banner("Build Quadrants Wheel")
def build_wheel(python: Command) -> None:
    extra = []

    cmake_args.writeback()
    u = platform.uname()
    match (u.system, u.machine):
        case ("Linux", "x86_64"):
            extra.extend(["-p", "manylinux_2_27_x86_64"])
        case ("Linux", "arm64") | ("Linux", "aarch64"):
            extra.extend(["-p", "manylinux_2_27_aarch64"])
        case ("Darwin", _):
            extra.extend(["-p", "macosx-11.0-arm64"])

    python("setup.py", "clean")

    with nice():
        python("setup.py", "bdist_wheel", *extra)


def setup_basic_build_env():
    u = platform.uname()
    setup_clang(as_compiler=False)
    if (u.system, u.machine) == ("Windows", "AMD64"):
        # Use MSVC on Windows
        setup_msvc()

    setup_llvm()
    if u.system == "Linux":
        # We support & test Vulkan shader debug printf on Linux
        # This is done through the validation layer
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


def action_configure():
    """
    Bootstrap the same toolchain as `action_wheel` (OS packages, clang,
    prebuilt LLVM, sccache discovery) and then stop after `cmake configure`.
    Produces ./build/compile_commands.json for downstream tools like
    clang-tidy. Does not build any C++ targets and does not produce a wheel.
    """
    setup_os_pkgs()
    _sccache, _python = setup_basic_build_env()
    handle_alternate_actions()
    configure_only()


def parse_args():
    parser = argparse.ArgumentParser()

    # Possible actions:
    #   wheel:     build the wheel (default)
    #   configure: cmake configure only, into ./build, producing
    #              compile_commands.json for tools like clang-tidy.
    help = 'Action, one of: "wheel" (default), "configure".'
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
        "configure": action_configure,
    }

    dispatch.get(options.action, action_notimpl)()

    return 0
