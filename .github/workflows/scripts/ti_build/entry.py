# -*- coding: utf-8 -*-

import argparse
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
    if (u.system, u.machine) == ("Windows", "AMD64"):
        # Use MSVC on Windows
        setup_clang(as_compiler=False)
        setup_msvc()
    elif u.system == "Linux":
        setup_clang(as_compiler=False)
    else:
        # Use Clang on all other platforms
        setup_clang()

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
