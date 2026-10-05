"""Install pinned Linux shader compilers compatible with the wheel build containers."""

import platform

from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home

GLSLANG_RELEASE = "glslang-15.4.0-20261005132022"
GLSLANG_ARCHIVES = {
    "x86_64": (
        "glslang-15.4.0-manylinux_2_28_x86_64.tar.xz",
        "77b4e1852ef75e1b9a42a2cf2fde602cb3339235752990dc32ce8fcc1c5eedb7",
    ),
    "aarch64": (
        "glslang-15.4.0-manylinux_2_34_aarch64.tar.xz",
        "badc8a6798680e559cfb68f5661deb7469958fa833524f04f4ea5d1644790f37",
    ),
}


@banner("Setup glslang")
def setup_glslang():
    if platform.system() != "Linux" or cmake_args.get_effective("QD_GLSLANG_EXECUTABLE"):
        return
    machine = platform.machine()
    if machine == "arm64":
        machine = "aarch64"
    if machine not in GLSLANG_ARCHIVES:
        return
    archive, checksum = GLSLANG_ARCHIVES[machine]
    prefix = get_cache_home() / GLSLANG_RELEASE / machine
    url = f"https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/releases/download/{GLSLANG_RELEASE}/{archive}"
    download_dep(url, prefix, strip=1, sha256=checksum)
    cmake_args["QD_GLSLANG_EXECUTABLE"] = str(prefix / "bin" / "glslangValidator")
