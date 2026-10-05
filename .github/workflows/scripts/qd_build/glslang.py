"""Install pinned Linux shader compilers compatible with the wheel build containers."""

import platform

from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home

GLSLANG_RELEASE = "glslang-15.4.0-202610051822"
GLSLANG_ARCHIVES = {
    "x86_64": (
        "glslang-15.4.0-manylinux_2_28_x86_64.tar.xz",
        "fdaab840fddc91acda447eaabb9d0f534177d40acff610bcefa81929a26954f0",
    ),
    "aarch64": (
        "glslang-15.4.0-manylinux_2_34_aarch64.tar.xz",
        "7fa035b0282e54598a26b28dafd0950c8a278cb89026b9d243f8dd79c5313ff1",
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
