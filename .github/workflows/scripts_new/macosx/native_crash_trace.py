"""Build and install a best-effort native fatal-signal stack trace for the crashing no-fill control."""

import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path


def build(folder, env):
    folder.mkdir(parents=True, exist_ok=True)
    header = Path(__file__).with_suffix(".h").resolve()
    source = folder / "crash_trace.cpp"
    source.write_text('#include <initializer_list>\n#include "native_crash_trace.h"\n')
    library = folder / "native_crash_trace.dylib"
    command = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-g",
        "-dynamiclib",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(header.parent),
        str(source),
        "-o",
        str(library),
    ]
    print("Building native crash trace:", command, flush=True)
    subprocess.run(command, check=True)
    env["QD_NATIVE_CRASH_LIBRARY"] = str(library.resolve())
    (folder / "native-build.json").write_text(
        json.dumps(dict(command=command, header_sha256=hashlib.sha256(header.read_bytes()).hexdigest()), indent=2)
        + "\n"
    )


_library = None


def install():
    global _library
    path = os.environ.get("QD_NATIVE_CRASH_LIBRARY")
    if not path or _library is not None:
        return
    _library = ctypes.CDLL(path)
    call = _library.native_install_crash_trace
    call.argtypes = []
    call.restype = ctypes.c_int
    assert call() == 0
