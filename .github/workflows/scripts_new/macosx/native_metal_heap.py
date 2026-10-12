"""Direct Metal heap/buffer controls, compiled only on the Mac runner."""

import ctypes as C
import hashlib
import json
import os
import subprocess
from pathlib import Path


class ProbeStats(C.Structure):
    _fields_ = [
        (key, C.c_uint64)
        for key in ("allocation_bytes", "alignment", "second_offset", "registry_id", "device_object", "command_status")
    ]


def build(output, env):
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix(".mm").resolve()
    library = output / "native_metal_heap.dylib"
    command = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-fobjc-arc",
        "-dynamiclib",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        str(source),
        "-o",
        str(library),
    ]
    print("Building direct Metal probe:", command, flush=True)
    subprocess.run(command, check=True)
    env["QD_NATIVE_METAL_LIBRARY"] = str(library.resolve())
    (output / "native-build.json").write_text(
        json.dumps(
            dict(
                command=command,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                runtime_header_sha256=hashlib.sha256(
                    source.with_name("native_metal_runtime.h").read_bytes()
                ).hexdigest(),
                library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


class NativeMetalHeap:
    def __init__(self, work):
        assert work in ("metal-heap", "metal-heap-small", "metal-buffer", "metal-runtime-heap", "metal-runtime-buffer")
        path = Path(os.environ["QD_NATIVE_METAL_LIBRARY"])
        self.library = C.CDLL(str(path))
        self.runtime = work.startswith("metal-runtime-")
        self.call = self.library.native_metal_runtime_cycle if self.runtime else self.library.native_metal_cycle
        self.call.argtypes = (
            [C.c_int, C.POINTER(ProbeStats)] if self.runtime else [C.c_uint64, C.c_int, C.POINTER(ProbeStats)]
        )
        self.call.restype = C.c_int
        self.work = work
        self.bytes = 16384 if work == "metal-heap-small" else 32 << 20
        self.use_heap = work not in ("metal-buffer", "metal-runtime-buffer")
        self.buffer_records = []
        self.metadata = dict(
            work=work,
            library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            api="Metal",
            application_shaders=0,
            probe_compute_shaders=1 if self.runtime else 0,
            runtime_pattern=self.runtime,
            allocation_bytes=self.bytes,
            placement_heap=self.use_heap,
            device_acquisition="MTLCopyAllDevices cached per process" if self.runtime else "MTLCopyAllDevices each cycle",
            autorelease_pool="per cycle",
        )

    def cycle(self):
        stats = ProbeStats()
        result = (
            self.call(self.use_heap, C.byref(stats))
            if self.runtime
            else self.call(self.bytes, self.use_heap, C.byref(stats))
        )
        assert result == 0, f"Native Metal probe failed at stage {result}"
        assert stats.allocation_bytes >= self.bytes and stats.command_status == 4
        if not self.buffer_records:
            self.buffer_records = [{key: getattr(stats, key) for key, _ in stats._fields_}]

    def close(self):
        pass
