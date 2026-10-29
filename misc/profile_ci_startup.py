"""Measure process startup and repeated runtime lifetimes using an installed wheel."""

import argparse
import cProfile
import ctypes
import gc
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time


START = time.perf_counter()


def measure(name, operation):
    start = time.perf_counter()
    result = operation()
    print(json.dumps({"phase": name, "seconds": time.perf_counter() - start}), flush=True)
    return result


def child(args):
    profiler = cProfile.Profile()
    profiler.enable()

    def import_quadrants():
        import quadrants

        return quadrants

    qd = measure("import_quadrants", import_quadrants)
    print(json.dumps({"quadrants": qd.__file__, "python": sys.version, "torch_imported": "torch" in sys.modules}), flush=True)
    for iteration in range(args.cycles):
        prefix = f"cycle_{iteration}"
        measure(
            f"{prefix}.init",
            lambda: qd.init(
                arch=getattr(qd, args.arch),
                enable_fallback=False,
                offline_cache=True,
                offline_cache_file_path=str(args.output / "kernel-cache"),
                device_memory_GB=0.3,
                log_level=qd.TRACE if args.trace else qd.ERROR,
            ),
        )

        @qd.kernel
        def increment(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
            for i in range(a.shape[0]):
                a[i] += 1

        array = measure(f"{prefix}.allocate", lambda: qd.ndarray(qd.i32, shape=1024))
        measure(f"{prefix}.fill", lambda: (array.fill(0), qd.sync()))
        measure(f"{prefix}.first_kernel", lambda: (increment(array), qd.sync()))

        def repeat():
            for _ in range(100):
                increment(array)
            qd.sync()

        measure(f"{prefix}.100_launches", repeat)
        result = measure(f"{prefix}.readback", array.to_numpy)
        assert (result == 101).all()
        del array
        measure(f"{prefix}.gc", gc.collect)
        measure(f"{prefix}.reset", qd.reset)
    profiler.disable()
    profiler.dump_stats(str(args.output / f"{args.label}.prof"))
    with (args.output / f"{args.label}-profile.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(70)
    print(json.dumps({"phase": "child_before_exit", "seconds": time.perf_counter() - START}), flush=True)


def driver_child(args):
    if args.torch_first:
        measure("import_torch", lambda: __import__("torch"))
    library_name, prefix = ("libamdhip64.so", "hip") if args.arch == "amdgpu" else ("libcuda.so.1", "cu")
    library = measure("load_driver_library", lambda: ctypes.CDLL(library_name))

    def invoke(name, types, values):
        function = getattr(library, prefix + name)
        function.argtypes = types
        function.restype = ctypes.c_int
        result = measure(prefix + name, lambda: function(*values))
        if result:
            raise RuntimeError(f"{prefix}{name} failed: {result}")

    version = ctypes.c_int()
    invoke("DriverGetVersion", [ctypes.POINTER(ctypes.c_int)], [ctypes.byref(version)])
    invoke("Init", [ctypes.c_uint], [0])
    count = ctypes.c_int()
    count_function = "GetDeviceCount" if args.arch == "amdgpu" else "DeviceGetCount"
    invoke(count_function, [ctypes.POINTER(ctypes.c_int)], [ctypes.byref(count)])
    context = ctypes.c_void_p()
    invoke("DevicePrimaryCtxRetain", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], [ctypes.byref(context), 0])
    print(json.dumps({"driver_version": version.value, "device_count": count.value}), flush=True)
    print(json.dumps({"phase": "child_before_exit", "seconds": time.perf_counter() - START}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["amdgpu", "cuda", "cpu"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--label", default="default")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--driver-only", action="store_true")
    parser.add_argument("--torch-first", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.child:
        if args.driver_only:
            driver_child(args)
        else:
            child(args)
        return
    for index in range(3):
        command = [
            sys.executable, __file__, "--child", "--arch", args.arch,
            "--output", str(args.output), "--cycles", str(args.cycles),
            "--label", f"{args.label}-{index}",
        ]
        if args.trace:
            command.append("--trace")
        if args.driver_only:
            command.append("--driver-only")
        if args.torch_first:
            command.append("--torch-first")
        measure(f"process_{index}.including_exit", lambda: subprocess.run(command, check=True, timeout=240))
    print(json.dumps({"enable_torch": os.environ.get("QD_ENABLE_TORCH", "default")}), flush=True)


if __name__ == "__main__":
    main()
