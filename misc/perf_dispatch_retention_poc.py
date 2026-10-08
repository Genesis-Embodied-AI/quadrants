"""Demonstrate ndarray retention by perf_dispatch's single-compatible-implementation cache.

Run with Python 3.10 and Quadrants installed:
    python misc/perf_dispatch_retention_poc.py --arch cpu
    python misc/perf_dispatch_retention_poc.py --arch cuda

The assertions describe the retention bug, so a version that fixes it should fail the retention assertion.
"""

import argparse
import gc
import hashlib
import inspect
import weakref

import quadrants as qd
from quadrants.lang import _perf_dispatch, impl


@qd.kernel
def touch(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for i in a:
        a[i] = 7.0


def allocate_and_call(call, elements, keyword=False):
    a = qd.ndarray(qd.f32, shape=(elements,))
    ref = weakref.ref(a)
    if keyword:
        call(a=a)
    else:
        call(a)
    qd.sync()
    assert a[0] == 7.0
    return ref  # Only a weak reference escapes this function.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mib", type=int, default=16, help="Size of each ndarray buffer in MiB")
    args = parser.parse_args()
    if args.mib <= 0:
        parser.error("--mib must be positive")
    arch = qd.cpu if args.arch == "cpu" else qd.cuda
    qd.init(arch=arch, enable_fallback=False, offline_cache=False)
    print(f"Quadrants version: {qd.__version__}; backend: {args.arch}", flush=True)
    source = inspect.getsource(_perf_dispatch.PerformanceDispatcher)
    print(f"PerformanceDispatcher source SHA256: {hashlib.sha256(source.encode()).hexdigest()}", flush=True)
    elements = args.mib * 1024 * 1024 // 4
    baseline = impl.get_runtime().prog._get_num_ndarrays()

    def report(label, ref, expected_alive):
        gc.collect()
        qd.sync()
        alive = ref() is not None
        count = impl.get_runtime().prog._get_num_ndarrays() - baseline
        print(f"{label}: alive={alive}, live_ndarrays={count}", flush=True)
        assert alive == expected_alive, label
        assert count == int(expected_alive), label

    direct_ref = allocate_and_call(touch, elements)
    report("direct kernel after scope exit", direct_ref, False)

    for keyword in (False, True):
        # A Python implementation wrapper avoids capturing the dispatcher in a registered kernel's closure.
        @qd.perf_dispatch(get_geometry_hash=lambda a: hash(a.shape))
        def dispatch(a):
            pass

        @dispatch.register
        def implementation(a):
            touch(a)

        label = "keyword" if keyword else "positional"
        ref = allocate_and_call(dispatch, elements, keyword=keyword)
        report(f"{label} dispatch after scope exit ({args.mib} MiB)", ref, True)
        if keyword:
            assert dispatch._cached_kwargs["a"] is ref()
        else:
            assert dispatch._cached_args[0] is ref()
        print(f"{label}: confirmed the cached argument is the surviving ndarray", flush=True)

        # Diagnostic intervention only: private cache attributes are not a supported cleanup API.
        dispatch._cached_args = None
        dispatch._cached_kwargs = None
        dispatch._cached_impl = None
        report(f"{label} after clearing dispatcher argument cache", ref, False)

    qd.reset()
    print("PASS: direct call releases; perf_dispatch retains; clearing its cache releases.", flush=True)


if __name__ == "__main__":
    main()
