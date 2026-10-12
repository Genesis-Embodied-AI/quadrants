"""Experimental ordered weak registry, plus a standalone kernel/closure lifetime probe.

Only runtime registry ownership changes. Live Python kernel wrappers retain their primal/adjoint kernels normally.
This is an attribution experiment, not a production patch. Test fixtures and GC policy remain unchanged.
"""

import gc
import json
import os
import weakref
from pathlib import Path


class WeakKernelRegistry:
    def __init__(self, kernels=()):
        self._values = weakref.WeakValueDictionary()
        for kernel in kernels:
            self.append(kernel)

    def append(self, kernel):
        self._values[id(kernel)] = kernel

    def __iter__(self):
        return iter(self._values.values())

    def __len__(self):
        return len(self._values)


def install():
    from quadrants.lang import impl

    if os.environ["QD_KERNEL_REGISTRY"] == "strong":
        return
    if os.environ["QD_KERNEL_REGISTRY"] != "weak":
        raise ValueError("Expected strong or weak registry")
    original = impl.PyQuadrants.__init__

    def initialize(self, kernels=None):
        original(self, kernels)
        if not isinstance(self.kernels, WeakKernelRegistry):
            self.kernels = WeakKernelRegistry(self.kernels)

    impl.PyQuadrants.__init__ = initialize
    impl.get_runtime().kernels = WeakKernelRegistry(impl.get_runtime().kernels)


def pytest_configure(config):
    install()


def lifetime_probe(output):
    import quadrants as qd
    from quadrants.lang import impl

    install()
    qd.init(arch=qd.vulkan)

    class Payload:
        value = 7

    def local_kernel():
        payload = Payload()

        @qd.kernel
        def temporary() -> qd.i32:
            return payload.value

        assert temporary() == 7
        return weakref.ref(payload), weakref.ref(temporary._primal)

    @qd.kernel
    def surviving(value: qd.i32) -> qd.i32:
        return value + 1

    baseline = len(impl.get_runtime().kernels)
    references = [local_kernel() for _ in range(12)]
    qd.reset()
    gc.collect()
    gc.collect()
    after = dict(
        kernels=len(impl.get_runtime().kernels),
        live_payloads=sum(payload() is not None for payload, kernel in references),
        live_temporary_kernels=sum(kernel() is not None for payload, kernel in references),
    )
    # A still-owned wrapper must remain callable over repeated reset/reinitialization cycles.
    for value in range(3):
        qd.init(arch=qd.vulkan)
        assert surviving(value) == value + 1
        qd.reset()
        gc.collect()
    result = dict(
        mode=os.environ["QD_KERNEL_REGISTRY"],
        baseline_kernels=baseline,
        after_discard_and_reset=after,
        surviving_kernel_checks=3,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import sys

    lifetime_probe(Path(sys.argv[1]))
