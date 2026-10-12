"""GPU-free ownership checks for the diagnostic registry."""

import gc
import weakref

from kernel_registry_probe import WeakKernelRegistry


class Kernel:
    pass


def test_live_entries_preserve_insertion_order():
    kernels = [Kernel() for _ in range(3)]
    registry = WeakKernelRegistry(kernels)
    assert list(registry) == kernels
    assert len(registry) == 3


def test_discarded_cycle_is_collectible():
    registry = WeakKernelRegistry()
    kernel = Kernel()
    kernel.cycle = kernel
    reference = weakref.ref(kernel)
    registry.append(kernel)
    del kernel
    gc.collect()
    assert reference() is None
    assert len(registry) == 0


def test_iteration_and_other_owners_keep_live_kernel():
    kernel = Kernel()
    registry = WeakKernelRegistry([kernel])
    reference = weakref.ref(kernel)
    iterator = iter(registry)
    current = next(iterator)
    del kernel
    gc.collect()
    assert current is reference()
    assert list(registry) == [current]
