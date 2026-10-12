"""Experimental runtime-identity guard for ndarray destruction, installed only in the candidate pytest arm.

The guard uses a weak reference so it does not extend runtime lifetime. Objects belonging to a previous runtime
cannot call the current runtime's deletion method, even if the native allocator has reused an ndarray address.
"""
from contextlib import contextmanager
import weakref


def install_guard(ndarray_class, get_runtime):
    original_init = ndarray_class.__init__
    original_del = ndarray_class.__del__

    def initialize(self, *args, **kwargs):
        self._experimental_owner_runtime = weakref.ref(get_runtime())
        original_init(self, *args, **kwargs)

    def destroy(self):
        owner = getattr(self, '_experimental_owner_runtime', None)
        runtime = get_runtime()
        if runtime is not None and owner is not None and owner() is runtime:
            original_del(self)

    ndarray_class.__init__ = initialize
    ndarray_class.__del__ = destroy


def pytest_configure(config):
    from quadrants.lang import impl
    from quadrants.lang._ndarray import Ndarray

    install_guard(Ndarray, impl.get_runtime)


@contextmanager
def without_forced_collections(test_root, output):
    fixture = test_root / 'tests/python/conftest.py'
    original = fixture.read_bytes()
    old = b'    yield\n    gc.collect()\n    gc.collect()\n'
    new = b'    yield\n'
    if original.count(old) != 1:
        raise RuntimeError('Expected exactly one double-collection fixture block')
    candidate = original.replace(old, new)
    (output / 'guarded-conftest.py').write_bytes(candidate)
    fixture.write_bytes(candidate)
    try:
        yield
    finally:
        fixture.write_bytes(original)
