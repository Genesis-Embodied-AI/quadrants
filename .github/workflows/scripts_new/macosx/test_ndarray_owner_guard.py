"""Check the experimental lifetime guard without requiring a native Quadrants build or GPU."""
import gc
from types import SimpleNamespace
import weakref

from ndarray_owner_guard import install_guard, without_forced_collections


def guarded_class():
    class Runtime:
        pass

    current = SimpleNamespace(runtime=Runtime())
    deleted = []

    class Array:
        def __init__(self, address):
            self.arr = address

        def __del__(self):
            if getattr(self, 'arr', None) is not None:
                deleted.append((current.runtime, self.arr))
                self.arr = None

    install_guard(Array, lambda: current.runtime)
    return Array, Runtime, current, deleted


def test_same_runtime_deletes_normally():
    array, _, current, deleted = guarded_class()
    value = array(1234)
    value.__del__()
    assert deleted == [(current.runtime, 1234)]


def test_old_runtime_cannot_delete_reused_address():
    array, runtime, current, deleted = guarded_class()
    old_runtime = current.runtime
    stale = array(1234)
    current.runtime = runtime()
    live = array(1234)
    stale.__del__()
    assert deleted == []
    live.__del__()
    assert deleted == [(current.runtime, 1234)]
    assert old_runtime is not current.runtime


def test_guard_does_not_keep_old_runtime_alive():
    array, runtime, current, deleted = guarded_class()
    owner = weakref.ref(current.runtime)
    stale = array(1234)
    current.runtime = runtime()
    gc.collect()
    assert owner() is None
    stale.__del__()
    assert deleted == []


def test_partially_initialized_object_is_ignored():
    array, _, _, deleted = guarded_class()
    partial = object.__new__(array)
    partial.__del__()
    assert deleted == []


def test_fixture_is_restored_even_after_failure(tmp_path):
    fixture = tmp_path / 'tests/python/conftest.py'
    fixture.parent.mkdir(parents=True)
    original = b'def fixture():\n    yield\n    gc.collect()\n    gc.collect()\n'
    fixture.write_bytes(original)
    try:
        with without_forced_collections(tmp_path, tmp_path):
            assert fixture.read_bytes() == b'def fixture():\n    yield\n'
            raise RuntimeError('simulated failure')
    except RuntimeError:
        pass
    assert fixture.read_bytes() == original
