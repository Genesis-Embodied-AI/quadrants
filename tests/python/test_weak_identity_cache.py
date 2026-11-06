import gc
import weakref
from dataclasses import dataclass

from quadrants.lang._weak_identity_cache import WeakIdentityCache


def test_unhashable_equal_keys_are_distinct_and_collected():
    @dataclass
    class Key:
        value: int

    cache = WeakIdentityCache()
    first, second = Key(1), Key(1)
    assert first == second
    assert cache.set(first, "first")
    assert cache.set(second, "second")
    assert cache.get(first) == "first"
    assert cache.get(second) == "second"
    key_id = id(first)
    ref = weakref.ref(first)
    del first
    gc.collect()
    assert ref() is None
    assert key_id not in cache._entries
    assert cache.get(second) == "second"


def test_stale_identity_and_callback_do_not_alias_or_evict_new_entry():
    class Key:
        pass

    cache = WeakIdentityCache()
    first, second = Key(), Key()
    cache.set(first, "first")
    old_ref, value = cache._entries[id(first)]
    # Simulate a recycled ID without relying on allocator timing.
    cache._entries[id(second)] = (old_ref, value)
    assert cache.get(second) is None
    cache.set(first, "replacement")
    old_ref.__callback__(old_ref)
    assert cache.get(first) == "replacement"


def test_non_weak_referenceable_key_is_not_retained():
    cache = WeakIdentityCache()
    key = []
    assert not cache.set(key, "value")
    assert cache.get(key) is None
    assert not cache._entries
