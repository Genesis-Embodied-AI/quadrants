"""Cache values by object identity without retaining the key objects."""

import weakref
from typing import Generic, TypeVar

_T = TypeVar("_T")


class WeakIdentityCache(Generic[_T]):
    """Support unhashable keys; cached values must not retain their keys, even indirectly."""

    def __init__(self) -> None:
        """Create an empty identity-indexed cache."""
        self._entries: dict[int, tuple[weakref.ReferenceType, _T]] = {}

    def get(self, obj: object) -> _T | None:
        """Return the cached value only if the entry still belongs to this exact object."""
        entry = self._entries.get(id(obj))
        if entry is not None and entry[0]() is obj:
            return entry[1]
        return None

    def set(self, obj: object, value: _T) -> bool:
        """Store a value until its key dies; return False if the key cannot be weakly referenced."""
        key = id(obj)

        def drop(ref: weakref.ReferenceType) -> None:
            """Remove only the entry belonging to this reference, even if its ID has been reused."""
            current = self._entries.get(key)
            if current is not None and current[0] is ref:
                del self._entries[key]

        try:
            ref = weakref.ref(obj, drop)
        except TypeError:
            return False
        self._entries[key] = (ref, value)
        return True
