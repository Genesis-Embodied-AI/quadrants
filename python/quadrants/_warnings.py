import warnings

_seen: set[str] = set()


def warn_once(message: str) -> None:
    global _seen
    if message in _seen:
        return
    _seen.add(message)
    warnings.warn(message)


def warn_uncacheable_weakref(error: TypeError) -> None:
    """Warn once when a Python object cannot serve as a weak cache key."""
    warn_once(
        f"{error}. Quadrants cannot use this object as a weak cache key. "
        "For a slotted dataclass, use @dataclass(slots=True, weakref_slot=True). "
        'For a regular class with __slots__, include "__weakref__" in its slots.'
    )
