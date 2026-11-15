import warnings

_seen: set[str] = set()


def warn_once(message: str) -> None:
    global _seen
    if message in _seen:
        return
    _seen.add(message)
    warnings.warn(message)
