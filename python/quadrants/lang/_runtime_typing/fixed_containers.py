"""Cache alias relationships for container references assumed fixed after first kernel use."""

import dataclasses
from typing import Any, TypeAlias
from weakref import ReferenceType, ref

from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

AliasKey: TypeAlias = tuple[tuple[int, ...], ...]
# Entries retain primitive arguments and weak references to other arguments to prevent recycled-ID hits.
AliasKeyCache: TypeAlias = dict[tuple[int, ...], tuple[AliasKey, tuple[object, ...]]]

_graph_ids: WeakIdentityCache[tuple[int, ...]] = WeakIdentityCache()


def _is_container(py_obj: Any) -> bool:
    """Recognize references whose replacement could change a dataclass access route."""
    return isinstance(py_obj, tuple) or is_dataclass_instance(py_obj) or is_data_oriented(py_obj)


def _collect_containers(
    py_obj: Any,
    identities: list[int],
    seen: set[int],
) -> None:
    """Collect stored container identities without following properties or arbitrary objects."""
    if not _is_container(py_obj):
        return

    identities.append(id(py_obj))
    if id(py_obj) in seen:
        return

    seen.add(id(py_obj))
    if isinstance(py_obj, tuple) and not is_dataclass_instance(py_obj):
        for child in py_obj:
            _collect_containers(child, identities, seen)
        return

    dataclass_owner = is_dataclass_instance(py_obj)
    members = (
        ((field.name, getattr(py_obj, field.name)) for field in dataclasses.fields(py_obj))
        if is_dataclass_instance(py_obj)
        else vars(py_obj).items()
    )
    for name, child in members:
        if not name.startswith("_qd_"):
            if not dataclass_owner or is_dataclass_instance(child) or is_data_oriented(child):
                _collect_containers(child, identities, seen)


def get_container_alias_key(
    args: tuple[Any, ...],
) -> tuple[tuple[int, ...], ...]:
    """Describe shared containers, assuming their references remain fixed after first use."""
    serials: dict[int, int] = {}
    result = []
    for py_obj in args:
        if not (is_dataclass_instance(py_obj) or is_data_oriented(py_obj)):
            result.append(())
            continue

        identities = _graph_ids.get(py_obj)
        if identities is None:
            collected: list[int] = []
            _collect_containers(py_obj, collected, set())
            identities = tuple(collected)
            try:
                _graph_ids[py_obj] = identities
            except TypeError:
                # Immutable named tuples can be owners but cannot be weakly referenced.
                pass
        result.append(tuple(serials.setdefault(identity, len(serials)) for identity in identities))
    return tuple(result)


def get_cached_container_alias_key(
    args: tuple[Any, ...], args_hash: tuple[int, ...], cache: AliasKeyCache
) -> AliasKey:
    """Reuse a whole-call alias key while its argument identities remain alive."""
    cached = cache.get(args_hash)
    if cached is not None:
        return cached[0]

    alias_key = get_container_alias_key(args)
    references: list[object] = []

    def evict(expired: ReferenceType[Any]) -> None:
        """Remove only the entry containing this expired weak reference."""
        entry = cache.get(args_hash)
        if entry is not None and any(item is expired for item in entry[1]):
            del cache[args_hash]

    for arg in args:
        if type(arg) in (int, float, bool, str, bytes, complex, type(None)):
            references.append(arg)
        else:
            try:
                references.append(ref(arg, evict))
            except TypeError:
                # Non-weakrefable containers must not be retained by the optimization.
                return alias_key

    # Scalar arguments can vary indefinitely while their owners stay alive. Bound this optional cache.
    if len(cache) >= 128:
        del cache[next(iter(cache))]
    cache[args_hash] = (alias_key, tuple(references))
    return alias_key
