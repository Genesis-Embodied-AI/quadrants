"""Describe shared containers so cached kernels retain compatible member bindings."""

import dataclasses
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeAlias
from weakref import ReferenceType, ref

from quadrants import _logging
from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

AliasKey: TypeAlias = tuple[tuple[int, ...], ...]
_ContainerIds: TypeAlias = tuple[int | None, ...]
DEFAULT_CONTAINER_ALIAS_CACHE_SIZE = 1024

_alias_key_by_container_ids: dict[_ContainerIds, tuple[AliasKey, tuple[ReferenceType, ...]]] = {}
_cache_capacity = DEFAULT_CONTAINER_ALIAS_CACHE_SIZE
_capacity_evictions = 0

# Container references are fixed after first use; scalar values do not affect these identities.
_graph_ids_by_instance: WeakIdentityCache[tuple[int, ...]] = WeakIdentityCache()


def _collect_container_ids(py_obj: Any) -> tuple[int, ...]:
    """Walk stored dataclass and data-oriented attributes, recording shared references and stopping at cycles."""
    identities: list[int] = []
    seen_ids: set[int] = set()
    pending = [py_obj]
    while pending:
        child = pending.pop()
        is_dataclass = is_dataclass_instance(child)
        if not is_dataclass and not is_data_oriented(child):
            continue

        identity = id(child)
        identities.append(identity)
        if identity in seen_ids:
            continue

        seen_ids.add(identity)
        if is_dataclass:
            pending.extend(getattr(child, field.name) for field in dataclasses.fields(child))
        else:
            members = child._asdict() if hasattr(child, "_asdict") else vars(child)
            pending.extend(value for name, value in members.items() if not name.startswith("_qd_"))
    return tuple(identities)


def get_container_alias_key(args: Sequence[Any]) -> AliasKey:
    """Describe sharing across arguments, reusing each container's cached graph.

    Container references must remain fixed after first use. Weak keys release graphs when their owners die.
    """
    serials: dict[int, int] = {}
    result: list[tuple[int, ...]] = []
    for arg in args:
        if not (is_dataclass_instance(arg) or is_data_oriented(arg)):
            result.append(())
            continue

        identities = _graph_ids_by_instance.get(arg)
        if identities is None:
            identities = _collect_container_ids(arg)
            try:
                _graph_ids_by_instance[arg] = identities
            except TypeError:
                # Named tuples cannot be weakly referenced; recompute rather than retaining them.
                pass
        result.append(tuple(serials.setdefault(identity, len(serials)) for identity in identities))
    return tuple(result)


def _evict_alias_key(container_ids: _ContainerIds, expired: ReferenceType[Any]) -> None:
    """Remove only the entry containing the expired reference, protecting against recycled object IDs."""
    entry = _alias_key_by_container_ids.get(container_ids)
    if entry is not None and any(item is expired for item in entry[1]):
        del _alias_key_by_container_ids[container_ids]


def get_cached_container_alias_key(args: tuple[Any, ...]) -> AliasKey:
    """Reuse the sharing pattern for the same containers in the same argument positions."""
    global _capacity_evictions
    container_ids = tuple(id(arg) if is_dataclass_instance(arg) or is_data_oriented(arg) else None for arg in args)
    cached = _alias_key_by_container_ids.get(container_ids)
    if cached is not None:
        return cached[0]

    alias_key = get_container_alias_key(args)
    try:
        drop = partial(_evict_alias_key, container_ids)
        references = tuple(ref(arg, drop) for arg, identity in zip(args, container_ids) if identity is not None)
    except TypeError:
        return alias_key

    if len(_alias_key_by_container_ids) >= _cache_capacity:
        del _alias_key_by_container_ids[next(iter(_alias_key_by_container_ids))]
        _capacity_evictions += 1
    _alias_key_by_container_ids[container_ids] = (alias_key, references)
    return alias_key


def set_container_alias_cache_size(size: int) -> None:
    """Set the shared cache capacity during qd.init, after the cache has been cleared."""
    global _cache_capacity
    if type(size) is not int or size < 1:
        raise ValueError("container_alias_cache_size must be a positive integer")
    _cache_capacity = size


def clear_container_alias_cache() -> None:
    """Report capacity evictions and clear cached argument combinations and container graphs."""
    global _capacity_evictions
    _alias_key_by_container_ids.clear()
    _graph_ids_by_instance.clear()
    if _capacity_evictions:
        _logging.warn(
            f"The container alias cache evicted {_capacity_evictions} entries at capacity {_cache_capacity}. "
            "See https://genesis-embodied-ai.github.io/quadrants/user_guide/compound_types.html#container-alias-cache"
        )
        _capacity_evictions = 0
