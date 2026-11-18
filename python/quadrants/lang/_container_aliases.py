"""Describe shared containers so cached kernels retain compatible member bindings."""

import atexit
import dataclasses
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeAlias
from weakref import ReferenceType, ref

from quadrants import _logging
from quadrants.lang import impl
from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

AliasKey: TypeAlias = tuple[tuple[int, ...], ...]

# Each key has one entry per kernel argument. None marks an argument that is neither a dataclass nor data-oriented.
_alias_key_by_container_ids: dict[tuple[int | None, ...], tuple[AliasKey, tuple[ReferenceType, ...]]] = {}
_cache_capacity: int | None = None
_num_cache_evictions = 0
_cleanup_hooks_registered = False

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
    global _cleanup_hooks_registered
    if not _cleanup_hooks_registered:
        impl.on_reset(_clear_container_alias_cache)
        atexit.register(_clear_container_alias_cache)
        _cleanup_hooks_registered = True

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


def _evict_alias_key(container_ids: tuple[int | None, ...], expired_ref: ReferenceType[Any]) -> None:
    """Remove only the entry containing the expired reference, protecting against recycled object IDs."""
    entry = _alias_key_by_container_ids.get(container_ids)
    if entry is not None and any(item is expired_ref for item in entry[1]):
        del _alias_key_by_container_ids[container_ids]


def cached_get_container_alias_key(py_args: tuple[Any, ...]) -> AliasKey:
    """Return a key describing which containers in py_args refer to the same object, including nested containers.

    The outer tuple has one entry per kernel argument. Each inner tuple lists the container labels encountered
    while walking that argument, including the argument itself. Labels are numbered from zero within the call.

    Example::

        from dataclasses import dataclass

        @dataclass
        class Grid:
            size: int

        @dataclass
        class State:
            grid: Grid

        grid = Grid(size=4)
        state = State(grid=grid)
        print(cached_get_container_alias_key((state, state.grid)))  # ((0, 1), (1,))

    Reuse the cached key when the same container objects occupy the same argument positions.
    """
    global _cache_capacity, _num_cache_evictions
    container_ids = tuple(id(arg) if is_dataclass_instance(arg) or is_data_oriented(arg) else None for arg in py_args)
    cached_alias_key_entry = _alias_key_by_container_ids.get(container_ids)
    if cached_alias_key_entry is not None:
        alias_key, _ = cached_alias_key_entry
        return alias_key

    alias_key = get_container_alias_key(py_args)

    drop_fn = partial(_evict_alias_key, container_ids)
    try:
        container_weak_refs = tuple(
            ref(arg, drop_fn) for arg, identity in zip(py_args, container_ids) if identity is not None
        )
    except TypeError:
        return alias_key

    if _cache_capacity is None:
        _cache_capacity = impl.get_runtime().container_alias_cache_size
    if len(_alias_key_by_container_ids) >= _cache_capacity:
        del _alias_key_by_container_ids[next(iter(_alias_key_by_container_ids))]
        _num_cache_evictions += 1
    _alias_key_by_container_ids[container_ids] = (alias_key, container_weak_refs)
    return alias_key


def _clear_container_alias_cache() -> None:
    """Report capacity evictions and clear cached argument combinations and container graphs."""
    global _cache_capacity, _num_cache_evictions
    _alias_key_by_container_ids.clear()
    _graph_ids_by_instance.clear()
    if _num_cache_evictions:
        _logging.warn(
            f"The container alias cache evicted {_num_cache_evictions} entries at capacity {_cache_capacity}. "
            "See https://genesis-embodied-ai.github.io/quadrants/user_guide/compound_types.html#container-alias-cache"
        )
        _num_cache_evictions = 0
    _cache_capacity = None
