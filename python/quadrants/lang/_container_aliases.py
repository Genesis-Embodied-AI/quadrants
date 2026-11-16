"""Describe shared containers so cached kernels retain compatible member bindings."""

import dataclasses
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeAlias
from weakref import ReferenceType, ref

from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

AliasKey: TypeAlias = tuple[tuple[int, ...], ...]
AliasKeyCache: TypeAlias = dict[tuple[int, ...], tuple[AliasKey, tuple[object, ...]]]

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
            pending.extend(
                getattr(child, field.name) for field in dataclasses.fields(child)
            )
        else:
            members = child._asdict() if hasattr(child, "_asdict") else vars(child)
            pending.extend(
                value for name, value in members.items() if not name.startswith("_qd_")
            )
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
        result.append(
            tuple(serials.setdefault(identity, len(serials)) for identity in identities)
        )
    return tuple(result)


def _evict_alias_key(
    cache: AliasKeyCache, args_hash: tuple[int, ...], expired: ReferenceType[Any]
) -> None:
    """Remove only the entry containing the expired reference, protecting against recycled object IDs."""
    entry = cache.get(args_hash)
    if entry is not None and any(item is expired for item in entry[1]):
        del cache[args_hash]


def get_cached_container_alias_key(
    args: tuple[Any, ...], args_hash: tuple[int, ...], cache: AliasKeyCache
) -> AliasKey:
    """Reuse a whole-call alias key while its argument identities remain alive."""
    cached = cache.get(args_hash)
    if cached is not None:
        return cached[0]

    alias_key = get_container_alias_key(args)
    references: list[object] = []
    for arg in args:
        if type(arg) in (int, float, bool, str, bytes, complex, type(None)):
            references.append(arg)
        else:
            try:
                references.append(ref(arg, partial(_evict_alias_key, cache, args_hash)))
            except TypeError:
                return alias_key

    # Changing scalar arguments can produce unlimited combinations while their owners stay alive.
    if len(cache) >= 128:
        del cache[next(iter(cache))]
    cache[args_hash] = (alias_key, tuple(references))
    return alias_key
