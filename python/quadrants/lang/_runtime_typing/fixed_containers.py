"""Fix container references at first kernel use and describe their alias relationships."""

import dataclasses
from typing import Any, TypeAlias
from weakref import ReferenceType, ref

from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

AliasKey: TypeAlias = tuple[tuple[int, ...], ...]
# Entries retain primitive arguments and weak references to other arguments to prevent recycled-ID hits.
AliasKeyCache: TypeAlias = dict[tuple[int, ...], tuple[AliasKey, tuple[object, ...]]]

_fixed_names: WeakIdentityCache[frozenset[str]] = WeakIdentityCache()
_graph_ids: WeakIdentityCache[tuple[int, ...]] = WeakIdentityCache()


def _is_container(py_obj: Any) -> bool:
    """Recognize references whose replacement could change a dataclass access route."""
    return isinstance(py_obj, (tuple, list, dict, set)) or is_dataclass_instance(py_obj) or is_data_oriented(py_obj)


def _install_assignment_guards(cls: type) -> None:
    """Wrap ordinary assignments and deletions, preserving the class's existing implementations."""
    if vars(cls).get("_qd_fixed_reference_guards", False):
        return

    original_setattr = cls.__setattr__
    original_delattr = cls.__delattr__

    def guarded_setattr(self: Any, name: str, value: Any) -> None:
        """Reject replacement of fixed references and introduction of new container attributes."""
        names = _fixed_names.get(self)
        if names is not None and not name.startswith("_qd_"):
            if name in names:
                if getattr(self, name) is not value:
                    raise QuadrantsRuntimeTypeError(
                        f"Container reference {type(self).__name__}.{name} is fixed after first kernel use."
                    )
            elif _is_container(value):
                raise QuadrantsRuntimeTypeError(
                    f"Cannot add container reference {type(self).__name__}.{name} after first kernel use."
                )
        original_setattr(self, name, value)

    def guarded_delattr(self: Any, name: str) -> None:
        """Reject deletion of a fixed container reference."""
        names = _fixed_names.get(self)
        if names is not None and name in names:
            raise QuadrantsRuntimeTypeError(
                f"Container reference {type(self).__name__}.{name} is fixed after first kernel use."
            )
        original_delattr(self, name)

    cls.__setattr__ = guarded_setattr
    cls.__delattr__ = guarded_delattr
    cls._qd_fixed_reference_guards = True


def _collect_containers(
    py_obj: Any,
    identities: list[int],
    seen: set[int],
    owners: list[tuple[Any, frozenset[str]]],
) -> None:
    """Validate stored containers and collect identities without following properties or arbitrary objects."""
    if isinstance(py_obj, (list, dict, set)):
        raise QuadrantsRuntimeTypeError(
            "Mutable Python containers in runtime container graphs are unsupported; use tuples."
        )
    if not _is_container(py_obj):
        return

    identities.append(id(py_obj))
    if id(py_obj) in seen:
        return

    seen.add(id(py_obj))
    if isinstance(py_obj, tuple) and not is_dataclass_instance(py_obj):
        for child in py_obj:
            _collect_containers(child, identities, seen, owners)
        return

    dataclass_owner = is_dataclass_instance(py_obj)
    members = (
        ((field.name, getattr(py_obj, field.name)) for field in dataclasses.fields(py_obj))
        if is_dataclass_instance(py_obj)
        else vars(py_obj).items()
    )
    fixed = set()
    for name, child in members:
        if not name.startswith("_qd_"):
            if _is_container(child):
                fixed.add(name)
            if not dataclass_owner or is_dataclass_instance(child) or is_data_oriented(child):
                _collect_containers(child, identities, seen, owners)
    owners.append((py_obj, frozenset(fixed)))


def get_container_alias_key(
    args: tuple[Any, ...],
) -> tuple[tuple[int, ...], ...]:
    """Lock references on first use and return an identity-independent description of shared containers."""
    serials: dict[int, int] = {}
    result = []
    for py_obj in args:
        if not (is_dataclass_instance(py_obj) or is_data_oriented(py_obj)):
            result.append(())
            continue

        identities = _graph_ids.get(py_obj)
        if identities is None:
            collected: list[int] = []
            owners: list[tuple[Any, frozenset[str]]] = []
            _collect_containers(py_obj, collected, set(), owners)
            for owner, names in owners:
                try:
                    _fixed_names[owner] = names
                except TypeError as error:
                    raise QuadrantsRuntimeTypeError("Runtime containers must support weak references.") from error
                _install_assignment_guards(type(owner))
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
