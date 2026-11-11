"""Find runtime dataclasses below data-oriented kernel arguments."""

from typing import Any, TypeAlias

from quadrants.lang._runtime_typing.runtime_dataclass import infer_dataclass_type
from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.util import is_data_oriented

DataclassPath: TypeAlias = tuple[str, ...]
# Data-oriented attribute layouts are fixed after their first kernel call, like ndarray member paths.
_paths_by_instance: WeakIdentityCache[tuple[DataclassPath, ...]] = WeakIdentityCache()


def nested_dataclass_paths(obj: Any) -> tuple[DataclassPath, ...]:
    """Find outermost dataclass members without retaining their owning objects."""
    cached = _paths_by_instance.get(obj)
    if cached is not None:
        return cached
    paths: list[DataclassPath] = []
    active_ids: set[int] = set()

    def walk(value: Any, path: DataclassPath) -> None:
        """Walk data-oriented members, stopping at a dataclass or an active cycle."""
        if hasattr(type(value), "__dataclass_fields__"):
            paths.append(path)
            return
        if not is_data_oriented(value) or id(value) in active_ids:
            return
        active_ids.add(id(value))
        try:
            members = value._asdict() if isinstance(value, tuple) and hasattr(value, "_asdict") else vars(value)
            for name, child in members.items():
                if not name.startswith("_qd_"):
                    walk(child, (*path, name))
        finally:
            active_ids.remove(id(value))

    walk(obj, ())
    result = tuple(paths)
    _paths_by_instance.set(obj, result)
    return result


def get_member_at_path(obj: Any, path: DataclassPath) -> Any:
    """Read an attribute chain from the current kernel argument."""
    for name in path:
        obj = getattr(obj, name)
    return obj


def infer_nested_dataclass_schemas(args: tuple[Any, ...], names: list[str]) -> list[tuple[int, DataclassPath, type]]:
    """Infer current schemas for dataclasses reached through data-oriented arguments."""
    schemas = []
    for index, obj in enumerate(args):
        if not is_data_oriented(obj) or hasattr(type(obj), "__dataclass_fields__"):
            continue
        for path in nested_dataclass_paths(obj):
            child = get_member_at_path(obj, path)
            if not hasattr(type(child), "__dataclass_fields__"):
                raise TypeError("A data-oriented dataclass member must remain a dataclass between kernel calls.")
            schema = infer_dataclass_type(child, ".".join((names[index], *path)))
            schemas.append((index, path, schema))
    return schemas
