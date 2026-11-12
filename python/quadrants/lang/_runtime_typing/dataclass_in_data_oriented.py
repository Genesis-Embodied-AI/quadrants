"""Find runtime dataclasses below data-oriented kernel arguments."""

from typing import Any

from quadrants.lang._runtime_typing.runtime_dataclass import (
    RuntimeDataclassSchema,
    cached_infer_dataclass_type,
    runtime_template_members,
)
from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

# Data-oriented attribute layouts are fixed after their first kernel call, like ndarray member paths.
_dataclass_paths_by_data_oriented_instance: WeakIdentityCache[tuple[tuple[str, ...], ...]] = WeakIdentityCache()


def _cached_get_dataclass_paths(data_oriented_obj: Any) -> tuple[tuple[str, ...], ...]:
    """Find outermost dataclass members without retaining their owning objects."""
    cached = _dataclass_paths_by_data_oriented_instance.get(data_oriented_obj)
    if cached is not None:
        return cached

    paths: list[tuple[str, ...]] = []
    active_ids: set[int] = set()
    def walk(py_obj: Any, path: tuple[str, ...]) -> None:
        """Walk data-oriented members, stopping at a dataclass or an active cycle."""
        if is_dataclass_instance(py_obj):
            paths.append(path)
            return

        if not is_data_oriented(py_obj):
            return

        if id(py_obj) in active_ids:
            return

        active_ids.add(id(py_obj))
        try:
            members = py_obj._asdict() if isinstance(py_obj, tuple) and hasattr(py_obj, "_asdict") else vars(py_obj)
            for name, child in members.items():
                if not name.startswith("_qd_"):
                    walk(child, (*path, name))
        finally:
            active_ids.remove(id(py_obj))

    walk(data_oriented_obj, ())
    result = tuple(paths)
    try:
        _dataclass_paths_by_data_oriented_instance[data_oriented_obj] = result
    except TypeError:
        # Named tuples can be data-oriented but cannot be weakly referenced.
        pass
    return result


def get_member_at_path(obj: Any, path: tuple[str, ...]) -> Any:
    """Read an attribute chain from the current kernel argument."""
    for name in path:
        obj = getattr(obj, name)
    return obj


def infer_nested_dataclass_schemas(
    args: tuple[Any, ...], arg_metas: list[ArgMetadata], template_slots: list[int]
) -> list[tuple[int, tuple[str, ...], type[RuntimeDataclassSchema]]]:
    """Infer current schemas for dataclasses reached through data-oriented arguments."""
    # Schemas for this launch, paired with argument indices and member paths.
    launch_arg_schemas: list[tuple[int, tuple[str, ...], type[RuntimeDataclassSchema]]] = []
    active_ids: set[int] = set()

    def visit_owner(index: int, owner: Any, owner_path: tuple[str, ...]) -> None:
        """Infer descendants and continue through any data-oriented members inside their schemas."""
        if id(owner) in active_ids or not is_data_oriented(owner):
            return

        active_ids.add(id(owner))
        try:
            for relative_path in _cached_get_dataclass_paths(owner):
                path = (*owner_path, *relative_path)
                child = get_member_at_path(owner, relative_path)
                if not hasattr(type(child), "__dataclass_fields__"):
                    raise TypeError("A data-oriented dataclass member must remain a dataclass between kernel calls.")
                schema = cached_infer_dataclass_type(child, ".".join((arg_metas[index].name, *path)))
                launch_arg_schemas.append((index, path, schema))
                for member_path, member in runtime_template_members(schema, child):
                    visit_owner(index, member, (*path, *member_path))
        finally:
            active_ids.remove(id(owner))

    for index in template_slots:
        if not is_dataclass_instance(args[index]):
            visit_owner(index, args[index], ())
    for index, meta in enumerate(arg_metas):
        if hasattr(meta.annotation, "_qd_runtime_origin"):
            for path, member in runtime_template_members(meta.annotation, args[index]):
                visit_owner(index, member, path)
    return launch_arg_schemas


def reject_undiscovered_dataclass_member(
    parent: Any, name: str, py_obj: Any, discovered_members: set[tuple[int, str]]
) -> None:
    """Reject compilation-time access to dataclasses missing from argument member discovery.

    Check the attribute, not the returned object's identity: a property can alias a stored dataclass yet require a
    different launch-binding path. This check does not run on already-compiled kernel launches.
    """
    if not is_dataclass_instance(py_obj):
        return

    if is_dataclass_instance(parent):
        discovered = name in type(parent).__dataclass_fields__
    elif is_data_oriented(parent):
        discovered = (id(parent), name) in discovered_members
    else:
        return

    if not discovered:
        raise QuadrantsRuntimeTypeError(
            f"Dataclass attribute {type(parent).__name__ + '.' + name!r} was not discovered as a runtime argument member. "
            "Dataclasses returned by properties or added after the first kernel call are unsupported. "
            "Store the dataclass in an ordinary instance attribute before the first kernel call."
        )


def reject_computed_dataclass_alias(py_obj: Any, bound_objects: set[int]) -> None:
    """Reject computed routes that could borrow an unrelated runtime dataclass binding."""
    if id(py_obj) in bound_objects:
        raise QuadrantsRuntimeTypeError(
            "A runtime dataclass alias reached through a call or indexing is unsupported. "
            "Access the dataclass through its ordinary stored attributes instead."
        )
