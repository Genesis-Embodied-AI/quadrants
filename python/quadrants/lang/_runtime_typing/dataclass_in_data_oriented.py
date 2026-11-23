"""Find runtime dataclasses below data-oriented kernel arguments."""

from typing import Any

from quadrants import _warnings
from quadrants.lang._runtime_typing.runtime_dataclass import (
    RuntimeDataclassSchema,
    cached_infer_dataclass_schema,
)
from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import (
    DataOriented,
    get_member_at_path,
    is_data_oriented,
    is_dataclass_instance,
)

# Data-oriented attribute layouts are fixed after their first kernel call, like ndarray member paths.
_dataclass_paths_by_data_oriented_instance: WeakIdentityCache[tuple[tuple[str, ...], ...]] = WeakIdentityCache()


def _collect_dataclass_paths_in_data_oriented(
    data_oriented_obj: DataOriented,
    path: tuple[str, ...],
    dataclass_paths: list[tuple[str, ...]],
    active_ids: set[int],
) -> None:
    """Record dataclass children of a data-oriented object."""
    if id(data_oriented_obj) in active_ids:
        return

    active_ids.add(id(data_oriented_obj))
    try:
        asdict = getattr(data_oriented_obj, "_asdict", None) if isinstance(data_oriented_obj, tuple) else None
        members = asdict() if asdict is not None else vars(data_oriented_obj)
        for name, child in members.items():
            if name.startswith("_qd_"):
                continue

            if is_dataclass_instance(child):
                dataclass_paths.append((*path, name))
            elif is_data_oriented(child):
                _collect_dataclass_paths_in_data_oriented(child, (*path, name), dataclass_paths, active_ids)
    finally:
        active_ids.remove(id(data_oriented_obj))


def _cached_collect_dataclass_paths_in_data_oriented(data_oriented_obj: DataOriented) -> tuple[tuple[str, ...], ...]:
    """Find outermost dataclass members."""
    cached = _dataclass_paths_by_data_oriented_instance.get(data_oriented_obj)
    if cached is not None:
        return cached

    paths: list[tuple[str, ...]] = []
    active_ids: set[int] = set()
    _collect_dataclass_paths_in_data_oriented(data_oriented_obj, (), paths, active_ids)
    dataclass_paths = tuple(paths)
    try:
        _dataclass_paths_by_data_oriented_instance[data_oriented_obj] = dataclass_paths
    except TypeError as error:
        # Named tuples can be data-oriented but cannot be weakly referenced.
        _warnings.warn_uncacheable_weakref(error)
    return dataclass_paths


def collect_dataclass_schemas_in_data_oriented_args(
    args: tuple[Any, ...], arg_metas: list[ArgMetadata], template_slots: list[int]
) -> list[tuple[int, tuple[str, ...], type[RuntimeDataclassSchema]]]:
    """Collect inferred schemas for dataclasses reached through data-oriented arguments."""
    # Schemas for this launch, paired with argument indices and member paths.
    launch_arg_schemas: list[tuple[int, tuple[str, ...], type[RuntimeDataclassSchema]]] = []
    for index in template_slots:
        py_arg = args[index]
        if not is_data_oriented(py_arg):
            continue

        data_oriented_obj = py_arg
        for path in _cached_collect_dataclass_paths_in_data_oriented(data_oriented_obj):
            outermost_dataclass = get_member_at_path(data_oriented_obj, path)
            if not hasattr(type(outermost_dataclass), "__dataclass_fields__"):
                raise TypeError("A data-oriented dataclass member must remain a dataclass between kernel calls.")
            schema = cached_infer_dataclass_schema(outermost_dataclass, ".".join((arg_metas[index].name, *path)))
            launch_arg_schemas.append((index, path, schema))
    return launch_arg_schemas


def reject_undiscovered_dataclass_member(
    parent: Any, name: str, py_obj: Any, discovered_members: set[tuple[int, str]]
) -> None:
    """Reject compilation-time access to dataclasses missing from argument member discovery.

    This can reject dataclasses reached through:
    - Properties or other descriptors, including properties returning an already-discovered dataclass.
    - Class-level attributes on data-oriented classes; discovery walks instance attributes.
    - Dynamic attributes supplied by __getattr__.
    - Attributes on a dataclass that are not declared dataclass fields.

    Check the attribute, not the returned object's identity: a property can alias a stored dataclass yet require a
    different launch-binding path. This check does not run on already-compiled kernel launches.
    """
    # FIXME: This function is symptomatic of attribute discovery and binding needing reorganization. Valid accesses
    # and missing runtime bindings reach the same fallback, making it confusing why this extra check is needed.
    # FIXME: Dataclass discovery should use the instance's actual runtime fields, not declared fields on its type.
    # FIXME: Data-oriented binding should work like dataclass binding, rather than using None bindings during discovery.
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
