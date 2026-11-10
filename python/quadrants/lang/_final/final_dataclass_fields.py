"""Discover Final dataclass fields, infer their schema annotations, and validate their containers."""

import dataclasses
import typing
from typing import TYPE_CHECKING, Any

from quadrants.lang._final.final_annotation import dataclass_field_is_final
from quadrants.lang._final.final_annotation import (  # pylint: disable=unused-import
    is_final_annotation as is_final_annotation,
)
from quadrants.lang._final.final_constant_values import final_scalar_key
from quadrants.lang.runtime_typing._deferred_errors import (
    create_deferred_member_type_for_invalid_dataclass_member,
)

if TYPE_CHECKING:
    from typing import _SpecialForm

    from quadrants.lang.runtime_typing._deferred_errors import DeferredMemberType
    from quadrants.lang.runtime_typing._runtime_dataclass import FinalValueKeyCache

# Final is a special-form object at runtime, although static checkers treat it as a type expression.
_FINAL_MARKER = typing.cast("_SpecialForm", typing.Final)

# Class identity maps to the retained class and its Final field names.
# This cache retains classes, assuming applications define relatively few dataclass types.
_final_field_names_by_dataclass_type_id: dict[int, tuple[type, frozenset[str]]] = {}


def get_final_field_names(dataclass_type: type) -> frozenset[str]:
    """Find and cache the names of fields marked Final."""
    cached_dataclass_type, cached_final_field_names = _final_field_names_by_dataclass_type_id.get(
        id(dataclass_type), (None, None)
    )
    if cached_dataclass_type is dataclass_type and cached_final_field_names is not None:
        return cached_final_field_names
    names = frozenset(
        field.name
        for field in dataclasses.fields(dataclass_type)
        if dataclass_field_is_final(dataclass_type, field.name, field.type)
    )
    _final_field_names_by_dataclass_type_id[id(dataclass_type)] = (dataclass_type, names)
    return names


def infer_final_field_annotation(
    value: object,
    non_frozen_ancestor_path: str | None,
    name: str,
    final_value_key_by_value_id: "FinalValueKeyCache | None" = None,
) -> "_SpecialForm | DeferredMemberType":
    """Infer a Final field's schema annotation from its runtime value and ancestor constraints."""
    if non_frozen_ancestor_path is not None:
        parent_name = non_frozen_ancestor_path.split(".", 1)[0]
        return create_deferred_member_type_for_invalid_dataclass_member(
            TypeError,
            f"{parent_name} is not frozen but reaches the ``Final`` field {non_frozen_ancestor_path}.{name}. "
            "No dataclass on the path to a used Final value may be reassignable.",
            value,
        )
    # Exact Python primitives cannot fail the existing value validator. Other values must still be checked on each
    # launch: an enum or primitive subclass can acquire mutable state even inside a frozen dataclass.
    value_type = type(value)
    if value_type is not bool and value_type is not int and value_type is not float and value_type is not str:
        try:
            key = final_scalar_key(value, live=True)
            if final_value_key_by_value_id is not None:
                final_value_key_by_value_id[id(value)] = (value, key)
        except TypeError as error:
            return create_deferred_member_type_for_invalid_dataclass_member(TypeError, str(error), value)
    # A schema needs only the marker; the specialization key and compiler retain the actual value and its type.
    return _FINAL_MARKER


def _describe_annotation(annotation: Any) -> str:
    """Return a readable name for an annotation."""
    return getattr(annotation, "__name__", None) or repr(annotation)


def _rebinding_is_prevented(dc_type: type) -> bool:
    """Check whether the dataclass declares frozen or unsafe_hash.

    A hashable class is not necessarily frozen: dataclass(eq=False) can inherit object.__hash__.
    """
    params = getattr(dc_type, "__dataclass_params__", None)
    if params is None:
        return dc_type.__hash__ is not None
    return params.frozen or params.unsafe_hash


# Cache nested Final paths by type identity. Retain each type to prevent id reuse.
_final_path_cache: "dict[int, tuple[type, tuple[str, ...] | None]]" = {}


def _first_final_path(dc_type: type, visiting: "frozenset[int]") -> "tuple[str, ...] | None":
    """Validate nested dataclass declarations and return the first Final field path, if any."""
    entry = _final_path_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity guard against a recycled ``id``
        return entry[1]
    if id(dc_type) in visiting:
        return None
    direct = final_field_names(dc_type)
    path: "tuple[str, ...] | None" = (min(direct),) if direct else None
    if path is None:
        visiting = visiting | {id(dc_type)}
        for field in dataclasses.fields(dc_type):
            if isinstance(field.type, type) and dataclasses.is_dataclass(field.type):
                child = _first_final_path(field.type, visiting)
                if child is not None:
                    path = (field.name,) + child
                    break
    _final_path_cache[id(dc_type)] = (dc_type, path)
    return path


def _build_final_plan(dc_type: type) -> "frozenset[str]":
    """Validate the dataclass's Final declarations and return their field names."""
    final_names = []
    for field in dataclasses.fields(dc_type):
        annotation = field.type
        if not dataclass_field_is_final(dc_type, field.name, annotation):
            # A ``Final``-like form that is not ``typing.Final`` would also silently become a runtime arg.
            origin = typing.get_origin(annotation)
            if origin is not None and "Final" in _describe_annotation(origin):
                raise TypeError(
                    f"{dc_type.__name__}.{field.name}: annotation {annotation!r} looks like a ``Final`` special form "
                    f"but is not ``typing.Final`` (got origin {origin!r}). Use ``typing.Final`` from the standard "
                    f"library."
                )
            continue
        final_names.append(field.name)

    if final_names and not _rebinding_is_prevented(dc_type):
        # ``lookup`` memoises on ``id(arg)``, so a reassignment after launch one would reuse the stale kernel.
        raise TypeError(
            f"{dc_type.__name__} has ``Final`` field(s) {sorted(final_names)} but is not frozen. A ``Final`` field's "
            f"value is baked into the compiled kernel, so it must not be reassignable - reassigning it would silently "
            f"keep using the kernel compiled for the old value. Declare the class as "
            f"``@dataclasses.dataclass(frozen=True)`` (or ``unsafe_hash=True`` if you must keep it mutable and accept "
            f"responsibility for never reassigning these fields)."
        )

    # Reject mutable ancestors too: replacing a child can change a nested compile-time constant.
    if not _rebinding_is_prevented(dc_type):
        for field in dataclasses.fields(dc_type):
            if not (isinstance(field.type, type) and dataclasses.is_dataclass(field.type)):
                continue
            nested = _first_final_path(field.type, frozenset({id(dc_type)}))
            if nested is None:
                continue
            leaf = ".".join((dc_type.__name__, field.name) + nested)
            raise TypeError(
                f"{dc_type.__name__} is not frozen but reaches the ``Final`` field {leaf} through its field "
                f"``{field.name}``. A ``Final`` value is baked into the compiled kernel, so no dataclass on the path "
                f"to it may be reassignable - rebinding ``{dc_type.__name__}.{field.name}`` would silently keep using "
                f"the kernel compiled for the previous value. Declare {dc_type.__name__} as "
                f"``@dataclasses.dataclass(frozen=True)`` (or ``unsafe_hash=True`` if you must keep it mutable and "
                f"accept responsibility for never reassigning ``{field.name}``)."
            )
    return frozenset(final_names)


# Cache Final field names by type identity. Retain each type to prevent id reuse.
_final_plan_cache: "dict[int, tuple[type, frozenset[str]]]" = {}


def final_field_names(dc_type: Any) -> "frozenset[str]":
    """Return cached Final field names, validating declarations on the first lookup."""
    entry = _final_plan_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity guard against a recycled ``id``
        return entry[1]
    names = _build_final_plan(dc_type)
    _final_plan_cache[id(dc_type)] = (dc_type, names)
    return names


def subtree_has_final_fields(dc_type: Any) -> bool:
    """Check whether a dataclass or any nested dataclass declares Final fields.

    Final-bearing subtrees must bypass instance launch caches because a constant's class behavior can change between
    launches.
    """
    return _first_final_path(dc_type, frozenset()) is not None
