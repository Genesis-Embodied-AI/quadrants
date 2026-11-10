"""Discover Final members without letting an unsupported sibling reject an inferred dataclass."""

import dataclasses
import typing
from typing import TYPE_CHECKING, Any

from quadrants.lang._deferred_errors import (
    create_deferred_member_type_for_invalid_dataclass_member,
)
from quadrants.lang._final_annotation import dataclass_field_is_final
from quadrants.lang._final_dataclass_fields import final_scalar_key

if TYPE_CHECKING:
    from typing import _SpecialForm

    from quadrants.lang._deferred_errors import DeferredMemberType
    from quadrants.lang._runtime_dataclass import FinalValueKeyCache


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


def runtime_final_type(
    value: object,
    non_frozen_ancestor_path: str | None,
    name: str,
    final_value_key_by_value_id: "FinalValueKeyCache | None" = None,
) -> "_SpecialForm | DeferredMemberType":
    """Return a valid Final annotation or a deferred diagnostic for the current arriving value/path."""
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


def runtime_final_key(value: object, final_value_key_by_value_id: "FinalValueKeyCache | None") -> tuple[Any, ...]:
    """Reuse a value key validated during this dispatch only; never retain it across kernel calls."""
    cached = final_value_key_by_value_id.get(id(value)) if final_value_key_by_value_id is not None else None
    if cached is not None and cached[0] is value:
        return cached[1]
    return final_scalar_key(value, live=True)
