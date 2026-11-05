"""Discover Final members without letting an unsupported sibling reject an inferred dataclass."""

import copy
import dataclasses

from quadrants.lang._deferred_dataclass import DeferredMemberType, deferred_member
from quadrants.lang._final_dataclass_fields import (
    final_field_names,
    final_scalar_key,
    is_final_annotation,
)

# Class identity maps to the retained class, validated Final field names, and deferred declaration errors.
_final_declaration_validation_by_dataclass_type_id: dict[
    int, tuple[type, frozenset[str], dict[str, DeferredMemberType]]
] = {}


def validate_final_field_declarations(source: type) -> tuple[frozenset[str], dict[str, DeferredMemberType]]:
    """Reuse legacy declaration validation, isolating unsupported Final members in a per-field fallback.

    Ordinary nested annotations must not introduce phantom children. Nested Final paths are checked against the
    arriving values by the runtime schema builder. Malformed declarations (bare or unresolved Final) keep their
    existing eager diagnostic; only real Final[T] members can defer unsupported-type or mutable-carrier errors.
    """
    cached = _final_declaration_validation_by_dataclass_type_id.get(id(source))
    if cached is not None and cached[0] is source:
        return cached[1], cached[2]
    fields = {}
    for field in dataclasses.fields(source):
        copied = copy.copy(field)
        if isinstance(field.type, type) and dataclasses.is_dataclass(field.type):
            copied.type = object
        fields[field.name] = copied
    attributes = {
        "__module__": source.__module__,
        "__annotations__": dict(source.__annotations__),
        "__dataclass_fields__": fields,
        "__dataclass_params__": source.__dataclass_params__,
        "_qd_runtime_origin": source,
    }
    direct = type(source.__name__, (), attributes)
    errors = {}
    try:
        names = final_field_names(direct)
    except TypeError:
        found = set()
        for name, field in fields.items():
            single = type(source.__name__, (), {**attributes, "__dataclass_fields__": {name: field}})
            try:
                found.update(final_field_names(single))
            except TypeError as error:
                if not is_final_annotation(field.type):
                    raise
                errors[name] = deferred_member(TypeError, str(error))
        names = frozenset(found)
    _final_declaration_validation_by_dataclass_type_id[id(source)] = (source, names, errors)
    return names, errors


def runtime_final_type(
    annotation, value, non_frozen_ancestor_path: str | None, name: str, final_value_key_by_value_id=None
):
    """Return a valid Final annotation or a deferred diagnostic for the current arriving value/path."""
    if non_frozen_ancestor_path is not None:
        parent_name = non_frozen_ancestor_path.split(".", 1)[0]
        return deferred_member(
            TypeError,
            f"{parent_name} is not frozen but reaches the ``Final`` field {non_frozen_ancestor_path}.{name}. "
            "No dataclass on the path to a used Final value may be reassignable.",
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
            return deferred_member(TypeError, str(error))
    return annotation


def runtime_final_key(value, final_value_key_by_value_id):
    """Reuse a value key validated during this dispatch only; never retain it across kernel calls."""
    cached = final_value_key_by_value_id.get(id(value)) if final_value_key_by_value_id is not None else None
    if cached is not None and cached[0] is value:
        return cached[1]
    return final_scalar_key(value, live=True)
