"""Validate Final constant values and construct their specialization keys."""

import enum
from typing import TYPE_CHECKING, Any

import numpy as np

from quadrants.lang._final.final_class_identity import (
    _is_baked_base_type,
    _is_exact_baked_type,
    _pack_f64,
    _subclass_identity,
    _unpack_u64,
)
from quadrants.lang._final.final_enum_validation import _reject_stateful_enum_member

if TYPE_CHECKING:
    from quadrants.lang._runtime_typing.runtime_dataclass import FinalValueKeyCache

# Separate constants that Python considers equal, such as True, 1, and an IntEnum member.
_FLOAT_KEY_TAG = "f64"
_ENUM_KEY_TAG = "enum"
_SCALAR_KEY_TAG = "scalar"


def _enum_member_map_key(cls: type, live: bool) -> tuple:
    """Build a key for every enum member's name and value.

    Include siblings because compiled code can access them through the selected member's class.
    """
    member_map = getattr(cls, "_member_map_", None)
    if not isinstance(member_map, dict):
        return ()
    return tuple(
        (name, validate_value_for_final_field_and_get_key(member.value, live)) for name, member in member_map.items()
    )


def validate_value_for_final_field_and_get_key(value: Any, live: bool = False) -> tuple[Any, ...]:
    """Validate a Final value and build its specialization key.

    Keep distinct constants separate even when Python considers them equal, such as True and 1 or an IntEnum member and
    its integer value. Float keys preserve exact bits, including signed zero and NaN payloads.

    Revalidate on each launch so a later change to a constant's class cannot silently reuse compiled code.
    """
    if type(value) is float:
        return (_FLOAT_KEY_TAG, _unpack_u64(_pack_f64(value))[0])
    if isinstance(value, enum.Enum):
        # Before the ``float``/``int`` branches so a mixed-in enum keys by identity, not value.
        _reject_stateful_enum_member(value)
        cls = type(value)
        return (
            _ENUM_KEY_TAG,
            *_subclass_identity(cls, live),
            value.name,
            validate_value_for_final_field_and_get_key(value.value, live),
            _enum_member_map_key(cls, live),
        )
    cls = type(value)
    if not _is_baked_base_type(cls):
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, which is not a supported "
            "compile-time constant. Use an exact builtin bool, int, float or str, a NumPy scalar analogue, "
            "or a supported enum member. Custom scalar subclasses are not supported."
        )
    if isinstance(value, np.floating):
        # Preserve the NumPy float's width, signed zero and NaN payload.
        return (_FLOAT_KEY_TAG, value.dtype.str, value.tobytes())
    if _is_exact_baked_type(cls):
        canonical = value
    elif isinstance(value, np.integer):
        canonical = value.item()
    elif isinstance(value, np.str_):
        canonical = str(value)
    elif isinstance(value, np.bool_):
        canonical = bool(value)
    else:
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, which is not a supported "
            "compile-time constant. Use a boolean, integer, floating-point or string scalar."
        )
    return (_SCALAR_KEY_TAG, cls.__module__, cls.__qualname__, canonical)


def get_cached_or_validate_value_for_final_field_and_get_key(
    value: object, final_value_key_by_value_id: "FinalValueKeyCache | None"
) -> tuple[Any, ...]:
    """Reuse a value key validated during this dispatch only; never retain it across kernel calls."""
    cached = final_value_key_by_value_id.get(id(value)) if final_value_key_by_value_id is not None else None
    if cached is not None and cached[0] is value:
        return cached[1]
    return validate_value_for_final_field_and_get_key(value, live=True)
