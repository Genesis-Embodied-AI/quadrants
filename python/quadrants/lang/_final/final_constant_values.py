"""Validate Final constant values and construct their specialization keys."""

import enum
from typing import Any

from quadrants.lang._final.final_class_identity import (
    _is_supported_builtin_scalar_type,
    _pack_f64,
    _subclass_identity,
    _unpack_u64,
)
from quadrants.lang._final.final_enum_validation import _reject_stateful_enum_member

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
    return tuple((name, final_scalar_key(member.value, live)) for name, member in member_map.items())


def final_scalar_key(value: Any, live: bool = False) -> tuple[Any, ...]:
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
            final_scalar_key(value.value, live),
            _enum_member_map_key(cls, live),
        )
    cls = type(value)
    if not _is_supported_builtin_scalar_type(cls):
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, which is not a supported "
            "compile-time constant. Use an exact builtin bool, int, float or str, or a supported enum member. "
            "Custom scalar subclasses and NumPy scalars are not supported."
        )
    return (_SCALAR_KEY_TAG, cls.__module__, cls.__qualname__, value)
