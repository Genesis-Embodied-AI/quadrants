"""Validate Final constant values and construct their specialization keys."""

import enum
from typing import TYPE_CHECKING, Any

import numpy as np

from quadrants.lang._final.final_class_identity import (
    _FRAMEWORK_PRIMITIVE_METACLASSES,
    _STRUCTURAL_CLASS_ATTRS,
    _is_baked_base_type,
    _is_exact_baked_type,
    _observable_metaclass_attr,
    _pack_f64,
    _slot_names,
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


def _reject_stateful_primitive_subclass(value: Any) -> None:
    """Reject scalar subclasses with unsupported instance state or class behavior."""
    cls = type(value)
    if _is_exact_baked_type(cls):
        return
    if not (isinstance(value, (int, float, str)) or isinstance(value, np.generic)):
        return  # not a baked-primitive / NumPy-scalar value at all
    if isinstance(value, np.generic) and _is_baked_base_type(cls):
        return  # an exact NumPy library scalar: a pure value, no user state/behavior
    if getattr(value, "__dict__", None):
        stateful = True
    else:
        stateful = False
        for klass in cls.__mro__:
            if any(slot not in ("__dict__", "__weakref__") and hasattr(value, slot) for slot in _slot_names(klass)):
                stateful = True
                break
    if stateful:
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, a subclass of a baked primitive that "
            f"carries extra per-instance state. A ``Final`` value is baked as a compile-time literal keyed by its "
            f"value, so state a kernel could read at compile time (e.g. ``cfg.x.unit``) would not select a "
            f"distinct specialization. Pass a plain ``bool`` / ``int`` / ``float`` / ``str`` (or an ``enum`` "
            f"member) instead."
        )
    # Inspect all user bases, including mixins after the primitive base in the MRO.
    for klass in cls.__mro__:
        if _is_baked_base_type(klass):
            continue
        for attr in vars(klass):
            if attr not in _STRUCTURAL_CLASS_ATTRS:
                raise TypeError(
                    f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, a subclass of a baked "
                    f"primitive that defines observable class-level behavior/state (e.g. attribute {attr!r}). "
                    f"The key identifies the subclass only by ``module``/``qualname``, which does not uniquely "
                    f"identify a dynamically created class, so two distinct same-named subclasses (e.g. from a "
                    f"factory) whose ``{attr}`` a kernel could read at compile time would select the same "
                    f"specialization. Pass a plain ``bool`` / ``int`` / ``float`` / ``str`` (or an ``enum`` "
                    f"member) instead."
                )
    # Metaclasses can expose additional behavior outside the class's own MRO.
    metacls: type = type(cls)
    for mcls in metacls.__mro__:
        if mcls in _FRAMEWORK_PRIMITIVE_METACLASSES:
            continue
        attr = _observable_metaclass_attr(mcls)
        if attr is not None:
            raise TypeError(
                f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, a subclass of a baked "
                f"primitive whose metaclass defines observable class-level behavior/state (e.g. attribute "
                f"{attr!r}, readable as ``cfg.x.__class__.{attr}``). ``module``/``qualname`` does not uniquely "
                f"identify a dynamically created class, so two same-named subclasses (e.g. from a factory) whose "
                f"metaclass ``{attr}`` a kernel could read would select the same specialization. Pass a plain "
                f"``bool`` / ``int`` / ``float`` / ``str`` (or an ``enum`` member) instead."
            )


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
    if isinstance(value, np.floating):
        # ``dtype.str`` + ``tobytes()`` preserves sign/NaN/width, in the float-tagged space.
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        if _is_baked_base_type(cls):  # an exact NumPy float scalar (``np.float32``/``np.float64``/...)
            return (_FLOAT_KEY_TAG, value.dtype.str, value.tobytes())
        return (_FLOAT_KEY_TAG, *_subclass_identity(cls, live), value.dtype.str, value.tobytes())
    if isinstance(value, float):
        # A ``float`` subclass (the exact builtin took the fast path); bit-encode but keep the subclass identity.
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        return (_FLOAT_KEY_TAG, *_subclass_identity(cls, live), _unpack_u64(_pack_f64(value))[0])
    # Include the exact type so equal values of different types select different specializations.
    _reject_stateful_primitive_subclass(value)
    cls = type(value)
    # Use the base value so a custom representation cannot hide differences in offline keys.
    if _is_exact_baked_type(cls):
        canonical = value
    elif isinstance(value, int):  # a Python ``int`` subclass (``bool`` cannot be subclassed)
        canonical = int.__int__(value)  # base slot: bypass an overridden ``__int__`` / ``__index__``
    elif isinstance(value, np.integer):  # NumPy integer scalar (no base int slot)
        canonical = value.item()  # override-proof extraction of the Python ``int``
    elif isinstance(value, str):  # a ``str`` subclass, including ``np.str_``; bypass any ``__str__`` override
        canonical = str.__str__(value)
    elif isinstance(value, np.bool_):  # NumPy boolean scalar -> plain ``bool``
        canonical = bool(value)
    else:
        # Reject arbitrary objects: equality and hashing need not capture their mutable state.
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, which is not a supported "
            f"compile-time constant. A ``Final`` value - or an ``enum`` member's underlying value - must be a "
            f"``bool`` / ``int`` / ``float`` / ``str`` (or a NumPy scalar analogue), or an ``enum`` member; an "
            f"arbitrary object cannot be keyed by value alone. Bake the specific scalar you need as a ``Final`` "
            f"field instead."
        )
    if _is_baked_base_type(cls):  # exact builtin primitive or exact NumPy scalar
        return (_SCALAR_KEY_TAG, cls.__module__, cls.__qualname__, canonical)
    # A behavior-free user subclass (``class Grams(int): pass``): carry the class-identity token like the other cases.
    return (_SCALAR_KEY_TAG, *_subclass_identity(cls, live), canonical)


def get_cached_or_validate_value_for_final_field_and_get_key(
    value: object, final_value_key_by_value_id: "FinalValueKeyCache | None"
) -> tuple[Any, ...]:
    """Reuse a value key validated during this dispatch only; never retain it across kernel calls."""
    cached = final_value_key_by_value_id.get(id(value)) if final_value_key_by_value_id is not None else None
    if cached is not None and cached[0] is value:
        return cached[1]
    return validate_value_for_final_field_and_get_key(value, live=True)
