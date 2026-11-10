"""Validate Final constant values and construct their specialization keys."""

import enum
from typing import Any

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

# Type tags keeping the key spaces disjoint: Python treats distinct constants as ``==`` with equal hashes (a float's
# bits vs an int; an ``IntEnum`` member vs its bare scalar; ``True`` vs ``1`` vs ``np.int64(1)``), so bare values
# would collide.
_FLOAT_KEY_TAG = "f64"
_ENUM_KEY_TAG = "enum"
_SCALAR_KEY_TAG = "scalar"


def _reject_stateful_primitive_subclass(value: Any) -> None:
    """Reject a subclass of a baked scalar (or NumPy scalar) that carries observable state the key cannot capture:
    per-instance state (``__dict__`` or a populated slot), or class-level behavior/state (a method/property/class var
    or overridden dunder, on the subclass chain or its metaclass). ``module``/``qualname`` does not identify a dynamic
    class, so two same-named factory subclasses would collide. Exact primitives/NumPy scalars and behavior-free
    subclasses (``class Meters(float): pass``) pass.
    """
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
    # Inspect every user class on the MRO, skipping trusted bases - not a scan-terminating break at the first one: a
    # mixin *after* the primitive base (``class Unit(int, Labels)`` -> MRO ``(Unit, int, Labels, object)``) is
    # observable too (``cfg.x.__class__.label``).
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
    # metaclass too: ``cfg.x.__class__.label`` -> ``type(cls).label``, off the MRO walk. ``: type`` keeps ``__mro__``
    # an instance read.
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
    """Hashable digest of ``cls``'s entire member map (each name + its value via ``final_scalar_key``). A baked
    ``Final`` member keys on this, not only the selected member, because a sibling is observable
    (``cfg.mode.__class__.OTHER.value``). ``()`` if there is no member map."""
    member_map = getattr(cls, "_member_map_", None)
    if not isinstance(member_map, dict):
        return ()
    return tuple((name, final_scalar_key(member.value, live)) for name, member in member_map.items())


def final_scalar_key(value: Any, live: bool = False) -> tuple[Any, ...]:
    """Collision-free key component for a baked ``Final`` value. Each value is *type-tagged*, because Python treats
    distinct compile-time constants as equal with equal hashes:

    - ``float`` (builtin/subclass/NumPy) -> exact IEEE-754 bits (so ``-0.0``/``0.0`` and NaN payloads stay distinct).
    - ``enum`` member -> ``(_ENUM_KEY_TAG, *class-identity, name, value-key, member-map)``, keyed by identity (an
      ``IntEnum`` member ``==`` its bare scalar); per-member-state members are rejected.
    - other scalar (``bool``/``int``/``str`` + NumPy) -> ``(_SCALAR_KEY_TAG, module, qualname, canonical-value)``, the
      value coerced via a base slot so the offline string is faithful under a nonstandard ``repr``.

    A subclass/enum carries class identity via ``_subclass_identity`` (``live`` = ``_ClassRef``, offline =
    process-stable), including the class kind (base + metaclass) so redefining the base or metaclass keys apart
    offline. An unsupported/arbitrary value is *rejected*, not keyed by its own ``__eq__``. Re-runs per launch for
    Final subtrees (not cached), catching a class turned behaviorful after launch one.
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
    # ``bool``/``int``/``str`` + NumPy analogues; the exact type completes the tag so value-equal but distinct-typed
    # constants (``True``/``1``/``np.int64(1)``) never share a specialization.
    _reject_stateful_primitive_subclass(value)
    cls = type(value)
    # Embed a *canonical* base value: the offline key ``str``-ifies this, so a subclass whose ``__repr__`` drops its
    # value must not serialize two distinct values to one string.
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
        # An arbitrary object is not key-determined by value (two ``==`` instances with differing attributes would
        # share a specialization) and may be mutable under the cached key, so reject rather than trust its
        # ``__eq__`` / ``__hash__``.
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
