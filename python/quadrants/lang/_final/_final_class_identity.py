"""Identify Final constant classes and snapshot their compile-time-observable structure."""

import enum
import itertools
import os
import struct
import sys
import types
import uuid
from typing import Any, NoReturn

import numpy as np

from quadrants.lang._weak_identity_cache import WeakIdentityCache

_MISSING = object()  # Distinguish missing attributes from attributes whose value is None.

# Prevent dynamic class keys from being reused across processes. Refresh this token after fork.
_PROCESS_NONCE = uuid.uuid4().hex


def _reseed_process_nonce() -> None:
    """Give a forked child a fresh token for offline class identity keys."""
    global _PROCESS_NONCE
    _PROCESS_NONCE = uuid.uuid4().hex


if hasattr(os, "register_at_fork"):  # POSIX only; ``spawn``/Windows re-import and reseed
    os.register_at_fork(after_in_child=_reseed_process_nonce)


# Pack/unpack a ``float`` as its exact IEEE-754 bits (see ``final_scalar_key``).
_pack_f64 = struct.Struct("<d").pack
_unpack_u64 = struct.Struct("<Q").unpack


# Exact baked primitives; an exact instance is a pure literal and skips the stateful-subclass check.
_EXACT_BAKED_TYPES = (bool, int, float, str)


def _is_exact_baked_type(cls: type) -> bool:
    """Check whether a class is exactly one of the supported built-in scalar types.

    Use identity because a metaclass can make distinct classes compare equal.
    """
    return any(cls is t for t in _EXACT_BAKED_TYPES)


# Distinguish static library types from user-defined subclasses, even if they spoof their module.
_HEAPTYPE_FLAG = 1 << 9

# Attributes allowed on an otherwise empty scalar subclass.
_STRUCTURAL_CLASS_ATTRS = frozenset(
    {
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__slots__",
        "__firstlineno__",
        "__static_attributes__",
    }
)

# Framework layers on a primitive subclass's metaclass MRO; only user ``type`` subclasses below them are inspected.
_FRAMEWORK_PRIMITIVE_METACLASSES = frozenset({type, object})


def _observable_metaclass_attr(mcls: type) -> "str | None":
    """Return the first unsupported metaclass attribute name, or None."""
    for name in vars(mcls):
        if name not in _STRUCTURAL_CLASS_ATTRS:
            return name
    return None


def _is_baked_base_type(klass: type) -> bool:
    """Check whether a class is an allowed built-in or NumPy scalar base.

    Static NumPy scalar types are allowed; a user subclass must not bypass validation by spoofing its module.
    """
    if _is_exact_baked_type(klass) or klass is object:
        return True
    is_heap_type = bool(klass.__flags__ & _HEAPTYPE_FLAG)
    return issubclass(klass, np.generic) and not is_heap_type


def _slot_names(klass: type) -> tuple[str, ...]:
    """Return the string slot names declared directly on a class."""
    slots = getattr(klass, "__slots__", ())
    if isinstance(slots, str):
        return (slots,)
    if not hasattr(slots, "__iter__"):
        return ()
    return tuple(s for s in slots if isinstance(s, str))


def _class_not_uniquely_identified(obj: Any) -> bool:
    """Check whether an object cannot be recovered by its module and qualified name."""
    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        return True
    module = sys.modules.get(module_name)
    if module is None:
        return True
    resolved: Any = module
    for part in qualname.split("."):
        if part == "<locals>":
            return True
        resolved = getattr(resolved, part, _MISSING)
        if resolved is _MISSING:
            return True
    return resolved is not obj


class _ClassRef:
    """Hold an object strongly and compare it by identity for an in-process cache key."""

    __slots__ = ("cls",)

    def __init__(self, cls: Any) -> None:
        """Retain the object used by this identity key."""
        self.cls = cls

    def __hash__(self) -> int:
        """Hash the retained object by identity."""
        return object.__hash__(self.cls)

    def __eq__(self, other: Any) -> bool:
        """Check whether both keys retain the same object."""
        return isinstance(other, _ClassRef) and self.cls is other.cls

    def __repr__(self) -> str:
        """Display the retained object's identity without invoking its representation."""
        return f"<_ClassRef 0x{id(self.cls):x}>"  # identity only; the live key is never serialized


# Assign unique serials to dynamic objects for offline cache keys.
# Weak references and identity checks prevent reuse after collection or confusion between objects that compare equal.
_dynamic_object_serials: WeakIdentityCache[int] = WeakIdentityCache()
_dynamic_class_serial_counter = itertools.count()


def _dynamic_class_serial(obj: Any) -> int:
    """Return a process-unique serial for an object, reusing it while the object is alive.

    Objects that cannot be weakly referenced receive a fresh serial on each call. This causes extra cache misses rather
    than unsafe reuse.
    """
    serial = _dynamic_object_serials.get(obj)
    if serial is None:
        serial = next(_dynamic_class_serial_counter)
        _dynamic_object_serials.set(obj, serial)
    return serial


def _identity_component(obj: Any, live: bool) -> _ClassRef | tuple[str, int] | None:
    """Return an identity key for live or offline caching.

    Offline keys use no token for objects recoverable by module and qualified name. Dynamic objects need a process-
    specific token to prevent incorrect reuse across processes.
    """
    if live:
        return _ClassRef(obj)
    return (_PROCESS_NONCE, _dynamic_class_serial(obj)) if _class_not_uniquely_identified(obj) else None


# Mutable class attributes whose values must be included in the cache key.
_KEYED_STRUCTURAL_ATTRS = (
    "__doc__",
    "__slots__",
    "__weakref__",
    "__firstlineno__",
    "__static_attributes__",
)
_ATTR_ABSENT = "<qd:attr-absent>"  # not a valid identifier, so it can never equal a real ``__slots__`` name/value

# Built-in descriptors can be keyed by type, name, and owner instead of their memory address.
_DESCRIPTOR_ATTR_TYPES = (
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.BuiltinFunctionType,
)

# Reject Python classes and callables as attribute values: their mutable behavior is not captured by the key.
_IDENTIFIED_ATTR_TYPES = (
    types.FunctionType,
    types.MethodType,
    classmethod,
    staticmethod,
    property,
    type,
)


def _reject_structural_subclass(val: Any) -> NoReturn:
    """Reject built-in subclasses whose extra state would be lost in a class attribute key."""
    cls = type(val)
    raise TypeError(
        f"A ``Final``-baked class carries a structural attribute holding a {cls.__module__}.{cls.__qualname__} "
        f"instance (a builtin subclass), which cannot be keyed faithfully: a kernel can read it at compile time "
        f"(``cfg.x.__class__.<attr>``) and the subclass may add per-instance state or identity, so reusing a "
        f"specialization across a change would be unsound. Use a plain builtin value instead."
    )


def _canonical_attr_value(val: Any, live: bool) -> Any:
    """Build a hashable snapshot of a class attribute, rejecting unsupported values.

    Preserve distinctions observable during compilation, including float bit patterns and container types. A lossy
    snapshot could reuse compiled code after an attribute changes.
    """
    if val is None:
        return None
    # Scalar subclasses and enums need state validation and class identity in their keys.
    if (isinstance(val, (bool, int, float, str)) and type(val) not in (bool, int, float, str)) or isinstance(
        val, enum.Enum
    ):
        # Scalar keys include class identity too. Import only for this recursive case, after both modules are loaded.
        # pylint: disable-next=import-outside-toplevel
        from quadrants.lang._final._final_constant_values import final_scalar_key

        return ("scalarsub", final_scalar_key(val, live))
    if isinstance(val, float):  # exact float (subclasses routed above); IEEE bits so ``0.0``/``-0.0`` never collapse
        return (type(val).__qualname__, _unpack_u64(_pack_f64(val))[0])
    if isinstance(val, complex):
        if type(val) is not complex:  # a ``complex`` subclass may add state/identity the bare bits drop
            _reject_structural_subclass(val)
        return (type(val).__qualname__, _unpack_u64(_pack_f64(val.real))[0], _unpack_u64(_pack_f64(val.imag))[0])
    if isinstance(val, (bool, int, str, bytes)):  # exact only: bool/int/str subclasses routed above, so a subclass here
        if type(val) not in (bool, int, str, bytes):  # is a ``bytes`` subclass, unkeyable by ``final_scalar_key``
            _reject_structural_subclass(val)
        return (type(val).__qualname__, val)  # tag the exact type: ``True``/``1`` compare equal but differ
    if isinstance(val, (tuple, list)):
        if type(val) not in (tuple, list):  # a container subclass may add per-instance state / a distinct identity
            _reject_structural_subclass(val)
        return (type(val).__qualname__, tuple(_canonical_attr_value(v, live) for v in val))
    if isinstance(val, (set, frozenset)):
        if type(val) not in (set, frozenset):
            _reject_structural_subclass(val)
        return (type(val).__qualname__, frozenset(_canonical_attr_value(v, live) for v in val))
    if isinstance(val, dict):
        if type(val) is not dict:
            _reject_structural_subclass(val)
        items = tuple((_canonical_attr_value(k, live), _canonical_attr_value(v, live)) for k, v in val.items())
        return ("dict", items)
    if isinstance(val, _DESCRIPTOR_ATTR_TYPES):
        # Include the descriptor's owner so descriptors borrowed from different classes remain distinct.
        owner = getattr(val, "__objclass__", None)
        identified = owner if owner is not None else val
        return (
            type(val).__qualname__,
            getattr(val, "__name__", None),
            getattr(identified, "__module__", None),
            getattr(identified, "__qualname__", None),
            _identity_component(identified, live),
        )
    if isinstance(val, _IDENTIFIED_ATTR_TYPES):
        # These objects can change behavior without changing identity, making cached compiled code stale.
        cls = type(val)
        raise TypeError(
            f"A ``Final``-baked class carries a structural attribute holding a {cls.__qualname__} "
            f"({getattr(val, '__module__', None)}.{getattr(val, '__qualname__', None)}), a class/callable whose "
            f"mutable behavior a kernel can read at compile time (``cfg.x.__class__.<attr>``) but the key cannot "
            f"capture, so reusing a specialization across a change to it would be unsound. Use a plain value "
            f"(scalar/str/tuple/list/set/dict of such) instead."
        )
    raise TypeError(
        f"A ``Final``-baked class carries a structural attribute holding {val!r} (type {type(val).__qualname__}), "
        f"which cannot be keyed faithfully. A kernel can read it at compile time (``cfg.x.__class__.<attr>``), so "
        f"reusing a specialization across a change to it would be unsound. Use a plain value (scalar/str/tuple/"
        f"list/set/dict of such)."
    )


def _attr_kind(klass: type, name: str, live: bool) -> Any:
    """Build a key for a class attribute, distinguishing missing attributes from present values."""
    if name not in klass.__dict__:
        return _ATTR_ABSENT
    return _canonical_attr_value(klass.__dict__[name], live)


def _kind_entry(klass: type, live: bool) -> tuple:
    """Build a key for one class's identity and observable structural attributes.

    Static built-in classes are immutable, so their structural values are omitted to keep keys small.
    """
    if not klass.__flags__ & _HEAPTYPE_FLAG:
        return (klass.__module__, klass.__qualname__)
    return (
        klass.__module__,
        klass.__qualname__,
        klass.__name__,
        *(_attr_kind(klass, name, live) for name in _KEYED_STRUCTURAL_ATTRS),
        _identity_component(klass, live),
    )


def _class_kind(cls: type, live: bool) -> tuple:
    """Build a key for a class's base classes and metaclasses.

    A class can keep its module and qualified name while its bases, metaclass, or structural attributes change.
    """
    metaclass: type = type(cls)  # annotate as instance so ``.__mro__`` is the tuple, not the descriptor (pyright)
    base_kind = tuple(_kind_entry(base, live) for base in cls.__mro__)
    meta_kind = tuple(_kind_entry(meta, live) for meta in metaclass.__mro__)
    return (base_kind, meta_kind)


def _subclass_identity(cls: type, live: bool) -> tuple:
    """Build a key that distinguishes scalar or enum classes and their observable structure."""
    return (cls.__module__, cls.__qualname__, _class_kind(cls, live), _identity_component(cls, live))
