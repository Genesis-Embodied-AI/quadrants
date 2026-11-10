"""Identify Final constant classes and snapshot their compile-time-observable structure."""

import enum
import itertools
import os
import struct
import sys
import types
import uuid
import weakref
from typing import Any, NoReturn

import numpy as np

_MISSING = object()  # Distinguish missing attributes from attributes whose value is None.

# Per-process token mixed into the *offline* key of a dynamically created class (see ``_subclass_identity``): its
# serial is only process-local, so the nonce makes such a class a guaranteed cross-process miss rather than a wrong
# reuse. Reseeded after ``fork`` (the child inherits the string); ``spawn`` re-imports and reseeds on its own.
_PROCESS_NONCE = uuid.uuid4().hex


def _reseed_process_nonce() -> None:
    """Give a forked child a fresh nonce for offline class identity keys."""
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
    """``cls`` *is* one of ``_EXACT_BAKED_TYPES`` (identity, never ``==``): a subclass whose metaclass fakes ``==`` to
    a builtin must not skip validation or be canonicalized, collapsing two distinct same-qualified factory classes."""
    return any(cls is t for t in _EXACT_BAKED_TYPES)


# ``Py_TPFLAGS_HEAPTYPE``: clear on C static types (builtins, NumPy scalars), set on any Python-defined type and not
# user-settable, so it tells a library scalar base from a user subclass spoofing ``__module__`` (see below).
_HEAPTYPE_FLAG = 1 << 9

# Names CPython auto-generates for a bare ``class X(base): pass``. A primitive subclass may carry only these; anything
# else is observable behavior we cannot key on. ``__firstlineno__`` / ``__static_attributes__`` are 3.13+.
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
    """Name of a user attribute on a metaclass layer (readable as ``cfg.x.__class__.<attr>``), or None; only
    ``_STRUCTURAL_CLASS_ATTRS`` are exempt. Even ``__eq__`` / ``__hash__`` count: the identity key stops them
    collapsing classes, but a kernel can still observe them.
    """
    for name in vars(mcls):
        if name not in _STRUCTURAL_CLASS_ATTRS:
            return name
    return None


def _is_baked_base_type(klass: type) -> bool:
    """True for a base a baked value may subclass without adding observable state: a builtin primitive, ``object``, or
    a *static* ``np.generic`` subclass. A NumPy scalar base is a static type (no ``Py_TPFLAGS_HEAPTYPE``), so a user
    ``class Foo(np.float64)`` (a heap type, even spoofing ``__module__``) cannot masquerade as one.
    """
    if _is_exact_baked_type(klass) or klass is object:
        return True
    is_heap_type = bool(klass.__flags__ & _HEAPTYPE_FLAG)
    return issubclass(klass, np.generic) and not is_heap_type


def _slot_names(klass: type) -> tuple[str, ...]:
    """Slot names declared by ``klass`` via ``__slots__`` (a tuple of strings). Robust to a post-creation rebind of
    ``__slots__`` to a non-iterable (e.g. ``None``) or to non-string entries: those create no real slot descriptors, so
    they contribute no names here (the rebind itself is captured by the class-kind key)."""
    slots = getattr(klass, "__slots__", ())
    if isinstance(slots, str):
        return (slots,)
    if not hasattr(slots, "__iter__"):
        return ()
    return tuple(s for s in slots if isinstance(s, str))


def _class_not_uniquely_identified(obj: Any) -> bool:
    """True if ``obj`` (a class, metaclass, or a callable/class bound as an attr value) cannot be recovered from its
    module + ``__qualname__`` (a ``<locals>`` qualname, a missing name, or it does not resolve back): it then shares
    its identity string with every sibling built the same way, yet is observably distinct (``cfg.x.__class__ is
    First``), so callers add an identity token to the key.
    """
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
    """A strong ref to an object (a class, or a callable/class bound as an attr value) that hashes/compares by *object
    identity* (never via a metaclass ``__eq__``/``__hash__``), for the in-process spec key: it pins the object so its
    ``id`` cannot be recycled, and stays distinct under a metaclass ``__eq__`` (rejected upstream anyway) or a reload.
    """

    __slots__ = ("cls",)

    def __init__(self, cls: Any) -> None:
        self.cls = cls

    def __hash__(self) -> int:
        return object.__hash__(self.cls)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _ClassRef) and self.cls is other.cls

    def __repr__(self) -> str:
        return f"<_ClassRef 0x{id(self.cls):x}>"  # identity only; the live key is never serialized


# Monotonic, never-reused serials for dynamically created objects (classes, or callables bound as attr values), used
# only in the *offline* key. Keyed by ``id`` with an identity (``is``) guard - not a ``WeakKeyDictionary``, whose
# ``get`` compares keys by ``__eq__``/``__hash__``, so a metaclass making distinct classes compare equal would alias
# two objects to one serial. ``id`` alone is unsafe (after an object is freed CPython recycles its address), so the
# entry holds a weakref: a callback drops it on death and the guard rejects a stale entry whose ``id`` was recycled.
_dynamic_object_serials: "dict[int, tuple[weakref.ref, int]]" = {}
_dynamic_class_serial_counter = itertools.count()


def _dynamic_class_serial(obj: Any) -> int:
    """A process-unique, non-recyclable serial for a dynamically created ``obj`` (class or callable), stable for its
    lifetime. An object that cannot be weakly referenced draws a fresh serial each call - a redundant miss, never a
    stale reuse.
    """
    key = id(obj)
    entry = _dynamic_object_serials.get(key)
    if entry is not None and entry[0]() is obj:  # identity guard: same live object, ``id`` not recycled
        return entry[1]
    serial = next(_dynamic_class_serial_counter)

    def _drop(ref: "weakref.ref", k: int = key) -> None:
        cur = _dynamic_object_serials.get(k)
        if cur is not None and cur[0] is ref:  # only our own entry, never one that recycled the ``id``
            del _dynamic_object_serials[k]

    try:
        ref = weakref.ref(obj, _drop)
    except TypeError:
        return serial
    _dynamic_object_serials[key] = (ref, serial)
    return serial


def _identity_component(obj: Any, live: bool) -> _ClassRef | tuple[str, int] | None:
    """Identity token separating objects (classes, metaclasses, or user callables/classes bound as attr values) that
    share a module/qualname - factory-built, reload-rebound, or ``<locals>``. ``live``: an id-pinning ``_ClassRef``
    (never serialized). offline: ``None`` when ``obj`` is uniquely recoverable from its module (stable cross-process),
    else ``(_PROCESS_NONCE, serial)`` - a guaranteed cross-process miss rather than a wrong reuse."""
    if live:
        return _ClassRef(obj)
    return (_PROCESS_NONCE, _dynamic_class_serial(obj)) if _class_not_uniquely_identified(obj) else None


# Structural attrs (from ``_STRUCTURAL_CLASS_ATTRS``) that are user-writable *and* readable at compile time
# (``cfg.x.__class__.<attr>``), so their *value* is folded into the class kind rather than exempted. ``__module__`` /
# ``__qualname__`` are omitted (already the entry's leading elements). Absence is itself observable, so it is kept
# distinct from an explicit ``None`` via ``_ATTR_ABSENT``. ``__firstlineno__`` / ``__static_attributes__`` are 3.13+
# (absent -> ``_ATTR_ABSENT`` on older runtimes); both are writable and observable, so their value is keyed too.
_KEYED_STRUCTURAL_ATTRS = (
    "__doc__",
    "__slots__",
    "__weakref__",
    "__firstlineno__",
    "__static_attributes__",
)
_ATTR_ABSENT = "<qd:attr-absent>"  # not a valid identifier, so it can never equal a real ``__slots__`` name/value

# Auto-generated slot descriptors (the common case: a class's own ``__dict__``/``__weakref__``). Their only stable,
# address-free content is ``(type, __name__)``; the owning class - separately keyed with its own identity - already
# separates two classes' same-named descriptors, so no per-value identity is needed.
_DESCRIPTOR_ATTR_TYPES = (
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.BuiltinFunctionType,
)

# User-bound Python classes/callables: rejected as structural attr values (their mutable behavior/state - ``__code__``
# / ``__defaults__`` / a class dict - is observable at compile time yet uncapturable by the key, and unvalidated by the
# source cache). Contrast ``_DESCRIPTOR_ATTR_TYPES``, whose behavior is fixed at the C level, so they stay keyable.
_IDENTIFIED_ATTR_TYPES = (
    types.FunctionType,
    types.MethodType,
    classmethod,
    staticmethod,
    property,
    type,
)


def _reject_structural_subclass(val: Any) -> NoReturn:
    """Reject a builtin *subclass* bound in a keyed structural attr. Scalar subclasses route through
    ``final_scalar_key`` (identity + stateful-subclass rejection); this covers the rest (``complex`` / ``bytes`` /
    container subclasses) that may add per-instance state or a distinct identity the bare value/elements would drop."""
    cls = type(val)
    raise TypeError(
        f"A ``Final``-baked class carries a structural attribute holding a {cls.__module__}.{cls.__qualname__} "
        f"instance (a builtin subclass), which cannot be keyed faithfully: a kernel can read it at compile time "
        f"(``cfg.x.__class__.<attr>``) and the subclass may add per-instance state or identity, so reusing a "
        f"specialization across a change would be unsound. Use a plain builtin value instead."
    )


def _canonical_attr_value(val: Any, live: bool) -> Any:
    """A deterministic, hashable, *lossless* snapshot of a class-dict value read at compile time
    (``cfg.x.__class__.<attr>``). Ints/str/bytes/None by value (type-tagged, so ``True``/``1`` differ); floats/complex
    by exact IEEE bits (``0.0``/``-0.0`` never collapse); tuples/lists/dicts recursively and sets order-independently,
    each keeping its container type (``()`` vs ``[]`` compare unequal). A scalar subclass routes through
    ``final_scalar_key`` (stateful-subclass rejection + subclass identity); an auto-generated slot descriptor reduces to
    an address-free ``(type, name)`` token carrying its owner's identity. A Python class/callable, a builtin subclass,
    or any other value whose observable content the key cannot capture is rejected rather than collapsed to a lossy
    token that could reuse a stale specialization after the value changes."""
    if val is None:
        return None
    # A scalar *subclass* (or enum member) may carry per-instance state or a distinct class identity that the bare
    # value discards, so route it through the full baked-scalar machinery (stateful-subclass rejection + subclass
    # identity + IEEE bits). Exact builtins below skip that and encode directly.
    if (isinstance(val, (bool, int, float, str)) and type(val) not in (bool, int, float, str)) or isinstance(
        val, enum.Enum
    ):
        # Scalar keys include class identity too. Import only for this recursive case, after both modules are loaded.
        # pylint: disable-next=import-outside-toplevel
        from quadrants.lang.final._final_constant_values import final_scalar_key

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
        # An auto-generated descriptor has an address-based ``repr`` (unkeyable) and its own ``__qualname__`` is just
        # the slot name, so its identity is its owner (``__objclass__``): a *foreign* descriptor (``A.__weakref__``
        # bound elsewhere) must carry ``A``'s module/qualname + identity so it never collides with ``B``'s. An own
        # descriptor's owner is this class (identity ``None`` when module-resolvable), keeping the offline key stable.
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
        # A Python-level class/callable carries mutable behavior/state (``__code__`` / ``__defaults__`` / a class
        # dict) that no ``module``/``qualname``/identity token captures, and - unlike a kernel or ``@qd.func`` - it is
        # not among the sources the source cache validates. So a *resolvable* one keeps the same offline key even after
        # its body changes between processes (``cfg.x.__class__.<attr>.__defaults__`` flips), which would reuse a stale
        # specialization. Reject rather than key it unsoundly.
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
    """Keyed value of one structural attr on ``klass`` itself, or ``_ATTR_ABSENT`` if it is not in the class dict
    (Python lets it be added/deleted after creation, and ``name in cls.__dict__`` is observable)."""
    if name not in klass.__dict__:
        return _ATTR_ABSENT
    return _canonical_attr_value(klass.__dict__[name], live)


def _kind_entry(klass: type, live: bool) -> tuple:
    """``(module, qualname, name, *keyed-structural-attr-values, identity)`` for one MRO entry - but only for a heap
    (Python) class; a static/builtin base's values are immutable, so they are omitted to avoid bloating the key (e.g.
    ``int``'s doc). ``__name__`` is a ``type`` getset (not in ``vars(cls)``, so the behavior scan misses it) that is
    independently reassignable from ``__qualname__``, and the trailing identity separates a factory-built base or
    metaclass that shares a module/qualname with its siblings - both observable (``cfg.x.__class__.__name__``,
    ``cfg.x.__class__.__class__ is ExpectedMeta``)."""
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
    """The class's compile-time-observable *kind* as ``(base MRO, metaclass MRO)``, each entry from ``_kind_entry``.
    Redefining a resolvable subclass's primitive/enum base (``class Unit(int)`` -> ``np.int64``, ``enum.Enum`` ->
    ``enum.IntEnum``) or its metaclass (``metaclass=EmptyMeta``, or a distinct factory-built one), or mutating a user
    class's ``__name__``/``__doc__``/``__slots__``/``__weakref__``, keeps module/qualname/canonical unchanged yet is
    observable (``cfg.x.__class__.__mro__[1]``, ``cfg.x.__class__.__class__``, ``cfg.x.__class__.<attr>``), so the key
    must separate them."""
    metaclass: type = type(cls)  # annotate as instance so ``.__mro__`` is the tuple, not the descriptor (pyright)
    base_kind = tuple(_kind_entry(base, live) for base in cls.__mro__)
    meta_kind = tuple(_kind_entry(meta, live) for meta in metaclass.__mro__)
    return (base_kind, meta_kind)


def _subclass_identity(cls: type, live: bool) -> tuple:
    """Class-identity component for a subclass of a baked type or an ``enum``: ``(module, qualname, kind, identity)``.
    ``module``/``qualname`` do not identify the class *object* (factory-built, or reload-rebound) yet
    ``cfg.x.__class__ is First`` is observable, so ``identity`` (see ``_identity_component``) separates distinct
    objects. ``kind`` (``_class_kind``: base + metaclass MRO) separates a resolvable subclass redefined with a
    different base or metaclass. Over-separating behaviour-equal kinds only costs a cache miss, never a wrong reuse."""
    return (cls.__module__, cls.__qualname__, _class_kind(cls, live), _identity_component(cls, live))
