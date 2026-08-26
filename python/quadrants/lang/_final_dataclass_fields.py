"""Helpers for the ``@dataclasses.dataclass`` kernel-arg path, including ``typing.Final[T]`` compile-time fields.

All ``Final`` resolution/validation is cached per dataclass type (``_final_plan_cache``, ``_final_path_cache``); the
per-launch hot path is a single ``dict.get`` and, with no ``Final`` fields, is untouched.
"""

import dataclasses
import enum
import itertools
import os
import struct
import sys
import types
import typing
import uuid
import weakref
from typing import Any

import numpy as np

_MISSING = object()  # sentinel distinct from ``None`` (a legal attribute value) while resolving a qualname

# Per-process token mixed into the *offline* key of a dynamically created class (see ``_subclass_identity``): its
# serial is only process-local, so the nonce makes such a class a guaranteed cross-process miss rather than a wrong
# reuse. Reseeded after ``fork`` (the child inherits the string); ``spawn`` re-imports and reseeds on its own.
_PROCESS_NONCE = uuid.uuid4().hex


def _reseed_process_nonce() -> None:
    global _PROCESS_NONCE
    _PROCESS_NONCE = uuid.uuid4().hex


if hasattr(os, "register_at_fork"):  # POSIX only; ``spawn``/Windows re-import and reseed
    os.register_at_fork(after_in_child=_reseed_process_nonce)

# Types allowed inside ``Final[T]``: baked as a literal, so ``T`` must be a stable-by-value compile-time constant.
# ``enum.Enum`` subclasses are allowed too (handled separately).
_FINAL_SCALAR_TYPES = frozenset({bool, int, float, str})

# Tailored hints for types where ``Final[T]`` is a likely mistake. Matched by name to avoid import cycles.
_FINAL_REJECT_HINTS: "dict[str, str]" = {
    "NdarrayType": "arrays are runtime data, not compile-time constants - drop the ``Final`` and annotate the field "
    "with ``qd.types.NDArray[dtype, ndim]`` as usual",
    "MatrixType": "matrices are runtime data, not compile-time constants - drop the ``Final``",
    "StructType": "``qd.dataclass`` structs are runtime data, not compile-time constants - drop the ``Final``",
    "Tensor": "``qd.Tensor`` is runtime data, not a compile-time constant - drop the ``Final``",
    # ``qd.Template`` and ``qd.types.annotations.template`` are both named ``Template``.
    "Template": "``Final[qd.Template]`` is redundant - a ``qd.Template`` field is already compile-time; use "
    "``Final[<python type>]`` (e.g. ``Final[int]``) instead",
}


def is_final_annotation(annotation: Any) -> bool:
    """True if ``annotation`` is a subscripted ``typing.Final[T]``.

    Bare ``typing.Final`` returns False (no origin) but is rejected by ``_build_final_plan``, not treated as a field.
    ``typing_extensions.Final`` is the same object (>=3.10).
    """
    return typing.get_origin(annotation) is typing.Final


def _describe_annotation(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or repr(annotation)


def _reject_hint_for(inner: Any) -> str | None:
    """Tailored remediation hint if ``Final[inner]`` names a rejectable type, else None (matched by name, no import)."""
    for name in (type(inner).__name__, _describe_annotation(inner)):
        hint = _FINAL_REJECT_HINTS.get(name)
        if hint is not None:
            return hint
    return None


def _validate_final_inner_type(dc_type: type, field_name: str, annotation: Any) -> None:
    """Raise unless ``Final[annotation]`` names a bakeable compile-time type. Only for a subscripted ``Final``."""
    inner = typing.get_args(annotation)
    if len(inner) != 1:
        raise TypeError(
            f"{dc_type.__name__}.{field_name}: ``typing.Final`` takes exactly one type argument, got "
            f"``Final[{', '.join(_describe_annotation(a) for a in inner)}]``."
        )
    inner_type = inner[0]

    if inner_type in _FINAL_SCALAR_TYPES:
        return
    if isinstance(inner_type, type) and issubclass(inner_type, enum.Enum):
        return

    hint = _reject_hint_for(inner_type)
    if hint is None:
        if isinstance(inner_type, type) and dataclasses.is_dataclass(inner_type):
            hint = (
                "nested dataclasses are walked structurally - drop the ``Final`` from this field and mark the "
                "leaf fields inside it as ``Final[...]`` instead"
            )
        else:
            allowed = ", ".join(sorted(t.__name__ for t in _FINAL_SCALAR_TYPES))
            hint = f"``Final[T]`` supports T in {{{allowed}}} or an ``enum.Enum`` subclass"
    raise TypeError(
        f"{dc_type.__name__}.{field_name}: ``Final[{_describe_annotation(inner_type)}]`` cannot be baked as a "
        f"Quadrants compile-time constant - {hint}."
    )


def _rebinding_is_prevented(dc_type: type) -> bool:
    """True if ``dc_type`` forbids field rebinding: ``frozen=True`` or ``unsafe_hash=True``.

    Not the ``__hash__ is not None`` proxy: ``@dataclass(eq=False)`` inherits ``object.__hash__`` and would look
    frozen, letting a baked ``Final`` constant be silently reassigned.
    """
    params = getattr(dc_type, "__dataclass_params__", None)
    if params is None:
        return dc_type.__hash__ is not None
    return params.frozen or params.unsafe_hash


# ``id(type) -> (type, path to a Final leaf or None)``. Identity-keyed with a strong-ref+``is`` guard (a metaclass
# ``__eq__`` must not merge distinct types); the stored type pins the ``id`` against recycling.
_final_path_cache: "dict[int, tuple[type, tuple[str, ...] | None]]" = {}


def _first_final_path(dc_type: type, visiting: "frozenset[int]") -> "tuple[str, ...] | None":
    """Field path from ``dc_type`` down to some ``Final`` leaf, or None. First hit wins; recursing eagerly validates
    every nested type. ``visiting`` holds ``id``s (not types) so a metaclass ``__eq__`` cannot fake "already visited".
    """
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


def _string_annotation_looks_final(dc_type: type, field_name: str, annotation: str) -> bool:
    """Best-effort: does the string annotation for ``field_name`` name ``typing.Final`` (possibly aliased)?

    Resolved per field, in the module + class that *define* it - not whole-class ``typing.get_type_hints``, which is
    all-or-nothing: one unresolvable sibling (e.g. a ``TYPE_CHECKING``-only import) would blind us to an aliased
    ``Final`` here. Tries the whole annotation, then (if only its inner type is unresolvable) the head symbol alone
    (``F[Bad]`` -> ``F`` -> ``Final``); falls back to a literal substring test if nothing resolves. ``eval`` carries
    the same trust as ``get_type_hints`` - both execute the defining module's own annotation expressions.
    """
    globalns: dict = {}
    localns: dict = {}
    for klass in dc_type.__mro__:
        if field_name in getattr(klass, "__annotations__", {}):
            module = sys.modules.get(klass.__module__)
            globalns = getattr(module, "__dict__", None) or {}
            localns = dict(vars(klass))
            break
    for expr in (annotation.strip(), annotation.split("[", 1)[0].strip()):
        try:
            resolved = eval(expr, globalns, localns)
        except Exception:
            continue
        return resolved is typing.Final or is_final_annotation(resolved)
    return "Final" in annotation


def _build_final_plan(dc_type: type) -> "frozenset[str]":
    """Validate every ``Final`` field on ``dc_type`` and return their names. Once per type (memoised)."""
    final_names = []
    for field in dataclasses.fields(dc_type):
        annotation = field.type
        if isinstance(annotation, str):
            # ``from __future__ import annotations`` stringizes ``field.type``; we cannot see ``Final`` through it, so
            # such a field would silently become a runtime arg. Detect a (possibly aliased) ``Final`` and reject.
            if _string_annotation_looks_final(dc_type, field.name, annotation):
                raise TypeError(
                    f"{dc_type.__name__}.{field.name}: annotation is the unresolved string {annotation!r}. Quadrants "
                    f"cannot see ``Final`` through a string annotation, so this field would silently become a runtime "
                    f"kernel argument. Remove ``from __future__ import annotations`` from the module defining "
                    f"{dc_type.__name__}, or annotate with the real type object."
                )
            continue
        if annotation is typing.Final:
            # Bare ``Final`` (no ``[T]``): ``is_final_annotation`` is False for it, so reject explicitly.
            raise TypeError(
                f"{dc_type.__name__}.{field.name}: bare ``typing.Final`` is not supported as a Quadrants "
                f"compile-time template field. Write ``Final[T]`` with a concrete type, e.g. "
                f"``{field.name}: Final[int]``."
            )
        if not is_final_annotation(annotation):
            # A ``Final``-like form that is not ``typing.Final`` would also silently become a runtime arg.
            origin = typing.get_origin(annotation)
            if origin is not None and "Final" in _describe_annotation(origin):
                raise TypeError(
                    f"{dc_type.__name__}.{field.name}: annotation {annotation!r} looks like a ``Final`` special form "
                    f"but is not ``typing.Final`` (got origin {origin!r}). Use ``typing.Final`` from the standard "
                    f"library."
                )
            continue
        _validate_final_inner_type(dc_type, field.name, annotation)
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

    # A mutable *ancestor* of a Final leaf is equally unsound (``outer.child = Inner(...)`` swaps a baked constant);
    # ``final_names`` covers only own fields, so reject every mutable dataclass on a path to a Final leaf.
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


# ``id(type) -> (type, frozenset of Final field names)``. Identity-keyed (a metaclass ``__eq__`` must not merge
# distinct types' schemas); the stored type is a strong ref pinning the ``id`` and verifying identity on lookup.
_final_plan_cache: "dict[int, tuple[type, frozenset[str]]]" = {}


def final_field_names(dc_type: Any) -> "frozenset[str]":
    """Cached set of ``Final`` field names on ``dc_type`` (validated on first sighting). Hot path: one ``dict.get`` +
    ``is``; callers short-circuit on the empty result. ``Any`` avoids a per-launch ``cast`` at the call site."""
    entry = _final_plan_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity guard against a recycled ``id``
        return entry[1]
    names = _build_final_plan(dc_type)
    _final_plan_cache[id(dc_type)] = (dc_type, names)
    return names


def subtree_has_final_fields(dc_type: Any) -> bool:
    """True if ``dc_type`` or any transitively nested dataclass has a ``Final`` field.

    Gates the per-instance launch caches (``_qd_spec_key``, ``_qd_dc_repr``): a ``Final`` value's class *behavior* can
    change after launch one (a plain ``enum`` later monkey-patched), so cache writers skip such dataclasses and the
    read falls through to a revalidating recompute. Transitive, since a cache hit also skips recursion into a leaf.
    """
    return _first_final_path(dc_type, frozenset()) is not None


# Pack/unpack a ``float`` as its exact IEEE-754 bits (see ``final_scalar_key``).
_pack_f64 = struct.Struct("<d").pack
_unpack_u64 = struct.Struct("<Q").unpack

# Type tags keeping the key spaces disjoint: Python treats distinct constants as ``==`` with equal hashes (a float's
# bits vs an int; an ``IntEnum`` member vs its bare scalar; ``True`` vs ``1`` vs ``np.int64(1)``), so bare values
# would collide.
_FLOAT_KEY_TAG = "f64"
_ENUM_KEY_TAG = "enum"
_SCALAR_KEY_TAG = "scalar"

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


def _slot_names(klass: type) -> tuple:
    """Slot names declared by ``klass`` via ``__slots__`` (a tuple of strings). Robust to a post-creation rebind of
    ``__slots__`` to a non-iterable (e.g. ``None``) or to non-string entries: those create no real slot descriptors, so
    they contribute no names here (the rebind itself is captured by the class-kind key)."""
    slots = getattr(klass, "__slots__", ())
    if isinstance(slots, str):
        return (slots,)
    if not hasattr(slots, "__iter__"):
        return ()
    return tuple(s for s in slots if isinstance(s, str))


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


# Standard per-member bookkeeping (not user state). A strict allowlist, not "skip all dunders", since a user could
# stash state under a dunder (``self.__unit__``). ``_inverted_`` is CPython's lazy inverted-Flag cache (>=3.11).
_ENUM_INTERNAL_MEMBER_ATTRS = frozenset(
    {"_name_", "_value_", "_sort_order_", "_inverted_", "__objclass__", "__dict__", "__weakref__"}
)


def _enum_member_state_attr(value: Any) -> "str | None":
    """Name of a user-defined per-member state attribute on ``value`` (in ``__dict__`` or a populated slot), or None."""
    d = getattr(value, "__dict__", None)
    if d:
        for k in d:
            if k not in _ENUM_INTERNAL_MEMBER_ATTRS:
                return k
    for klass in type(value).__mro__:
        for slot in _slot_names(klass):
            if slot in _ENUM_INTERNAL_MEMBER_ATTRS:
                continue
            if hasattr(value, slot):  # a declared-but-unset slot raises on access, so only *populated* slots count
                return slot
    return None


# Library enum bases; only user classes below them are inspected for observable behavior.
_FRAMEWORK_ENUM_CLASSES = frozenset(
    c
    for c in (getattr(enum, n, None) for n in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "ReprEnum"))
    if isinstance(c, type)
)

# Library enum metaclasses (``EnumMeta`` / >=3.11 ``EnumType``); only user subclasses below them are inspected.
_FRAMEWORK_ENUM_METACLASSES = frozenset(
    c for c in (type, object, getattr(enum, "EnumMeta", None), getattr(enum, "EnumType", None)) if isinstance(c, type)
)

# Dunders whose override is observable at compile time. One in an enum's own dict is a candidate override; on >=3.11
# the machinery copies the mix-in's ``__str__`` / ``__format__`` / ``__repr__`` in, so ``_dunder_copied_from_base``
# filters those out.
_OBSERVABLE_DUNDERS = frozenset(
    {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__bool__",
        "__int__",
        "__index__",
        "__float__",
        "__complex__",
        "__str__",
        "__repr__",
        "__format__",
        "__bytes__",
        "__len__",
        "__length_hint__",
        "__contains__",
        "__iter__",
        "__next__",
        "__reversed__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__getattr__",
        "__getattribute__",
        "__call__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__matmul__",
        "__rmatmul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__mod__",
        "__rmod__",
        "__divmod__",
        "__rdivmod__",
        "__pow__",
        "__rpow__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__and__",
        "__rand__",
        "__or__",
        "__ror__",
        "__xor__",
        "__rxor__",
        "__lshift__",
        "__rlshift__",
        "__rshift__",
        "__rrshift__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
    }
)


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


def _identity_component(obj: Any, live: bool):
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


def _reject_structural_subclass(val: Any):
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


def _canonical_attr_value(val: Any, live: bool):
    """A deterministic, hashable, *lossless* snapshot of a class-dict value read at compile time
    (``cfg.x.__class__.<attr>``). Ints/str/bytes/None by value (type-tagged, so ``True``/``1`` differ); floats/complex
    by exact IEEE bits (``0.0``/``-0.0`` never collapse); tuples/lists/dicts recursively and sets order-independently,
    each keeping its container type (``()`` vs ``[]``
    compare unequal); an auto-generated slot descriptor reduces to an address-free ``(type, name)`` token (the owning
    class carries identity); a user-bound class/callable also carries an identity component so distinct same-named
    ``<locals>`` objects never collide. Anything else is rejected rather than collapsed to a lossy token that could
    reuse a stale specialization after the value changes."""
    if val is None:
        return None
    # A scalar *subclass* (or enum member) may carry per-instance state or a distinct class identity that the bare
    # value discards, so route it through the full baked-scalar machinery (stateful-subclass rejection + subclass
    # identity + IEEE bits). Exact builtins below skip that and encode directly.
    if (isinstance(val, (bool, int, float, str)) and type(val) not in (bool, int, float, str)) or isinstance(
        val, enum.Enum
    ):
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


def _attr_kind(klass: type, name: str, live: bool):
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


def _dunder_copied_from_base(klass: type, name: str, value: Any) -> bool:
    """True if ``klass.__dict__[name]`` is the *same object* as a strict base's: on >=3.11 the machinery copies the
    mix-in's ``__str__`` / ``__format__`` / ``__repr__`` in verbatim, whereas a real override is a fresh object.
    """
    for base in klass.__mro__[1:]:
        if base.__dict__.get(name, _MISSING) is value:
            return True
    return False


def _enum_probe_classes() -> "list[type]":
    """One fresh enum of each framework kind plus a *direct* ``int`` / ``str`` mix-in, to learn on the running Python
    which sunder/dunder names the machinery injects and which track the bases/mix-in. The direct-mix-in probes matter:
    a hook like ``_value_repr_`` holds the mix-in's value, so framework subclasses alone would misjudge a user
    ``class M(int, enum.Enum)``."""

    class _E(enum.Enum):
        A = 1
        B = 2

    class _IE(enum.IntEnum):
        A = 1
        B = 2

    class _IF(enum.IntFlag):
        A = 1
        B = 2

    class _DMI(int, enum.Enum):  # a hand-written ``IntEnum`` (direct ``int`` mix-in)
        A = 1
        B = 2

    probes: list = [_E, _IE, _IF, _DMI]
    _flag_base = getattr(enum, "Flag", None)
    if _flag_base is not None:

        class _F(_flag_base):
            A = 1
            B = 2

        probes.append(_F)
    _str_enum_base = getattr(enum, "StrEnum", None)  # 3.11+; ``getattr`` (not ``enum.StrEnum``) keeps pylint happy
    if _str_enum_base is not None:

        class _SE(_str_enum_base):
            A = "a"
            B = "b"

        probes.append(_SE)

        class _DMS(str, enum.Enum):  # a hand-written ``StrEnum`` (direct ``str`` mix-in)
            A = "a"
            B = "b"

        probes.append(_DMS)
    return probes


def _dict_entry_is_inherited_default(klass: type, name: str, member: Any) -> bool:
    """True if ``klass.__dict__[name]`` is the inherited default (machinery-copied), not user-authored. Compares
    *unwrapped* callable identity: 3.12+ copies ``_generate_next_value_`` as a fresh wrapper, so a wrapper-identity
    test would flag every plain enum."""
    target = getattr(member, "__func__", member)
    for base in klass.__mro__[1:]:
        base_member = vars(base).get(name, _MISSING)
        if base_member is not _MISSING:
            return getattr(base_member, "__func__", base_member) is target
    return False


def _enum_machinery_class_dict(cls: type) -> "dict[str, Any] | None":
    """The own ``__dict__`` the machinery produces for a *member-free* class with ``cls``'s exact bases/metaclass:
    every injected hook at its value for this mix-in shape, no members. ``_observable_class_dict_attr`` compares
    against it to tell a user override from bookkeeping, even for mix-in-dependent hooks a nearest-base check cannot
    judge. Built via the metaclass's ``__prepare__`` (``EnumMeta`` rejects a plain ``dict``); None if unbuildable."""
    mcls = type(cls)
    try:
        # ``__prepare__`` returns the metaclass's own namespace (``_EnumDict``), which must be passed through verbatim.
        # Pyright types it as ``MutableMapping``, hence the local ignore.
        namespace = mcls.__prepare__("_qd_enum_probe", cls.__bases__)
        reference = mcls("_qd_enum_probe", cls.__bases__, namespace)  # pyright: ignore[reportArgumentType]
    except Exception:  # pylint: disable=broad-except  # any build failure -> conservative allowlist fallback
        return None
    return dict(vars(reference))


def _compute_enum_generated_class_attrs() -> "frozenset[str]":
    """Union across enum kinds of sunder/dunder names the machinery generates, learned by probing the running Python.
    Used only as the fallback allowlist when the per-class reference cannot be built."""

    class _Plain:
        pass

    def _generated(probe: type) -> "set[str]":
        # Keep enum-valued machinery sunders (``_boundary_``); exclude members/aliases (enum-valued, plain name).
        return {
            n for n, v in vars(probe).items() if not isinstance(v, enum.Enum) or (n.startswith("_") and n.endswith("_"))
        }

    names: "set[str]" = _generated(_Plain)
    for probe in _enum_probe_classes():
        names.update(_generated(probe))
    return frozenset(names)


def _compute_enum_reference_checked_attrs() -> "frozenset[str]":
    """Machinery sunder/dunder names whose value is fixed by the bases/mix-in (``_value_repr_`` / ``__new__`` /
    ``_generate_next_value_`` / copied operator dunders / ``_boundary_`` / ...): a clean enum's entry equals a
    member-free rebuild, so ``_observable_class_dict_attr`` can flag an override by comparison. Member-*derived*
    bookkeeping differs even for a clean enum (a distinct container) and is excluded; a name qualifies only if it
    matches in every probe carrying it."""
    appears: "dict[str, int]" = {}
    matches: "dict[str, int]" = {}
    for probe in _enum_probe_classes():
        reference = _enum_machinery_class_dict(probe)
        if reference is None:
            continue
        for name, member in vars(probe).items():
            if not (name.startswith("_") and name.endswith("_")):  # members/aliases are non-sunder -> skipped here
                continue
            appears[name] = appears.get(name, 0) + 1
            default = reference.get(name, _MISSING)
            if default is not _MISSING and getattr(member, "__func__", member) is getattr(default, "__func__", default):
                matches[name] = matches.get(name, 0) + 1
    return frozenset(name for name, count in appears.items() if matches.get(name, 0) == count) - _STRUCTURAL_CLASS_ATTRS


# Fallback allowlist (see ``_compute_enum_generated_class_attrs``); the primary path uses the per-class reference.
_ENUM_GENERATED_CLASS_ATTRS = _compute_enum_generated_class_attrs()

# Names fixed by the bases/mix-in, so an override is caught by comparing to a member-free reference.
_ENUM_REFERENCE_CHECKED_ATTRS = _compute_enum_reference_checked_attrs()

# Fallback allowlist used only when the member-free reference cannot be built: exempt only when the stored object *is*
# the inherited default (``_dict_entry_is_inherited_default``); the reference path also catches mix-in-dependent hooks.
_ENUM_OVERRIDABLE_HOOK_NAMES = frozenset({"_generate_next_value_"})


def _generated_attr_is_user_override(klass: type, name: str, member: Any, reference: "dict[str, Any] | None") -> bool:
    """True if ``klass.__dict__[name]`` is a *user* override of a reference-checked hook (unwrapped-callable identity
    vs the member-free ``reference``). Absent from the reference => not an override; ``reference is None`` =>
    inherited-default allowlist fallback."""
    if reference is None:
        return name in _ENUM_OVERRIDABLE_HOOK_NAMES and not _dict_entry_is_inherited_default(klass, name, member)
    default = reference.get(name, _MISSING)
    if default is _MISSING:
        return False
    return getattr(member, "__func__", member) is not getattr(default, "__func__", default)


def _is_official_enum_member(klass: type, name: str, member: Any) -> bool:
    """True if ``klass.__dict__[name]`` is a real member/alias of ``klass`` (its own name in ``_member_map_``), not a
    user-added enum-valued attribute (``Mode.X = Mode.A``) that is observable but absent from the key."""
    member_map = klass.__dict__.get("_member_map_")
    return isinstance(member_map, dict) and member_map.get(name, _MISSING) is member


def _observable_class_dict_attr(klass: type) -> "str | None":
    """Name of a user-authored observable attribute in ``klass.__dict__`` (method/property/class var, a real
    operator-dunder override, a user sunder/dunder hook or override of a machinery hook, or a user-added enum-valued
    attribute), or None. Whether a sunder/dunder is machinery bookkeeping is judged against *this* class's member-free
    reference (not the global union across kinds), so a name generated only for another kind - ``_boundary_`` on
    ``Flag`` - cannot exempt a user attribute of that name on an unrelated enum.
    """
    reference: Any = _MISSING  # member-free machinery reference; built lazily, at most once per call
    for name, member in vars(klass).items():
        is_sunder_dunder = name.startswith("_") and name.endswith("_")
        if isinstance(member, enum.Enum) and not is_sunder_dunder:
            if _is_official_enum_member(klass, name, member):
                continue  # a machinery member/alias - keyed by member identity
            return name  # a user-added enum-valued attribute (``Mode.X = Mode.A``), absent from the key
        if name in _OBSERVABLE_DUNDERS:
            if _dunder_copied_from_base(klass, name, member):
                continue  # base/data-type dunder copied by the machinery (>=3.11)
            return name  # a genuine user operator/behavior dunder override
        if is_sunder_dunder:
            if name in _STRUCTURAL_CLASS_ATTRS:
                continue  # structural names (some absent from the programmatic reference); always exempt
            if reference is _MISSING:
                reference = _enum_machinery_class_dict(klass)
            if reference is not None:
                if name not in reference:
                    return name  # a user-added sunder/dunder the machinery does not generate for this shape
                # A bases/mix-in-fixed hook (incl. ``_boundary_``) is a user override iff it no longer holds the
                # machinery's value.
                if name in _ENUM_REFERENCE_CHECKED_ATTRS and _generated_attr_is_user_override(
                    klass, name, member, reference
                ):
                    return name
                # Otherwise untouched machinery bookkeeping for this shape: member-derived data or a matched hook,
                # exempt. Member-derived data is a function of the member map, already folded into the keys via
                # ``_enum_member_map_key``. Directly reassigning these private enum internals (``Mode._member_names_ =
                # [...]``) is unsupported - like the ``_qd_`` names reserved for Quadrants - so we do not guard it.
                continue
            # Reference unbuildable: best-effort fallback to the global-union allowlist.
            if name in _ENUM_GENERATED_CLASS_ATTRS:
                if _generated_attr_is_user_override(klass, name, member, None):
                    return name
                continue
            return name  # a user sunder/dunder hook (``_missing_`` / ``_repr_html_`` / ...)
        return name  # a user method / property / class var
    return None


def _enum_class_behavior_attr(enum_cls: type) -> "str | None":
    """Name of a user-defined class-level attribute on ``enum_cls``, one of its user bases (*including a non-enum
    mixin*), or its metaclass, or None. Two same-named factory enums whose ``label`` / ``__eq__`` differ key
    identically otherwise, yet ``qd.static(cfg.mode.label == ...)`` differs; the metaclass is walked too
    (``cfg.mode.__class__.label``).
    """
    for klass in enum_cls.__mro__:
        # Framework internals: the mixed-in primitive, ``object`` / a NumPy base, and library enum bases. Every user
        # base is inspected, enum or not: a non-enum mixin can add observable behavior too.
        if _is_baked_base_type(klass) or klass in _FRAMEWORK_ENUM_CLASSES:
            continue
        attr = _observable_class_dict_attr(klass)
        if attr is not None:
            return attr
    enum_metacls: type = type(enum_cls)  # ``: type`` keeps ``__mro__`` an instance read
    for mcls in enum_metacls.__mro__:
        if mcls in _FRAMEWORK_ENUM_METACLASSES:
            continue
        attr = _observable_metaclass_attr(mcls)
        if attr is not None:
            return attr
    return None


def _reject_stateful_enum_member(value: Any) -> None:
    """Reject an ``enum`` member with observable state the key cannot capture: per-member state, or class-level
    behavior two factory enums could define differently. Plain enums (and unnamed ``IntFlag`` composites) pass.
    """
    cls = type(value)
    # Check *every* member, not just the selected one: a sibling's per-member state (``Mode.B.unit``) is observable
    # (``cfg.mode.__class__.B.unit``) yet the member-map key records only each member's value, so any member carrying
    # state must be rejected. ``value`` is appended in case it is an ``IntFlag`` composite (absent from the map).
    member_map = getattr(cls, "_member_map_", None)
    members = list(member_map.values()) if isinstance(member_map, dict) else []
    members.append(value)
    for member in members:
        extra = _enum_member_state_attr(member)
        if extra is not None:
            member_name = getattr(member, "name", None) or "<value>"
            raise TypeError(
                f"A ``Final`` field received a member of {cls.__module__}.{cls.__qualname__}: ``{member_name}`` "
                f"has user-defined per-member state (e.g. attribute {extra!r}). A ``Final`` value is baked as a "
                f"compile-time literal keyed by member identity, so per-member state a kernel could read (e.g. "
                f"``cfg.mode.unit``) would not select a distinct specialization. Use a plain ``enum`` (state-free "
                f"members), or bake the needed value as a separate ``Final`` field."
            )
    behavior = _enum_class_behavior_attr(cls)
    if behavior is not None:
        raise TypeError(
            f"A ``Final`` field received a member of {cls.__module__}.{cls.__qualname__}, an ``enum`` class that "
            f"defines observable class-level behavior (e.g. attribute {behavior!r}). ``module``/``qualname`` does "
            f"not uniquely identify a dynamically created class, so two same-named enum classes (e.g. from a "
            f"factory) whose ``{behavior}`` a kernel could read (``cfg.mode.{behavior}``) would select the same "
            f"specialization. Use a plain ``enum`` (no user methods/properties/class vars), or bake the needed "
            f"value as a separate ``Final`` field."
        )


def _enum_member_map_key(cls: type, live: bool) -> tuple:
    """Hashable digest of ``cls``'s entire member map (each name + its value via ``final_scalar_key``). A baked
    ``Final`` member keys on this, not only the selected member, because a sibling is observable
    (``cfg.mode.__class__.OTHER.value``). ``()`` if there is no member map."""
    member_map = getattr(cls, "_member_map_", None)
    if not isinstance(member_map, dict):
        return ()
    return tuple((name, final_scalar_key(member.value, live)) for name, member in member_map.items())


def final_scalar_key(value: Any, live: bool = False) -> Any:
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
