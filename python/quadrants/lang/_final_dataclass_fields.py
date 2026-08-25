"""Helpers for the ``@dataclasses.dataclass`` kernel-arg path, including ``typing.Final[T]`` compile-time fields.

Perf: all ``Final`` resolution/validation runs once per dataclass type and is cached (``_final_plan_cache``,
``_final_path_cache``). The per-launch hot path does a single ``dict.get`` on the type and, in the common no-Final
case, takes the pre-existing path with no per-launch reflection (an ``isinstance`` is a ~100-200ns MRO walk vs a
~10ns ``type(x) is Y`` pointer compare).
"""

import dataclasses
import enum
import itertools
import os
import struct
import sys
import typing
import uuid
import weakref
from typing import Any

import numpy as np

_MISSING = object()  # sentinel for "attribute not found" while resolving a qualname (``None`` is a legal value)

# Random per-process token folded into the *offline* key of a non-resolvable (locally/dynamically created) class (see
# ``_subclass_identity``). Its companion serial (``_dynamic_class_serial``) is only process-local, so without the
# nonce two workers building same-named dynamic classes would serialize identical offline keys and one could load the
# other's kernel; the nonce makes a dynamic class a guaranteed cross-process miss, never a wrong reuse. Must be
# reseeded in a ``fork``ed child (which inherits the string *and* the serial counter); ``spawn`` re-imports and
# reseeds anyway.
_PROCESS_NONCE = uuid.uuid4().hex


def _reseed_process_nonce() -> None:
    global _PROCESS_NONCE
    _PROCESS_NONCE = uuid.uuid4().hex


if hasattr(os, "register_at_fork"):  # POSIX only; ``spawn``/Windows re-import and reseed on their own
    os.register_at_fork(after_in_child=_reseed_process_nonce)

# ``T`` values permitted inside ``Final[T]``: baked into the kernel as a literal and folded into both key spaces, so
# ``T`` must be meaningful as a compile-time literal and hash/``repr`` by value stably across processes. Membership is
# by exact type. ``enum.Enum`` subclasses are also permitted (resolved separately): Genesis stores ``IntEnum`` members
# in ``int``-annotated fields, and an enum member is a valid, stably-repr'able literal.
_FINAL_SCALAR_TYPES = frozenset({bool, int, float, str})

# Types worth a tailored error, because ``Final[T]`` on them is a plausible mistake with a clear better alternative.
# Resolved lazily to dodge import cycles (``_ndarray`` imports back into ``lang``).
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
    """Return True if ``annotation`` is a ``typing.Final[T]`` special form.

    On a frozen dataclass kernel arg this marks the field's value as a compile-time constant: baked into the kernel
    (``qd.static(config.field)`` is legal), folded into the spec + fastcache keys (distinct values => distinct
    kernels), and not declared as a runtime scalar arg.

    Bare ``typing.Final`` returns False (``typing.get_origin`` is ``None``) but is still not an ordinary field:
    ``_build_final_plan`` rejects it outright. ``typing_extensions.Final`` is the same object on all supported Pythons
    (>=3.10) so it is accepted transparently.
    """
    return typing.get_origin(annotation) is typing.Final


def _describe_annotation(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or repr(annotation)


def _reject_hint_for(inner: Any) -> str | None:
    """Return a tailored remediation hint when ``Final[inner]`` names a type we want to reject well, else None.

    Matched on the *type name* (not by importing the classes) to avoid import cycles; runs once per dataclass type.
    """
    for name in (type(inner).__name__, _describe_annotation(inner)):
        hint = _FINAL_REJECT_HINTS.get(name)
        if hint is not None:
            return hint
    return None


def _validate_final_inner_type(dc_type: type, field_name: str, annotation: Any) -> None:
    """Raise a clear error unless ``Final[annotation]`` names a type we can bake as a compile-time literal.

    Only called after ``is_final_annotation`` confirmed a subscripted ``Final``, so ``typing.get_args`` is non-empty.
    """
    inner = typing.get_args(annotation)
    if len(inner) != 1:
        raise TypeError(
            f"{dc_type.__name__}.{field_name}: ``typing.Final`` takes exactly one type argument, got "
            f"``Final[{', '.join(_describe_annotation(a) for a in inner)}]``."
        )
    inner_type = inner[0]

    if inner_type in _FINAL_SCALAR_TYPES:
        return
    # ``issubclass`` is fine here: once per dataclass type, never per launch.
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
    """Return True if ``dc_type`` prevents (or has explicitly disclaimed) rebinding of its fields.

    True for ``frozen=True`` (prevents it outright) and ``unsafe_hash=True`` (an explicit value-stability assertion the
    machinery already honours). Deliberately *not* the ``__hash__ is not None`` proxy: ``@dataclass(eq=False)`` inherits
    ``object.__hash__`` and would read a plain mutable class as frozen, letting a baked ``Final`` constant be
    reassigned with no recompilation - exactly what this check prevents.
    """
    params = getattr(dc_type, "__dataclass_params__", None)
    if params is None:
        return dc_type.__hash__ is not None
    return params.frozen or params.unsafe_hash


# Memo of ``id(dataclass type) -> (type, field path down to some Final leaf)`` (path ``None`` if none), used to reject
# mutable ancestors. Identity-keyed with a strong-ref+``is`` guard like ``_final_plan_cache`` (metaclass ``__eq__``
# must not merge distinct types). Once per type; never per launch.
_final_path_cache: "dict[int, tuple[type, tuple[str, ...] | None]]" = {}


def _first_final_path(dc_type: type, visiting: "frozenset[int]") -> "tuple[str, ...] | None":
    """Return the field path from ``dc_type`` down to some ``Final`` leaf, or None if the subtree contains none.

    First hit wins (only a witness is needed to name one offending path). Recursing eagerly validates every nested
    type at the top-level call. ``visiting`` holds ``id(type)`` (not the types) to guard a self-referential graph
    without a metaclass ``__eq__`` making distinct types look "already visited"; the walk keeps each on-path type
    alive so its ``id`` is stable.
    """
    entry = _final_path_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity check: guard against a recycled ``id``
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


def _resolved_type_hints(dc_type: Any) -> "dict[str, Any] | None":
    """``typing.get_type_hints(dc_type)`` (real objects, ``Final`` preserved), or None if unresolvable. Lets
    ``_build_final_plan`` detect an *aliased* ``Final`` behind a string annotation (``Final as F; x: F[int]`` stores
    ``"F[int]"``, which a substring test for ``Final`` misses). Off the hot path; on any failure returns None so the
    caller falls back to its substring check."""
    try:
        return typing.get_type_hints(dc_type)
    except Exception:
        return None


def _build_final_plan(dc_type: type) -> "frozenset[str]":
    """Validate every ``Final`` field on ``dc_type`` and return the set of Final-annotated field names.

    Called once per dataclass type (memoised in ``_final_plan_cache``), so all reflection here stays off the hot path.
    """
    resolved_hints: "dict[str, Any] | None" = None
    resolved_computed = False
    final_names = []
    for field in dataclasses.fields(dc_type):
        annotation = field.type
        if isinstance(annotation, str):
            # ``from __future__ import annotations`` leaves ``field.type`` a string. The kernel-arg path assumes
            # resolved types, so rather than half-support it, flag the correctness trap: a field the user believes is a
            # compile-time constant that we would lower as a runtime arg. Resolve hints (lazily) to catch an *aliased*
            # ``Final`` (``Final as F`` -> ``"F[int]"``), falling back to a substring test if resolution fails.
            if not resolved_computed:
                resolved_hints = _resolved_type_hints(dc_type)
                resolved_computed = True
            resolved = resolved_hints.get(field.name) if resolved_hints is not None else None
            looks_final = is_final_annotation(resolved) if resolved is not None else ("Final" in annotation)
            if looks_final:
                raise TypeError(
                    f"{dc_type.__name__}.{field.name}: annotation is the unresolved string {annotation!r}. Quadrants "
                    f"cannot see ``Final`` through a string annotation, so this field would silently become a runtime "
                    f"kernel argument. Remove ``from __future__ import annotations`` from the module defining "
                    f"{dc_type.__name__}, or annotate with the real type object."
                )
            continue
        if annotation is typing.Final:
            # Bare ``Final`` with no ``[T]``: ``is_final_annotation`` returns False for it, so it would otherwise be
            # lowered as a runtime arg and die later in ``cook_dtype``. Reject the unsupported spelling with a clear
            # message.
            raise TypeError(
                f"{dc_type.__name__}.{field.name}: bare ``typing.Final`` is not supported as a Quadrants "
                f"compile-time template field. Write ``Final[T]`` with a concrete type, e.g. "
                f"``{field.name}: Final[int]``."
            )
        if not is_final_annotation(annotation):
            # Catch a ``Final``-like special form that is not ``typing.Final`` (e.g. a future ``typing_extensions`` that
            # stops aliasing the stdlib object); silently lowering it to a runtime arg is the same correctness trap.
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
        # A mutable carrier contradicts ``Final``: ``TemplateMapper.lookup`` memoises on ``id(arg)``, so relaunching
        # the same instance after a reassignment would silently reuse the kernel baked with the old value.
        raise TypeError(
            f"{dc_type.__name__} has ``Final`` field(s) {sorted(final_names)} but is not frozen. A ``Final`` field's "
            f"value is baked into the compiled kernel, so it must not be reassignable - reassigning it would silently "
            f"keep using the kernel compiled for the old value. Declare the class as "
            f"``@dataclasses.dataclass(frozen=True)`` (or ``unsafe_hash=True`` if you must keep it mutable and accept "
            f"responsibility for never reassigning these fields)."
        )

    # A mutable *ancestor* of a Final leaf is just as unsound (a mutable ``Outer`` holding a frozen ``Inner`` with a
    # ``Final`` field still lets ``outer.child = Inner(...)`` swap a baked constant), and ``final_names`` covers only
    # this class's own fields. Reject every mutable dataclass on a path down to a Final leaf.
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


# Memo of ``id(dataclass type) -> (type, frozenset of Final field names)``. Keyed on ``id`` (not ``type``), so a
# metaclass ``__eq__``/``__hash__`` making two distinct types compare equal cannot merge their (possibly different)
# Final schemas. The stored ``type`` is a strong ref: it pins the type (so its ``id`` cannot be recycled) and lets a
# lookup verify identity (``entry[0] is dc_type``) before trusting the plan.
_final_plan_cache: "dict[int, tuple[type, frozenset[str]]]" = {}


def final_field_names(dc_type: Any) -> "frozenset[str]":
    """Return the cached set of ``Final``-annotated field names on ``dc_type``, validating on first sighting.

    Hot-path contract: one ``dict.get`` + one ``is`` check; callers short-circuit on the empty result so Final-free
    dataclasses run the pre-existing path untouched. Typed ``Any`` because ``_extract_arg`` calls it with its
    loosely-typed ``annotation`` (already known to be a dataclass type) and narrowing would need a per-launch ``cast``.
    """
    entry = _final_plan_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity check: guard against a recycled ``id``
        return entry[1]
    names = _build_final_plan(dc_type)
    _final_plan_cache[id(dc_type)] = (dc_type, names)
    return names


def subtree_has_final_fields(dc_type: Any) -> bool:
    """True if ``dc_type`` - or any dataclass nested transitively beneath it - declares a ``Final`` field.

    Gates the per-instance launch caches (``_qd_spec_key``, ``_qd_dc_repr``): serving them verbatim is sound only when
    the baked key cannot change after launch one, but a ``Final`` value's *class behavior* can (a plain ``enum``
    accepted first can have ``Mode.__eq__`` monkey-patched before launch two, observable via ``qd.static(cfg.mode ==
    1)`` yet invisible to the fixed key). So the cache *writers* never store a cache for such a dataclass; the read
    then misses and falls through to a revalidating recompute, leaving Final-free dataclasses untouched.

    Transitive because an early cache hit also skips the recursion into a nested Final leaf. Answered by
    ``_first_final_path`` (memoised), so after first sighting this is one ``dict.get`` + ``is`` check.
    """
    return _first_final_path(dc_type, frozenset()) is not None


# Precomputed packers for encoding a ``float`` by its exact IEEE-754 bits (see ``final_scalar_key``).
_pack_f64 = struct.Struct("<d").pack
_unpack_u64 = struct.Struct("<Q").unpack

# Type tags that keep the three key spaces disjoint. Python treats several distinct constants as ``==`` with equal
# hashes, so bare values would collide: a float's bit-encoding vs a bare int (``_FLOAT_KEY_TAG``); an ``IntEnum`` /
# ``StrEnum`` member vs its bare scalar or a same-valued member of another enum (``_ENUM_KEY_TAG``); ``True`` vs ``1``
# vs ``np.int64(1)`` (``_SCALAR_KEY_TAG``, which also tags the exact type). Any short, process-stable marker works.
_FLOAT_KEY_TAG = "f64"
_ENUM_KEY_TAG = "enum"
_SCALAR_KEY_TAG = "scalar"

# Exact baked primitive types: an instance of exactly one is a pure literal, so it skips the stateful-subclass check.
_EXACT_BAKED_TYPES = (bool, int, float, str)


def _is_exact_baked_type(cls: type) -> bool:
    """``cls`` *is* one of ``_EXACT_BAKED_TYPES`` - identity, never equality. A subclass whose metaclass makes it
    compare ``==`` to a builtin must not be mistaken for an exact builtin: equality-based ``in`` would skip its
    validation *and* canonicalize it, collapsing two distinct same-qualified factory classes even though
    ``cfg.x.__class__ is First`` is observable. ``is`` cannot be spoofed."""
    return any(cls is t for t in _EXACT_BAKED_TYPES)


# ``Py_TPFLAGS_HEAPTYPE`` (``1 << 9``): set by the interpreter on every ``class``/``type(...)`` type, clear on
# C-defined *static* types (builtin primitives, NumPy's own scalars). Not user-settable, so it tells a library scalar
# base from a user subclass even one spoofing ``__module__ = "numpy"`` (see ``_is_baked_base_type``).
_HEAPTYPE_FLAG = 1 << 9

# Class-dict names CPython auto-generates for a bare ``class X(base): pass`` subclass. A primitive subclass may carry
# *only* these; anything else (method, property, class var, overridden dunder) is observable behavior we cannot key on
# (see ``_reject_stateful_primitive_subclass``). ``__firstlineno__`` / ``__static_attributes__`` are 3.13+.
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

# Framework metaclass layers on a primitive subclass's *metaclass* MRO (a plain subclass's metaclass is exactly
# ``type``); only user-authored ``type`` subclasses below it are inspected for observable behavior.
_FRAMEWORK_PRIMITIVE_METACLASSES = frozenset({type, object})


def _observable_metaclass_attr(mcls: type) -> "str | None":
    """Name of a user-authored observable attribute in a metaclass layer's ``__dict__`` (readable as
    ``cfg.x.__class__.<attr>``), or None; only ``_STRUCTURAL_CLASS_ATTRS`` are exempt. Shared by the enum and
    primitive-subclass metaclass walks.

    Even ``__eq__`` / ``__ne__`` / ``__hash__`` are rejected: the identity key (``_ClassRef``) prevents them from
    collapsing classes or unhashing the key, but a kernel can still *observe* them (``qd.static(cfg.x.__class__ ==
    Expected)``) and mutating their state after launch one would reuse a stale specialization under the fixed key.
    """
    for name in vars(mcls):
        if name not in _STRUCTURAL_CLASS_ATTRS:
            return name
    return None


def _is_baked_base_type(klass: type) -> bool:
    """True for a library base a baked value can subclass without adding observable state: a builtin primitive,
    ``object``, or a *static* ``np.generic`` subclass (a NumPy scalar). Their attributes are framework internals, so
    the walk in ``_reject_stateful_primitive_subclass`` stops here.

    A NumPy scalar base is recognised by ``issubclass(klass, np.generic)`` plus the absence of ``Py_TPFLAGS_HEAPTYPE``
    (a static C type), never by the mutable ``__module__`` string - so a user ``class Foo(np.float64)`` (always a heap
    type, even spoofing ``__module__ = "numpy"``) cannot masquerade as a trusted base.
    """
    if _is_exact_baked_type(klass) or klass is object:
        return True
    # A genuine NumPy scalar base is a static ``np.generic`` subclass; a user subclass is a heap type.
    is_heap_type = bool(klass.__flags__ & _HEAPTYPE_FLAG)
    return issubclass(klass, np.generic) and not is_heap_type


def _reject_stateful_primitive_subclass(value: Any) -> None:
    """Reject a *subclass* of a baked scalar (``bool``/``int``/``float``/``str`` or a NumPy scalar) whose instance
    carries observable state a kernel could read but the key (typed value + ``module``/``qualname``) cannot capture:

    - *Per-instance* state (``class TaggedFloat(float)`` with a ``unit``, in ``__dict__`` or a populated slot): two
      instances equal in value but not state would bake different kernels yet select one specialization.
    - *Class-level* behavior/state (a factory's ``float`` subclasses whose ``unit`` property or overridden
      ``__eq__``/``__int__``/``__repr__`` differ): ``module``/``qualname`` does not uniquely identify a dynamic class,
      so two same-named subclasses collide. Any class member outside ``_STRUCTURAL_CLASS_ATTRS`` (including an
      overridden dunder) is rejected, and the *metaclass* is inspected the same way (``cfg.x.__class__.label``
      resolves to ``type(cls).label``), skipping the framework ``type``/``object`` layers.

    No bounded, process-stable way to serialise arbitrary state exists, so reject rather than mis-specialise. Exact
    primitives, exact NumPy scalars, and behavior-free subclasses (``class Meters(float): pass``) are unaffected.
    """
    cls = type(value)
    if _is_exact_baked_type(cls):
        return
    if not (isinstance(value, (int, float, str)) or isinstance(value, np.generic)):
        return  # not a baked-primitive / NumPy-scalar value at all
    if isinstance(value, np.generic) and _is_baked_base_type(cls):
        return  # an exact NumPy library scalar (a static ``np.generic`` type): a pure value, no user state/behavior
    if getattr(value, "__dict__", None):
        stateful = True
    else:
        stateful = False
        for klass in cls.__mro__:
            slots = getattr(klass, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            if any(slot not in ("__dict__", "__weakref__") and hasattr(value, slot) for slot in slots):
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
    for klass in cls.__mro__:  # subclass chain above the base type; stop at the library base (its attrs are fine)
        if _is_baked_base_type(klass):
            break
        for attr in vars(klass):
            # Only the auto-generated structural members are exempt; a method/property/class var or an overridden
            # dunder counts.
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
    # ``cfg.x.__class__.label`` resolves to ``type(cls).label``, invisible to the subclass MRO walk, so the metaclass
    # must be inspected too (as ``_enum_class_behavior_attr`` does). ``: type`` keeps ``__mro__`` an instance read.
    metacls: type = type(cls)  # user-authored layers sit below ``type`` (``class Unit(int, metaclass=M)``)
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


# Allowlist of ``enum`` member attribute names that are standard bookkeeping, not per-member state. A strict allowlist
# (not "skip all dunders") on purpose: a user can stash state under a dunder name (``self.__unit__``). ``_name_`` /
# ``_value_`` / ``_sort_order_`` / ``__objclass__`` are member fields; ``_inverted_`` is the value-derived cache
# CPython lazily stores on an inverted ``Flag``/``IntFlag`` member (>=3.11); ``__dict__`` / ``__weakref__`` are slots.
_ENUM_INTERNAL_MEMBER_ATTRS = frozenset(
    {"_name_", "_value_", "_sort_order_", "_inverted_", "__objclass__", "__dict__", "__weakref__"}
)


def _enum_member_state_attr(value: Any) -> "str | None":
    """Name of a user-defined per-member state attribute on ``value``, or None if it carries only enum bookkeeping
    (see ``_ENUM_INTERNAL_MEMBER_ATTRS``). State can live in ``__dict__`` or a populated slot; both are inspected. Any
    name off the allowlist (including a user dunder like ``__unit__``) counts as observable state.
    """
    d = getattr(value, "__dict__", None)
    if d:
        for k in d:
            if k not in _ENUM_INTERNAL_MEMBER_ATTRS:
                return k
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in _ENUM_INTERNAL_MEMBER_ATTRS:
                continue
            if hasattr(value, slot):  # a declared-but-unset slot raises on access, so only *populated* slots count
                return slot
    return None


# Standard-library enum base classes. A user enum's MRO contains these too; only the user-authored classes below them
# are inspected for observable class-level behavior.
_FRAMEWORK_ENUM_CLASSES = frozenset(
    c
    for c in (getattr(enum, n, None) for n in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "ReprEnum"))
    if isinstance(c, type)
)

# Framework metaclass layers on an enum's *metaclass* MRO (a plain enum's metaclass is exactly ``EnumMeta`` / >=3.11
# ``EnumType``); only user-authored ``EnumMeta`` subclasses below them are inspected (``_enum_class_behavior_attr``).
_FRAMEWORK_ENUM_METACLASSES = frozenset(
    c for c in (type, object, getattr(enum, "EnumMeta", None), getattr(enum, "EnumType", None)) if isinstance(c, type)
)

# Dunders that make behavior observable at compile time (comparison/hashing/conversion/container/callable/arithmetic).
# One in a user enum's own dict is a *candidate* override two factory enums could define differently (``cfg.mode ==
# 1``). NOTE: on >=3.11 the machinery copies the mix-in's ``__str__`` / ``__format__`` / ``__repr__`` onto the
# subclass dict, so ``_dunder_copied_from_base`` filters those out and only a genuine fresh override is rejected.
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


def _class_not_uniquely_identified(cls: type) -> bool:
    """True if ``cls`` cannot be recovered from its ``module``/``qualname`` - i.e. locally/dynamically created
    (qualname contains ``<locals>``, or ``module.qualname`` does not resolve back to it). Such a class shares its
    identity string with every other class built the same way, yet distinct class objects are observably distinct
    (``cfg.x.__class__ is First``), so callers add a class-identity token to the key (a ``_ClassRef`` in-process, a
    ``_dynamic_class_serial`` offline).
    """
    module = sys.modules.get(cls.__module__)
    if module is None:
        return True
    obj: Any = module
    for part in cls.__qualname__.split("."):
        if part == "<locals>":
            return True
        obj = getattr(obj, part, _MISSING)
        if obj is _MISSING:
            return True
    return obj is not cls


class _ClassRef:
    """A strong reference to a class that hashes and compares by *object identity*, never via the metaclass. Embedded
    in the in-process (``live``) spec key:

    - It pins ``cls`` while the key lives, so the class cannot be collected and its ``id`` recycled by a later
      same-named factory class (which would collide).
    - Its ``__hash__`` / ``__eq__`` use ``object`` identity, so a metaclass ``__eq__`` cannot merge two classes and
      ``__hash__ = None`` cannot make the key unhashable. (Such (meta)class behavior is rejected upstream anyway; this
      is defense-in-depth and also keeps a module-reload rebind distinct.)
    """

    __slots__ = ("cls",)

    def __init__(self, cls: type) -> None:
        self.cls = cls

    def __hash__(self) -> int:
        return object.__hash__(self.cls)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _ClassRef) and self.cls is other.cls

    def __repr__(self) -> str:
        return f"<_ClassRef 0x{id(self.cls):x}>"  # identity only; the live key is never serialized


# Monotonic, never-reused serials for locally/dynamically created classes, used only in the *offline* key (see
# ``_subclass_identity``). ``id(cls)`` is unusable there: the on-disk artifact outlives the class, and after
# ``qd.reset()`` collects a class CPython can hand its freed address to the next same-qualified factory class, which
# would then serialize an identical ``(nonce, id)`` key and load the dead class's kernel. A serial is never handed out
# twice, so a later class keys distinctly even at a recycled address. Held *weakly* (this component pins nothing); the
# counter only advances so a slot is never reissued. Classes reaching here hash/compare by identity, safe as keys.
_dynamic_class_serials: "weakref.WeakKeyDictionary[type, int]" = weakref.WeakKeyDictionary()
_dynamic_class_serial_counter = itertools.count()


def _dynamic_class_serial(cls: type) -> int:
    """A process-unique, non-recyclable serial for a locally/dynamically created ``cls``, stable for its lifetime.

    Distinct class objects always get distinct serials (the counter only advances) even at a recycled address, so the
    offline key never mistakes a later same-qualified class for a dead one. A class that cannot be weakly referenced
    (already rejected upstream) draws a fresh serial each call - a redundant miss, never a stale reuse.
    """
    serial = _dynamic_class_serials.get(cls)
    if serial is not None:
        return serial
    serial = next(_dynamic_class_serial_counter)
    try:
        _dynamic_class_serials[cls] = serial
    except TypeError:
        pass
    return serial


def _subclass_identity(cls: type, live: bool) -> tuple:
    """Key component identifying a user *subclass* of a baked base type or an ``enum`` class. ``module``/``qualname``
    never uniquely identify the *class object* (two factory-built classes share both, and a module-level name can be
    rebound by a reload), yet ``cfg.x.__class__ is First`` is observable, so distinct class objects must key apart.
    ``live`` selects the strategy:

    - ``live=True`` (in-process spec key): a ``_ClassRef(cls)`` token - a strong ref that pins ``cls`` (so its ``id``
      cannot be recycled) yet hashes/compares by object identity (distinct classes stay apart across a reload or under
      a metaclass ``__eq__`` / ``__hash__ = None``). Process-local; must never reach the offline key.
    - ``live=False`` (offline fastcache key, ``str``-ified by ``args_hasher``): ``None`` for a resolvable (module-level)
      class - a stable cross-process string - and ``(_PROCESS_NONCE, _dynamic_class_serial(cls))`` for a non-resolvable
      one: the serial separates dynamic classes within this process (never recycled, unlike ``id``) and the nonce makes
      them a guaranteed cross-process miss. A reloaded module-level class still cannot reuse across processes, which is
      unavoidable and safe (the old object does not exist there).
    """
    if live:
        return (cls.__module__, cls.__qualname__, _ClassRef(cls))
    identity = (_PROCESS_NONCE, _dynamic_class_serial(cls)) if _class_not_uniquely_identified(cls) else None
    return (cls.__module__, cls.__qualname__, identity)


def _dunder_copied_from_base(klass: type, name: str, value: Any) -> bool:
    """True if ``klass.__dict__[name]`` is the *exact same object* as the attribute in one of ``klass``'s strict
    bases. On >=3.11 the enum machinery copies the mix-in's ``__str__`` / ``__format__`` / ``__repr__`` onto the
    subclass dict verbatim; that base method is not a user override, whereas a genuine override is a fresh object
    appearing in no base's ``__dict__``.
    """
    for base in klass.__mro__[1:]:
        if base.__dict__.get(name, _MISSING) is value:
            return True
    return False


def _enum_probe_classes() -> "list[type]":
    """One freshly-built enum of each framework kind (plain / ``IntEnum`` / ``IntFlag`` / ``Flag`` / ``StrEnum``) plus
    a *direct* ``int`` / ``str`` mix-in, used to learn on the *running* Python which sunder/dunder names the machinery
    injects and which track the bases/mix-in. The direct-mix-in probes matter: a hook like ``_value_repr_`` holds the
    mix-in's value (not an enum-base copy), so framework subclasses alone would misjudge a user ``class M(int,
    enum.Enum)``."""

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
    """True if ``klass.__dict__[name]`` is the inherited default (the machinery copied an inherited hook in), not a
    user-authored one. Compares *unwrapped* callable identity (``staticmethod``/``classmethod`` -> ``__func__``): 3.12+
    ``EnumMeta`` copies ``_generate_next_value_`` as a *fresh* wrapper around the same function, so a wrapper-identity
    test would wrongly flag every plain enum. A genuine override supplies a different function (returns False)."""
    target = getattr(member, "__func__", member)
    for base in klass.__mro__[1:]:
        base_member = vars(base).get(name, _MISSING)
        if base_member is not _MISSING:
            return getattr(base_member, "__func__", base_member) is target
    return False


def _enum_machinery_class_dict(cls: type) -> "dict[str, Any] | None":
    """The own ``__dict__`` the enum machinery would produce for a *member-free* class with ``cls``'s exact bases and
    metaclass: every injected sunder/dunder (``_value_repr_`` / ``__new__`` / ``_new_member_`` / ...) at its value for
    *this* mix-in shape, with no user members/hooks. ``_observable_class_dict_attr`` compares ``cls``'s own entry
    against this to tell a user override (``Mode._value_repr_ = ...``) from untouched bookkeeping - even for
    mix-in-dependent hooks that are not verbatim base copies (so ``_dict_entry_is_inherited_default`` cannot judge
    them: a direct ``class M(int, enum.Enum)`` legitimately carries ``int``'s ``_value_repr_``).

    Built via the metaclass's own ``__prepare__`` namespace (``EnumMeta`` rejects a plain ``dict`` body). Returns None
    if unbuildable (a base refusing re-subclassing, an objecting ``__init_subclass__``, ...); the caller then falls
    back to ``_ENUM_OVERRIDABLE_HOOK_NAMES``. Off the hot path, built at most once per class per validation."""
    mcls = type(cls)
    try:
        # ``__prepare__`` returns the metaclass's own namespace (``EnumMeta`` -> ``_EnumDict``), which must be passed
        # through verbatim (``EnumMeta`` rejects a plain ``dict``). Pyright types it as ``MutableMapping``, hence the
        # local ignore.
        namespace = mcls.__prepare__("_qd_enum_probe", cls.__bases__)
        reference = mcls("_qd_enum_probe", cls.__bases__, namespace)  # pyright: ignore[reportArgumentType]
    except Exception:  # pylint: disable=broad-except  # any build failure -> conservative allowlist fallback
        return None
    return dict(vars(reference))


def _compute_enum_generated_class_attrs() -> "frozenset[str]":
    """Names the class machinery (``type`` + enum) puts in a *user* class's own ``__dict__``: generic bookkeeping
    (``__doc__`` / ``__module__`` / ``__qualname__`` / ..., plus 3.13+ ``__firstlineno__`` / ``__static_attributes__``)
    and enum bookkeeping (``_member_map_`` / ``_value2member_map_`` / ``_generate_next_value_`` / ``__new__`` / ...).

    Computed once by probing a plain class and each framework enum kind, so the set tracks the *running* Python (3.13
    added several dunders). A sunder/dunder NOT here is a user hook (``_missing_``, ``_repr_html_``, ...) that
    ``_enum_class_behavior_attr`` rejects. (Machinery-copied operator dunders are handled earlier via
    ``_OBSERVABLE_DUNDERS`` + ``_dunder_copied_from_base``.)
    """

    class _Plain:
        pass

    def _generated(probe: type) -> "set[str]":
        # Exclude enum members/aliases (enum-valued, plain name) but keep enum-valued *machinery sunders* like
        # ``_boundary_`` (a ``FlagBoundary`` member on ``IntFlag``): bookkeeping, not user members.
        return {
            n for n, v in vars(probe).items() if not isinstance(v, enum.Enum) or (n.startswith("_") and n.endswith("_"))
        }

    names: "set[str]" = _generated(_Plain)
    for probe in _enum_probe_classes():
        names.update(_generated(probe))
    return frozenset(names)


def _compute_enum_reference_checked_attrs() -> "frozenset[str]":
    """The subset of machinery-generated sunder/dunder names whose value is fixed by an enum's *bases / mix-in* (not
    its members): ``_value_repr_`` / ``__new__`` / ``_new_member_`` / ``_generate_next_value_`` / copied operator
    dunders / ... For these a clean enum's own entry equals a *member-free* rebuild of the same bases
    (``_enum_machinery_class_dict``), so ``_observable_class_dict_attr`` can flag a user override by comparison.

    Member-*derived* bookkeeping (``_member_map_`` / ``_value2member_map_`` / ...) differs from the reference even for
    a clean enum and is a function of the already-keyed members, so it is excluded; structural names too. A name
    qualifies only if in *every* probe carrying it the stored object *is* (unwrapped) the reference's, so the set
    tracks the running Python and never mistakes member data for a hook."""
    appears: "dict[str, int]" = {}
    matches: "dict[str, int]" = {}
    for probe in _enum_probe_classes():
        reference = _enum_machinery_class_dict(probe)
        if reference is None:
            continue
        for name, member in vars(probe).items():
            if isinstance(member, enum.Enum) or not (name.startswith("_") and name.endswith("_")):
                continue
            appears[name] = appears.get(name, 0) + 1
            default = reference.get(name, _MISSING)
            if default is not _MISSING and getattr(member, "__func__", member) is getattr(default, "__func__", default):
                matches[name] = matches.get(name, 0) + 1
    return frozenset(name for name, count in appears.items() if matches.get(name, 0) == count) - _STRUCTURAL_CLASS_ATTRS


# Sunder/dunder names the machinery puts in a user enum/class dict; anything else sunder/dunder there is a user hook.
# Computed once at import so it matches the running Python.
_ENUM_GENERATED_CLASS_ATTRS = _compute_enum_generated_class_attrs()

# The generated names fixed by an enum's bases/mix-in, so a user override is caught by comparing against a member-free
# reference (see ``_observable_class_dict_attr``). Member-derived data + structural names stay exempt.
_ENUM_REFERENCE_CHECKED_ATTRS = _compute_enum_reference_checked_attrs()

# Fallback allowlist used only when the member-free reference (``_enum_machinery_class_dict``) cannot be built. The
# machinery copies these defaults into every subclass dict, so the name alone cannot distinguish a user override; they
# are exempt only when the stored object *is* the inherited default (see ``_dict_entry_is_inherited_default``). The
# preferred reference path also catches mix-in-dependent hooks this allowlist omits.
_ENUM_OVERRIDABLE_HOOK_NAMES = frozenset({"_generate_next_value_"})


def _generated_attr_is_user_override(klass: type, name: str, member: Any, reference: "dict[str, Any] | None") -> bool:
    """True if ``klass.__dict__[name]`` (a reference-checked machinery sunder/dunder) is a *user* override, comparing
    unwrapped callable identity (``staticmethod``/``classmethod`` -> ``__func__``).

    Against the member-free ``reference``: present-and-differing => override; absent from the reference => not an
    override. If ``reference`` is None (unbuildable) it degrades to the inherited-default allowlist, still catching
    ``_generate_next_value_``."""
    if reference is None:
        return name in _ENUM_OVERRIDABLE_HOOK_NAMES and not _dict_entry_is_inherited_default(klass, name, member)
    default = reference.get(name, _MISSING)
    if default is _MISSING:
        return False
    return getattr(member, "__func__", member) is not getattr(default, "__func__", default)


def _is_official_enum_member(klass: type, name: str, member: Any) -> bool:
    """True if ``klass.__dict__[name]`` (an ``enum`` member) is a machinery-defined member/alias of ``klass`` (``name``
    is its own name in ``klass._member_map_``), not a user-added enum-valued attribute (``Mode.X = Mode.A``). Members
    and aliases live in ``_member_map_`` and are captured by the class/name/value key; a user-added attribute never
    enters it yet is observable as ``cfg.mode.__class__.X``, so it must be reported. A missing/non-dict ``_member_map_``
    yields False (so the attribute is reported too)."""
    member_map = klass.__dict__.get("_member_map_")
    return isinstance(member_map, dict) and member_map.get(name, _MISSING) is member


def _observable_class_dict_attr(klass: type) -> "str | None":
    """Name of a user-authored observable attribute in ``klass.__dict__`` (a method/property/class var, a genuine
    operator-dunder override, a user sunder/dunder hook including an override of a machinery hook like ``_value_repr_``,
    or a user-added enum-valued attribute), or None. Machinery members/aliases (``_is_official_enum_member``),
    base-copied dunders (>=3.11, ``_dunder_copied_from_base``), and *untouched* machinery sunders/dunders are skipped.
    Used on an enum class's own MRO and its metaclass MRO.

    A machinery name is not exempt by name alone: the machinery copies overridable hooks (``_value_repr_``, ...) into
    every enum dict, so a user reassignment would slip through; each is compared against a member-free reference
    (``_enum_machinery_class_dict``), reporting only a genuine override. An enum-valued attribute is exempt only when a
    real member/alias, not a user-added ``Mode.X``.
    """
    reference: Any = _MISSING  # member-free machinery reference; built lazily, at most once per call
    for name, member in vars(klass).items():
        is_sunder_dunder = name.startswith("_") and name.endswith("_")
        if isinstance(member, enum.Enum) and not is_sunder_dunder:  # a member name holding an enum value
            if _is_official_enum_member(klass, name, member):
                continue  # a machinery member/alias - keyed by member identity
            return name  # a user-added enum-valued attribute (``Mode.X = Mode.A``), absent from the key
        if name in _OBSERVABLE_DUNDERS:
            if _dunder_copied_from_base(klass, name, member):
                continue  # base/data-type dunder copied by the machinery (>=3.11), not a user override
            return name  # a genuine user operator/behavior dunder override
        if is_sunder_dunder:  # ``_boundary_`` (enum-valued *machinery* sunder) is exempted here, not above
            if name in _ENUM_GENERATED_CLASS_ATTRS:
                if name in _ENUM_REFERENCE_CHECKED_ATTRS:
                    if reference is _MISSING:
                        reference = _enum_machinery_class_dict(klass)
                    # A bases/mix-in-fixed hook is exempt only while it still holds the machinery's value; a user
                    # override (distinct object) is observable behavior the fixed key cannot capture.
                    if _generated_attr_is_user_override(klass, name, member, reference):
                        return name
                continue  # untouched bookkeeping: member-derived data, structural, or a matched hook
            return name  # a user sunder/dunder hook (``_missing_`` / ``_repr_html_`` / ...)
        return name  # a user method / property / class var
    return None


def _enum_class_behavior_attr(enum_cls: type) -> "str | None":
    """Name of a user-defined class-level attribute (method/property/class var/operator dunder) on ``enum_cls``, one of
    its user-authored bases (*including a non-enum mixin*), or its *metaclass*, or None. ``module``/``qualname``/name/
    value do not uniquely identify a dynamic enum class, so two same-named factory enums whose ``label`` or ``__eq__``
    differ key identically while ``qd.static(cfg.mode.label == "x")`` / ``qd.static(cfg.mode == 1)`` differ; a non-enum
    mixin's behavior (``class Mode(Labels, enum.Enum)``) is observable as ``cfg.mode.label`` too.

    The *metaclass* is inspected as well (``cfg.mode.__class__.label`` -> ``type(enum_cls).label``, off the MRO walk);
    only user-authored layers, so a plain enum (metaclass exactly ``EnumMeta``) is unaffected.
    """
    for klass in enum_cls.__mro__:
        # Skip the mixed-in primitive, ``object`` / a NumPy base, and the library enum bases (framework internals).
        # Every user-authored base is inspected, enum or not: a non-enum mixin can add observable behavior too.
        if _is_baked_base_type(klass) or klass in _FRAMEWORK_ENUM_CLASSES:
            continue
        attr = _observable_class_dict_attr(klass)
        if attr is not None:
            return attr
    # ``: type`` keeps ``__mro__`` an instance read (a class object), not a read of the ``type.__mro__`` descriptor.
    enum_metacls: type = type(enum_cls)  # user-authored metaclass layers below the framework ``EnumMeta``
    for mcls in enum_metacls.__mro__:
        if mcls in _FRAMEWORK_ENUM_METACLASSES:
            continue
        attr = _observable_metaclass_attr(mcls)
        if attr is not None:
            return attr
    return None


def _reject_stateful_enum_member(value: Any) -> None:
    """Reject an ``enum`` member carrying observable state the key (``module``/``qualname``/name/value) cannot capture,
    like ``_reject_stateful_primitive_subclass``. Two sources are rejected:

    - *per-member* state (an attribute on the member, in ``__dict__`` or a populated slot), and
    - *class-level* behavior (a user method/property/class var on the enum class) two factory enums could define
      differently while sharing ``module``/``qualname``.

    Plain enums (and unnamed ``IntFlag`` composites) carry only name/value bookkeeping and pass both.
    """
    cls = type(value)
    extra = _enum_member_state_attr(value)
    if extra is not None:
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}.{value.name}, an ``enum`` member "
            f"with user-defined per-member state (e.g. attribute {extra!r}). A ``Final`` value is baked as a "
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
    """A hashable digest of ``cls``'s *entire* member map - every member/alias name paired with its value routed
    through ``final_scalar_key`` (so ``True`` vs ``1`` cannot collide and an unsupported/mutable value is rejected). A
    baked ``Final`` member keys on this, not only the selected member, because a *sibling* is observable
    (``cfg.mode.__class__.OTHER.value``): without it, changing another member's value would leave the key unchanged and
    reuse a stale specialization. Preserves insertion order, so it is stable for a fixed definition.

    Returns ``()`` if ``cls`` has no member map (defensive); off the hot path, so re-walking the small map is cheap."""
    member_map = getattr(cls, "_member_map_", None)
    if not isinstance(member_map, dict):
        return ()
    return tuple((name, final_scalar_key(member.value, live)) for name, member in member_map.items())


def final_scalar_key(value: Any, live: bool = False) -> Any:
    """Return a collision-free key component for a baked ``Final`` field value.

    Every value becomes a *type-tagged* component, because Python treats several distinct compile-time constants as
    equal (with equal hashes) and annotations are not enforced, so one field can receive any of them. Encodings:

    - a ``float`` (builtin/subclass/NumPy scalar) -> exact IEEE-754 bits under ``_FLOAT_KEY_TAG`` (a subclass adds its
      class identity; a NumPy scalar uses ``(<dtype str>, <raw bytes>)``). Bits, not value, so ``-0.0``/``0.0`` and
      NaNs differing only in sign/payload stay distinct and widths never alias.
    - an ``enum`` member -> ``(_ENUM_KEY_TAG, *class-identity, name, final_scalar_key(value), member-map)``. An
      ``IntEnum``/``StrEnum`` member is ``==`` to its bare scalar and to a same-valued member of another class, so
      keying on identity keeps them distinct. ``name`` + value separate same-named members of two factory-shared
      classes; the value recurses (so ``True`` vs ``1`` cannot collide and an unsupported value is rejected). Members
      with per-member state are rejected. The member-map (``_enum_member_map_key``) folds in every sibling, observable
      as ``cfg.mode.__class__.OTHER.value``.
    - every remaining scalar (``bool``/``int``/``str`` and NumPy analogues) -> ``(_SCALAR_KEY_TAG, module, qualname,
      canonical-value)``. ``True == 1 == np.int64(1)`` but bake observably different constants, so the exact type is
      tagged. A subclass is accepted only if behavior-free, and its value is coerced via a base slot
      (``int.__int__``/``str.__str__``, ``.item()`` for NumPy) so the offline string is faithful even under a
      nonstandard ``repr``.

    A subclass or ``enum`` is keyed by class identity via ``_subclass_identity`` (``cfg.x.__class__`` is observable).
    ``live=True`` uses a ``_ClassRef`` identity token (distinct classes never collide, and its retained ref pins the
    ``id``); ``live=False`` (default, offline) uses a process-stable component. A value that is none of the above (an
    arbitrary/mutable object) is *rejected* rather than keyed by its own ``__eq__`` / ``__hash__``.

    Off the hot path for Final-free dataclasses (cached, never reach here). A Final-bearing subtree deliberately does
    not cache (``subtree_has_final_fields``), so this re-runs per launch and catches a value whose class turned
    behaviorful (e.g. a monkey-patched enum) after launch one.
    """
    if type(value) is float:
        return (_FLOAT_KEY_TAG, _unpack_u64(_pack_f64(value))[0])
    if isinstance(value, enum.Enum):
        # Checked before the ``float``/``int`` branches so a mixed-in enum keys by identity, not value. The key carries
        # both ``name`` (``None`` for an unnamed ``IntFlag`` composite) and the value, recursed through
        # ``final_scalar_key`` so ``True`` vs ``1`` cannot collide and a mutable/unsupported value is rejected.
        # ``module``/``qualname``/name/value do not uniquely identify the class *object* (factory-built or reload-
        # rebound), so ``_subclass_identity`` adds a ``_ClassRef`` (in-process) or ``(nonce, serial)`` / ``None``
        # (offline). The whole member map (``_enum_member_map_key``) is folded in because a sibling is observable
        # (``cfg.mode.__class__.OTHER.value``); for a resolvable class (``None`` identity) it is the only guard against
        # a changed sibling.
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
        # ``dtype.str`` + ``tobytes()`` preserves sign bit, NaN payload and width, in the same tagged space as the
        # builtin-float branch. An exact NumPy float is keyed by dtype+bytes; a stateful subclass is rejected and a
        # behavior-free one additionally carries its class identity.
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        if _is_baked_base_type(cls):  # an exact NumPy float scalar (``np.float32``/``np.float64``/...)
            return (_FLOAT_KEY_TAG, value.dtype.str, value.tobytes())
        return (_FLOAT_KEY_TAG, *_subclass_identity(cls, live), value.dtype.str, value.tobytes())
    if isinstance(value, float):
        # A ``float`` subclass (the exact builtin took the fast path above). Bit-encode like a plain float but keep the
        # subclass identity so it is not confused with a plain float or another factory subclass; a stateful subclass
        # is rejected.
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        return (_FLOAT_KEY_TAG, *_subclass_identity(cls, live), _unpack_u64(_pack_f64(value))[0])
    # ``bool`` / ``int`` / ``str`` and their NumPy analogues. The exact type completes the tag so value-equal but
    # distinct-typed constants (``True`` vs ``1``, ``np.int64(1)`` vs ``1``) never share a specialization; a stateful
    # subclass is rejected.
    _reject_stateful_primitive_subclass(value)
    cls = type(value)
    # Embed a *canonical* base value, not the object: the offline key ``str``-ifies this tuple, so a subclass whose
    # ``__repr__`` drops its value would serialize two distinct values to one string. Coercing keeps the string
    # faithful and stable.
    if _is_exact_baked_type(cls):
        canonical = value
    elif isinstance(value, int):  # a Python ``int`` subclass (``bool`` cannot be subclassed)
        canonical = int.__int__(value)  # base slot: bypass an overridden ``__int__`` / ``__index__``
    elif isinstance(value, np.integer):  # NumPy integer scalar (not a Python ``int`` subclass, so no base int slot)
        canonical = value.item()  # native, override-proof extraction of the underlying Python ``int``
    elif isinstance(value, str):  # a ``str`` subclass, including ``np.str_``; bypass any ``__str__`` override
        canonical = str.__str__(value)
    elif isinstance(value, np.bool_):  # NumPy boolean scalar -> plain ``bool``
        canonical = bool(value)
    else:
        # Annotations are not enforced, so a ``Final`` field (or an enum member's value routed here) can be any type.
        # An arbitrary object is not fully key-determined by value: two ``==`` instances with differing attributes
        # would share a specialization while ``qd.static(cfg.x.tag == 1)`` reads the difference, and a mutable value
        # could change under the cached ``_qd_spec_key``. Reject rather than trust its ``__eq__`` / ``__hash__``.
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, which is not a supported "
            f"compile-time constant. A ``Final`` value - or an ``enum`` member's underlying value - must be a "
            f"``bool`` / ``int`` / ``float`` / ``str`` (or a NumPy scalar analogue), or an ``enum`` member; an "
            f"arbitrary object cannot be keyed by value alone. Bake the specific scalar you need as a ``Final`` "
            f"field instead."
        )
    if _is_baked_base_type(cls):  # exact builtin primitive or exact NumPy scalar
        return (_SCALAR_KEY_TAG, cls.__module__, cls.__qualname__, canonical)
    # A behavior-free user subclass (``class Grams(int): pass``): ``_subclass_identity`` carries the class-identity
    # token so two same-named subclasses key apart, matching the enum and float-subclass branches.
    return (_SCALAR_KEY_TAG, *_subclass_identity(cls, live), canonical)
