"""Helpers for the ``@dataclasses.dataclass`` kernel-arg path, including ``typing.Final[T]`` compile-time template
fields.

PERF NOTE: everything about ``Final`` resolution - including the validation that rejects mutable carriers - is
computed **once per dataclass type** and cached (``_final_plan_cache``, ``_final_path_cache``). Callers on the
per-launch hot path (``_extract_arg``, ``args_hasher.dataclass_to_repr``) do a single ``dict.get`` keyed on the
dataclass type and then, in the overwhelmingly common no-Final-field case, take a branch that is byte-for-byte the
pre-existing code path. No ``isinstance`` / ``typing.get_origin`` / ``dataclasses.fields`` call happens per launch.
See the module docstring of ``_template_mapper_hotpath.py`` for why that matters (``isinstance`` is a ~100-200ns MRO
walk vs a ~10ns pointer comparison for ``type(x) is Y``).
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

# A random token minted once per interpreter process. It is folded into the *offline* (cross-process) key component of
# a non-uniquely-resolvable (locally / dynamically created) class - see ``_subclass_identity``. The per-class serial
# that accompanies it (see ``_dynamic_class_serial``) is only process-local and restarts from zero in every process,
# so two workers building same-named dynamic classes would otherwise serialize identical offline keys and one could
# load a kernel baked for the other's (distinct) class. The nonce makes such a key unique to this process: a dynamic
# class is a guaranteed cross-process cache *miss*, never a wrong reuse, matching the documented "dynamic classes
# don't reuse another process's cached kernel" contract.
#
# It MUST be reseeded in a ``fork``ed child: the child inherits the parent's string *and* its serial counter, so a
# fresh dynamic class in one child can draw the same serial as a same-qualified one in a sibling and serialize an
# identical offline key. ``os.register_at_fork`` mints a fresh token in every child (``uuid4`` re-reads OS entropy, so
# each child - and the parent - differ), restoring the guarantee. ``spawn`` re-imports this module, so it reseeds
# anyway.
_PROCESS_NONCE = uuid.uuid4().hex


def _reseed_process_nonce() -> None:
    global _PROCESS_NONCE
    _PROCESS_NONCE = uuid.uuid4().hex


if hasattr(os, "register_at_fork"):  # POSIX only; ``spawn``/Windows re-import the module and reseed on their own
    os.register_at_fork(after_in_child=_reseed_process_nonce)

# ``T`` values permitted inside ``Final[T]``. A Final field's value is baked into the compiled kernel as a literal and
# folded into both the in-process template spec key and the cross-process fastcache key, so ``T`` must be something
# that (a) is meaningful as a compile-time literal in generated code and (b) hashes and ``repr``s by value, stably
# across processes. ``bool`` precedes ``int`` only for readability - membership is by exact type so ordering is
# irrelevant.
#
# ``enum.Enum`` subclasses are permitted too (resolved separately, since membership here is by exact type): Genesis
# declares ``integrator: int`` / ``ccd_algorithm: int`` etc. but stores ``IntEnum`` members in them, and an IntEnum
# member is both a valid literal and stably repr'able.
_FINAL_SCALAR_TYPES = frozenset({bool, int, float, str})

# Types that are specifically worth a tailored error, because ``Final[T]`` on them is a plausible user mistake with a
# clear better alternative. Resolved lazily to dodge import cycles (``_ndarray`` imports back into ``lang``).
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

    ``typing.Final[T]`` on a field of a frozen ``@dataclasses.dataclass`` kernel argument marks the field's value as a
    compile-time constant: the value is baked into the compiled kernel (so ``qd.static(config.field)`` is legal), it is
    folded into the template spec key and the fastcache key (so distinct values compile distinct kernels), and it is
    NOT declared as a runtime scalar kernel arg.

    Bare ``typing.Final`` (with no ``[T]``) returns False here, because ``typing.get_origin`` yields ``None`` for it.
    That does NOT make it an ordinary field: ``_build_final_plan`` rejects the bare spelling outright, since Quadrants
    needs the wrapped type and silently lowering it to a runtime arg dies later in ``cook_dtype``.

    ``typing_extensions.Final`` is the same object as ``typing.Final`` on every Python version Quadrants supports
    (>=3.10), so it is accepted transparently; ``_build_final_plan`` raises a clear error if a future divergence ever
    makes that untrue.
    """
    return typing.get_origin(annotation) is typing.Final


def _describe_annotation(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or repr(annotation)


def _reject_hint_for(inner: Any) -> str | None:
    """Return a tailored remediation hint when ``Final[inner]`` names a type we specifically want to reject well.

    Matched on the *type name* rather than by importing the classes themselves, both to avoid import cycles and
    because the check runs once per dataclass type at compile time, where the cost of a string compare is irrelevant.
    """
    for name in (type(inner).__name__, _describe_annotation(inner)):
        hint = _FINAL_REJECT_HINTS.get(name)
        if hint is not None:
            return hint
    return None


def _validate_final_inner_type(dc_type: type, field_name: str, annotation: Any) -> None:
    """Raise a clear error unless ``Final[annotation]`` names a type we can bake as a compile-time literal.

    Only called once ``is_final_annotation`` has confirmed the annotation is a subscripted ``Final``, so
    ``typing.get_args`` is guaranteed non-empty here - Python rejects ``Final[()]`` at subscript time, and bare
    ``Final`` is caught earlier in ``_build_final_plan``.
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
    # ``issubclass`` here is fine: this runs once per dataclass type at compile time, never per launch.
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

    ``frozen=True`` prevents it outright. ``unsafe_hash=True`` does not, but it is an explicit assertion by the user
    that instances are value-stable, and the surrounding dataclass machinery already takes them at their word (see the
    ``is_frozen`` key cache in ``_extract_arg``), so honour it here for consistency.

    Deliberately *not* the codebase-wide ``__hash__ is not None`` proxy: ``@dataclass(eq=False)`` inherits
    ``object.__hash__``, so that proxy reads a plain mutable class as frozen. For a ``Final`` field that means the
    baked constant can be reassigned with no error and no recompilation (``TemplateMapper.lookup`` memoises on
    ``id(arg)``), which is precisely what this check exists to prevent - and unlike ``unsafe_hash=True``, turning off
    ``__eq__`` is not a statement about value stability.
    """
    params = getattr(dc_type, "__dataclass_params__", None)
    if params is None:
        return dc_type.__hash__ is not None
    return params.frozen or params.unsafe_hash


# Memo of ``id(dataclass type) -> (type, field path down to some Final leaf)`` (path ``None`` if the subtree holds
# none), used to reject mutable ancestors. Identity-keyed with a strong-ref+``is`` guard for the same reason as
# ``_final_plan_cache`` (metaclass ``__eq__`` must not merge distinct types). Same once-per-type lifecycle; never
# consulted per launch.
_final_path_cache: "dict[int, tuple[type, tuple[str, ...] | None]]" = {}


def _first_final_path(dc_type: type, visiting: "frozenset[int]") -> "tuple[str, ...] | None":
    """Return the field path from ``dc_type`` down to some ``Final`` leaf, or None if the subtree contains none.

    Only a witness is needed (to name one offending path in the error), so the first hit wins. Recursing through
    ``final_field_names`` means every nested type is validated eagerly at the top-level call rather than lazily when
    the hot path first walks that far down.

    ``visiting`` guards against a self-referential type graph. Quadrants cannot actually lower a recursive dataclass
    kernel arg, but validation must fail with a real error rather than a ``RecursionError``. It holds ``id(type)``,
    not the types themselves, so a metaclass that makes two distinct nested types compare equal cannot make one look
    "already visited" - which would return ``None`` early and let a mutable ancestor of a ``Final`` leaf slip past the
    rejection above. Every type on the current recursion path is held alive by the walk, so its ``id`` is stable.
    """
    entry = _final_path_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity check: guard against a recycled ``id`` (belt & braces)
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
    """``typing.get_type_hints(dc_type)`` (real objects, with ``Final`` preserved) or None if the annotations cannot be
    resolved. Lets ``_build_final_plan`` detect an *aliased* ``Final`` hidden behind a string annotation -
    ``from typing import Final as F; x: F[int]`` under ``from __future__ import annotations`` stores ``"F[int]"``,
    which a substring test for the literal name ``Final`` would miss. Off the hot path (``_build_final_plan`` is
    memoised per type). Any resolution failure (an unresolvable forward reference, an exotic annotation object, ...)
    falls back to None so the caller can apply its best-effort substring check instead of crashing."""
    try:
        return typing.get_type_hints(dc_type)
    except Exception:
        return None


def _build_final_plan(dc_type: type) -> "frozenset[str]":
    """Validate every ``Final`` field on ``dc_type`` and return the set of Final-annotated field names.

    Called once per dataclass type (memoised in ``_final_plan_cache``), so all of the reflection here - including
    ``dataclasses.fields``, ``typing.get_origin`` and ``issubclass`` - stays entirely off the per-launch hot path.
    """
    resolved_hints: "dict[str, Any] | None" = None
    resolved_computed = False
    final_names = []
    for field in dataclasses.fields(dc_type):
        annotation = field.type
        if isinstance(annotation, str):
            # ``from __future__ import annotations`` (or an explicit string annotation) leaves ``field.type`` as an
            # unresolved string. The pre-existing dataclass kernel-arg path already assumes resolved types, so rather
            # than half-supporting it, flag the one case where silently ignoring it would be a correctness trap: a
            # field the user believes is a compile-time constant but which we would lower as a runtime arg. Resolve the
            # class's hints (once, lazily) so an *aliased* ``Final`` (``Final as F`` -> ``"F[int]"``) is caught, not
            # just the literal name; fall back to the substring test only when the annotations cannot be resolved.
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
            # Bare ``Final`` with no ``[T]``. ``typing.get_origin(typing.Final)`` is ``None``, so this does not look
            # like a Final annotation to ``is_final_annotation`` and the field would otherwise be treated as an
            # ordinary runtime one - reaching ``decl_scalar_arg`` and dying in ``cook_dtype`` with
            # ``ValueError: Invalid data type typing.Final``. That is exactly the confusing failure this feature
            # exists to remove for ``Final[T]``, so reject the unsupported spelling here with a clear message.
            raise TypeError(
                f"{dc_type.__name__}.{field.name}: bare ``typing.Final`` is not supported as a Quadrants "
                f"compile-time template field. Write ``Final[T]`` with a concrete type, e.g. "
                f"``{field.name}: Final[int]``."
            )
        if not is_final_annotation(annotation):
            # Catch a ``Final``-like special form that is not ``typing.Final`` - e.g. if a future ``typing_extensions``
            # release stops aliasing the stdlib object. Silently treating such a field as a runtime one would be a
            # correctness trap of exactly the kind described just above.
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
        # ``Final`` asserts the value never changes, and Quadrants bakes it into compiled code accordingly - so a
        # mutable carrier is a contradiction we reject rather than silently tolerate. Rebinding it is not merely
        # untidy: ``TemplateMapper.lookup`` memoises the specialisation on ``id(arg)``, so re-launching with the same
        # instance after a reassignment silently reuses the kernel compiled with the old value.
        raise TypeError(
            f"{dc_type.__name__} has ``Final`` field(s) {sorted(final_names)} but is not frozen. A ``Final`` field's "
            f"value is baked into the compiled kernel, so it must not be reassignable - reassigning it would silently "
            f"keep using the kernel compiled for the old value. Declare the class as "
            f"``@dataclasses.dataclass(frozen=True)`` (or ``unsafe_hash=True`` if you must keep it mutable and accept "
            f"responsibility for never reassigning these fields)."
        )

    # A mutable *ancestor* of a Final field is just as unsound as a mutable carrier, and is not covered by the check
    # above because ``final_names`` only describes this class's own fields. Given a frozen ``Inner`` holding
    # ``n: Final[int]``, a mutable ``Outer`` holding ``child: Inner`` still lets ``outer.child = Inner(n=9)`` change a
    # baked constant, and the ``id(arg)``-keyed lookup cache then hands back the specialisation compiled for the old
    # child. Reject every mutable dataclass on a path down to a Final leaf.
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


# Memo of ``id(dataclass type) -> (type, frozenset of Final field names)``. Keyed on the type's *identity* (its
# ``id``), never on ``type`` itself, so a metaclass that overrides ``__eq__``/``__hash__`` to make two distinct
# dataclass types compare equal cannot merge their (possibly different) Final schemas - a plain ``dict[type, ...]``
# would return the first type's plan for the second. The stored ``type`` is a strong reference: it both pins the type
# (so its ``id`` cannot be recycled while the entry lives) and lets a lookup verify identity (``entry[0] is dc_type``)
# before trusting the cached plan. Bounded by the number of distinct dataclass types the process passes to a kernel.
_final_plan_cache: "dict[int, tuple[type, frozenset[str]]]" = {}


def final_field_names(dc_type: Any) -> "frozenset[str]":
    """Return the cached set of ``Final``-annotated field names on ``dc_type``, validating on first sighting.

    Hot-path contract: one ``dict.get`` + one ``is`` check. Callers should short-circuit on the empty result so that
    dataclasses with no ``Final`` fields (the overwhelmingly common case) run the pre-existing code path untouched.

    ``dc_type`` is typed ``Any`` rather than ``type`` because ``_extract_arg`` calls this with its loosely-typed
    ``annotation`` parameter (a union covering every kernel-arg annotation shape), having already established that it
    is a dataclass type via the ``__dataclass_fields__`` probe. Narrowing at that call site would need a
    ``typing.cast``, which is a real function call on a per-launch path.
    """
    entry = _final_plan_cache.get(id(dc_type))
    if entry is not None and entry[0] is dc_type:  # identity check: guard against a recycled ``id`` (belt & braces)
        return entry[1]
    names = _build_final_plan(dc_type)
    _final_plan_cache[id(dc_type)] = (dc_type, names)
    return names


def subtree_has_final_fields(dc_type: Any) -> bool:
    """True if ``dc_type`` - or any dataclass nested (transitively) beneath it - declares a ``Final`` field.

    Gates the per-instance caches the launch hot path keeps on frozen dataclasses (``_qd_spec_key`` for the in-process
    spec key, ``_qd_dc_repr`` for the offline fastcache repr): both are served verbatim on later launches, which is
    sound only when nothing about the baked key can change after the first launch. A ``Final`` value's *class behavior*
    can: a plain ``enum`` accepted on launch one can have ``Mode.__eq__`` monkey-patched before launch two - a kernel
    observes that via ``qd.static(cfg.mode == 1)`` but the cached key cannot notice it (the class identity is
    unchanged). So a dataclass whose subtree bakes any ``Final`` value must recompute each launch, re-running
    ``final_scalar_key``'s validation (which then rejects the now-behaviorful class). The cache *writers* gate on this:
    they simply never store a cache for such a dataclass, so the read always misses and falls through to the
    revalidating recompute - leaving the steady-state cost of Final-free dataclasses untouched.

    Transitive because an early cache hit also skips the recursion into nested dataclasses: a ``Final`` leaf nested
    under an otherwise-Final-free ancestor still has to disable that ancestor's cache. Answered by ``_first_final_path``
    (memoised per type), so after first sighting this is a single ``dict.get`` + ``is`` check.
    """
    return _first_final_path(dc_type, frozenset()) is not None


# Precomputed packers for encoding a ``float`` by its exact IEEE-754 bits (see ``final_scalar_key``).
_pack_f64 = struct.Struct("<d").pack
_unpack_u64 = struct.Struct("<Q").unpack

# Namespaces the float bit-encoding so it lives in its own value space and can never be equal to a bare ``int`` key
# component (see the third bullet in ``final_scalar_key``). Any short, process-stable marker works; it must only be
# a type that a plain ``Final`` value can never be, and a 1-char ``str`` inside a 2-tuple satisfies that.
_FLOAT_KEY_TAG = "f64"

# Same idea for enum members: ``IntEnum`` / ``StrEnum`` members are ``==`` (with equal hashes) to their bare
# ``int`` / ``str`` value, so keying on their class + member identity under this tag keeps them disjoint from
# scalar keys and from same-valued members of other enum classes (see the enum bullet in ``final_scalar_key``).
_ENUM_KEY_TAG = "enum"

# And for every remaining scalar (``bool`` / ``int`` / ``str`` and their NumPy analogues): tagging with the exact
# type keeps value-equal but distinct-typed constants apart (``True`` vs ``1`` vs ``np.int64(1)``), which the bare
# value cannot do because Python gives them equal ``==`` and equal hashes (see the scalar bullet below).
_SCALAR_KEY_TAG = "scalar"

# Exact baked primitive types: an instance of exactly one of these is a pure literal with no extra per-instance
# state, so it never needs the stateful-subclass check below.
_EXACT_BAKED_TYPES = (bool, int, float, str)


def _is_exact_baked_type(cls: type) -> bool:
    """``cls`` *is* one of ``_EXACT_BAKED_TYPES`` - an identity test, never equality. A *subclass* whose metaclass makes
    it compare ``==`` to a builtin (``class X(int, metaclass=M)`` with ``M.__eq__`` returning ``X == int``) must not be
    mistaken for an exact builtin: equality-based ``in`` membership would then skip the subclass / metaclass validation
    *and* canonicalize ``X`` as an exact builtin, collapsing two distinct same-qualified factory classes onto one live
    key even though ``cfg.x.__class__ is First`` is observable at compile time. ``is`` cannot be spoofed."""
    return any(cls is t for t in _EXACT_BAKED_TYPES)


# ``Py_TPFLAGS_HEAPTYPE`` (``1 << 9``): the interpreter sets this on every type created by a ``class`` statement or a
# ``type(...)`` call, and leaves it clear on C-defined *static* types (the builtin primitives and NumPy's own scalar
# types). It is not user-settable, so it distinguishes a library-provided scalar base from a user subclass even one
# that spoofs ``__module__ = "numpy"`` (see ``_is_baked_base_type``).
_HEAPTYPE_FLAG = 1 << 9

# Class-dict names CPython auto-generates for a bare ``class X(base): pass`` (or ``__slots__``-only) subclass. A
# primitive subclass may carry *only* these; anything else (a method, property, class var, or an overridden dunder
# such as ``__eq__`` / ``__int__``) is observable class-level behavior we cannot key on - see
# ``_reject_stateful_primitive_subclass``. ``__firstlineno__`` / ``__static_attributes__`` are auto-added on 3.13+.
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

# Framework metaclass layers on a primitive subclass's *metaclass* MRO. A plain subclass's metaclass is exactly
# ``type``; only a user-authored ``type`` subclass sits below it, so only those layers are inspected for observable
# metaclass-level behavior (see the metaclass walk in ``_reject_stateful_primitive_subclass``).
_FRAMEWORK_PRIMITIVE_METACLASSES = frozenset({type, object})


def _observable_metaclass_attr(mcls: type) -> "str | None":
    """Name of a user-authored observable attribute in a metaclass layer's ``__dict__`` - readable at compile time as
    ``cfg.x.__class__.<attr>`` - or None. Only the auto-generated structural names (``_STRUCTURAL_CLASS_ATTRS``, since a
    metaclass is an ordinary ``type`` subclass) are exempt; anything else is state/behavior the fixed class-identity
    key cannot capture. Shared by the enum and primitive-subclass metaclass walks.

    Note that even a metaclass ``__eq__`` / ``__ne__`` / ``__hash__`` is rejected. The class-identity key uses
    ``id`` / ``is`` / ``object.__hash__`` (see ``_ClassRef``), so those operators cannot collapse two distinct classes
    or make the key unhashable - but a kernel can still *observe* them via ``qd.static(cfg.x.__class__ == Expected)``,
    and mutating the state they consult after the first launch would leave the (fixed) key unchanged and reuse the
    stale specialization. Identity-safe keying only prevents dict collisions; it does not make the operators
    unobservable, so equality/hash behavior on the metaclass is not exempt.
    """
    for name in vars(mcls):
        if name not in _STRUCTURAL_CLASS_ATTRS:
            return name
    return None


def _is_baked_base_type(klass: type) -> bool:
    """True for a library base a baked value can subclass without adding observable state/behavior: a builtin
    primitive, ``object``, or a NumPy-provided scalar type (any *static* subclass of ``np.generic``). Their own
    attributes are framework internals, not user state/behavior, so the subclass walk in
    ``_reject_stateful_primitive_subclass`` stops here and an exact instance needs no checking.

    A NumPy scalar base is recognised by inheritance plus *type nature* - ``issubclass(klass, np.generic)`` and the
    absence of ``Py_TPFLAGS_HEAPTYPE`` - never by the mutable ``__module__`` string. NumPy's own scalar types are all
    C-defined static types, whereas a user subclass (``class Foo(np.float64)``) is always a heap type even if it sets
    ``__module__ = "numpy"``; keying off the flag rather than the string keeps such a subclass from masquerading as a
    trusted base (its state/behavior is inspected and it is keyed by class identity like any other user subclass).
    """
    if _is_exact_baked_type(klass) or klass is object:
        return True
    # A genuine NumPy scalar base is a static (C-defined) ``np.generic`` subclass; a user subclass - even one that
    # spoofs ``__module__ = "numpy"`` - is a ``class``-statement heap type (``Py_TPFLAGS_HEAPTYPE`` set).
    is_heap_type = bool(klass.__flags__ & _HEAPTYPE_FLAG)
    return issubclass(klass, np.generic) and not is_heap_type


def _reject_stateful_primitive_subclass(value: Any) -> None:
    """Reject a *subclass* of a baked scalar type (``bool``/``int``/``float``/``str`` or a NumPy scalar) whose
    instance carries observable state a kernel could read but the key cannot capture - per-instance or class-level.

    A ``Final`` value is baked as a compile-time literal keyed by its (typed) value plus the subclass
    ``module``/``qualname``. That is not enough when a subclass carries more than its value:

    - *Per-instance* state - e.g. ``class TaggedFloat(float)`` with a ``unit`` attribute (or the same over
      ``np.float64``) - is not described by the numeric value, so two instances with an equal value but different
      state would bake different kernels yet select the same specialization. State can live in ``__dict__`` or a
      populated ``__slots__`` slot.
    - *Class-level* behavior/state - e.g. a factory returning ``float`` subclasses whose ``unit`` property closes
      over different values, or that override ``__eq__`` / ``__int__`` / ``__repr__`` differently - is not captured
      either: ``module``/``qualname`` does not uniquely identify a dynamically created class, so two distinct
      same-named subclasses (whose ``cfg.x.unit`` or ``cfg.x == 1`` a kernel could read) would collide. Any class
      member other than the auto-generated structural ones (``_STRUCTURAL_CLASS_ATTRS``) is therefore rejected -
      including an overridden operator/conversion/repr dunder, since it too is observable. The subclass's *metaclass*
      is inspected the same way (``cfg.x.__class__.label`` resolves to ``type(cls).label``, invisible to the subclass
      MRO walk), skipping the framework ``type`` / ``object`` layers - mirroring ``_enum_class_behavior_attr``.

    There is no bounded, process-stable way to serialise arbitrary state/behavior, so we reject rather than silently
    mis-specialise. Exact primitives, exact NumPy library scalars (their attrs are library internals, not user
    state) and behavior-free stateless subclasses (``class Meters(float): pass``) are unaffected. Runs once per
    instance, off the steady-state path.
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
            # Only the auto-generated structural members are exempt; a method / property / class var, or an
            # overridden operator / conversion / repr dunder (``__eq__``, ``__int__``, ``__repr__``, ...) all count.
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
    # ``cfg.x.__class__.label`` resolves to ``type(cls).label``, which the subclass MRO walk never sees. A custom
    # metaclass can carry observable state/behavior that two same-named factory subclasses define differently, or
    # that is mutated between launches, while the key (keyed only on the subclass ``module``/``qualname``) stays
    # fixed - so it must be inspected too, exactly as ``_enum_class_behavior_attr`` does for enums. The ``: type``
    # annotation keeps the ``__mro__`` access an instance read (a class *object*), not a read of the descriptor.
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


# Exhaustive allowlist of ``enum`` member attribute names that are standard bookkeeping, NOT user-defined per-member
# state. This is a strict allowlist rather than a "skip all dunders" test on purpose: a user can stash observable
# state under a dunder-looking name (``self.__unit__``, read as ``cfg.mode.__unit__``), so only these exact names
# are exempt. ``_name_`` / ``_value_`` / ``_sort_order_`` and the ``__objclass__`` back-pointer are the member
# fields; ``_inverted_`` is the value-derived cache CPython lazily stores on a ``Flag`` / ``IntFlag`` member the
# first time it is inverted (``~Perm.R``, Python >=3.11), so a member inverted before reaching a ``Final`` field is
# still accepted. ``__dict__`` / ``__weakref__`` are structural slot names (containers, not state).
_ENUM_INTERNAL_MEMBER_ATTRS = frozenset(
    {"_name_", "_value_", "_sort_order_", "_inverted_", "__objclass__", "__dict__", "__weakref__"}
)


def _enum_member_state_attr(value: Any) -> "str | None":
    """Return the name of a user-defined per-member state attribute on ``value``, or None if it carries only enum
    bookkeeping (see ``_ENUM_INTERNAL_MEMBER_ATTRS``). State can live in the member's ``__dict__`` or, when the enum
    declares ``__slots__``, in a populated slot that never appears in ``__dict__``; both are inspected (as the
    primitive-subclass check does), since checking only one would miss the other. Any name not on the allowlist -
    including a user-defined dunder like ``__unit__`` - is treated as observable state.
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


# Standard-library enum base classes. A user enum's MRO also contains these (plus the mixed-in primitive and
# ``object``); only the user-authored enum classes below them are inspected for observable class-level behavior.
_FRAMEWORK_ENUM_CLASSES = frozenset(
    c
    for c in (getattr(enum, n, None) for n in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "ReprEnum"))
    if isinstance(c, type)
)

# Framework metaclass layers on an enum's *metaclass* MRO. A plain enum's metaclass is exactly ``enum.EnumMeta`` (aka
# ``EnumType`` on Python >=3.11); only a user-authored ``EnumMeta`` *subclass* sits below it, and only those layers are
# inspected for observable metaclass-level behavior (see ``_enum_class_behavior_attr``).
_FRAMEWORK_ENUM_METACLASSES = frozenset(
    c for c in (type, object, getattr(enum, "EnumMeta", None), getattr(enum, "EnumType", None)) if isinstance(c, type)
)

# Dunder methods that make an object's *behavior* observable at compile time (comparison, hashing, conversion,
# container / callable / arithmetic protocols). One appearing in a user enum's own class dict is a *candidate* user
# override that two same-named factory enums could define differently (``cfg.mode == 1``). NOTE: on Python >=3.11 the
# enum machinery itself copies the mixed-in data type's ``__str__`` / ``__format__`` / ``__repr__`` onto an
# ``IntEnum`` / ``StrEnum`` / ``IntFlag`` subclass's dict, so presence alone is not enough:
# ``_dunder_copied_from_base`` filters those base-copied entries out, and only a genuine (fresh) override is rejected.
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
    """True if ``cls`` cannot be recovered from its ``module``/``qualname`` - i.e. it is locally/dynamically created
    (its qualname contains ``<locals>``, or ``module.qualname`` does not resolve back to this class object). Such a
    class shares its identity string with every other class built the same way, so distinct class objects would key
    identically on ``(module, qualname, ...)`` alone: two recreated enum classes have distinct members
    (``First.A is not Second.A``; for a plain ``Enum`` also ``First.A != Second.A``), and two recreated primitive
    subclasses are still observably distinct (``cfg.x.__class__ is First``). Callers add a class-identity token to the
    key in that case to keep them distinct (a ``_ClassRef`` in-process, a ``_dynamic_class_serial`` offline). Runs once
    per instance (cached), off the steady-state launch path.
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
    """A strong reference to a class that hashes and compares by *object identity*, never delegating to the class's
    metaclass. Embedded in the in-process (``live``) spec key, it serves two ends that a bare ``cls`` cannot serve at
    once:

    - It pins ``cls`` for as long as the key lives in the template mapper, so the class object cannot be garbage
      collected and its ``id`` cannot be recycled by a later same-named factory class (which would otherwise collide).
    - Its ``__hash__`` / ``__eq__`` use ``object`` identity, so a metaclass with a custom ``__eq__`` cannot merge two
      distinct classes and a metaclass with ``__hash__ = None`` cannot make the whole spec key unhashable
      (``self.mapping[key]`` would otherwise raise ``TypeError``). A value whose (meta)class carries such observable
      behavior is in fact rejected upstream (see ``_observable_metaclass_attr``); this identity-based hashing is
      defense-in-depth and is what also keeps a normal module-reload rebind (same ``qualname``, new ``id``) distinct.

    ``object.__hash__(self.cls)`` is the identity hash regardless of the metaclass; ``is`` gives identity equality.
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


# Monotonic, never-reused serial numbers for locally/dynamically created classes, used only in the *offline*
# fastcache key (see ``_subclass_identity``). ``id(cls)`` is unusable there: it identifies the class only while the
# class is alive, but the on-disk artifact keyed by it outlives the class. Once ``qd.reset()`` drops the live spec key
# whose ``_ClassRef`` pinned such a class, the class can be collected while its artifact remains, and CPython can hand
# the freed address to the *next* same-qualified factory class in this process - which would then serialize an
# identical ``(nonce, id)`` offline key and load the dead class's kernel, even though ``qd.static(cfg.x.__class__ is
# First)`` distinguishes them at compile time. A serial drawn from an ever-increasing counter is never handed out
# twice, so a later class keys distinctly even at a recycled address. The map holds classes *weakly* (unlike the live
# ``_ClassRef``, this component must pin nothing), so a collected class's entry simply disappears; the counter only
# advances, so its slot is never reissued. Classes reaching here have already passed the stateful-subclass /
# observable-metaclass rejection, so ``cls`` hashes and compares by object identity (``type``'s defaults) - safe as a
# ``WeakKeyDictionary`` key.
_dynamic_class_serials: "weakref.WeakKeyDictionary[type, int]" = weakref.WeakKeyDictionary()
_dynamic_class_serial_counter = itertools.count()


def _dynamic_class_serial(cls: type) -> int:
    """A process-unique, non-recyclable serial for a locally/dynamically created ``cls``, stable for its lifetime.

    Distinct class objects always get distinct serials (the counter only advances), even if one is collected and the
    next is allocated at the same address, so the offline key can never mistake a later same-qualified class for a
    dead one. A class that cannot be weakly referenced (a pathological metaclass, already rejected upstream) draws a
    fresh serial each call - never a stale reuse, only a redundant cross-process miss.
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
    """Key component identifying a user *subclass* of a baked base type (a ``float``/``int``/``str`` or NumPy scalar
    subclass), or an ``enum`` class. ``module``/``qualname`` name it, but they never uniquely identify the *class
    object*: two behavior-free classes built by the same factory share both, and even a module-level name can be
    rebound to a fresh class object (a module reload / reassignment) that resolves the same way - yet
    ``cfg.x.__class__ is First`` is observable at compile time, so distinct class objects must key distinctly.

    The two key consumers need different strategies, selected by ``live``:

    - ``live=True`` (the in-process template spec key): the identity component is a ``_ClassRef(cls)`` token. It holds a
      *strong reference* to ``cls`` (so this key, living in the template mapper for the specialization's lifetime, pins
      the class and its ``id`` cannot be recycled by a later same-named factory class) yet hashes and compares purely by
      object identity - so distinct class objects key apart even across a reload that transiently shares ``module`` /
      ``qualname``, a metaclass with a custom ``__eq__`` cannot merge two classes, and a metaclass with ``__hash__ =
      None`` cannot make the whole spec key unhashable. This token is process-local, so it must never reach the offline
      key.
    - ``live=False`` (the offline fastcache key, ``str``-ified by ``args_hasher``): the component must make another
      process reuse a *resolvable* class's cached kernel while never wrongly reusing one for a *non-resolvable* class.
      So it is ``None`` for a uniquely resolvable (typically module-level) class - keeping the string stable across
      processes - and ``(_PROCESS_NONCE, _dynamic_class_serial(cls))`` for a non-resolvable (locally/dynamically
      created) one: the serial separates distinct dynamic classes *within* this process and is never recycled (unlike
      ``id(cls)``, which the allocator can reissue to a later same-qualified class once the original is collected after
      ``qd.reset()`` - see ``_dynamic_class_serial``), while the per-process nonce makes the string unique *across*
      processes (serials restart from zero in each process), so a dynamic class is always a cross-process cache miss,
      never a wrong reuse. This still cannot reuse a reloaded module-level class's kernel across processes, which is
      unavoidable without process-stable identity and is safe (the old class object does not exist in the other
      process).
    """
    if live:
        return (cls.__module__, cls.__qualname__, _ClassRef(cls))
    identity = (_PROCESS_NONCE, _dynamic_class_serial(cls)) if _class_not_uniquely_identified(cls) else None
    return (cls.__module__, cls.__qualname__, identity)


def _dunder_copied_from_base(klass: type, name: str, value: Any) -> bool:
    """True if ``klass.__dict__[name]`` is the *exact same object* as the attribute in one of ``klass``'s strict
    bases. On Python >=3.11 the enum machinery copies the mixed-in data type's ``__str__`` / ``__format__`` /
    ``__repr__`` onto an ``IntEnum`` / ``StrEnum`` / ``IntFlag`` subclass's own ``__dict__`` (so members format like
    the bare value) - e.g. ``CcdAlgorithm.__dict__["__format__"] is IntEnum.__dict__["__format__"]``. That copy is a
    base method verbatim, not user-authored behavior, so it must not be treated as a class-level override. A genuine
    user override is a fresh object that appears in no base's ``__dict__``.
    """
    for base in klass.__mro__[1:]:
        if base.__dict__.get(name, _MISSING) is value:
            return True
    return False


def _enum_probe_classes() -> "list[type]":
    """One freshly-built enum of each framework kind (plain / ``IntEnum`` / ``IntFlag`` / ``Flag`` / ``StrEnum``) plus a
    *direct* ``int`` / ``str`` mix-in, used to learn - on the *running* Python, not a hand-maintained list that drifts -
    which sunder/dunder names the machinery injects into a user enum's own dict and which of those track the bases /
    mix-in rather than the members. The direct-mix-in probes matter: a hook like ``_value_repr_`` holds the mix-in's
    value, not a copy from any enum base, so a probe set of framework subclasses alone would misjudge a user's
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
    """True if ``klass.__dict__[name]`` (``member``) is the inherited default for ``name`` - i.e. the machinery merely
    copied an inherited hook into the class dict, rather than the class introducing a distinct (user-authored) one.

    Compares the *unwrapped* callable identity (``staticmethod`` / ``classmethod`` expose it as ``__func__``): CPython
    3.12+ ``EnumMeta`` copies ``_generate_next_value_`` into every enum as a *fresh* ``staticmethod`` wrapper around the
    same inherited function, so a wrapper-identity (``is``) test would see every plain enum as a user override and
    wrongly reject it. A genuine user override supplies a different underlying function, so this still returns False for
    it (the caller then treats it as observable)."""
    target = getattr(member, "__func__", member)
    for base in klass.__mro__[1:]:
        base_member = vars(base).get(name, _MISSING)
        if base_member is not _MISSING:
            return getattr(base_member, "__func__", base_member) is target
    return False


def _enum_machinery_class_dict(cls: type) -> "dict[str, Any] | None":
    """The own ``__dict__`` the enum machinery would produce for a *member-free* class with ``cls``'s exact bases and
    metaclass: every sunder/dunder the machinery injects (``_value_repr_`` / ``__new__`` / ``_member_type_`` /
    ``_new_member_`` / ``_generate_next_value_`` / ...) already set to its value for *this* mix-in shape, with no user
    members and no user hooks. Comparing ``cls``'s own entry against this is how ``_observable_class_dict_attr`` tells a
    user override of a machinery hook (``Mode._value_repr_ = ...``, observable and unkeyable) from untouched machinery
    bookkeeping - even for mix-in-dependent hooks whose value is *not* a verbatim base copy, which
    ``_dict_entry_is_inherited_default`` therefore cannot judge (a direct ``class M(int, enum.Enum)`` legitimately
    carries ``int``'s ``_value_repr_``, which appears in no enum base).

    Built through the metaclass's own ``__prepare__`` namespace, since ``EnumMeta`` rejects a plain ``dict`` body.
    Returns ``None`` if the reference cannot be built (a base that refuses re-subclassing, a mix-in
    ``__init_subclass__`` that objects, an exotic metaclass, ...); the caller then falls back to the
    ``_ENUM_OVERRIDABLE_HOOK_NAMES`` allowlist. Off the steady-state hot path (only Final-bearing configs reach here,
    and they deliberately re-validate per launch), built at most once per class per validation."""
    mcls = type(cls)
    try:
        # ``__prepare__`` returns the metaclass's own namespace (``EnumMeta`` -> ``_EnumDict``), which must be passed
        # through verbatim - wrapping it in a plain ``dict`` is exactly what ``EnumMeta`` rejects. Pyright types it as
        # a ``MutableMapping`` rather than the ``dict`` the class-creation call expects, hence the local ignore.
        namespace = mcls.__prepare__("_qd_enum_probe", cls.__bases__)
        reference = mcls("_qd_enum_probe", cls.__bases__, namespace)  # pyright: ignore[reportArgumentType]
    except Exception:  # pylint: disable=broad-except  # any build failure -> conservative allowlist fallback
        return None
    return dict(vars(reference))


def _compute_enum_generated_class_attrs() -> "frozenset[str]":
    """Names the class machinery (generic ``type`` + enum) puts in a *user* class's own ``__dict__``: generic
    bookkeeping (``__doc__`` / ``__module__`` / ``__qualname__`` / ``__dict__`` / ``__weakref__``, and on Python 3.13+
    ``__firstlineno__`` / ``__static_attributes__``) plus enum-generated bookkeeping (``_member_map_`` /
    ``_value2member_map_`` / ``_generate_next_value_`` / ``__new__`` / version-specific ``_hashable_values_`` / ...).

    Computed once by probing a plain class and each framework enum kind with real ``class`` syntax, so the set tracks
    the *running* Python version rather than a hand-maintained list that silently drifts (3.13 added several dunders).
    A sunder/dunder in a user enum's own dict that is NOT here is therefore a user-authored hook (``_missing_``,
    ``_repr_html_`` on 3.13+, an overriding ``_numeric_repr_``, ...): observable behavior the Final key cannot capture,
    so ``_enum_class_behavior_attr`` rejects it. (Machinery-copied *operator* dunders like ``IntFlag.__and__`` are
    handled earlier via ``_OBSERVABLE_DUNDERS`` + ``_dunder_copied_from_base`` and never reach that check.)
    """

    class _Plain:
        pass

    names: "set[str]" = {n for n, v in vars(_Plain).items() if not isinstance(v, enum.Enum)}
    for probe in _enum_probe_classes():
        names.update(n for n, v in vars(probe).items() if not isinstance(v, enum.Enum))
    return frozenset(names)


def _compute_enum_reference_checked_attrs() -> "frozenset[str]":
    """The subset of machinery-generated sunder/dunder names whose value is fixed by an enum's *bases / mix-in* rather
    than its members: ``_value_repr_`` / ``__new__`` / ``_member_type_`` / ``_new_member_`` / ``_generate_next_value_``
    / copied operator dunders / ... For these a clean enum's own entry equals the value in a *member-free* rebuild of
    the same bases (``_enum_machinery_class_dict``), so ``_observable_class_dict_attr`` can flag a user override by
    comparing against that reference.

    Member-*derived* bookkeeping (``_member_map_`` / ``_value2member_map_`` / ``_member_names_`` / version-specific
    ``_hashable_values_`` / ``_flag_mask_`` / ...) differs from the member-free reference even for a clean enum, so it
    is deliberately *excluded*: it is a function of the (already keyed) members, never a user hook. Structural names
    (``__module__`` / ``__qualname__`` / ``__doc__`` / ...) vary legitimately per class and are excluded too.

    A name qualifies only if, in *every* probe whose own dict carries it, the stored object *is* (unwrapped) the
    member-free reference's - so the set tracks the running Python (3.11 added ``_value_repr_`` / ``_use_args_`` / ...)
    and never mistakes member-derived data for a hook."""
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


# Sunder/dunder names the machinery itself puts in a user enum/class dict; anything else sunder/dunder in that dict is
# a user hook (see ``_compute_enum_generated_class_attrs``). Computed once at import so it matches the running Python.
_ENUM_GENERATED_CLASS_ATTRS = _compute_enum_generated_class_attrs()

# The generated names whose value is set from an enum's bases/mix-in (not its members), so a user *override* is caught
# by comparing against a member-free machinery reference (see ``_compute_enum_reference_checked_attrs`` and
# ``_observable_class_dict_attr``). The complementary generated names (member-derived data + structural) stay exempt.
_ENUM_REFERENCE_CHECKED_ATTRS = _compute_enum_reference_checked_attrs()

# Fallback allowlist of machinery names a user can nonetheless *override* with behavior a kernel could reach through a
# baked member (``qd.static(cfg.mode._generate_next_value_(...))``), used only when the member-free machinery reference
# (see ``_enum_machinery_class_dict``) cannot be built. The machinery *copies* the inherited default of these into every
# subclass's own dict, so the name alone appears even on a plain enum and cannot distinguish a user override - so these
# are exempted only when the stored object *is* the inherited default (see ``_dict_entry_is_inherited_default``). The
# preferred path compares against the reference and so also catches mix-in-dependent hooks this allowlist omits.
_ENUM_OVERRIDABLE_HOOK_NAMES = frozenset({"_generate_next_value_"})


def _generated_attr_is_user_override(klass: type, name: str, member: Any, reference: "dict[str, Any] | None") -> bool:
    """True if ``klass.__dict__[name]`` (a reference-checked machinery sunder/dunder) is a *user* override rather than
    the machinery's own value, comparing unwrapped callable identity (``staticmethod`` / ``classmethod`` ->
    ``__func__``, as the enum metaclass wraps ``_generate_next_value_`` afresh per class - see
    ``_dict_entry_is_inherited_default``).

    Against the member-free machinery ``reference``: a present name is an override iff its unwrapped object differs from
    the reference's; a name absent from the reference is treated as not-an-override (a member-free rebuild simply did
    not produce it for this shape). If ``reference`` is ``None`` (unbuildable) the check degrades to the
    inherited-default allowlist, which still catches ``_generate_next_value_``."""
    if reference is None:
        return name in _ENUM_OVERRIDABLE_HOOK_NAMES and not _dict_entry_is_inherited_default(klass, name, member)
    default = reference.get(name, _MISSING)
    if default is _MISSING:
        return False
    return getattr(member, "__func__", member) is not getattr(default, "__func__", default)


def _observable_class_dict_attr(klass: type) -> "str | None":
    """Name of a user-authored observable attribute in ``klass.__dict__`` - a method / property / class var, a
    genuinely-overridden observable operator dunder, or a user sunder/dunder hook (including a user *override* of a
    machinery-generated hook such as ``_value_repr_`` / ``__new__`` / ``_generate_next_value_``) - or None. Enum
    members/aliases, observable dunders the enum machinery merely copies from a base (Python >=3.11, see
    ``_dunder_copied_from_base``), and *untouched* machinery-generated sunder/dunder names (``_member_map_`` /
    ``__new__`` / ``__doc__`` / 3.13's ``__firstlineno__`` / ...) are skipped. Used for an enum class's own MRO and its
    metaclass MRO.

    A machinery-generated name is not exempted by name alone: the machinery *copies* several overridable hooks
    (``_value_repr_``, ``_generate_next_value_``, ...) into every enum's own dict, so a user reassignment (observable as
    ``cfg.mode._value_repr_`` yet invisible to the fixed Final key) would otherwise slip through. Each such entry is
    compared against a member-free machinery reference (``_enum_machinery_class_dict``) built once per class, so only a
    genuine user override is reported while mix-in-derived defaults and member-derived bookkeeping stay exempt.
    """
    reference: Any = _MISSING  # member-free machinery reference; built lazily, at most once per call
    for name, member in vars(klass).items():
        if isinstance(member, enum.Enum):  # an enum member or alias defined on the class
            continue
        if name in _OBSERVABLE_DUNDERS:
            if _dunder_copied_from_base(klass, name, member):
                continue  # enum machinery copied a base/data-type dunder (Python >=3.11), not a user override
            return name  # a genuine user operator/behavior dunder override
        if name.startswith("_") and name.endswith("_"):
            if name in _ENUM_GENERATED_CLASS_ATTRS:
                if name in _ENUM_REFERENCE_CHECKED_ATTRS:
                    if reference is _MISSING:
                        reference = _enum_machinery_class_dict(klass)
                    # A bases/mix-in-fixed hook (``_value_repr_``, ``__new__``, ...) is exempt only while it still
                    # holds the machinery's own value for this class; a user override (a distinct object) is
                    # observable behavior the fixed Final key cannot capture.
                    if _generated_attr_is_user_override(klass, name, member, reference):
                        return name
                continue  # untouched bookkeeping: member-derived data (``_member_map_``), structural, or matched hook
            return name  # a user-authored sunder/dunder hook (``_missing_`` / ``_repr_html_`` / ...) - observable
        return name  # a user method / property / class var
    return None


def _enum_class_behavior_attr(enum_cls: type) -> "str | None":
    """Return the name of a user-defined class-level attribute (method / property / class var / operator dunder) on
    ``enum_cls``, one of its user-authored bases - *including a non-enum mixin* - or its *metaclass*, or None.
    ``module``/``qualname``/member-name/value do not uniquely identify a dynamically created enum class, so two
    same-named factory enums whose ``label`` property closes over different strings (or whose ``__eq__`` differs) key
    identically while ``qd.static(cfg.mode.label == "x")`` / ``qd.static(cfg.mode == 1)`` differ. The same holds for
    behavior inherited from a non-enum mixin (``class Mode(Labels, enum.Enum)`` with ``Labels.label``): it is observable
    as ``cfg.mode.label`` yet absent from the key, so it must be inspected too.

    The *metaclass* is inspected as well: ``cfg.mode.__class__.label`` can resolve to ``type(enum_cls).label``, which
    the plain MRO walk never sees (the metaclass is not on ``enum_cls.__mro__``). A custom ``EnumMeta`` subclass can
    carry observable state/behavior that two same-named factory metaclasses define differently, again absent from the
    key. Only user-authored metaclass layers are inspected; the framework metaclass (``EnumMeta`` / ``EnumType``,
    ``type``, ``object``) is skipped, so a plain enum (whose metaclass is exactly ``EnumMeta``) is unaffected.
    """
    for klass in enum_cls.__mro__:
        # Skip the mixed-in primitive data type (``int``/``str``/... - a baked base), ``object`` / a NumPy base, and
        # the library's own enum base classes; their attributes are framework internals, not user behavior. Every
        # *user-authored* base is inspected whether or not it is itself an ``Enum``: a non-enum mixin can add
        # observable behavior/state (``cfg.mode.label``) that two same-named factory mixins could define differently,
        # exactly like an attribute on the enum class.
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
    """Reject an ``enum`` member that carries observable state the key cannot capture, for the same reason as
    ``_reject_stateful_primitive_subclass``: the key records only ``module``/``qualname``/name/value, so anything a
    kernel could additionally read (``qd.static(cfg.mode.unit == "m")``) - or that differs across processes for the
    offline key - would not select a distinct specialization. Two sources of such state are rejected:

    - *per-member* state (an attribute set on the member, in ``__dict__`` or a populated slot), and
    - *class-level* behavior (a user method / property / class var on the enum class), which two same-named enum
      classes built by a factory could define differently while sharing ``module``/``qualname``.

    Plain enums (and unnamed ``IntFlag`` composites) carry only name/value bookkeeping and pass both. Runs once
    per instance, off the steady-state launch path.
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


def final_scalar_key(value: Any, live: bool = False) -> Any:
    """Return a collision-free key component for a baked ``Final`` field value.

    Every value is turned into a *type-tagged* key component, because Python treats several distinct compile-time
    constants as equal (with equal hashes) and annotations are not enforced at runtime, so one ``Final`` field can
    receive any of them across launches. The encodings:

    - a ``float`` (builtin, a subclass, or a NumPy floating scalar) -> its exact IEEE-754 bits under
      ``_FLOAT_KEY_TAG`` (builtin ``(_FLOAT_KEY_TAG, <bits as int>)``; a subclass adds its class identity - see
      ``_subclass_identity``; a NumPy scalar uses ``(_FLOAT_KEY_TAG, <dtype str>, <raw bytes>)``). Bits (not value)
      so ``-0.0``/``0.0`` (equal, equal hash) and NaNs differing only in sign/payload (all ``str``-ed to ``"nan"``)
      stay distinct, and widths never alias.
    - an ``enum`` member -> ``(_ENUM_KEY_TAG, *class-identity, name, final_scalar_key(value))``. An ``IntEnum`` /
      ``StrEnum`` member is ``==`` to its bare scalar and to a same-valued member of another enum class (and
      ``str(member)`` is just the scalar on Python >=3.11), so keying on identity keeps them distinct. Both ``name``
      and the member value are kept: ``name`` (``None`` for an unnamed ``IntFlag`` composite) plus the value
      separates same-named members of two classes that share ``module``/``qualname`` (e.g. an enum rebuilt by a
      local factory). The value is itself run through ``final_scalar_key`` so a raw ``True`` vs ``1`` (``==``, equal
      hash) cannot collide and so an unsupported / mutable member value is rejected. A member carrying user-defined
      per-member state is rejected (identity alone cannot capture that state).
    - every remaining scalar (``bool`` / ``int`` / ``str`` and NumPy analogues) ->
      ``(_SCALAR_KEY_TAG, module, qualname, canonical-value)``. ``True == 1 == np.int64(1)`` with equal hashes, but
      they bake observably different Python constants (e.g. ``config.value is True``), so the exact type is tagged.
      A ``bool`` / ``int`` / ``str`` *subclass* is accepted only if it is behavior-free (see
      ``_reject_stateful_primitive_subclass``); its value is then coerced to the plain base type via a *base slot*
      (``int.__int__`` / ``str.__str__``, ``.item()`` for NumPy) so the offline-cache string (built by ``str``-ing
      this tuple) is faithful and process-stable even if a future/allowed subclass has a nonstandard ``repr``.

    A user *subclass* (of ``float``/``int``/``str`` or a NumPy scalar) or an ``enum`` is keyed by its class identity
    via ``_subclass_identity``, not just its value, since ``cfg.x.__class__`` is observable at compile time. ``live``
    selects the identity strategy: the in-process spec key (``live=True``) keys on a ``_ClassRef(cls)`` identity token
    so distinct classes never collide - even across a module reload or under a metaclass that makes two classes ``==``
    (or one with ``__hash__ = None``) - while its retained class object pins ``cls`` so its ``id`` cannot be recycled
    while the specialization is cached; the offline fastcache key (``live=False``, the default) uses a process-stable
    component instead (``None`` for a resolvable class, a per-process ``(nonce, serial)`` for a locally/dynamically
    created one, so it is a guaranteed cross-process miss rather than a possible wrong reuse). Annotations are not
    enforced at runtime, so a value that is none of the above (an arbitrary object, or a mutable container) is
    *rejected* with a clear ``TypeError`` rather than keyed by its own ``__eq__`` / ``__hash__``. Such an object could
    select the wrong specialization or change under the cached ``_qd_spec_key`` after first launch.

    This is off the steady-state hot path for Final-free dataclasses (which serve a cached ``_qd_spec_key`` /
    ``_qd_dc_repr`` and never reach here), so the ``isinstance`` probes cost nothing there. A dataclass whose subtree
    bakes a ``Final`` value deliberately does *not* cache (see ``subtree_has_final_fields``): it recomputes each launch
    so this validation re-runs, catching a value whose class turned behaviorful (e.g. a monkey-patched enum) after the
    first launch. So for Final-bearing configs this runs per launch by design, not once per instance.
    """
    if type(value) is float:
        return (_FLOAT_KEY_TAG, _unpack_u64(_pack_f64(value))[0])
    if isinstance(value, enum.Enum):
        # Checked before the ``float``/``int`` branches so a mixed-in enum (``IntEnum``/``StrEnum``, or an exotic
        # ``float`` mix-in) keys by identity, not by its value. A member with user-defined per-member state is
        # rejected, since identity alone would not capture that state. The key carries BOTH ``name`` and ``value``:
        # ``name`` identifies the canonical member (``None`` for an unnamed ``IntFlag`` composite). ``value`` is
        # routed through ``final_scalar_key`` itself, not embedded raw: two factory members named ``A`` valued
        # ``True`` vs ``1`` are ``==`` with equal hashes, so a raw value would still collide - recursing type-tags
        # them apart (and bit-encodes a float value, etc.). Recursing also rejects a mutable / unsupported member
        # value (e.g. a ``list``), which could otherwise change under the cached ``_qd_spec_key``.
        #
        # ``module``/``qualname``/name/value still do not uniquely identify the class *object*: two plain ``Enum``
        # classes built by a factory can share all four yet have distinct members (``First.A`` is not ``Second.A``,
        # and for a plain ``Enum`` ``First.A != Second.A``), and a module-level name can be rebound to a fresh class
        # by a reload - either way a kernel branching on ``cfg.mode == First.A`` needs distinct specializations.
        # ``_subclass_identity`` keys on a ``_ClassRef(cls)`` identity token in-process (so all such cases stay
        # distinct, even under a metaclass with a custom ``==`` / ``__hash__ = None``, and the retained class pins its
        # ``id``) and on a per-process ``(nonce, serial)`` / ``None`` offline. Every supported value encodes to a
        # hashable key, so the tuple stays hashable.
        _reject_stateful_enum_member(value)
        cls = type(value)
        return (_ENUM_KEY_TAG, *_subclass_identity(cls, live), value.name, final_scalar_key(value.value, live))
    if isinstance(value, np.floating):
        # ``dtype.str`` (e.g. ``"<f4"``) + ``tobytes()`` preserves sign bit, NaN payload and width, and stays in the
        # same tagged space as the builtin-float branch so it can never equal a bare int / str key component. An
        # exact NumPy float is a pure value keyed by dtype+bytes alone; a *user subclass* carrying state/behavior is
        # rejected here, and an accepted (behavior-free) subclass additionally carries its class identity so two
        # distinct same-named factory subclasses do not collide (as for the builtin-float / scalar branches below).
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        if _is_baked_base_type(cls):  # an exact NumPy float scalar (``np.float32``/``np.float64``/...)
            return (_FLOAT_KEY_TAG, value.dtype.str, value.tobytes())
        return (_FLOAT_KEY_TAG, *_subclass_identity(cls, live), value.dtype.str, value.tobytes())
    if isinstance(value, float):
        # A ``float`` subclass (the exact builtin ``float`` took the fast path above). Bit-encode like a plain float
        # so signed zeros stay distinct, but keep the subclass identity so it is not confused with a plain ``float``
        # of the same value (nor with another same-named factory subclass). A subclass carrying extra observable
        # state is not fully described by its numeric value, so it is rejected rather than silently mis-specialised.
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        return (_FLOAT_KEY_TAG, *_subclass_identity(cls, live), _unpack_u64(_pack_f64(value))[0])
    # ``bool`` / ``int`` / ``str`` and their NumPy analogues. The exact type completes the tag so value-equal but
    # distinct-typed constants (``True`` vs ``1``, ``np.int64(1)`` vs ``1``) never share a specialization. As with
    # float subclasses, an ``int`` / ``str`` subclass carrying extra observable state is rejected.
    _reject_stateful_primitive_subclass(value)
    cls = type(value)
    # Embed a *canonical* base value rather than the object itself: the offline cache key is built by ``str``-ing
    # this tuple (args_hasher), so a subclass (or NumPy scalar) whose ``__repr__`` does not preserve its value
    # (e.g. ``OddInt.__repr__`` returning a constant) would otherwise serialize two distinct values to one string.
    # Coercing to the plain base type keeps the in-process key correct and makes the string faithful and stable.
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
        # Annotations are not enforced at runtime, so a ``Final`` field - or an ``enum`` member's value, routed here
        # by the enum branch - can be an object of any type. Everything above is a supported baked primitive (or its
        # NumPy analogue) whose value fully determines the key. An arbitrary object is not: two instances that are
        # ``==`` with equal hashes but carry differing attributes would select the same specialization while
        # compile-time code (``qd.static(cfg.x.tag == 1)``) reads the difference, and a mutable value (e.g. a
        # ``list``) could even change under the cached ``_qd_spec_key``. Reject rather than let an unsupported
        # object's ``__eq__`` / ``__hash__`` control key equality.
        raise TypeError(
            f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, which is not a supported "
            f"compile-time constant. A ``Final`` value - or an ``enum`` member's underlying value - must be a "
            f"``bool`` / ``int`` / ``float`` / ``str`` (or a NumPy scalar analogue), or an ``enum`` member; an "
            f"arbitrary object cannot be keyed by value alone. Bake the specific scalar you need as a ``Final`` "
            f"field instead."
        )
    if _is_baked_base_type(cls):  # exact builtin primitive or exact NumPy scalar
        return (_SCALAR_KEY_TAG, cls.__module__, cls.__qualname__, canonical)
    # A behavior-free user subclass (``class Grams(int): pass``): ``module``/``qualname`` do not uniquely identify the
    # class object, so ``_subclass_identity`` carries a ``_ClassRef(cls)`` token (in-process) or a per-process
    # ``(nonce, serial)`` / None (offline) - two distinct same-named subclasses (``cfg.x.__class__ is First``) then key
    # apart, matching the enum and float-subclass branches above.
    return (_SCALAR_KEY_TAG, *_subclass_identity(cls, live), canonical)
