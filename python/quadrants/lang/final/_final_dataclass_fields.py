"""Discover and validate Final dataclass fields and their containers."""

import dataclasses
import sys
import typing
from typing import Any

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
