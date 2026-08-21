"""Helpers for the ``@dataclasses.dataclass`` kernel-arg path, including ``typing.Final[T]`` compile-time template
fields.

PERF NOTE: everything about ``Final`` resolution - including the validation that rejects mutable carriers - is
computed **once per dataclass type** and cached (``_final_plan_cache``, ``_final_path_cache``). Callers on the
per-launch hot path (``_extract_arg``, ``args_hasher.dataclass_to_repr``) do a
single ``dict.get`` keyed on the dataclass type and then, in the overwhelmingly common no-Final-field case, take a
branch that is byte-for-byte the pre-existing code path. No ``isinstance`` / ``typing.get_origin`` /
``dataclasses.fields`` call happens per launch. See the module docstring of ``_template_mapper_hotpath.py`` for why
that matters (``isinstance`` is a ~100-200ns MRO walk vs a ~10ns pointer comparison for ``type(x) is Y``).
"""

import dataclasses
import enum
import struct
import typing
from typing import Any

import numpy as np

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


# Memo of ``dataclass type -> field path down to some Final leaf`` (``None`` if the subtree holds none), used to reject
# mutable ancestors. Same once-per-type lifecycle as ``_final_plan_cache``; never consulted per launch.
_final_path_cache: "dict[type, tuple[str, ...] | None]" = {}


def _first_final_path(dc_type: type, visiting: "frozenset[type]") -> "tuple[str, ...] | None":
    """Return the field path from ``dc_type`` down to some ``Final`` leaf, or None if the subtree contains none.

    Only a witness is needed (to name one offending path in the error), so the first hit wins. Recursing through
    ``final_field_names`` means every nested type is validated eagerly at the top-level call rather than lazily when
    the hot path first walks that far down.

    ``visiting`` guards against a self-referential type graph. Quadrants cannot actually lower a recursive dataclass
    kernel arg, but validation must fail with a real error rather than a ``RecursionError``.
    """
    if dc_type in _final_path_cache:
        return _final_path_cache[dc_type]
    if dc_type in visiting:
        return None
    direct = final_field_names(dc_type)
    path: "tuple[str, ...] | None" = (min(direct),) if direct else None
    if path is None:
        visiting = visiting | {dc_type}
        for field in dataclasses.fields(dc_type):
            if isinstance(field.type, type) and dataclasses.is_dataclass(field.type):
                child = _first_final_path(field.type, visiting)
                if child is not None:
                    path = (field.name,) + child
                    break
    _final_path_cache[dc_type] = path
    return path


def _build_final_plan(dc_type: type) -> "frozenset[str]":
    """Validate every ``Final`` field on ``dc_type`` and return the set of Final-annotated field names.

    Called once per dataclass type (memoised in ``_final_plan_cache``), so all of the reflection here - including
    ``dataclasses.fields``, ``typing.get_origin`` and ``issubclass`` - stays entirely off the per-launch hot path.
    """
    final_names = []
    for field in dataclasses.fields(dc_type):
        annotation = field.type
        if isinstance(annotation, str):
            # ``from __future__ import annotations`` (or an explicit string annotation) leaves ``field.type`` as an
            # unresolved string. The pre-existing dataclass kernel-arg path already assumes resolved types, so rather
            # than half-supporting it, flag the one case where silently ignoring it would be a correctness trap: a
            # field the user believes is a compile-time constant but which we would lower as a runtime arg.
            if "Final" in annotation:
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
            nested = _first_final_path(field.type, frozenset({dc_type}))
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


# Memo of ``dataclass type -> frozenset of Final field names``. Keyed on the type object, so it is bounded by the
# number of distinct dataclass types the process ever passes to a kernel. An empty frozenset (the common case) is a
# meaningful cached result, so callers must distinguish it from a cache miss via ``.get(...) is None``.
_final_plan_cache: "dict[type, frozenset[str]]" = {}


def final_field_names(dc_type: Any) -> "frozenset[str]":
    """Return the cached set of ``Final``-annotated field names on ``dc_type``, validating on first sighting.

    Hot-path contract: one ``dict.get``. Callers should short-circuit on the empty result so that dataclasses with no
    ``Final`` fields (the overwhelmingly common case) run the pre-existing code path untouched.

    ``dc_type`` is typed ``Any`` rather than ``type`` because ``_extract_arg`` calls this with its loosely-typed
    ``annotation`` parameter (a union covering every kernel-arg annotation shape), having already established that it
    is a dataclass type via the ``__dataclass_fields__`` probe. Narrowing at that call site would need a
    ``typing.cast``, which is a real function call on a per-launch path.
    """
    names = _final_plan_cache.get(dc_type)
    if names is None:
        names = _build_final_plan(dc_type)
        _final_plan_cache[dc_type] = names
    return names


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


def _reject_stateful_primitive_subclass(value: Any) -> None:
    """Reject a ``float`` / ``int`` / ``str`` *subclass* instance that carries observable state a kernel could read
    but the key cannot capture - either per-instance state or class-level behavior/state.

    A ``Final`` value is baked as a compile-time literal keyed by its (typed) value plus the subclass
    ``module``/``qualname``. That is not enough when a subclass carries more than its value:

    - *Per-instance* state - e.g. ``class TaggedFloat(float)`` with a ``unit`` attribute - is not described by the
      numeric value, so two instances with an equal value but different state would bake different kernels yet
      select the same specialization. State can live in ``__dict__`` or a populated ``__slots__`` slot.
    - *Class-level* behavior/state - e.g. a factory returning ``float`` subclasses whose ``unit`` property closes
      over different values - is not captured either: ``module``/``qualname`` does not uniquely identify a
      dynamically created class, so two distinct same-named subclasses (whose ``cfg.x.unit`` a kernel could read)
      would collide. Overriding a value-conversion / repr dunder (``__int__``, ``__repr__``, ...) is fine - the key
      is built from a base slot, never those - so only *non-dunder* class attributes count.

    There is no bounded, process-stable way to serialise arbitrary state/behavior, so we reject rather than silently
    mis-specialise. Exact primitives, NumPy scalars (library internals, not user state) and behavior-free stateless
    subclasses (``class Meters(float): pass``) are unaffected. Runs once per instance, off the steady-state path.
    """
    if type(value) in _EXACT_BAKED_TYPES or not isinstance(value, (int, float, str)):
        return
    if isinstance(value, np.generic):  # NumPy scalar (e.g. ``np.str_``); its class attrs are library internals
        return
    cls = type(value)
    if getattr(value, "__dict__", None):
        stateful = True
    else:
        stateful = False
        for klass in type(value).__mro__:
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
    for klass in cls.__mro__:  # subclass chain above the base primitive; stop at the base (its own attrs are fine)
        if klass in _EXACT_BAKED_TYPES or klass is object:
            break
        for attr in vars(klass):
            # Dunders (incl. handled conversion overrides like __int__ / __repr__) are not class-level state.
            if not (attr.startswith("__") and attr.endswith("__")):
                raise TypeError(
                    f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}, a subclass of a baked "
                    f"primitive that defines observable class-level behavior/state (e.g. attribute {attr!r}). "
                    f"The key identifies the subclass only by ``module``/``qualname``, which does not uniquely "
                    f"identify a dynamically created class, so two distinct same-named subclasses (e.g. from a "
                    f"factory) whose ``{attr}`` a kernel could read at compile time would select the same "
                    f"specialization. Pass a plain ``bool`` / ``int`` / ``float`` / ``str`` (or an ``enum`` "
                    f"member) instead."
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


def _reject_stateful_enum_member(value: Any) -> None:
    """Reject an ``enum`` member carrying user-defined per-member state, for the same reason as
    ``_reject_stateful_primitive_subclass``: the key records only the member identity, so per-member state a kernel
    can read at compile time (``qd.static(cfg.mode.unit == "m")``) - or that differs across processes for the
    offline key - would not select a distinct specialization. Plain enums (and unnamed ``IntFlag`` composites)
    carry only name/value bookkeeping and are unaffected. Runs once per instance, off the steady-state launch path.
    """
    extra = _enum_member_state_attr(value)
    if extra is None:
        return
    cls = type(value)
    raise TypeError(
        f"A ``Final`` field received {cls.__module__}.{cls.__qualname__}.{value.name}, an ``enum`` member with "
        f"user-defined per-member state (e.g. attribute {extra!r}). A ``Final`` value is baked as a compile-time "
        f"literal keyed by member identity, so per-member state a kernel could read (e.g. ``cfg.mode.unit``) "
        f"would not select a distinct specialization. Use a plain ``enum`` (state-free members), or bake the "
        f"needed value as a separate ``Final`` field."
    )


def final_scalar_key(value: Any) -> Any:
    """Return a process-stable, collision-free key component for a baked ``Final`` field value.

    Every value is turned into a *type-tagged* key component, because Python treats several distinct compile-time
    constants as equal (with equal hashes) and annotations are not enforced at runtime, so one ``Final`` field can
    receive any of them across launches. The encodings:

    - a ``float`` (builtin, a subclass, or a NumPy floating scalar) -> its exact IEEE-754 bits under
      ``_FLOAT_KEY_TAG`` (builtin ``(_FLOAT_KEY_TAG, <bits as int>)``; a subclass adds its ``module``/``qualname``;
      a NumPy scalar uses ``(_FLOAT_KEY_TAG, <dtype str>, <raw bytes>)``). Bits (not value) so ``-0.0``/``0.0``
      (equal, equal hash) and NaNs differing only in sign/payload (all ``str``-ed to ``"nan"``) stay distinct, and
      widths never alias.
    - an ``enum`` member -> ``(_ENUM_KEY_TAG, module, qualname, name, final_scalar_key(value))``. An ``IntEnum`` /
      ``StrEnum`` member is ``==`` to its bare scalar and to a same-valued member of another enum class (and
      ``str(member)`` is just the scalar on Python >=3.11), so keying on identity keeps them distinct. Both ``name``
      and the member value are kept: ``name`` (``None`` for an unnamed ``IntFlag`` composite) plus the value
      separates same-named members of two classes that share ``module``/``qualname`` (e.g. an enum rebuilt by a
      local factory). The value is itself run through ``final_scalar_key`` so a raw ``True`` vs ``1`` (``==``, equal
      hash) cannot collide and so an unsupported / mutable member value is rejected. A member carrying user-defined
      per-member state is rejected (identity alone cannot capture that state).
    - every remaining scalar (``bool`` / ``int`` / ``str`` and NumPy analogues) ->
      ``(_SCALAR_KEY_TAG, module, qualname, canonical-value)``. ``True == 1 == np.int64(1)`` with equal hashes, but
      they bake observably different Python constants (e.g. ``config.value is True``), so the exact type is tagged;
      the value is coerced to its plain base type via a *base slot* (``int.__int__`` / ``str.__str__``, ``.item()``
      for NumPy), never the subclass's own dunder, so a subclass with a misleading ``__int__`` / ``__str__`` /
      ``__repr__`` cannot collapse two distinct values to one key (in-process or in the offline-cache string).

    Annotations are not enforced at runtime, so a value that is none of the above (an arbitrary object, or a mutable
    container) is *rejected* with a clear ``TypeError`` rather than keyed by its own ``__eq__`` / ``__hash__``. Such
    an object could select the wrong specialization or change under the cached ``_qd_spec_key`` after first launch.

    Everything here runs once per instance (Final keys/reprs are cached), never on the steady-state launch path, so
    the ``isinstance`` probes are off the hot path.
    """
    if type(value) is float:
        return (_FLOAT_KEY_TAG, _unpack_u64(_pack_f64(value))[0])
    if isinstance(value, enum.Enum):
        # Checked before the ``float``/``int`` branches so a mixed-in enum (``IntEnum``/``StrEnum``, or an exotic
        # ``float`` mix-in) keys by identity, not by its value. A member with user-defined per-member state is
        # rejected, since identity alone would not capture that state. The key carries BOTH ``name`` and ``value``:
        # ``name`` identifies the canonical member (``None`` for an unnamed ``IntFlag`` composite), while ``value``
        # separates same-named members of two classes that share ``module``/``qualname`` - e.g. an enum rebuilt by a
        # local factory, whose qualname is ``<factory>.<locals>.Local``. The value is routed through
        # ``final_scalar_key`` itself, not embedded raw: two factory members named ``A`` valued ``True`` vs ``1``
        # are ``==`` with equal hashes, so a raw value would still collide - recursing type-tags them apart (and
        # bit-encodes a float value, etc.). Recursing also rejects a mutable / unsupported member value (e.g. a
        # ``list``): such a value could otherwise change under the cached ``_qd_spec_key`` after first launch and
        # so must not be accepted at all. Every supported value encodes to a hashable key, so the tuple stays
        # hashable.
        _reject_stateful_enum_member(value)
        cls = type(value)
        return (_ENUM_KEY_TAG, cls.__module__, cls.__qualname__, value.name, final_scalar_key(value.value))
    if isinstance(value, np.floating):
        # ``dtype.str`` (e.g. ``"<f4"``) + ``tobytes()`` preserves sign bit, NaN payload and width, and stays in the
        # same tagged space as the builtin-float branch so it can never equal a bare int / str key component.
        return (_FLOAT_KEY_TAG, value.dtype.str, value.tobytes())
    if isinstance(value, float):
        # A ``float`` subclass (the exact builtin ``float`` took the fast path above). Bit-encode like a plain float
        # so signed zeros stay distinct, but keep the subclass ``module``/``qualname`` so it is not confused with a
        # plain ``float`` of the same value. A subclass carrying extra observable state is not fully described by
        # its numeric value, so it is rejected rather than silently mis-specialised.
        _reject_stateful_primitive_subclass(value)
        cls = type(value)
        return (_FLOAT_KEY_TAG, cls.__module__, cls.__qualname__, _unpack_u64(_pack_f64(value))[0])
    # ``bool`` / ``int`` / ``str`` and their NumPy analogues. The exact type completes the tag so value-equal but
    # distinct-typed constants (``True`` vs ``1``, ``np.int64(1)`` vs ``1``) never share a specialization. As with
    # float subclasses, an ``int`` / ``str`` subclass carrying extra observable state is rejected.
    _reject_stateful_primitive_subclass(value)
    cls = type(value)
    # Embed a *canonical* base value rather than the object itself: the offline cache key is built by ``str``-ing
    # this tuple (args_hasher), so a subclass (or NumPy scalar) whose ``__repr__`` does not preserve its value
    # (e.g. ``OddInt.__repr__`` returning a constant) would otherwise serialize two distinct values to one string.
    # Coercing to the plain base type keeps the in-process key correct and makes the string faithful and stable.
    if cls in _EXACT_BAKED_TYPES:
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
    return (_SCALAR_KEY_TAG, cls.__module__, cls.__qualname__, canonical)
