"""Reject Final enum members whose observable state or behavior cannot be captured by their keys."""

import enum
from typing import Any

from quadrants.lang._final.final_class_identity import (
    _STRUCTURAL_CLASS_ATTRS,
    _is_baked_base_type,
    _observable_metaclass_attr,
    _slot_names,
)

_MISSING = object()  # Distinguish missing enum attributes from attributes whose value is None.

# Allowed enum bookkeeping. Other attributes, including user-defined dunders, count as state.
_ENUM_INTERNAL_MEMBER_ATTRS = frozenset(
    {"_name_", "_value_", "_sort_order_", "_inverted_", "__objclass__", "__dict__", "__weakref__"}
)


def _enum_member_state_attr(value: Any) -> "str | None":
    """Return the first unsupported enum member attribute name, or None."""
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

# Potential behavior overrides; inherited methods copied by Python are filtered out.
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


def _dunder_copied_from_base(klass: type, name: str, value: Any) -> bool:
    """Check whether a class attribute is the same object as an attribute on a base class."""
    for base in klass.__mro__[1:]:
        if base.__dict__.get(name, _MISSING) is value:
            return True
    return False


def _enum_probe_classes() -> "list[type]":
    """Create sample enums that reveal which attributes Python generates.

    Probe direct int/str mixins too: their generated hooks can differ from those on IntEnum or StrEnum.
    """

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
    """Check whether an enum attribute matches an inherited default.

    Compare unwrapped callables because Python can copy an inherited hook into a fresh wrapper.
    """
    target = getattr(member, "__func__", member)
    for base in klass.__mro__[1:]:
        base_member = vars(base).get(name, _MISSING)
        if base_member is not _MISSING:
            return getattr(base_member, "__func__", base_member) is target
    return False


def _enum_machinery_class_dict(cls: type) -> "dict[str, Any] | None":
    """Rebuild an enum without members to obtain its generated attributes, or return None."""
    mcls = type(cls)
    try:
        # Preserve the enum metaclass's namespace object; converting it to a plain dict would break construction.
        namespace = mcls.__prepare__("_qd_enum_probe", cls.__bases__)
        reference = mcls("_qd_enum_probe", cls.__bases__, namespace)  # pyright: ignore[reportArgumentType]
    except Exception:  # pylint: disable=broad-except  # any build failure -> conservative allowlist fallback
        return None
    return dict(vars(reference))


def _compute_enum_generated_class_attrs() -> "frozenset[str]":
    """Collect attribute names generated for the sample enum classes.

    Used as a fallback when a member-free reference enum cannot be constructed.
    """

    class _Plain:
        pass

    def _generated(probe: type) -> "set[str]":
        # Keep enum-valued machinery sunders (``_boundary_``); exclude members/aliases (enum-valued, plain name).
        """Collect generated enum attributes, excluding ordinary members and aliases."""
        return {
            n for n, v in vars(probe).items() if not isinstance(v, enum.Enum) or (n.startswith("_") and n.endswith("_"))
        }

    names: "set[str]" = _generated(_Plain)
    for probe in _enum_probe_classes():
        names.update(_generated(probe))
    return frozenset(names)


def _compute_enum_reference_checked_attrs() -> "frozenset[str]":
    """Find enum attributes whose values can be checked against a member-free reference.

    Exclude member-derived bookkeeping: it can legitimately differ between an enum and its member-free reference.
    """
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

# If a reference cannot be built, allow these hooks only when they match the inherited default.
_ENUM_OVERRIDABLE_HOOK_NAMES = frozenset({"_generate_next_value_"})


def _generated_attr_is_user_override(klass: type, name: str, member: Any, reference: "dict[str, Any] | None") -> bool:
    """Check whether an enum attribute overrides its generated default."""
    if reference is None:
        return name in _ENUM_OVERRIDABLE_HOOK_NAMES and not _dict_entry_is_inherited_default(klass, name, member)
    default = reference.get(name, _MISSING)
    if default is _MISSING:
        return False
    return getattr(member, "__func__", member) is not getattr(default, "__func__", default)


def _is_official_enum_member(klass: type, name: str, member: Any) -> bool:
    """Check whether an attribute is a registered enum member or alias."""
    member_map = klass.__dict__.get("_member_map_")
    return isinstance(member_map, dict) and member_map.get(name, _MISSING) is member


def _observable_class_dict_attr(klass: type) -> "str | None":
    """Return the first unsupported enum class attribute name, or None.

    Compare against this enum's reference. An attribute generated for another enum kind must not exempt a user attribute
    here.
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
                # Check hooks against their generated defaults.
                if name in _ENUM_REFERENCE_CHECKED_ATTRS and _generated_attr_is_user_override(
                    klass, name, member, reference
                ):
                    return name
                # Member-derived bookkeeping is already represented by the member-map key.
                # Direct mutation of private enum internals is unsupported.
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
    """Find unsupported attributes on an enum, its bases, or its metaclass."""
    for klass in enum_cls.__mro__:
        # Skip library bases but inspect all user bases, including non-enum mixins.
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
    """Reject enum members with unsupported instance state or class behavior."""
    cls = type(value)
    # Reject state on sibling members too: compiled code can access them through the selected member's class.
    # Include the selected value separately for unnamed IntFlag composites.
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
