"""Recognize Final markers without resolving or evaluating their ignored inner annotations."""

import ast
import sys
import types
import typing
from typing import Any

_MISSING = object()


def is_final_annotation(annotation: Any) -> bool:
    """Recognize bare Final and Final[T]; typing_extensions uses the same marker on supported Python versions."""
    return annotation is typing.Final or typing.get_origin(annotation) is typing.Final


def dataclass_field_is_final(dataclass_type: type, field_name: str, annotation: Any) -> bool:
    """Recognize a field's Final marker, including when the annotation is a string."""
    if not isinstance(annotation, str):
        return is_final_annotation(annotation)
    try:
        node = ast.parse(annotation, mode="eval").body
        while isinstance(node, ast.Constant) and isinstance(node.value, str):
            node = ast.parse(node.value, mode="eval").body
    except SyntaxError:
        return False
    if isinstance(node, ast.Subscript):
        node = node.value
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return False
    parts.append(node.id)
    parts.reverse()
    owner = next(
        (cls for cls in dataclass_type.__mro__ if field_name in vars(cls).get("__annotations__", {})), dataclass_type
    )
    value = _MISSING
    for cls in owner.__mro__:
        value = vars(cls).get(parts[0], _MISSING)
        if value is not _MISSING:
            break
    if value is _MISSING:
        module = sys.modules.get(owner.__module__)
        value = vars(module).get(parts[0], _MISSING) if module is not None else _MISSING
    if value is _MISSING:
        # Canonical spellings can also occur in function-local classes whose local imports are no longer available.
        return parts in (["Final"], ["typing", "Final"], ["typing_extensions", "Final"])
    for part in parts[1:]:
        if not isinstance(value, (types.ModuleType, type)):
            return False
        value = vars(value).get(part, _MISSING)
    return is_final_annotation(value)
