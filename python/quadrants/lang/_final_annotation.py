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
    """Recognize a field's Final marker, including when the annotation is a string.

    Why do we read annotations if member types come from runtime values?
        We read only the Final marker, which makes a member a compile-time constant. Its inner type is ignored.

    Why does this need the dataclass type and field name?
        A string can use aliases defined in the class or its module. An inherited field must use the namespace
        of the class that declared it. Non-string annotations go directly to is_final_annotation instead.

    Why not resolve the whole annotation with Python's standard annotation helpers, then ignore its arguments?
        That works when every name resolves, but an inner type may be imported only under TYPE_CHECKING and
        therefore be unavailable at runtime. For example, "Final[SomeType]" can describe an arriving integer
        even when SomeType is unavailable. We must recognize Final without resolving or evaluating SomeType.
        Resolving the whole expression could also execute code inside the ignored brackets.

    Why parse a string into an AST instead of splitting it?
        An AST describes Python syntax without evaluating it. It handles whitespace, parentheses, quoted
        strings, and nested subscripts. For "Final[int]", we keep the outer subscript's value, Name("Final").
        For "module.typing.Final", we extract the dotted name. Both paths produce a list of name components.

    Why not first remove square brackets with a regex such as r"\\[.*\\]"?
        That greedy pattern would turn "Final[int] | list[str]" into "Final", incorrectly accepting a union.
        The AST lets us strip a subscript only when it is the outermost expression, without writing a parser.

    Why keep the full dotted name rather than just the string "Final"?
        some_module.Final might be unrelated to typing.Final. Conversely, an alias such as Constant can refer
        to the real marker. We resolve the actual object before testing whether it is Final or Final[T].

    What are annotation_node and parts used for?
        annotation_node tracks our current position in the syntax tree. parts accumulates the names, initially
        in reverse order. For module.typing.Final, the completed list is ["module", "typing", "Final"].
        The AST stores "Final" as an Attribute.attr string here, not as a Name("Final") node.
        We resolve module in the declaring class or module namespace, then follow its typing and Final attributes.
        If the first name is unavailable, the canonical spellings Final, typing.Final, and typing_extensions.Final
        are accepted as a fallback. The function's final output is a boolean, not the node or list of names.
    """
    if not isinstance(annotation, str):
        return is_final_annotation(annotation)
    try:
        annotation_node = ast.parse(annotation, mode="eval").body
        while isinstance(annotation_node, ast.Constant) and isinstance(annotation_node.value, str):
            annotation_node = ast.parse(annotation_node.value, mode="eval").body
    except SyntaxError:
        return False
    # "Final[int]" parses as:
    # Subscript
    # |-- value: Name("Final")  <- keep this node
    # `-- slice: Name("int")    <- discard the inner annotation
    # After this if, annotation_node is Name("Final") in this example.
    if isinstance(annotation_node, ast.Subscript):
        annotation_node = annotation_node.value
    # "module.typing.Final" parses as:
    # Attribute
    # |-- value: Attribute
    # |   |-- value: Name("module")
    # |   `-- attr: "typing"           <- a string
    # `-- attr: "Final"                <- a string
    # The loop collects "Final", then "typing", and stops at Name("module"). Below, we append "module"
    # and reverse the list to get ["module", "typing", "Final"]. This lets us look up module first,
    # then its typing attribute, then that object's Final attribute.
    parts = []
    while isinstance(annotation_node, ast.Attribute):
        parts.append(annotation_node.attr)
        annotation_node = annotation_node.value
    if not isinstance(annotation_node, ast.Name):
        return False
    parts.append(annotation_node.id)
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
