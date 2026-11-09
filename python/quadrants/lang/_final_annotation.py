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


def unwrap_and_parse_string_constants(ast_tree: ast.expr) -> ast.expr:
    """Unwrap annotations that contain an extra layer of quotes.

    For example, with postponed annotations:

    ```python
    from __future__ import annotations

    class Config:
        value: "Final[int]"
    ```

    The stored annotation can be the string "'Final[int]'". Parsing it once produces:

    ```text
    Constant(value="Final[int]")
    ```

    The loop parses that constant's string value again, producing the actual annotation tree:

    ```text
    Subscript
    ├── value: Name("Final")
    └── slice: Name("int")
    ```

    It repeats while another string layer remains. No expression is evaluated.
    Non-string nodes pass through unchanged. Invalid syntax inside a string raises SyntaxError.
    """
    unwrapped_node = ast_tree
    while isinstance(unwrapped_node, ast.Constant) and isinstance(unwrapped_node.value, str):
        unwrapped_node = ast.parse(unwrapped_node.value, mode="eval").body
    return unwrapped_node


def dataclass_field_is_final(dataclass_type: type, field_name: str, annotation: Any) -> bool:
    """Recognize a field's Final marker, including when the annotation is a string.

    Can you give an overview of the process we are following? Is it something like:
    - convert the AST back into the path of strings, a.b.Final
    - somehow look this up in the module, and see whether it 'is' Final?
        Yes. Parse the string, discard an outer subscript, and extract ["a", "b", "Final"]. Resolve a in the
        declaring class, its bases, or its defining module, then follow b and Final. Check the resulting object.
        If the first name is unavailable, the canonical spellings Final, typing.Final, and typing_extensions.Final
        are accepted as a fallback.

    Why do we need to parse into an AST? Is it because we are assuming the input to this function is a string?
        Only string annotations take this path. Actual typing objects go directly to is_final_annotation.
        The AST describes the string's Python syntax without evaluating it.

    But if we already have a string, why can't we do simply input_string.split()?
        Splitting on dots works for "typing.Final", but annotations can contain parentheses, whitespace,
        quoted strings, and nested subscripts. The AST handles those forms without a custom string parser.

    Does Python not already provide us functions to do this stuff for us?
    What do other people use for reading string annotations?
        The standard options include typing.get_type_hints and inspect.get_annotations(eval_str=True).
        Python 3.14 adds annotationlib.get_annotations. These do not directly provide the operation we need:
        resolve just the outer marker without resolving or evaluating its inner annotation.

    Why can't we resolve the whole annotation in a standard way, then ignore the type inside Final?
        An inner type imported only under TYPE_CHECKING is unavailable at runtime:

        if TYPE_CHECKING:
            from somewhere import SomeType

        Resolving "Final[SomeType]" would fail before we could discard SomeType, even if the arriving value
        is a supported constant. Resolving only Final avoids that failure and does not execute expressions
        inside the ignored brackets. The arriving value supplies the type instead.

    What if we removed anything inside pairs of square brackets first? Like remove something like '\\[.*\\]'?
        That greedy regex turns "Final[int] | list[str]" into "Final", incorrectly recognizing a union as a
        Final field. The AST lets us remove a subscript only when it is the outermost expression.

    So AST parsing is the easier way to do this?
        Yes. Python handles the syntax, and we inspect the outer expression without evaluating its contents.
        Resolving names and aliases in the correct class or module is a separate step.
    """
    if not isinstance(annotation, str):
        return is_final_annotation(annotation)
    # Catch syntax errors from both parsing calls: the initial parse and parsing unwrapped string constants.
    try:
        ast_tree = ast.parse(annotation, mode="eval").body
        annotation_node = unwrap_and_parse_string_constants(ast_tree)
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
