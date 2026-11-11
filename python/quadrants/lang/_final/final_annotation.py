"""Recognize Final markers without resolving or evaluating their ignored inner annotations."""

import ast
import sys
import types
import typing
from typing import Any

from quadrants.lang._name_utils import extract_dotted_name_parts

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
    |-- value: Name("Final")
    `-- slice: Name("int")
    ```

    It repeats while another string layer remains. No expression is evaluated.
    Non-string nodes pass through unchanged. Invalid syntax inside a string raises SyntaxError.
    """
    unwrapped_node = ast_tree
    while isinstance(unwrapped_node, ast.Constant) and isinstance(unwrapped_node.value, str):
        unwrapped_node = ast.parse(unwrapped_node.value, mode="eval").body
    return unwrapped_node


def strip_annotation_subscript(annotation_node: ast.expr) -> ast.expr:
    """Return the outer subscript's value node, or the input node unchanged if it is not a subscript.

    Input: the expression tree for "Final[int]".
    Subscript
    |-- value: Name("Final")  <- keep this node
    `-- slice: Name("int")    <- discard the inner annotation

    Output: Name("Final") in this example. The input tree is not modified.
    """
    if isinstance(annotation_node, ast.Subscript):
        return annotation_node.value
    return annotation_node


def find_field_declaring_dataclass(dataclass_type: type, field_name: str) -> type:
    """Return the class declaring the field's annotation, or dataclass_type if no declaration is found."""
    # An inherited field may have its annotation declared on a base class rather than dataclass_type itself.
    return next(
        (cls for cls in dataclass_type.__mro__ if field_name in vars(cls).get("__annotations__", {})), dataclass_type
    )


def lookup_name_in_class_hierarchy(declaring_dataclass: type, name: str) -> object:
    """Return a name's stored value from the declaring class or its bases, or _MISSING if absent."""
    # Searching from dataclass_type could pick up a subclass's override:
    #
    # @dataclasses.dataclass
    # class Base:
    #     Marker = typing.Final
    #     value: "Marker[int]"
    #
    # @dataclasses.dataclass
    # class Child(Base):
    #     Marker = int
    #
    # For Child(7), the field's annotation was declared in Base.
    #
    # - Searching from Child finds Marker = int.
    # - Searching from Base finds Marker = typing.Final.
    for cls in declaring_dataclass.__mro__:
        value = vars(cls).get(name, _MISSING)
        if value is not _MISSING:
            return value
    return _MISSING


def lookup_name_in_defining_module(declaring_dataclass: type, name: str) -> object:
    """Return a name's value from the declaring class's module, or _MISSING if the name or module is absent."""
    # vars(Config) contains only attributes stored directly on Config. It doesn't include names from the
    # surrounding module.
    #
    # import typing
    #
    # class Config:
    #     local_alias = typing.Final
    #
    # vars(Config)["local_alias"]  # typing.Final
    # vars(Config)["typing"]       # KeyError
    #
    # Python can use the module-level name typing while executing the class body, but that doesn't make it
    # a class attribute. Only the assignment to local_alias creates one here.
    module = sys.modules.get(declaring_dataclass.__module__)
    return vars(module).get(name, _MISSING) if module is not None else _MISSING


def is_canonical_final_name(name_parts: list[str]) -> bool:
    """Recognize the three canonical Final spellings used when the first name cannot be resolved."""
    # If the first name could not be found in either the class or its module, recognize these spellings by name:
    #
    # "Final"
    # "typing.Final"
    # "typing_extensions.Final"
    #
    # This supports cases such as a function-local import:
    #
    # def make_config():
    #     from typing import Final
    #
    #     @dataclasses.dataclass(frozen=True)
    #     class Config:
    #         value: "Final[int]"
    #
    #     return Config(7)
    #
    # Later, that local Final binding is not available through Config or its module.
    # The fallback nevertheless treats "Final" as the marker.
    return name_parts in (["Final"], ["typing", "Final"], ["typing_extensions", "Final"])


def resolve_attribute_path(root_object: object, attribute_names: list[str]) -> object:
    """Follow stored attributes through modules/classes; return _MISSING for absent or unsupported paths."""
    value = root_object
    for name in attribute_names:
        if not isinstance(value, (types.ModuleType, type)):
            return _MISSING
        value = vars(value).get(name, _MISSING)
    return value


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
        unwrapped_node = unwrap_and_parse_string_constants(ast_tree)
    except SyntaxError:
        return False
    annotation_node = strip_annotation_subscript(unwrapped_node)
    name_parts = extract_dotted_name_parts(annotation_node)
    if name_parts is None:
        return False
    declaring_dataclass = find_field_declaring_dataclass(dataclass_type, field_name)
    root_object = lookup_name_in_class_hierarchy(declaring_dataclass, name_parts[0])
    if root_object is _MISSING:
        root_object = lookup_name_in_defining_module(declaring_dataclass, name_parts[0])
    if root_object is _MISSING:
        return is_canonical_final_name(name_parts)
    resolved_annotation = resolve_attribute_path(root_object, name_parts[1:])
    return is_final_annotation(resolved_annotation)
