import ast
import dataclasses
import inspect
from typing import Any

from quadrants.lang import util
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang.ast import (
    ASTTransformerFuncContext,
)
from quadrants.lang.kernel_arguments import ArgMetadata


def _populate_struct_locals_from_params_dict(basename: str, struct_locals, struct_type) -> None:
    """
    We are populating struct locals from a type included in function parameters, or one of their subtypes

    struct_locals will be a list of all possible unpacked variable names we can form from the struct.
    basename is used to take into account the parent struct's name. For example, lets say we have:

    @dataclasses.dataclass
    class StructAB:
        a:
        b:
        struct_cd: StructCD

    @dataclasses.dataclass
    class StructCD:
        c:
        d:
        struct_ef: StructEF

    @dataclasses.dataclass
    class StructEF:
        e:
        f:

    ... and the function parameters look like: `def foo(struct_ab: StructAB)`

    then all possible variables we could form from this are:
    - struct_ab.a
    - struct_ab.b
    - struct_ab.struct_cd.c
    - struct_ab.struct_cd.d
    - struct_ab.struct_cd.strucdt_ef.e
    - struct_ab.struct_cd.strucdt_ef.f

    And the members of struct_locals should be:
    - __qd_struct_ab__qd_a
    - __qd_struct_ab__qd_b
    - __qd_struct_ab__qd_struct_cd__qd_c
    - __qd_struct_ab__qd_struct_cd__qd_d
    - __qd_struct_ab__qd_struct_cd__qd_struct_ef__qd_e
    - __qd_struct_ab__qd_struct_cd__qd_struct_ef__qd_f
    """
    for field in dataclasses.fields(struct_type):
        child_name = create_flat_name(basename, field.name)
        if dataclasses.is_dataclass(field.type):
            _populate_struct_locals_from_params_dict(child_name, struct_locals, field.type)
        else:
            struct_locals.add(child_name)


def extract_struct_locals_from_context(ctx: ASTTransformerFuncContext) -> set[str]:
    """
    Provides meta information for later tarnsformation of nodes in AST

    - Uses ctx.func.func to get the function signature.
    - Searches this for any dataclasses:
      - If it finds any dataclasses, then converts them into expanded names.
      - E.g. my_struct: MyStruct, and MyStruct contains a, b, c would become:
          {"__qd_my_struct_a", "__qd_my_struct_b, "__qd_my_struct_c"}
    """
    struct_locals = set()
    assert ctx.func is not None
    sig = inspect.signature(ctx.func.func)
    parameters = sig.parameters
    for param_name, parameter in parameters.items():
        if dataclasses.is_dataclass(parameter.annotation):
            for field in dataclasses.fields(parameter.annotation):
                child_name = create_flat_name(param_name, field.name)
                if dataclasses.is_dataclass(field.type):
                    _populate_struct_locals_from_params_dict(child_name, struct_locals, field.type)
                    continue
                struct_locals.add(child_name)
    return struct_locals


def expand_func_arguments(
    used_py_dataclasses_parameters_enforcing: set[str] | None, arguments: list[ArgMetadata]
) -> list[ArgMetadata]:
    """
    Used to expand arguments for @qd.func
    """
    expanded_arguments = []
    for i, argument in enumerate(arguments):
        if dataclasses.is_dataclass(argument.annotation):
            for field in dataclasses.fields(argument.annotation):
                child_name = create_flat_name(argument.name, field.name)
                if (
                    used_py_dataclasses_parameters_enforcing is not None
                    and child_name not in used_py_dataclasses_parameters_enforcing
                ):
                    continue
                if dataclasses.is_dataclass(field.type):
                    new_arg = ArgMetadata(
                        annotation=field.type,
                        name=child_name,
                        default=argument.default,
                    )
                    child_args = expand_func_arguments(used_py_dataclasses_parameters_enforcing, [new_arg])
                    expanded_arguments += child_args
                else:
                    new_argument = ArgMetadata(
                        annotation=field.type,
                        name=child_name,
                    )
                    expanded_arguments.append(new_argument)
        else:
            if (
                not argument.name.startswith("__qd_")
                or used_py_dataclasses_parameters_enforcing is None
                or argument.name in used_py_dataclasses_parameters_enforcing
            ):
                expanded_arguments.append(argument)
    return expanded_arguments


class FlattenAttributeNameTransformer(ast.NodeTransformer):
    def __init__(self, struct_locals: set[str]) -> None:
        self.struct_locals = struct_locals

    def visit_Attribute(self, node):
        flat_name = FlattenAttributeNameTransformer._flatten_attribute_name(node)
        if not flat_name or flat_name not in self.struct_locals:
            return self.generic_visit(node)
        return ast.copy_location(ast.Name(id=flat_name, ctx=node.ctx), node)

    @staticmethod
    def _flatten_attribute_name(node: ast.Attribute) -> str | None:
        """
        see unpack_ast_struct_expressions docstring for more explanation
        """
        if isinstance(node.value, ast.Name):
            return create_flat_name(node.value.id, node.attr)
        if isinstance(node.value, ast.Attribute):
            child_flat_name = FlattenAttributeNameTransformer._flatten_attribute_name(node.value)
            if not child_flat_name:
                return None
            return create_flat_name(child_flat_name, node.attr)
        return None


def unpack_ast_struct_expressions(tree: ast.Module, struct_locals: set[str]) -> ast.Module:
    """
    Transform nodes in AST, to flatten access to struct members

    Examples of things we will transform/flatten:

    # my_struct_ab.a
    # Attribute(value=Name())
    Attribute(
        value=Name(id='my_struct_ab', ctx=Load()),
        attr='a',
        ctx=Load())
    =>
    # __qd_my_struct_ab__qd_a
    Name(id='__qd_my_struct_ab__qd_a', ctx=Load()

    # my_struct_ab.struct_cd.d
    # Attribute(value=Attribute(value=Name()))
    Attribute(
        value=Attribute(
            value=Name(id='my_struct_ab', ctx=Load()),
            attr='struct_cd',
            ctx=Load()),
        attr='d',
        ctx=Load())
        visit_attribute
    =>
    # __qd_my_struct_ab__qd_struct_cd__qd_d
    Name(id='__qd_my_struct_ab__qd_struct_cd__qd_d', ctx=Load()

    # my_struct_ab.struct_cd.struct_ef.f
    # Attribute(value=Attribute(value=Name()))
    Attribute(
        value=Attribute(
            value=Attribute(
            value=Name(id='my_struct_ab', ctx=Load()),
            attr='struct_cd',
            ctx=Load()),
            attr='struct_ef',
            ctx=Load()),
        attr='f',
        ctx=Load())
    =>
    # __qd_my_struct_ab__qd_struct_cd__qd_struct_ef__qd_f
    Name(id='__qd_my_struct_ab__qd_struct_cd__qd_struct_ef__qd_f', ctx=Load()
    """
    transformer = FlattenAttributeNameTransformer(struct_locals=struct_locals)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree


def populate_global_vars_from_dataclass(
    param_name: str,
    param_type: Any,
    py_arg: Any,
    global_vars: dict[str, Any],
):
    for field in dataclasses.fields(param_type):
        child_value = getattr(py_arg, field.name)
        flat_name = create_flat_name(param_name, field.name)
        if dataclasses.is_dataclass(field.type):
            populate_global_vars_from_dataclass(
                param_name=flat_name,
                param_type=field.type,
                py_arg=child_value,
                global_vars=global_vars,
            )
        elif util.is_qd_template(field.type):
            global_vars[flat_name] = child_value
