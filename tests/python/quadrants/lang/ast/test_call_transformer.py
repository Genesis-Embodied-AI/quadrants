import ast
import dataclasses

import pytest

import quadrants as qd
from quadrants.lang._func_base import FuncBase
from quadrants.lang._pruning import Pruning
from quadrants.lang.ast.ast_transformer_utils import (
    ASTTransformerFuncContext,
    ASTTransformerGlobalContext,
)
from quadrants.lang.ast.ast_transformers.call_transformer import CallTransformer

from tests import test_utils


@dataclasses.dataclass
class MyStructAB:
    a: qd.types.NDArray[qd.i32, 1]
    b: qd.types.NDArray[qd.i32, 1]


@dataclasses.dataclass
class MyStructCD:
    c: qd.types.NDArray[qd.i32, 1]
    d: qd.types.NDArray[qd.i32, 1]
    my_struct_ab: MyStructAB


@dataclasses.dataclass
class MyStructEF:
    e: qd.types.NDArray[qd.i32, 1]
    f: qd.types.NDArray[qd.i32, 1]
    my_struct_cd: MyStructCD


def dump_ast_list(nodes: tuple[ast.stmt, ...]) -> str:
    res_l = []
    for node in nodes:
        res_l.append(ast.dump(node, indent=2))
    return "[\n" + ",\n".join(res_l) + "\n]"


@pytest.mark.parametrize(
    "args_in, expected_args_out",
    [
        (
            [ast.Name(id="my_struct_ab", ctx=ast.Load(), ptr=MyStructAB)],
            [
                ast.Name(id="__ti_my_struct_ab__ti_a", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ab__ti_b", ctx=ast.Load()),
            ],
        ),
        (
            [ast.Name(id="my_struct_cd", ctx=ast.Load(), ptr=MyStructCD)],
            [
                ast.Name(id="__ti_my_struct_cd__ti_c", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_cd__ti_d", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_cd__ti_my_struct_ab__ti_a", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_cd__ti_my_struct_ab__ti_b", ctx=ast.Load()),
            ],
        ),
        (
            [ast.Name(id="my_struct_ef", ctx=ast.Load(), ptr=MyStructEF)],
            (
                ast.Name(id="__ti_my_struct_ef__ti_e", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_f", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_c", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_d", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_my_struct_ab__ti_a", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_my_struct_ab__ti_b", ctx=ast.Load()),
            ),
        ),
    ],
)
@test_utils.test()
def test_expand_Call_dataclass_args(args_in: tuple[ast.stmt, ...], expected_args_out: tuple[ast.stmt, ...]) -> None:
    for arg in args_in:
        arg.lineno = 1
        arg.end_lineno = 2
        arg.col_offset = 1
        arg.end_col_offset = 2

    pruning = Pruning(kernel_used_parameters=None)

    class MockGlobalContext(ASTTransformerGlobalContext):
        def __init__(self):
            self.pruning = pruning

    mock_global_context = MockGlobalContext()

    class MockFunc(FuncBase):
        def __init__(self) -> None:
            self.func_id = 1

    mock_func = MockFunc()

    class MockContext(ASTTransformerFuncContext):
        def __init__(self):
            self.used_py_dataclass_parameters_enforcing = None
            self.global_context = mock_global_context
            self.func = mock_func

    ctx = MockContext()
    args_added, args_out = CallTransformer._expand_Call_dataclass_args(ctx, args_in)
    assert len(args_out) > 0
    assert len(args_added) > 0
    assert dump_ast_list(expected_args_out) == dump_ast_list(args_out)
    assert dump_ast_list(expected_args_out) == dump_ast_list(args_added)
