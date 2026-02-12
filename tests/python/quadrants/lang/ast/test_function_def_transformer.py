import dataclasses
from typing import Any

import pytest

import quadrants as ti
import quadrants._test_tools.dataclass_test_tools as dataclass_test_tools
from quadrants.lang.ast.ast_transformers.function_def_transformer import (
    FunctionDefTransformer,
)

from tests import test_utils


@dataclasses.dataclass
class MyStructAB:
    a: ti.types.NDArray[ti.i32, 1]
    b: ti.types.NDArray[ti.i32, 1]


@dataclasses.dataclass
class MyStructCD:
    c: ti.types.NDArray[ti.i32, 1]
    d: ti.types.NDArray[ti.i32, 1]
    my_struct_ab: MyStructAB


@dataclasses.dataclass
class MyStructEF:
    e: ti.types.NDArray[ti.i32, 1]
    f: ti.types.NDArray[ti.i32, 1]
    my_struct_cd: MyStructCD


class NDArrayBuilder:
    def __init__(self, dtype: Any, shape: tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = shape

    def build(self) -> ti.types.NDArray:
        return ti.ndarray(self.dtype, self.shape)


@pytest.mark.parametrize(
    "argument_name, argument_type, expected_variables",
    [
        (
            "my_struct_1",
            MyStructAB,
            {
                "__ti_my_struct_1__ti_a": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_1__ti_b": ti.types.NDArray[ti.i32, 1],
            },
        ),
        (
            "my_struct_2",
            MyStructCD,
            {
                "__ti_my_struct_2__ti_c": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_2__ti_d": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_2__ti_my_struct_ab__ti_a": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_2__ti_my_struct_ab__ti_b": ti.types.NDArray[ti.i32, 1],
            },
        ),
        (
            "my_struct_3",
            MyStructEF,
            {
                "__ti_my_struct_3__ti_e": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_3__ti_f": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_3__ti_my_struct_cd__ti_c": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_3__ti_my_struct_cd__ti_d": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_3__ti_my_struct_cd__ti_my_struct_ab__ti_a": ti.types.NDArray[ti.i32, 1],
                "__ti_my_struct_3__ti_my_struct_cd__ti_my_struct_ab__ti_b": ti.types.NDArray[ti.i32, 1],
            },
        ),
    ],
)
@test_utils.test()
def test_process_func_arg(argument_name: str, argument_type: Any, expected_variables: dict[str, Any]) -> None:
    class MockContext:
        def __init__(self) -> None:
            self.variables: dict[str, Any] = {}

        def create_variable(self, name: str, data: Any) -> None:
            assert name not in self.variables
            self.variables[name] = data

    data = dataclass_test_tools.build_struct(argument_type)
    ctx = MockContext()
    FunctionDefTransformer._transform_func_arg(
        ctx,
        argument_name,
        argument_type,
        data,
    )
    # since these should both be flat, we can just loop over both
    assert set(ctx.variables.keys()) == set(expected_variables.keys())
    for k, expected_obj in expected_variables.items():
        if isinstance(expected_obj, ti.types.NDArray):
            actual = ctx.variables[k]
            assert isinstance(actual, (ti.ScalarNdarray,))
            assert len(actual.shape) == expected_obj.ndim
            assert actual.dtype == expected_obj.dtype
        else:
            raise Exception("unexpected expected type", expected_obj)
