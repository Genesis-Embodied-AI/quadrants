import dataclasses
from typing import Any

import pytest

import quadrants as qd
import quadrants._test_tools.dataclass_test_tools as dataclass_test_tools
from quadrants.lang.ast.ast_transformers.function_def_transformer import (
    FunctionDefTransformer,
)

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


class NDArrayBuilder:
    def __init__(self, dtype: Any, shape: tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = shape

    def build(self) -> qd.types.NDArray:
        return qd.ndarray(self.dtype, self.shape)


@pytest.mark.parametrize(
    "argument_name, argument_type, expected_variables",
    [
        (
            "my_struct_1",
            MyStructAB,
            {
                "__qd_my_struct_1__qd_a": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_1__qd_b": qd.types.NDArray[qd.i32, 1],
            },
        ),
        (
            "my_struct_2",
            MyStructCD,
            {
                "__qd_my_struct_2__qd_c": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_2__qd_d": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_2__qd_my_struct_ab__qd_a": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_2__qd_my_struct_ab__qd_b": qd.types.NDArray[qd.i32, 1],
            },
        ),
        (
            "my_struct_3",
            MyStructEF,
            {
                "__qd_my_struct_3__qd_e": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_3__qd_f": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_3__qd_my_struct_cd__qd_c": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_3__qd_my_struct_cd__qd_d": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_3__qd_my_struct_cd__qd_my_struct_ab__qd_a": qd.types.NDArray[qd.i32, 1],
                "__qd_my_struct_3__qd_my_struct_cd__qd_my_struct_ab__qd_b": qd.types.NDArray[qd.i32, 1],
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
        if isinstance(expected_obj, qd.types.NDArray):
            actual = ctx.variables[k]
            assert isinstance(actual, (qd.ScalarNdarray,))
            assert len(actual.shape) == expected_obj.ndim
            assert actual.dtype == expected_obj.dtype
        else:
            raise Exception("unexpected expected type", expected_obj)
