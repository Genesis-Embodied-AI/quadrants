import dataclasses
from typing import Any

import pytest

import quadrants as qd
from quadrants.lang.ast.ast_transformers.function_def_transformer import (
    FunctionDefTransformer,
)

from tests.helpers import dataclass_helpers, test_utils


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


@pytest.mark.parametrize("argument_type", [MyStructAB, MyStructCD, MyStructEF])
@test_utils.test()
def test_process_func_arg_preserves_dataclass(argument_type: type) -> None:
    class MockContext:
        def __init__(self) -> None:
            self.variables: dict[str, Any] = {}

        def create_variable(self, name: str, data: Any) -> None:
            assert name not in self.variables
            self.variables[name] = data

    argument_name = "state"
    data = dataclass_helpers.build_struct(argument_type)
    ctx = MockContext()
    FunctionDefTransformer._transform_func_arg(
        ctx,
        argument_name,
        argument_type,
        data,
    )
    assert set(ctx.variables) == {argument_name}
    assert ctx.variables[argument_name] is data
