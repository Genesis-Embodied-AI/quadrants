from dataclasses import dataclass
from unittest.mock import Mock

from quadrants.lang.ast.ast_transformers.function_def_transformer import (
    FunctionDefTransformer,
)


def test_process_func_arg_preserves_dataclass() -> None:
    """The compiler creates one local variable, "state", holding the caller's original nested dataclass."""

    @dataclass
    class Grid:
        size: int

    @dataclass
    class State:
        grid: Grid

    state = State(Grid(4))
    ctx = Mock(fn_param_names=set())
    FunctionDefTransformer._transform_func_arg(ctx, "state", State, state)

    ctx.create_variable.assert_called_once()
    name, value = ctx.create_variable.call_args.args
    assert name == "state"
    assert value is state
