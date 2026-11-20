import ast
from dataclasses import dataclass
from unittest.mock import Mock

from quadrants.lang._runtime_typing.inline_arguments import bind_inline_arguments
from quadrants.lang.kernel_arguments import ArgMetadata


def test_process_func_arg_preserves_dataclass() -> None:
    """The compiler creates one local variable, "state", holding the caller's original nested dataclass."""

    @dataclass
    class Grid:
        size: int

    @dataclass
    class State:
        grid: Grid

    state = State(Grid(4))
    ctx = Mock(fn_param_names=set(), py_args=(state,))
    ctx.func.arg_metas = [ArgMetadata(int, "state")]
    ctx.kernel_compile_context.current_pass.runtime_schema_by_id = {id(state): (State, "state")}
    bind_inline_arguments(ctx, ast.parse("def read(state): pass").body[0])

    ctx.create_variable.assert_called_once()
    name, value = ctx.create_variable.call_args.args
    assert name == "state"
    assert value is state
