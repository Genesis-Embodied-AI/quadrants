from typing import Any, Callable

from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import (
    Function as FunctionCxx,
)
from quadrants._lib.core.quadrants_python import FunctionKey
from quadrants.lang import _kernel_impl_dataclass, impl, ops
from quadrants.lang.any_array import AnyArray
from quadrants.lang.ast import (
    transform_tree,
)
from quadrants.lang.ast.ast_transformer_utils import ReturnStatus
from quadrants.lang.exception import (
    QuadrantsSyntaxError,
    QuadrantsTypeError,
)
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.types import (
    ndarray_type,
    primitive_types,
    template,
)
from quadrants.types.enums import AutodiffMode

from ._func_base import FuncBase

# Define proxy for fast lookup
_NONE = AutodiffMode.NONE


class Func(FuncBase):
    function_counter = 1  # MUST start from >= 1, because 0 means "kernel".

    def __init__(self, _func: Callable, _classfunc=False, _pyfunc=False, is_real_function=False) -> None:
        super().__init__(
            func=_func,
            is_classfunc=_classfunc,
            is_kernel=False,
            is_classkernel=False,
            is_real_function=is_real_function,
            func_id=Func.function_counter,
        )
        Func.function_counter += 1
        self.compiled: dict[int, Callable] = {}  # only for real funcs
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.is_real_function = is_real_function
        self.cxx_function_by_id: dict[int, FunctionCxx] = {}
        self.has_print = False

    def __call__(self: "Func", *py_args, **kwargs) -> Any:
        runtime = impl.get_runtime()
        global_context = runtime._current_global_context
        current_kernel = global_context.current_kernel if global_context is not None else None
        py_args = self.fuse_args(
            is_func=True, is_pyfunc=self.pyfunc, py_args=py_args, kwargs=kwargs, global_context=global_context
        )

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise QuadrantsSyntaxError("Quadrants functions cannot be called from Python-scope.")
            return self.func(*py_args)

        assert current_kernel is not None
        assert global_context is not None
        if self.is_real_function:
            if current_kernel.autodiff_mode != _NONE:
                raise QuadrantsSyntaxError("Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(impl.current_cfg().raise_on_templated_floats, py_args)
            key = FunctionKey(self.func.__name__, self.func_id, instance_id)
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=py_args, arg_features=arg_features)
            return self.func_call_rvalue(key=key, args=py_args)
        tree, ctx = self.get_tree_and_ctx(
            is_kernel=False,
            py_args=py_args,
            ast_builder=current_kernel.ast_builder(),
            is_real_function=self.is_real_function,
        )

        struct_locals = _kernel_impl_dataclass.extract_struct_locals_from_context(ctx)

        tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(tree, struct_locals=struct_locals)
        ret = transform_tree(tree, ctx)
        if not self.is_real_function:
            if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                raise QuadrantsSyntaxError("Function has a return type but does not have a return statement")

        return ret

    def func_call_rvalue(self, key: FunctionKey, args: tuple[Any, ...]) -> Any:
        # Skip the template args, e.g., |self|
        assert self.is_real_function
        non_template_args = []
        dbg_info = _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
        for i, kernel_arg in enumerate(self.arg_metas):
            anno = kernel_arg.annotation
            if not isinstance(anno, template):
                if id(anno) in primitive_types.type_ids:
                    non_template_args.append(ops.cast(args[i], anno))
                elif isinstance(anno, primitive_types.RefType):
                    non_template_args.append(_qd_core.make_reference(args[i].ptr, dbg_info))
                elif isinstance(anno, ndarray_type.NdarrayType):
                    if not isinstance(args[i], AnyArray):
                        raise QuadrantsTypeError(
                            f"Expected ndarray in the kernel argument for argument {kernel_arg.name}, got {args[i]}"
                        )
                    non_template_args += _qd_core.get_external_tensor_real_func_args(args[i].ptr, dbg_info)
                else:
                    non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        compiling_callable = impl.get_runtime().compiling_callable
        assert compiling_callable is not None
        func_call = compiling_callable.ast_builder().insert_func_call(
            self.cxx_function_by_id[key.instance_id], non_template_args, dbg_info
        )
        if self.return_type is None:
            return None
        func_call = Expr(func_call)
        ret = []

        for i, return_type in enumerate(self.return_type):
            if id(return_type) in primitive_types.type_ids:
                ret.append(Expr(_qd_core.make_get_element_expr(func_call.ptr, (i,), dbg_info)))
            elif isinstance(return_type, (StructType, MatrixType)):
                ret.append(return_type.from_quadrants_object(func_call, (i,)))
            else:
                raise QuadrantsTypeError(f"Unsupported return type for return value {i}: {return_type}")
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def do_compile(self, key: FunctionKey, args: tuple[Any, ...], arg_features: tuple[Any, ...]) -> None:
        """
        only for real func
        """
        tree, ctx = self.get_tree_and_ctx(
            is_kernel=False,
            py_args=args,
            arg_features=arg_features,
            is_real_function=self.is_real_function,
        )
        fn = impl.get_runtime().prog.create_function(key)

        def func_body():
            old_callable = impl.get_runtime().compiling_callable
            impl.get_runtime()._compiling_callable = fn
            ctx.ast_builder = fn.ast_builder()
            transform_tree(tree, ctx)
            impl.get_runtime()._compiling_callable = old_callable

        self.cxx_function_by_id[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.cxx_function_by_id[key.instance_id].set_function_body(func_body)
