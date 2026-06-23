# type: ignore

import ast
import dataclasses
from typing import Any, Callable

from quadrants._lib.core.quadrants_python import (
    BoundaryMode,
    DataTypeCxx,
)
from quadrants._tensor import (
    _TENSOR_T_FIELD_MARKER,
    _TENSOR_T_NDARRAY_MARKER,
)
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang import (
    _ndarray,
    any_array,
    expr,
    impl,
    kernel_arguments,
    matrix,
)
from quadrants.lang import ops as qd_ops
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang.ast.ast_transformer_utils import (
    ASTTransformerFuncContext,
)
from quadrants.lang.ast.ast_transformers.checkpoint_transformer import (
    CheckpointTransformer,
)
from quadrants.lang.ast.symbol_resolver import ASTResolver
from quadrants.lang.buffer_view import BufferView
from quadrants.lang.exception import (
    QuadrantsSyntaxError,
)
from quadrants.lang.matrix import MatrixType
from quadrants.lang.stream import stream_parallel
from quadrants.lang.struct import StructType
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    to_quadrants_type,
)
from quadrants.types import annotations, buffer_view_type, ndarray_type, primitive_types


class FunctionDefTransformer:
    @staticmethod
    def _decl_and_create_variable(
        ctx: ASTTransformerFuncContext,
        annotation: Any,
        name: str,
        this_arg_features: tuple[tuple[Any, ...], ...] | None,
        prefix_name: str,
    ) -> tuple[bool, Any]:
        full_name = prefix_name + "_" + name
        if not isinstance(annotation, primitive_types.RefType):
            ctx.kernel_args.append(name)
        # qd.Tensor value-dispatch. The first slot of this_arg_features is a string marker placed by
        # _template_mapper_hotpath. The annotation is the wrapper class itself (``qd.Tensor``).
        if annotation is _TensorClass:
            assert this_arg_features is not None
            marker = this_arg_features[0]
            if marker == _TENSOR_T_NDARRAY_MARKER:
                raw_element_type, ndim, needs_grad, boundary, layout = this_arg_features[1:]
                return False, (
                    kernel_arguments.decl_ndarray_arg,
                    (
                        to_quadrants_type(raw_element_type),
                        ndim,
                        full_name,
                        needs_grad,
                        BoundaryMode(boundary),
                        layout,
                    ),
                )
            if marker == _TENSOR_T_FIELD_MARKER:
                # Field branch: behave exactly like a template arg.
                if name in ctx.template_vars:
                    return True, ctx.template_vars[name]
                assert ctx.global_vars is not None
                return True, ctx.global_vars.get(name)
            raise AssertionError(f"unknown qd.Tensor marker: {marker!r}")
        if annotation == annotations.template or isinstance(annotation, annotations.template):
            if name in ctx.template_vars:
                return True, ctx.template_vars[name]
            assert ctx.global_vars is not None
            return True, ctx.global_vars.get(name)
        if isinstance(annotation, annotations.sparse_matrix_builder):
            return False, (
                kernel_arguments.decl_sparse_matrix,
                (
                    to_quadrants_type(this_arg_features),
                    full_name,
                ),
            )
        if isinstance(annotation, buffer_view_type.BufferViewType):
            assert this_arg_features is not None
            raw_element_type, ndim, needs_grad, boundary = this_arg_features
            arr = kernel_arguments.decl_ndarray_arg(
                to_quadrants_type(raw_element_type),
                ndim,
                full_name + "_buf",
                needs_grad,
                BoundaryMode(boundary),
            )
            offset = kernel_arguments.decl_scalar_arg(primitive_types.i32, full_name + "_offset")
            size = kernel_arguments.decl_scalar_arg(primitive_types.i32, full_name + "_size")
            return True, BufferView(arr, offset, size)
        if isinstance(annotation, ndarray_type.NdarrayType):
            assert this_arg_features is not None
            raw_element_type: DataTypeCxx
            ndim: int
            needs_grad: bool
            boundary: int
            # Tensors layout is the trailing slot; None for legacy / identity.
            raw_element_type, ndim, needs_grad, boundary, layout = this_arg_features
            return False, (
                kernel_arguments.decl_ndarray_arg,
                (
                    to_quadrants_type(raw_element_type),
                    ndim,
                    full_name,
                    needs_grad,
                    BoundaryMode(boundary),
                    layout,
                ),
            )
        if isinstance(annotation, MatrixType):
            return True, kernel_arguments.decl_matrix_arg(annotation, name)
        if isinstance(annotation, StructType):
            return True, kernel_arguments.decl_struct_arg(annotation, name)
        return True, kernel_arguments.decl_scalar_arg(annotation, name)

    @staticmethod
    def _transform_kernel_arg(
        ctx: ASTTransformerFuncContext,
        argument_name: str,
        argument_type: Any,
        this_arg_features: tuple[Any, ...],
    ) -> None:
        pruning = ctx.global_context.pruning
        func_id = ctx.func.func_id
        if dataclasses.is_dataclass(argument_type):
            ctx.create_variable(argument_name, argument_type)
            for field_idx, field in enumerate(dataclasses.fields(argument_type)):
                flat_name = create_flat_name(argument_name, field.name)
                if pruning.enforcing and flat_name not in pruning.used_vars_by_func_id[func_id]:
                    continue
                # if a field is a dataclass, then feed back into process_kernel_arg recursively
                if dataclasses.is_dataclass(field.type):
                    FunctionDefTransformer._transform_kernel_arg(
                        ctx,
                        flat_name,
                        field.type,
                        this_arg_features[field_idx],
                    )
                elif isinstance(field.type, type) and getattr(field.type, "_data_oriented", False):
                    # ``@qd.data_oriented`` field type inside a typed-dataclass kernel arg. The two patterns are
                    # semantically incompatible at this layer: dataclass kernel-arg recursion uses annotations to
                    # flatten leaf fields into per-leaf kernel args at compile time, but data_oriented containers don't
                    # carry per-attribute type annotations — they need a value-driven walk
                    # (``_predeclare_struct_ndarrays``), which only fires for ``qd.template()`` / ``qd.Tensor``
                    # annotations. Rather than silently miscompile, raise a clear error pointing users to the
                    # recommended pattern.
                    raise QuadrantsSyntaxError(
                        f"Kernel arg {argument_name!r}: field {field.name!r} has @qd.data_oriented type "
                        f"{field.type.__name__!r}, which cannot be flattened into a typed-dataclass kernel arg. "
                        f"Use ``{argument_name}: qd.template()`` for the outer kernel arg annotation instead; "
                        f"data_oriented contents (including nested ndarrays) are walked at kernel-compile time via "
                        f"the template path."
                    )
                else:
                    result, obj = FunctionDefTransformer._decl_and_create_variable(
                        ctx,
                        field.type,
                        flat_name,
                        this_arg_features[field_idx],
                        "",
                    )
                    if result:
                        ctx.create_variable(flat_name, obj)
                    else:
                        decl_type_func, type_args = obj
                        obj = decl_type_func(*type_args)
                        ctx.create_variable(flat_name, obj)
        else:
            result, obj = FunctionDefTransformer._decl_and_create_variable(
                ctx,
                argument_type,
                argument_name,
                this_arg_features if ctx.arg_features is not None else None,
                "",
            )
            if not result:
                decl_type_func, type_args = obj
                obj = decl_type_func(*type_args)
            ctx.create_variable(argument_name, obj)

    @staticmethod
    def _transform_as_kernel(ctx: ASTTransformerFuncContext, node: ast.FunctionDef, args: ast.arguments) -> None:
        assert ctx.func is not None
        assert ctx.arg_features is not None
        if node.returns is not None:
            if not isinstance(node.returns, ast.Constant):
                assert ctx.func.return_type is not None
                for return_type in ctx.func.return_type:
                    kernel_arguments.decl_ret(return_type)
        compiling_callable = impl.get_runtime().compiling_callable
        assert compiling_callable is not None
        compiling_callable.finalize_rets()

        for i in range(len(args.args)):
            arg_meta = ctx.func.arg_metas[i]
            FunctionDefTransformer._transform_kernel_arg(
                ctx,
                arg_meta.name,
                arg_meta.annotation,
                ctx.arg_features[i] if ctx.arg_features is not None else (),
            )

        FunctionDefTransformer._predeclare_struct_ndarrays(ctx)
        compiling_callable.finalize_params()
        # remove original args
        node.args.args = []

    @staticmethod
    def _predeclare_struct_ndarrays(ctx: ASTTransformerFuncContext) -> None:
        """Walk template args that are structs and pre-declare any ``Ndarray`` attributes as kernel args (via
        ``decl_ndarray_arg``) so they are registered before ``finalize_params``. The resulting ``AnyArray`` objects are
        cached on the global context for later lookup by ``build_Attribute``.

        Also stores ``(arg_id, template_arg_idx, attr_chain)`` tuples in
        ``ctx.global_context.struct_ndarray_launch_info`` so the launch path can populate the corresponding slots in the
        launch context.
        """
        from quadrants.lang.util import cook_dtype  # pylint: disable=C0415

        cache = ctx.global_context.ndarray_to_any_array
        launch_info = ctx.global_context.struct_ndarray_launch_info

        # ``_seen`` set guards against attribute-graph cycles in user containers (e.g. Genesis ``sim.solver.sim is
        # sim``). Without it this walker recurses infinitely on the back-edge and blows the Python stack at compile
        # time. Tracked by ``id(obj)`` to avoid relying on ``__hash__`` for arbitrary user types.
        def _walk_obj(obj, arg_idx, path, seen):
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            if is_dataclass_instance(obj):
                for field in dataclasses.fields(obj):
                    child = getattr(obj, field.name)
                    if isinstance(child, _TensorClass):
                        child = child._unwrap()
                    if isinstance(child, _ndarray.Ndarray):
                        _register_ndarray(child, arg_idx, (*path, field.name))
                    elif is_dataclass_instance(child) or is_data_oriented(child):
                        _walk_obj(child, arg_idx, (*path, field.name), seen)
            else:
                for attr_name, attr_val in vars(obj).items():
                    if isinstance(attr_val, _TensorClass):
                        attr_val = attr_val._unwrap()
                    if isinstance(attr_val, _ndarray.Ndarray):
                        _register_ndarray(attr_val, arg_idx, (*path, attr_name))
                    elif is_dataclass_instance(attr_val) or is_data_oriented(attr_val):
                        _walk_obj(attr_val, arg_idx, (*path, attr_name), seen)

        def _register_ndarray(nd, arg_idx, attr_chain):
            key = id(nd)
            if key in cache:
                return
            from quadrants._lib import core as _qd_core  # pylint: disable=C0415

            element_type = cook_dtype(nd.element_type)
            ndim = len(nd._physical_shape)
            needs_grad = nd.grad is not None
            layout = getattr(nd, "_qd_layout", None)
            name = f"__qd_struct_nd_{key}"
            arg_id_vec = impl.get_runtime().compiling_callable.insert_ndarray_param(
                element_type, ndim, name, needs_grad
            )
            arr = any_array.AnyArray(
                _qd_core.make_external_tensor_expr(element_type, ndim, arg_id_vec, needs_grad, BoundaryMode.UNSAFE),
                _qd_layout=layout,
            )
            cache[key] = arr
            launch_info.append((arg_id_vec[0], arg_idx, attr_chain))

        assert ctx.py_args is not None
        for i, arg_meta in enumerate(ctx.func.arg_metas):
            anno = arg_meta.annotation
            is_template = anno is annotations.template or isinstance(anno, annotations.template)
            is_tensor_anno = anno is _TensorClass
            if not (is_template or is_tensor_anno):
                continue
            val = ctx.py_args[i]
            if isinstance(val, _TensorClass):
                val = val._unwrap()
            if isinstance(val, _ndarray.Ndarray):
                continue
            if is_dataclass_instance(val):
                _walk_obj(val, i, (), set())
            elif hasattr(val, "__dict__"):
                _walk_obj(val, i, (), set())

    @staticmethod
    def _unwrap_tensor(data: Any) -> Any:
        """Unwrap a ``qd.Tensor`` wrapper to its bare impl, if needed."""
        if isinstance(data, _TensorClass):
            return data._unwrap()
        return data

    @staticmethod
    def _transform_func_arg(
        ctx: ASTTransformerFuncContext,
        argument_name: str,
        argument_type: Any,
        data: Any,
    ) -> None:
        # Template arguments are passed by reference.
        if isinstance(argument_type, annotations.template):
            ctx.create_variable(argument_name, data)
            return None

        # qd.Tensor in @qd.func context: polymorphic pass-by-reference. No template-mapper features are available
        # (those only exist for top-level @qd.kernel args). Unwrap any Tensor wrapper, then create the variable
        # directly — ndarray and field impls are both valid pass-by-reference arguments.
        if argument_type is _TensorClass:
            data = FunctionDefTransformer._unwrap_tensor(data)
            _cache = getattr(getattr(ctx, "global_context", None), "ndarray_to_any_array", None)
            promoted = _cache.get(id(data)) if _cache else None
            ctx.create_variable(argument_name, promoted if promoted is not None else data)
            return None

        if dataclasses.is_dataclass(argument_type):
            for field in dataclasses.fields(argument_type):
                flat_name = create_flat_name(argument_name, field.name)
                data_child = FunctionDefTransformer._unwrap_tensor(getattr(data, field.name))
                if isinstance(
                    data_child,
                    (
                        _ndarray.ScalarNdarray,
                        matrix.VectorNdarray,
                        matrix.MatrixNdarray,
                        any_array.AnyArray,
                    ),
                ):
                    # qd.Tensor struct fields skip check_matched (the Tensor class has no such method — it is
                    # polymorphic).
                    if field.type is not _TensorClass and hasattr(field.type, "check_matched"):
                        field.type.check_matched(data_child.get_type(), field.name)
                    _cache = getattr(
                        getattr(ctx, "global_context", None),
                        "ndarray_to_any_array",
                        None,
                    )
                    promoted = _cache.get(id(data_child)) if _cache else None
                    ctx.create_variable(flat_name, promoted if promoted is not None else data_child)
                elif dataclasses.is_dataclass(data_child):
                    FunctionDefTransformer._transform_func_arg(
                        ctx,
                        flat_name,
                        field.type,
                        data_child,
                    )
                else:
                    raise QuadrantsSyntaxError(
                        f"Argument {field.name} of type {argument_type} {field.type} is not recognized."
                    )
            return None

        # Ndarray arguments are passed by reference.
        if isinstance(argument_type, (ndarray_type.NdarrayType)):
            if not isinstance(
                data,
                (
                    _ndarray.ScalarNdarray,
                    matrix.VectorNdarray,
                    matrix.MatrixNdarray,
                    any_array.AnyArray,
                ),
            ):
                raise QuadrantsSyntaxError(f"Argument {argument_name} of type {argument_type} is not recognized.")
            argument_type.check_matched(data.get_type(), argument_name)
            ctx.create_variable(argument_name, data)
            return None

        # BufferView arguments are passed by reference.
        # Dtype validation happens at the kernel boundary (_template_mapper_hotpath._extract_arg),
        # not here — data.arr is an Expr node during func compilation, not a real Ndarray.
        if isinstance(argument_type, buffer_view_type.BufferViewType):
            if not isinstance(data, BufferView):
                raise QuadrantsSyntaxError(f"Argument {argument_name} expects a BufferView, got {type(data).__name__}")
            ctx.create_variable(argument_name, data)
            return None

        # Matrix arguments are passed by value.
        if isinstance(argument_type, (MatrixType)):
            # "data" is expected to be an Expr here,
            # so we simply call "impl.expr_init_func(data)" to perform:
            #
            # TensorType* t = alloca()
            # assign(t, data)
            #
            # We created local variable "t" - a copy of the passed-in argument "data"
            if not isinstance(data, expr.Expr) or not data.ptr.is_tensor():
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix, but got {type(data)}."
                )

            element_shape = data.ptr.get_rvalue_type().shape()
            if len(element_shape) != argument_type.ndim:
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with ndim {argument_type.ndim}, but got {len(element_shape)}."
                )

            assert argument_type.ndim > 0
            if element_shape[0] != argument_type.n:
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with n {argument_type.n}, but got {element_shape[0]}."
                )

            if argument_type.ndim == 2 and element_shape[1] != argument_type.m:
                raise QuadrantsSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with m {argument_type.m}, but got {element_shape[0]}."
                )

            ctx.create_variable(argument_name, impl.expr_init_func(data))
            return None

        if id(argument_type) in primitive_types.type_ids:
            ctx.create_variable(argument_name, impl.expr_init_func(qd_ops.cast(data, argument_type)))
            return None
        # Create a copy for non-template arguments,
        # so that they are passed by value.
        var_name = argument_name
        ctx.create_variable(var_name, impl.expr_init_func(data))
        return None

    @staticmethod
    def _bind_intermediate_dataclass_sentinels(ctx: ASTTransformerFuncContext, basename: str, dc_type: Any) -> None:
        """Recursively bind every nested dataclass node ``__qd_<basename>__qd_<field>`` to its dataclass type, so
        AST-flattened intermediate names (e.g. ``s.child`` rewritten to ``Name("__qd_s__qd_child")`` by
        FlattenAttributeNameTransformer) resolve at lookup time and call-site expansion in
        ``_expand_Call_dataclass_args`` triggers correctly.

        Mirrors what ``_transform_kernel_arg`` already does kernel-side via its recursion (each recursive call's first
        action is ``ctx.create_variable(flat_name, field.type)``). On the func side these intermediates are missing,
        because ``_transform_func_arg`` is invoked once per *leaf* arg (``fuse_args`` has already expanded the dataclass
        into leaf arg-metas by then).
        """
        for field in dataclasses.fields(dc_type):
            if dataclasses.is_dataclass(field.type):
                child_name = create_flat_name(basename, field.name)
                ctx.create_variable(child_name, field.type)
                FunctionDefTransformer._bind_intermediate_dataclass_sentinels(ctx, child_name, field.type)

    @staticmethod
    def _transform_as_func(ctx: ASTTransformerFuncContext, node: ast.FunctionDef, args: ast.arguments) -> None:
        # pylint: disable=import-outside-toplevel
        from quadrants.lang.kernel_impl import Func

        assert isinstance(ctx.func, Func)
        assert ctx.py_args is not None
        for py_arg_i, py_arg in enumerate(ctx.py_args):
            argument = ctx.func.arg_metas_expanded[py_arg_i]
            FunctionDefTransformer._transform_func_arg(ctx, argument.name, argument.annotation, py_arg)

        # deal with dataclasses
        for v in ctx.func.orig_arguments:
            if dataclasses.is_dataclass(v.annotation):
                ctx.create_variable(v.name, v.annotation)
                FunctionDefTransformer._bind_intermediate_dataclass_sentinels(ctx, v.name, v.annotation)

    @staticmethod
    def build_FunctionDef(
        ctx: ASTTransformerFuncContext,
        node: ast.FunctionDef,
        build_stmts: Callable[[ASTTransformerFuncContext, list[ast.stmt]], None],
    ) -> None:
        if ctx.visited_funcdef:
            raise QuadrantsSyntaxError(
                f"Function definition is not allowed in 'qd.{'kernel' if ctx.is_kernel else 'func'}'."
            )
        ctx.visited_funcdef = True

        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        if ctx.is_kernel:  # qd.kernel
            FunctionDefTransformer._transform_as_kernel(ctx, node, args)

        if ctx.only_parse_function_def:
            return None

        if not ctx.is_kernel:  # qd.func
            assert ctx.py_args is not None
            assert ctx.func is not None
            if ctx.is_real_function:
                FunctionDefTransformer._transform_as_kernel(ctx, node, args)
            else:
                FunctionDefTransformer._transform_as_func(ctx, node, args)

        if ctx.is_kernel:
            FunctionDefTransformer._validate_stream_parallel_exclusivity(node.body, ctx.global_vars)
            FunctionDefTransformer._validate_graph_do_while_structure(node.body)
            kernel = ctx.global_context.current_kernel
            if kernel is not None:
                # Reset before walking the body so re-materialisations (e.g. when a templated kernel is compiled with a
                # different argument shape) start from an empty list. Mirrors how `graph_do_while_arg` gets overwritten
                # unconditionally during AST traversal.
                kernel.checkpoint_yield_on_args = []
                kernel.checkpoint_user_labels_by_cp_id = []
                # Auto-wrap pass for `@qd.kernel(graph=True, checkpoints=True)` kernels. Mutates `node.body` in place so
                # every top-level for-loop (and every for-loop inside a `qd.graph_do_while` body) that the user did not
                # already wrap in a `with qd.checkpoint(...)` gets wrapped in a synthetic implicit no-yield checkpoint.
                # Implicit checkpoints share the same dense source-order internal cp_id space as explicit ones, but
                # carry `None` in `checkpoint_user_labels_by_cp_id` so they never appear in `GraphStatus.checkpoint` /
                # `kernel.resume(from_checkpoint=...)`. Runs here (after coverage instrumentation has already injected
                # its top-level `_qd_cov[i] = 1` probes, which are bare assigns that the wrap pass intentionally leaves
                # alone) so that the regular `build_stmts` walk below sees a uniform stream of `with qd.checkpoint(...)`
                # blocks and bare prologue stmts.
                if kernel.use_checkpoints:
                    node.body = CheckpointTransformer.auto_wrap_for_loops(node.body)

        with ctx.variable_scope_guard():
            build_stmts(ctx, node.body)

        return None

    @staticmethod
    def _is_stream_parallel_with(stmt: ast.stmt, global_vars: dict[str, Any]) -> bool:
        if not isinstance(stmt, ast.With):
            return False
        if len(stmt.items) != 1:
            return False
        item = stmt.items[0]
        if not isinstance(item.context_expr, ast.Call):
            return False
        func_node = item.context_expr.func
        if ASTResolver.resolve_to(func_node, stream_parallel, global_vars):
            return True
        resolved = ASTResolver.resolve_value(func_node, global_vars)
        if resolved is not None:
            return getattr(resolved, "__name__", None) == "stream_parallel" and getattr(
                resolved, "__module__", ""
            ).startswith("quadrants")
        if isinstance(func_node, ast.Attribute) and func_node.attr == "stream_parallel":
            return True
        if isinstance(func_node, ast.Name) and func_node.id == "stream_parallel":
            return True
        return False

    @staticmethod
    def _is_graph_do_while_while(stmt: ast.stmt) -> bool:
        """Syntactic check matching ASTTransformer._is_graph_do_while_call: a ``while qd.graph_do_while(var):`` loop."""
        if not isinstance(stmt, ast.While):
            return False
        test = stmt.test
        if not isinstance(test, ast.Call):
            return False
        func = test.func
        if isinstance(func, ast.Attribute) and func.attr == "graph_do_while":
            return True
        if isinstance(func, ast.Name) and func.id == "graph_do_while":
            return True
        return False

    @staticmethod
    def _is_checkpoint_with(stmt: ast.With) -> bool:
        """Syntactic check matching CheckpointTransformer: a ``with qd.checkpoint(...):`` block. Accepted as a
        well-formed statement inside a graph_do_while body (the checkpoint itself enforces its own restrictions)."""
        if not isinstance(stmt, ast.With) or len(stmt.items) != 1:
            return False
        ctx = stmt.items[0].context_expr
        if not isinstance(ctx, ast.Call):
            return False
        func = ctx.func
        if isinstance(func, ast.Attribute) and func.attr == "checkpoint":
            return True
        if isinstance(func, ast.Name) and func.id == "checkpoint":
            return True
        return False

    @staticmethod
    def _is_loop_config_call(stmt: ast.stmt) -> bool:
        """A bare ``qd.loop_config(...)`` statement. It only configures the next for-loop (block_dim, serialize, ...) by
        mutating ast_builder state and emits no offloaded task, so it does not break graph_do_while level tagging and is
        allowed where bare statements otherwise are not."""
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        func = stmt.value.func
        if isinstance(func, ast.Attribute) and func.attr == "loop_config":
            return True
        if isinstance(func, ast.Name) and func.id == "loop_config":
            return True
        return False

    @staticmethod
    def _is_graph_parallel_context_with(stmt: ast.stmt) -> bool:
        """Syntactic check matching ASTTransformer._is_graph_parallel_context_call: a
        ``with qd.graph_parallel_context():`` fork/join region."""
        if not isinstance(stmt, ast.With) or len(stmt.items) != 1:
            return False
        ctx_expr = stmt.items[0].context_expr
        if not isinstance(ctx_expr, ast.Call):
            return False
        func = ctx_expr.func
        if isinstance(func, ast.Attribute) and func.attr == "graph_parallel_context":
            return True
        if isinstance(func, ast.Name) and func.id == "graph_parallel_context":
            return True
        return False

    @staticmethod
    def _is_parallel_section_with(stmt: ast.stmt) -> bool:
        """Syntactic check matching ASTTransformer._is_parallel_section_call: a ``with qd.graph_parallel(...):``
        section of a ``qd.graph_parallel_context()`` region."""
        if not isinstance(stmt, ast.With) or len(stmt.items) != 1:
            return False
        ctx_expr = stmt.items[0].context_expr
        if not isinstance(ctx_expr, ast.Call):
            return False
        func = ctx_expr.func
        if isinstance(func, ast.Attribute) and func.attr == "graph_parallel":
            return True
        if isinstance(func, ast.Name) and func.id == "graph_parallel":
            return True
        return False

    @staticmethod
    def _validate_graph_do_while_structure(body: list[ast.stmt]) -> None:
        """If a kernel uses qd.graph_do_while() anywhere, enforce the structural rules that remain after
        per-statement graph-region tagging landed (see the offloader's region-boundary flushing and
        ``GraphRegionTag`` in the C++ IR).

        Bare statements -- assignments, ``@qd.func`` calls, ``qd.loop_config(...)`` directives, ``if``
        branches -- are now legal at the kernel top level *and* inside graph_do_while bodies: each lowers to
        a serial task tagged with the graph_do_while level it was written at, so it runs exactly once at the
        kernel top level and every iteration inside a loop (issue #744). The old ``for _ in range(1):``
        wrapping is no longer required (it still works, since a one-trip for-loop is just an offloaded task).

        What this still checks is genuine well-formedness the offloader / runtime cannot express: a
        graph_do_while ``while``-loop may not carry an ``else`` clause, and a bare runtime ``while`` (other
        than ``qd.graph_do_while()``) is not a valid statement in a graph kernel. The placement rule that a
        graph_do_while loop may only sit at the kernel top level or directly inside another graph_do_while
        body (not inside a for-loop) is enforced elsewhere during lowering."""
        uses_gdw = any(FunctionDefTransformer._is_graph_do_while_while(n) for stmt in body for n in ast.walk(stmt))
        if not uses_gdw:
            return
        FunctionDefTransformer._validate_graph_do_while_stmt_list(body, is_kernel_top=True)

    @staticmethod
    def _validate_graph_do_while_stmt_list(stmts: list[ast.stmt], is_kernel_top: bool) -> None:
        for i, stmt in enumerate(stmts):
            if FunctionDefTransformer._is_docstring(stmt, i):
                continue
            if FunctionDefTransformer._is_coverage_probe(stmt):
                continue
            if FunctionDefTransformer._is_loop_config_call(stmt):
                # qd.loop_config(...) only tunes the next for-loop; it emits no serial task, so it is safe here.
                continue
            if isinstance(stmt, ast.For):
                # A for-loop is an offloaded task; its body is ordinary loop code (unrestricted).
                continue
            if FunctionDefTransformer._is_graph_do_while_while(stmt):
                if stmt.orelse:
                    raise QuadrantsSyntaxError("'else' clause for 'while' not supported in Quadrants kernels")
                FunctionDefTransformer._validate_graph_do_while_stmt_list(stmt.body, is_kernel_top=False)
                continue
            # Bare statements are now correctly region-tagged by the offloader, so they are permitted at the
            # kernel top level and inside graph_do_while bodies. `@qd.func` calls (bare `Expr`) inline at the
            # current level, keeping their internal parallel for-loops parallel; bare assignments / directives
            # lower to a serial task tagged at this level (run-once at the top level, every iteration inside a
            # loop).
            if isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            if isinstance(stmt, ast.If):
                # Recurse so a malformed graph_do_while nested inside an `if` is still caught.
                FunctionDefTransformer._validate_graph_do_while_stmt_list(stmt.body, is_kernel_top=is_kernel_top)
                FunctionDefTransformer._validate_graph_do_while_stmt_list(stmt.orelse, is_kernel_top=is_kernel_top)
                continue
            if isinstance(stmt, ast.With) and FunctionDefTransformer._is_checkpoint_with(stmt):
                # `with qd.checkpoint(...)` is a legal placement for graph kernels (the checkpoint may sit at the kernel
                # top level or inside any graph_do_while level). Recurse so a malformed graph_do_while nested inside a
                # checkpoint body is still caught; the checkpoint's own body restrictions are enforced by
                # `CheckpointTransformer.build_checkpoint_with`.
                FunctionDefTransformer._validate_graph_do_while_stmt_list(stmt.body, is_kernel_top=is_kernel_top)
                continue
            if FunctionDefTransformer._is_graph_parallel_context_with(stmt):
                # A `with qd.graph_parallel_context()` region groups concurrent `with qd.graph_parallel()`
                # sections; it is a legal sibling of for-loops / checkpoints. Its body must be
                # `qd.graph_parallel` section blocks (optionally under `if qd.static(...)`); the full check
                # is in ASTTransformer._build_graph_parallel_context_with. Each `qd.graph_parallel` section's
                # body is task territory, validated here with the in-loop rules. Descend through `if` members
                # so `qd.graph_parallel` sections inside an optional `if qd.static(...)` are reached too.
                pending = list(stmt.body)
                while pending:
                    member = pending.pop()
                    if FunctionDefTransformer._is_parallel_section_with(member):
                        FunctionDefTransformer._validate_graph_do_while_stmt_list(member.body, is_kernel_top=False)
                    elif isinstance(member, ast.If):
                        pending.extend(member.body)
                        pending.extend(member.orelse)
                continue
            where = "the kernel body" if is_kernel_top else "a qd.graph_do_while() body"
            raise QuadrantsSyntaxError(
                f"When a kernel uses qd.graph_do_while(), {where} may not contain a {type(stmt).__name__} "
                f"statement. Allowed: for-loops, qd.graph_do_while() while-loops, bare assignments, and "
                f"@qd.func calls (freely mixed and nested). [offending stmt {i}: {type(stmt).__name__}]"
            )

    @staticmethod
    def _is_docstring(stmt: ast.stmt, index: int) -> bool:
        return index == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str))

    @staticmethod
    def _is_coverage_probe(stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        return (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id.startswith("_qd_cov")
        )

    @staticmethod
    def _validate_stream_parallel_exclusivity(body: list[ast.stmt], global_vars: dict[str, Any]) -> None:
        if not any(FunctionDefTransformer._is_stream_parallel_with(s, global_vars) for s in body):
            return
        for i, stmt in enumerate(body):
            if FunctionDefTransformer._is_docstring(stmt, i):
                continue
            if FunctionDefTransformer._is_coverage_probe(stmt):
                continue
            if not FunctionDefTransformer._is_stream_parallel_with(stmt, global_vars):
                stmt_desc = f"{type(stmt).__name__}"
                if isinstance(stmt, ast.With) and stmt.items:
                    ctx_expr = stmt.items[0].context_expr
                    if isinstance(ctx_expr, ast.Call) and isinstance(ctx_expr.func, ast.Attribute):
                        stmt_desc += f"(with {ast.dump(ctx_expr.func)})"
                raise QuadrantsSyntaxError(
                    "When using qd.stream_parallel(), all top-level statements "
                    "in the kernel must be 'with qd.stream_parallel():' blocks. "
                    f"Move non-parallel code to a separate kernel. "
                    f"[stmt {i}: {stmt_desc}, body_len={len(body)}]"
                )
