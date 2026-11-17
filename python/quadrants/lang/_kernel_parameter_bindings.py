"""Register kernel parameters and populate their values from Python argument attributes."""

from collections.abc import Callable
from dataclasses import _FIELDS
from typing import TYPE_CHECKING, Any, Literal

from quadrants._lib import core as _qd_core
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants.lang import expr
from quadrants.lang._kernel_types import KernelBatchedArgType
from quadrants.lang.util import cook_dtype
from quadrants.types import ndarray_type, template
from quadrants.types.utils import is_signed

if TYPE_CHECKING:
    from quadrants.lang._pruning import Pruning
    from quadrants.lang.kernel_arguments import ArgMetadata

ScalarKind = Literal["f", "i", "u"]
ParameterKind = Literal["scalar", "ndarray", "matrix", "struct", "sparse"]
_FLOAT, _INT, _UINT = KernelBatchedArgType.FLOAT, KernelBatchedArgType.INT, KernelBatchedArgType.UINT


class KernelParameterLaunchBindings:
    """Keep only the records needed to supply registered parameters on each launch."""

    def __init__(self) -> None:
        """Create empty records for direct arguments, array members, and scalar members."""
        self.arg_metas: list["ArgMetadata"] = []
        self.used_dataclass_members: set[str] = set()
        self.ndarray_launch_infos: list[tuple[int, int, tuple[str, ...]]] = []
        self.launch_infos: list[tuple[int, int, tuple[str, ...], ScalarKind]] = []

    def populate_launch_buffer(
        self,
        args: tuple[Any, ...],
        launch_ctx: _qd_core.KernelLaunchContext,
        launch_ctx_buffer: dict[KernelBatchedArgType, list],
        callbacks: list[Callable[[], Any]],
    ) -> bool:
        """Populate all registered argument groups, returning whether the resulting launch context is cacheable."""
        # FuncBase imports compiler contexts, so import its existing value handlers after initialization.
        from quadrants.lang._func_base import FuncBase
        from quadrants.lang._ndarray import Ndarray

        actual_argument_slot = 0
        is_launch_ctx_cacheable = True
        template_num = 0
        i_out = 0
        for i_in, val in enumerate(args):
            needed_ = self.arg_metas[i_in].annotation
            if needed_ is template or type(needed_) is template:
                template_num += 1
                i_out += 1
                continue
            if (
                val is None
                and self.arg_metas[i_in].optional
                and type(self.arg_metas[i_in].annotation) is ndarray_type.NdarrayType
            ):
                # Optional ndarray + None is specialized away at compile time, so it consumes no runtime arg.
                continue
            # An all-Field dataclass unwraps to zero launch-context slots, so skip _recursive_set_args(). That also
            # skips its provided-is-needed type check, so only shortcut when _qd_all_field is cached True for THIS
            # exact annotation (proof the check already ran and passed for it) - hence the per-annotation id() key.
            _all_field_cache = getattr(val, "_qd_all_field", None)
            if _all_field_cache is not None and getattr(needed_, _FIELDS, None) is not None:
                _af_hit = _all_field_cache.get(id(needed_))
                if _af_hit is not None and _af_hit[0] is needed_ and _af_hit[1]:
                    continue
            num_args_, is_launch_ctx_cacheable_ = FuncBase._recursive_set_args(
                self.used_dataclass_members,
                self.arg_metas[i_in].name,
                launch_ctx,
                launch_ctx_buffer,
                needed_,
                type(val),
                val,
                i_out - template_num,
                actual_argument_slot,
                callbacks,
            )
            i_out += num_args_
            is_launch_ctx_cacheable &= is_launch_ctx_cacheable_

        for arg_id, argument_index, attr_chain in self.ndarray_launch_infos:
            py_obj = args[argument_index]
            for attr_name in attr_chain:
                py_obj = getattr(py_obj, attr_name)
            if type(py_obj) in _TENSOR_WRAPPER_TYPES:
                py_obj = py_obj._unwrap()
            assert isinstance(py_obj, Ndarray), f"Expected Ndarray at {attr_chain}, got {type(py_obj)}"
            if py_obj.grad is None:
                launch_ctx_buffer[KernelBatchedArgType.QD_ARRAY].append((arg_id, py_obj.arr))
            else:
                launch_ctx_buffer[KernelBatchedArgType.QD_ARRAY_WITH_GRAD].append((arg_id, py_obj.arr, py_obj.grad.arr))

        for arg_id, argument_index, attr_chain, kind in self.launch_infos:
            py_obj = args[argument_index]
            for attr_name in attr_chain:
                py_obj = getattr(py_obj, attr_name)
            if kind == "f":
                launch_ctx_buffer[_FLOAT].append((arg_id, float(py_obj)))
            else:
                # A float replacing an integer would silently lose its fractional part.
                if type(py_obj) is float:
                    raise TypeError(
                        f"Primitive member '{'.'.join(attr_chain)}' of a @qd.data_oriented(template_primitives=False) "
                        f"object was lifted as an integer kernel argument (its type at first compile), but is now a "
                        f"float ({py_obj!r}); coercing it would truncate. Keep the member's type stable across launches, "
                        f"or use the default template_primitives=True to re-specialise the kernel per type."
                    )
                if kind == "u":
                    launch_ctx_buffer[_UINT].append((arg_id, int(py_obj)))
                else:
                    launch_ctx_buffer[_INT].append((arg_id, int(py_obj)))

        return is_launch_ctx_cacheable and not self.launch_infos


class KernelParameterBindings:
    """Own parameter registration, compiler values, and launch records for all kernel argument kinds."""

    def __init__(self, pruning: "Pruning") -> None:
        """Start one compilation pass with empty member mappings and launch records."""
        self.pruning = pruning
        self.arg_info_by_member: dict[tuple[int, str], tuple[str, int, tuple[str, ...], ScalarKind]] = {}
        self.arg_expr_by_member: dict[tuple[int, str], expr.Expr] = {}
        self.launch_bindings = KernelParameterLaunchBindings()

    def add_kernel_parameter(
        self,
        kind: ParameterKind,
        dtype: Any,
        name: str,
        *,
        source: tuple[int, tuple[str, ...]] | None = None,
        member: tuple[int, str] | None = None,
        ndim: int = 0,
        needs_grad: bool = False,
        boundary: Any = None,
        layout: Any = None,
    ) -> Any:
        """Create a parameter and its compiler value, recording member sources for launch when supplied.

        Direct Python arguments use arg_metas and the existing typed value handlers at launch. A member source
        supplies an argument index and attribute chain. Scalar members also use member identity for pruning.
        """
        from quadrants.lang.kernel_arguments import create_kernel_parameter

        scalar_kind: ScalarKind = "f"
        if member is not None:
            assert kind == "scalar" and source is not None
            if member in self.arg_info_by_member:
                return self.arg_expr_by_member.get(member)

            cooked = cook_dtype(dtype)
            scalar_kind = "f" if _qd_core.is_real(cooked) else ("i" if is_signed(cooked) else "u")
            self.arg_info_by_member[member] = (name, *source, scalar_kind)
            # The kernel ID includes uses inside inlined functions in pruning and fastcache metadata.
            if not self.pruning.enforcing or not self.pruning.is_used(self.pruning.KERNEL_FUNC_ID, name):
                return None

        arg_id, value = create_kernel_parameter(
            kind, dtype, name, ndim=ndim, needs_grad=needs_grad, boundary=boundary, layout=layout
        )
        if source is not None:
            if kind == "ndarray":
                self.launch_bindings.ndarray_launch_infos.append((arg_id[0], *source))
            else:
                assert member is not None
                self.arg_expr_by_member[member] = value
                self.launch_bindings.launch_infos.append((arg_id[0], *source, scalar_kind))
        return value
