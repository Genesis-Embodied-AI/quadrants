"""Register kernel parameters and populate their values from Python argument attributes."""

from typing import TYPE_CHECKING, Any, Literal

from quadrants._lib import core as _qd_core
from quadrants.lang import expr, impl
from quadrants.lang._kernel_types import KernelBatchedArgType
from quadrants.lang.util import cook_dtype
from quadrants.types.utils import is_signed

if TYPE_CHECKING:
    from quadrants.lang._pruning import Pruning

ScalarKind = Literal["f", "i", "u"]
_FLOAT, _INT, _UINT = KernelBatchedArgType.FLOAT, KernelBatchedArgType.INT, KernelBatchedArgType.UINT


class KernelParameterLaunchBindings:
    """Keep only the records needed to supply registered parameters on each launch."""

    def __init__(self) -> None:
        """Create an empty list of parameter slots and their Python value sources."""
        self.launch_infos: list[tuple[int, int, tuple[str, ...], ScalarKind]] = []

    def populate_launch_buffer(self, args: tuple[Any, ...], launch_ctx_buffer: dict[KernelBatchedArgType, list]) -> None:
        """Read current scalar values and append them to the existing batched argument buffers."""
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


class KernelParameterBindings:
    """Own scalar parameter registration, read expressions, and launch records during compilation."""

    def __init__(self, pruning: "Pruning") -> None:
        """Start one compilation pass with empty member mappings and launch records."""
        self.pruning = pruning
        self.arg_info_by_member: dict[tuple[int, str], tuple[str, int, tuple[str, ...], ScalarKind]] = {}
        self.arg_expr_by_member: dict[tuple[int, str], expr.Expr] = {}
        self.launch_bindings = KernelParameterLaunchBindings()

    def add_kernel_parameter(
        self, parent: object, attr_name: str, argument_index: int, attr_chain: tuple[str, ...], dtype: _qd_core.DataTypeCxx
    ) -> None:
        """Register a scalar member and, when used, create its parameter, read expression, and launch record.

        Discovery records the member's source so compilation can mark accesses as used. The enforcing pass adds
        only used parameters. Launch code retains launch_bindings without retaining this compilation object.
        """
        # Interned scalar values can be shared by unrelated members; identify the parent and attribute instead.
        member = (id(parent), attr_name)
        if member in self.arg_info_by_member:
            return

        cooked = cook_dtype(dtype)
        kind: ScalarKind = "f" if _qd_core.is_real(cooked) else ("i" if is_signed(cooked) else "u")
        flat_name = f"__qd_doprim_{argument_index}__{'__'.join(attr_chain)}"
        self.arg_info_by_member[member] = (flat_name, argument_index, attr_chain, kind)
        # The kernel ID includes uses inside inlined functions in pruning and fastcache metadata.
        if not self.pruning.enforcing or not self.pruning.is_used(self.pruning.KERNEL_FUNC_ID, flat_name):
            return

        runtime = impl.get_runtime()
        arg_id = runtime.compiling_callable.insert_scalar_param(cooked, flat_name)
        debug_info = _qd_core.DebugInfo(runtime.get_current_src_info())
        self.arg_expr_by_member[member] = expr.Expr(
            _qd_core.make_arg_load_expr(arg_id, cooked, False, create_load=True, dbg_info=debug_info)
        )
        self.launch_bindings.launch_infos.append((arg_id[0], argument_index, attr_chain, kind))
