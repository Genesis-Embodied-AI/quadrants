"""Register kernel parameters as data records and process their launch values in batches."""

from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from quadrants._lib import core as _qd_core
from quadrants.lang import expr
from quadrants.lang._kernel_types import KernelBatchedArgType, ParameterKind
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.util import cook_dtype
from quadrants.types.utils import is_signed

if TYPE_CHECKING:
    from quadrants.lang._pruning import Pruning

ScalarKind = Literal["f", "i", "u"]


class SourceProjection(Enum):
    """Select the whole value or one native parameter of a buffer view."""

    VALUE = auto()
    BUFFER_ARRAY = auto()
    BUFFER_OFFSET = auto()
    BUFFER_SIZE = auto()


class ValuePolicy(Enum):
    """Select a shared value-processing system without attaching behavior to individual parameters."""

    TYPED = auto()
    MEMBER_FLOAT = auto()
    MEMBER_INT = auto()
    MEMBER_UINT = auto()


class ParameterSource(NamedTuple):
    """Locate a value relative to the Python kernel arguments."""

    argument_index: int
    attributes: tuple[str, ...] = ()
    projection: SourceProjection = SourceProjection.VALUE


class ParameterRecord(NamedTuple):
    """Describe one native parameter, its source, and its launch validation and caching policy."""

    slot: int
    source: ParameterSource
    annotation: Any
    name: str
    cacheable: bool | None
    member_ndarray: bool = False


class DataclassGuard(NamedTuple):
    """Preserve type checks on containers even when none of their members become parameters."""

    source: ParameterSource
    annotation: type


def _read_source(args: tuple[Any, ...], source: ParameterSource) -> Any:
    """Read an argument or attribute chain, then select a buffer-view component if requested."""
    py_obj = args[source.argument_index]
    for name in source.attributes:
        py_obj = getattr(py_obj, name)
    if source.projection is SourceProjection.BUFFER_ARRAY:
        return py_obj.get_ndarray()

    if source.projection is SourceProjection.BUFFER_OFFSET:
        return py_obj.offset

    if source.projection is SourceProjection.BUFFER_SIZE:
        return py_obj.size

    return py_obj


class KernelParameterLaunchBindings:
    """Own parameter records grouped by value policy; no compiler expressions or application objects are retained."""

    def __init__(self) -> None:
        """Create empty parameter groups and container validation records."""
        self.records_by_policy: dict[ValuePolicy, list[ParameterRecord]] = {}
        self.dataclass_guards: list[DataclassGuard] = []
        self.cacheable = True

    def member_ndarray_launch_infos(self) -> list[tuple[int, int, tuple[str, ...]]]:
        """Derive the member-array view used by pruning and array-rebinding invalidation."""
        return [
            (record.slot, record.source.argument_index, record.source.attributes)
            for records in self.records_by_policy.values()
            for record in records
            if record.member_ndarray
        ]

    def populate_launch_buffer(
        self,
        args: tuple[Any, ...],
        launch_ctx: _qd_core.KernelLaunchContext,
        launch_ctx_buffer: dict[KernelBatchedArgType, list],
        callbacks: list[Callable[[], Any]],
    ) -> bool:
        """Validate containers, then process each parameter group into the existing batched launch buffers."""
        # FuncBase imports compiler contexts; its existing typed codecs are available after initialization.
        from quadrants.lang._func_base import FuncBase

        for source, annotation in self.dataclass_guards:
            py_obj = _read_source(args, source)
            if type(py_obj) is not annotation and not isinstance(py_obj, annotation):
                raise QuadrantsRuntimeTypeError("needed", annotation, "cannot be assigned the provided", type(py_obj))

        cacheable = self.cacheable
        for record in self.records_by_policy.get(ValuePolicy.TYPED, ()):
            py_obj = _read_source(args, record.source)
            # Native slots are recorded during compilation, rather than recomputed by a dataclass walk at launch.
            _, value_cacheable = FuncBase._recursive_set_args(
                set(),
                record.name,
                launch_ctx,
                launch_ctx_buffer,
                record.annotation,
                record.annotation if record.source.attributes else type(py_obj),
                py_obj,
                record.slot,
                0,
                callbacks,
            )
            cacheable &= value_cacheable if record.cacheable is None else record.cacheable

        # These policies preserve the existing coercions for opted-in data-oriented scalar members.
        for policy, buffer_kind in (
            (ValuePolicy.MEMBER_FLOAT, KernelBatchedArgType.FLOAT),
            (ValuePolicy.MEMBER_INT, KernelBatchedArgType.INT),
            (ValuePolicy.MEMBER_UINT, KernelBatchedArgType.UINT),
        ):
            records = self.records_by_policy.get(policy)
            if not records:
                continue

            cacheable = False
            for record in records:
                py_obj = _read_source(args, record.source)
                if policy is ValuePolicy.MEMBER_FLOAT:
                    value = float(py_obj)
                else:
                    if type(py_obj) is float:
                        raise TypeError(
                            f"Primitive member '{'.'.join(record.source.attributes)}' of a "
                            f"@qd.data_oriented(template_primitives=False) object was lifted as an integer kernel "
                            f"argument (its type at first compile), but is now a float ({py_obj!r}); coercing it would "
                            "truncate. Keep the member's type stable across launches, or use the default "
                            "template_primitives=True to re-specialise the kernel per type."
                        )
                    value = int(py_obj)
                launch_ctx_buffer[buffer_kind].append((record.slot, value))
        return cacheable


class KernelParameterBindings:
    """Create native parameters and compiler values, recording their value sources for the launch systems."""

    def __init__(self, pruning: "Pruning") -> None:
        """Start one compilation pass with empty compiler mappings and launch records."""
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
        source: ParameterSource | None = None,
        launch_annotation: Any = None,
        cacheable: bool | None = None,
        member: tuple[int, str] | None = None,
        member_ndarray: bool = False,
        ndim: int = 0,
        needs_grad: bool = False,
        boundary: Any = None,
        layout: Any = None,
    ) -> Any:
        """Create a parameter and read expression, and register its source in a shared launch-processing group.

        Scalar members are first recorded for pruning; only used members get native parameters. Every parameter
        with a source uses the same record format, whether supplied directly or reached through a container.
        """
        from quadrants.lang.kernel_arguments import create_kernel_parameter

        policy = ValuePolicy.TYPED
        if member is not None:
            assert kind is ParameterKind.SCALAR and source is not None
            if member in self.arg_info_by_member:
                return self.arg_expr_by_member.get(member)

            cooked = cook_dtype(dtype)
            scalar_kind: ScalarKind = "f" if _qd_core.is_real(cooked) else ("i" if is_signed(cooked) else "u")
            policy = {"f": ValuePolicy.MEMBER_FLOAT, "i": ValuePolicy.MEMBER_INT, "u": ValuePolicy.MEMBER_UINT}[
                scalar_kind
            ]
            self.arg_info_by_member[member] = (name, source.argument_index, source.attributes, scalar_kind)
            if not self.pruning.enforcing or not self.pruning.is_used(self.pruning.KERNEL_FUNC_ID, name):
                return None

        arg_id, value = create_kernel_parameter(
            kind, dtype, name, ndim=ndim, needs_grad=needs_grad, boundary=boundary, layout=layout
        )
        if member is not None:
            self.arg_expr_by_member[member] = value
        if source is not None:
            record = ParameterRecord(
                arg_id[0],
                source,
                dtype if launch_annotation is None else launch_annotation,
                name,
                cacheable,
                member_ndarray,
            )
            self.launch_bindings.records_by_policy.setdefault(policy, []).append(record)
        return value
