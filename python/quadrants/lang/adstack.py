"""Experimental forward measurements for stack-based reverse AD.

Explicit gradient calls consume records in reverse invocation order for each kernel. The initial implementation
uses the largest measured depth for every stack and supports CPU, CUDA, Vulkan, and Metal.
"""

from dataclasses import dataclass, field
from numbers import Number
from typing import Any

from quadrants._lib import core
from quadrants.lang import impl
from quadrants.lang._ndarray import ScalarNdarray
from quadrants.types.enums import AutodiffMode
from quadrants.types.primitive_types import i32, i64


@dataclass
class _Record:
    arguments: tuple
    buffer: ScalarNdarray
    capacity: int
    keepalive: tuple


@dataclass
class _Scope:
    capacities: list[int] = field(default_factory=list)
    records: list["_Invocation"] = field(default_factory=list)


class _Invocation:
    """A reusable measured invocation; release it when its arguments are no longer needed."""

    def __init__(self, kernel: Any, record: _Record):
        self._kernel = kernel
        self._record: _Record | None = record
        self._program = impl.get_runtime().prog

    def _remove_pending(self):
        pending = getattr(self._kernel, "_adstack_pending", [])
        pending[:] = [item for item in pending if item is not self]

    def backward(self):
        record = self._record
        if record is None:
            raise RuntimeError("This adstack invocation has been released")
        if impl.get_runtime().prog is not self._program:
            raise RuntimeError("Adstack invocations cannot be used after qd.reset() or qd.init()")
        if _arguments(record.keepalive) != record.arguments:
            raise RuntimeError("Counted invocation argument metadata changed before backward")
        if getattr(impl.get_runtime(), "adstack_scope", None) is not None:
            raise RuntimeError("Run backward outside the qd.adstack context")
        self._remove_pending()
        # A previous failed replay must not poison a subsequent attempt after inputs and gradients are restored.
        record.buffer.arr.write_int([1], 0)
        variant = _variant(self._kernel, 2)
        variant.adstack_record = record
        try:
            result = variant(*record.keepalive)
            impl.get_runtime().sync()
            if int(record.buffer.arr.read_int([1])):
                raise RuntimeError(
                    "Adstack capacity exceeded during backward replay. Replay required more stack space than counted "
                    f"during forward (capacity {record.capacity}). Discard or reset this backward's gradients."
                )
            return result
        finally:
            variant.adstack_record = None

    def release(self):
        """Drop the measurement and argument references; subsequent backward calls fail."""
        self._remove_pending()
        self._record = None
        self._kernel = None
        self._program = None


class _AdstackContext:
    def __enter__(self):
        runtime = impl.get_runtime()
        config = impl.current_cfg()
        if config.arch not in (core.cuda, core.x64, core.vulkan, core.metal):
            raise RuntimeError("qd.adstack prototype supports CUDA, x64, Vulkan, and Metal only")
        if config.ad_stack_size:
            raise RuntimeError("qd.adstack prototype requires automatic ad_stack_size=0")
        if getattr(runtime, "adstack_scope", None) is not None:
            raise RuntimeError("Nested qd.adstack contexts are not supported")
        scope = _Scope()
        runtime.adstack_scope = scope
        return scope

    def __exit__(self, exc_type, exc_value, traceback):
        impl.get_runtime().adstack_scope = None


adstack = _AdstackContext()


def _arguments(args):
    # Kernel binding and launch code validate supported types. Only identity/metadata matching belongs here;
    # comparing array values would perform device work and would not detect loss of historical replay inputs.
    return tuple(
        (
            (type(arg), arg)
            if isinstance(arg, Number)
            else (
                type(arg),
                id(arg),
                str(getattr(arg, "dtype", None)),
                tuple(getattr(arg, "shape", ())),
            )
        )
        for arg in args
    )


def _variant(kernel, mode):
    # A distinct Kernel object owns its specialization caches; persistent keys also distinguish the three modes.
    from quadrants.lang.kernel import Kernel  # pylint: disable=import-outside-toplevel

    name = "_adstack_count_variant" if mode == 1 else "_adstack_reverse_variant"
    variant = getattr(kernel, name, None)
    if variant is None:
        variant = Kernel(kernel.func, kernel.autodiff_mode, _is_classkernel=kernel.is_classkernel)
        variant.quadrants_callable = getattr(kernel, "_adstack_source_callable", kernel.quadrants_callable)
        variant.adstack_mode = mode
        if mode == 1:
            variant.grad = kernel.grad
            kernel.grad._adstack_source_callable = kernel.quadrants_callable
        setattr(kernel, name, variant)
    return variant


def dispatch(kernel, args, kwargs):
    """Return (handled, result); ordinary kernel calls remain on their existing launch path."""
    runtime = impl.get_runtime()
    scope = getattr(runtime, "adstack_scope", None)
    pending = getattr(kernel, "_adstack_pending", None)
    counting = scope is not None and kernel.autodiff_mode == AutodiffMode.NONE
    if not counting and pending is None:
        return False, None
    if kernel.use_graph or kernel.use_checkpoints or kwargs.get("qd_stream") is not None:
        raise RuntimeError("qd.adstack prototype requires non-graph kernels without custom streams")
    kwargs = dict(kwargs)
    kwargs.pop("qd_stream", None)
    args = kernel.fuse_args(is_func=False, is_pyfunc=False, py_args=args, kwargs=kwargs, global_context=None)
    signature = _arguments(args)
    if counting:
        assert scope is not None
        # Allocation and clearing are implementation work, not forward invocations to record on a Tape.
        tape = runtime.target_tape
        runtime.target_tape = None
        try:
            buffer = ScalarNdarray(
                i32 if impl.current_cfg().arch in (core.vulkan, core.metal) else i64,
                (2,),
            )
        finally:
            runtime.target_tape = tape
        variant = _variant(kernel, 1)
        variant.adstack_record = _Record(signature, buffer, 0, args)
        try:
            result = variant(*args)
            runtime.sync()
            if impl.current_cfg().arch in (core.vulkan, core.metal) and int(buffer.arr.read_int([1])):
                raise RuntimeError("Measured adstack depth exceeds the backend capacity limit")
            capacity = max(1, int(buffer.arr.read_int([0])))
        finally:
            variant.adstack_record = None
        record = _Record(signature, buffer, capacity, args)
        if not hasattr(kernel.grad, "_adstack_pending"):
            kernel.grad._adstack_pending = []
        invocation = _Invocation(kernel.grad, record)
        kernel.grad._adstack_pending.append(invocation)
        scope.records.append(invocation)
        scope.capacities.append(capacity)
        return True, result

    if not pending:
        raise RuntimeError("No pending counted forward invocation for this backward")
    invocation = pending[-1]
    if invocation._record.arguments != signature:
        raise RuntimeError("Counted backward calls must match forward arguments in reverse invocation order")
    return True, invocation.backward()
