"""Experimental forward measurements for stack-based reverse AD.

Explicit gradient calls consume records in reverse invocation order for each kernel. The initial implementation
uses the largest measured depth for every stack and supports CPU, CUDA, Vulkan, and Metal with offline caching off.
"""

from dataclasses import dataclass

from quadrants._lib import core
from quadrants.lang import impl
from quadrants.lang._ndarray import Ndarray, ScalarNdarray
from quadrants.types.enums import AutodiffMode
from quadrants.types.primitive_types import i32, i64


@dataclass
class _Record:
    arguments: tuple
    buffer: ScalarNdarray
    capacity: int
    keepalive: tuple


class _AdstackContext:
    def __enter__(self):
        runtime = impl.get_runtime()
        config = impl.current_cfg()
        if config.arch not in (core.cuda, core.x64, core.vulkan, core.metal):
            raise RuntimeError("qd.adstack prototype supports CUDA, x64, Vulkan, and Metal only")
        if config.offline_cache:
            raise RuntimeError("qd.adstack prototype requires qd.init(offline_cache=False)")
        if config.ad_stack_size:
            raise RuntimeError("qd.adstack prototype requires automatic ad_stack_size=0")
        if getattr(runtime, "adstack_scope", None) is not None:
            raise RuntimeError("Nested qd.adstack contexts are not supported")
        runtime.adstack_scope = self
        self.capacities = []
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        impl.get_runtime().adstack_scope = None


adstack = _AdstackContext()


def _arguments(args):
    signature = []
    for arg in args:
        if isinstance(arg, Ndarray):
            signature.append(("ndarray", id(arg), arg.dtype, arg.shape))
        elif isinstance(arg, (int, float)):
            signature.append((type(arg), arg))
        else:
            raise RuntimeError("qd.adstack prototype accepts qd.ndarray and scalar arguments only")
    return tuple(signature)


def _variant(kernel, mode):
    # A distinct Kernel object owns its own specialization caches. Both persistent cache layers are bypassed.
    from quadrants.lang.kernel import Kernel  # pylint: disable=import-outside-toplevel

    name = "_adstack_count_variant" if mode == 1 else "_adstack_reverse_variant"
    variant = getattr(kernel, name, None)
    if variant is None:
        variant = Kernel(kernel.func, kernel.autodiff_mode)
        variant.adstack_mode = mode
        if mode == 1:
            variant.grad = kernel.grad
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
    if kwargs or kernel.use_graph or kernel.use_checkpoints:
        raise RuntimeError("qd.adstack prototype requires positional arguments and non-graph kernels")
    signature = _arguments(args)
    if counting:
        assert scope is not None
        # Allocation and clearing are implementation work, not forward invocations to record on a Tape.
        tape = runtime.target_tape
        runtime.target_tape = None
        try:
            buffer = ScalarNdarray(i32 if impl.current_cfg().arch in (core.vulkan, core.metal) else i64, (2,))
        finally:
            runtime.target_tape = tape
        variant = _variant(kernel, 1)
        variant.adstack_record = _Record(signature, buffer, 0, args)
        try:
            result = variant(*args)
            runtime.sync()
            capacity = max(1, int(buffer.arr.read_int([0])))
        finally:
            variant.adstack_record = None
        record = _Record(signature, buffer, capacity, args)
        if not hasattr(kernel.grad, "_adstack_pending"):
            kernel.grad._adstack_pending = []
        kernel.grad._adstack_pending.append(record)
        scope.capacities.append(capacity)
        return True, result

    if not pending:
        raise RuntimeError("No pending counted forward invocation for this backward")
    record = pending[-1]
    if record.arguments != signature:
        raise RuntimeError("Counted backward calls must match forward arguments in reverse invocation order")
    pending.pop()  # Records are single-use, including failed backwards.
    variant = _variant(kernel, 2)
    variant.adstack_record = record
    try:
        result = variant(*args)
        runtime.sync()
        if int(record.buffer.arr.read_int([1])):
            raise RuntimeError(
                "Adstack capacity exceeded during backward replay. Replay required more stack space than counted "
                f"during forward (capacity {record.capacity}). Discard or reset this backward's gradients."
            )
        return True, result
    finally:
        variant.adstack_record = None
