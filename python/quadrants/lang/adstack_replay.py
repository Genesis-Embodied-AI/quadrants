"""Measure the reduced primal replay immediately before a stack-based backward."""

from dataclasses import dataclass

from quadrants._lib import core
from quadrants.lang import impl
from quadrants.lang._ndarray import ScalarNdarray
from quadrants.types.primitive_types import i32, i64


@dataclass
class _Record:
    buffer: ScalarNdarray
    capacity: int = 0


def _variant(kernel, mode):
    from quadrants.lang.kernel import Kernel  # pylint: disable=import-outside-toplevel

    name = "_adstack_count_variant" if mode == 1 else "_adstack_reverse_variant"
    variant = getattr(kernel, name, None)
    if variant is None:
        variant = Kernel(kernel.func, kernel.autodiff_mode, _is_classkernel=kernel.is_classkernel)
        variant.quadrants_callable = getattr(kernel, "_adstack_source_callable", kernel.quadrants_callable)
        variant.adstack_mode = mode
        setattr(kernel, name, variant)
    return variant


def _read_measurement(runtime, arch, record):
    if arch in (core.x64, core.arm64, core.cuda):
        return core.read_adstack_replay_measurement(runtime.prog, record.buffer.arr)
    return record.buffer.to_numpy()


def backward(kernel, args, qd_stream=None):
    runtime = impl.get_runtime()
    arch = impl.current_cfg().arch
    if arch not in (core.x64, core.arm64, core.cuda, core.vulkan, core.metal):
        raise RuntimeError("Adstack replay counting supports x64, ARM64, CUDA, Vulkan, and Metal")
    if kernel.use_graph or kernel.use_checkpoints or qd_stream is not None:
        raise RuntimeError("Adstack replay counting requires non-graph kernels without custom streams")
    tape = runtime.target_tape
    runtime.target_tape = None
    try:
        record = kernel._adstack_replay_scratch
        if record is None:
            record = _Record(ScalarNdarray(i32 if arch in (core.vulkan, core.metal) else i64, (2,)))
            kernel._adstack_replay_scratch = record
        else:
            if arch in (core.x64, core.arm64, core.cuda):
                # Zero both i64 words via the byte-size-aware u32 fill, without launching a Python fill kernel.
                runtime.prog.fill_uint(record.buffer.arr, 0)
            else:
                record.buffer.fill(0)
            record.capacity = 0
        count = _variant(kernel, 1)
        count.adstack_record = record
        try:
            count(*args)
            measurement = _read_measurement(runtime, arch, record)
            if int(measurement[1]):
                raise RuntimeError("Measured adstack depth exceeds the backend capacity limit")
            record.capacity = max(1, int(measurement[0]))
        finally:
            count.adstack_record = None
        reverse = _variant(kernel, 2)
        reverse.adstack_record = record
        try:
            result = reverse(*args)
            if int(_read_measurement(runtime, arch, record)[1]):
                raise RuntimeError(
                    "Adstack capacity exceeded during backward replay. "
                    f"Measured capacity {record.capacity}. Discard or reset this backward's gradients."
                )
            return result
        finally:
            reverse.adstack_record = None
    finally:
        runtime.target_tape = tape
