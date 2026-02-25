# type: ignore

from quadrants._lib import core as _qd_core
from quadrants.lang import impl


def clock_counter():
    """
    Returns the current value of a hardware cycle counter.

    All backends return raw clock cycles or ticks, NOT nanoseconds.
    The counter frequency varies by hardware and may change dynamically
    (e.g., due to GPU boost or thermal throttling).

    Supported backends:
    - CUDA: Per-streaming-multiprocessor cycle counter (increments every SM clock cycle).
      See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function
    - AMDGPU: GPU cycle counter
    - Vulkan: Device clock in cycles (requires VK_KHR_shader_clock, else returns 0)
    - CPU (x64/arm64): Processor timestamp counter (constant rate on modern CPUs)

    Unsupported backends (returns 0):
    - Metal

    Use this for relative timing measurements within the same backend and run.
    Comparing cycle counts across different backends or hardware is not meaningful.
    """
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("cuda_clock_i64", with_runtime_context=False)
    if arch == _qd_core.amdgpu:
        return impl.call_internal("amdgpu_clock_i64", with_runtime_context=False)
    if arch == _qd_core.vulkan:
        return impl.call_internal("spirv_clock_i64", with_runtime_context=False)
    if arch == _qd_core.x64 or arch == _qd_core.arm64:
        return impl.call_internal("cpu_clock_i64", with_runtime_context=False)
    # No-op if not supported
    return 0


def clock_freq_hz():
    """
    Returns the clock speed in Hz of the compute device, i.e. GPU. Throws NotImplementedError on
    unsupported architectures.

    Note that this is the nominal speed, NOT the current dynamic speed.

    To set to fixed speed, per AI (untested):

    # Lock GPU clock and memory clock to specific values
    sudo nvidia-smi -lgc <gpu_clock_mhz>
    sudo nvidia-smi -lmc <memory_clock_mhz>

    # Example: Lock to 1200 MHz GPU clock
    sudo nvidia-smi -lgc 1200
    """
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        clock_rate_khz = _qd_core.query_int64("cuda_clock_rate_khz")
        return float(clock_rate_khz * 1000)
    raise NotImplementedError(f"{clock_freq_hz.__name__} not implemented for arch {arch.name}")


__all__ = [
    "clock_counter",
    "clock_freq_hz",
]
