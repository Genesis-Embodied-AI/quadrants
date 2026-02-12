import pytest

import quadrants as ti

from tests import test_utils

clock_freq_supported_archs = ti.cuda


def _arch_supports_clock(arch):
    """Check if the architecture supports the clock intrinsic."""
    if arch == ti.vulkan:
        # Vulkan: check device capability at runtime
        device_caps = ti.lang.impl.get_runtime().prog.get_device_caps()
        return device_caps.get(ti._lib.core.DeviceCapability.spirv_has_shader_clock) and device_caps.get(
            ti._lib.core.DeviceCapability.spirv_has_int64
        )
    # CPU and CUDA/AMDGPU always support int64
    return arch in (ti.cuda, ti.amdgpu, ti.x64, ti.arm64)


@test_utils.test()
def test_clock_monotonic():
    arch = ti.lang.impl.get_runtime().prog.config().arch

    dtype = ti.i64 if _arch_supports_clock(arch) else ti.i32
    a = ti.field(dtype=dtype, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(serialize=True, block_dim=1)
        for i in range(32):
            x = ti.random() * 0.5 + 0.5
            for j in range((i + 1) * 2000):
                x = ti.sin(x * 1.0001 + j * 1e-6) + 1.2345
            if x < 10.0:
                a[i] = ti.clock_counter()

    foo()

    if _arch_supports_clock(arch):
        for i in range(1, 31):
            assert a[i - 1] < a[i] < a[i + 1]
    else:
        # On unsupported backends, clock returns 0
        for i in range(1, 31):
            assert a[i] == 0


@test_utils.test(arch=ti.cuda)
def test_clock_accuracy():
    a = ti.field(dtype=ti.i64, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=1)
        for i in range(32):
            start = ti.clock_counter()
            x = ti.random() * 0.5 + 0.5
            for j in range((i + 1) * 2000):
                x = ti.sin(x * 1.0001 + j * 1e-6) + 1.2345
            if x < 10.0:
                a[i] = ti.clock_counter() - start

    foo()

    for i in range(1, 31):
        assert a[i - 1] < a[i] < a[i + 1]
        assert -1 < a[i] / a[0] - (i + 1) < 1


@test_utils.test(arch=clock_freq_supported_archs)
def test_clock_freq_hz_cuda():
    clock_rate_hz = ti.clock_freq_hz()
    assert clock_rate_hz > 0, "CUDA clock speed should be greater than 0"
    assert clock_rate_hz > 100e6, f"CUDA clock speed {clock_rate_hz} Hz seems too low"
    assert clock_rate_hz < 5e9, f"CUDA clock speed {clock_rate_hz} Hz seems too high"


@test_utils.test(exclude=clock_freq_supported_archs)
def test_clock_freq_hz_unsupported():
    with pytest.raises(NotImplementedError):
        ti.clock_freq_hz()
