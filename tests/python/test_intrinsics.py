import pytest

import quadrants as qd

from tests import test_utils

clock_freq_supported_archs = qd.cuda


def _arch_supports_clock(arch):
    """Check if the architecture supports the clock intrinsic."""
    if arch == qd.vulkan:
        # Vulkan: check device capability at runtime
        device_caps = qd.lang.impl.get_runtime().prog.get_device_caps()
        return device_caps.get(qd._lib.core.DeviceCapability.spirv_has_shader_clock) and device_caps.get(
            qd._lib.core.DeviceCapability.spirv_has_int64
        )
    # CPU and CUDA/AMDGPU always support int64
    return arch in (qd.cuda, qd.amdgpu, qd.x64, qd.arm64)


@test_utils.test()
def test_clock_monotonic():
    arch = qd.lang.impl.get_runtime().prog.config().arch

    dtype = qd.i64 if _arch_supports_clock(arch) else qd.i32
    a = qd.field(dtype=dtype, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(serialize=True, block_dim=1)
        for i in range(32):
            x = qd.random() * 0.5 + 0.5
            for j in range((i + 1) * 2000):
                x = qd.sin(x * 1.0001 + j * 1e-6) + 1.2345
            if x < 10.0:
                a[i] = qd.clock_counter()

    foo()

    if _arch_supports_clock(arch):
        for i in range(1, 31):
            assert a[i - 1] < a[i] < a[i + 1]
    else:
        # On unsupported backends, clock returns 0
        for i in range(1, 31):
            assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_clock_accuracy():
    """Verify that clock_counter() measures elapsed cycles proportional to work done.

    Launches 32 threads as a single warp, each doing a different number of LCG iterations
    (thread i does (i+1)*200000). Asserts strict monotonicity across threads and that
    a[i]/a[0] ≈ (i+1), confirming clock_counter() tracks real computational work.
    """
    a = qd.field(dtype=qd.i64, shape=32)
    state = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def measure_sequence_timings():
        # 32 threads = one CUDA warp, so all threads share the same SM clock for comparable timing
        for i in range(32):
            # Read from a field so the compiler can't constant-fold the deterministic LCG sequence
            x = state[i]
            start = qd.i64(0)
            for j in range((i + 1) * 200000):
                # LCG: constant cost per iteration (pure integer arithmetic) and uniform output
                # over [0, 2^31), making `x > 10` true >99.999% of the time but not provably
                # always true, so the compiler can't optimize away the conditional store.
                x = (1664527 * x + 1013904223) % 2147483647
                # Start timing after 10 warmup iterations so all operations (LCG, store, clock
                # counter) are warmed up before we begin measuring.
                if j == 10:
                    start = qd.clock_counter()
                # x > 10 is almost always true for LCG; this data-dependent condition prevents the
                # compiler from dead-code-eliminating the clock read and field write.
                # Writing the end time inside the loop (not after it) is essential: in a warp, all
                # threads execute in lockstep, so a post-loop clock read would fire at the warp's
                # reconvergence point and give all threads the same value. By writing each
                # iteration, a[i] captures the clock at thread i's last iteration.
                if x > 10:
                    a[i] = qd.clock_counter() - start
            # Write x back to prevent dead-code elimination of the entire LCG loop
            state[i] = x

    measure_sequence_timings()

    for i in range(1, 31):
        assert a[i - 1] < a[i] < a[i + 1]
        ratio = a[i] / a[0]
        expected = i + 1
        assert abs(ratio - expected) / expected < 0.2  # 20% tolerance


@test_utils.test(arch=clock_freq_supported_archs)
def test_clock_freq_hz_cuda():
    clock_rate_hz = qd.clock_freq_hz()
    assert clock_rate_hz > 0, "CUDA clock speed should be greater than 0"
    assert clock_rate_hz > 100e6, f"CUDA clock speed {clock_rate_hz} Hz seems too low"
    assert clock_rate_hz < 5e9, f"CUDA clock speed {clock_rate_hz} Hz seems too high"


@test_utils.test(exclude=clock_freq_supported_archs)
def test_clock_freq_hz_unsupported():
    with pytest.raises(NotImplementedError):
        qd.clock_freq_hz()
