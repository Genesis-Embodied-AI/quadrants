"""Tests that the SPIR-V we emit for Vulkan satisfies spirv-val.

These are spec-conformance regression tests. They guard against codegen
bugs that produce invalid SPIR-V which most drivers tolerate (so end-to-end
tests pass) but which can cause crashes / miscompiles on stricter drivers.

Concretely, the first regression here motivated this file: previously
`IRBuilder::get_array_type()` unconditionally emitted `OpDecorate ArrayStride`
on every array type, including arrays used as the pointee of `Function`-
or `Workgroup`-scope `OpVariable`s. Per Vulkan's standalone-SPIR-V rules
(VUID-StandaloneSpirv-None-10684), explicit-layout decorations like
`ArrayStride` are only permitted on types used in explicit-layout storage
classes (Block/BufferBlock-backed StorageBuffer/Uniform/PushConstant).
Most Vulkan drivers silently accept the bogus decoration, but the
NVIDIA proprietary driver's SPIR-V->NVVM compiler (driver 580.76.05 on
Blackwell) crashes on certain kernels that use it. spirv-val rejects it
in any case.

The kernels below are designed to exercise the codegen paths most likely
to produce subtly-invalid SPIR-V (function-scope local Matrix/Vector
allocations, Workgroup-scope arrays, struct-of-array binding types).
If a future codegen change reintroduces a spec violation in any of these
paths, spirv-val will fail and these tests will catch it before the
buggy SPIR-V reaches a real driver.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

import quadrants as qd

from tests import test_utils

_vk_on_mac = (qd.vulkan, "Darwin")


def _find_spirv_val() -> str | None:
    """Locate the spirv-val binary, or return None if not available."""
    found = shutil.which("spirv-val")
    if found is not None:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        candidate = pathlib.Path(sdk) / "bin" / "spirv-val"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _validate_dumped_spirv(tmp_path: pathlib.Path) -> None:
    """Run spirv-val on every *_after_opt.spirv emitted under tmp_path.

    Skips the test if spirv-val isn't installed / discoverable.
    Asserts that at least one SPIR-V file was dumped (otherwise the test
    is silently a no-op) and that every dumped file passes validation.
    """
    spirv_val = _find_spirv_val()
    if spirv_val is None:
        pytest.skip("spirv-val not found in PATH or $VULKAN_SDK/bin")

    spirv_files = sorted(tmp_path.glob("*_after_opt.spirv"))
    assert len(spirv_files) > 0, f"No *_after_opt.spirv dumped under {tmp_path} -- did QD_DUMP_IR=1 work?"

    failures = []
    for asm_file in spirv_files:
        # The dumped file is SPIR-V *assembly* text; spirv-val needs the
        # binary form. spirv-as (sibling tool) does the conversion.
        spirv_as = shutil.which("spirv-as") or (str(pathlib.Path(spirv_val).parent / "spirv-as") if spirv_val else None)
        if spirv_as is None or not os.path.isfile(spirv_as):
            pytest.skip("spirv-as not found alongside spirv-val")
        bin_file = asm_file.with_suffix(".spv")
        as_res = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.3", str(asm_file), "-o", str(bin_file)],
            capture_output=True,
            text=True,
        )
        if as_res.returncode != 0:
            failures.append(f"spirv-as failed on {asm_file.name}:\n{as_res.stderr}")
            continue
        val_res = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.3", str(bin_file)],
            capture_output=True,
            text=True,
        )
        if val_res.returncode != 0:
            failures.append(f"spirv-val failed on {asm_file.name}:\n{val_res.stdout}{val_res.stderr}")

    assert not failures, "spirv-val rejected emitted SPIR-V:\n\n" + "\n\n".join(failures)


@test_utils.test(arch=[qd.vulkan], exclude=[_vk_on_mac], offline_cache=False)
def test_spirv_val_local_matrix_temp_in_parallel_loop(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Function-scope local Matrix temp -> OpVariable with Function storage class.

    Regression test for the ArrayStride-on-Function-scope-array bug.

    The outer `for i in range(n)` is a parallel range-for, so the kernel
    body becomes the per-thread offloaded task. The inner per-thread
    Matrix temp escapes scalarization (the optimizer can't fully prove
    each element gets stored & loaded in isolation across the inner
    loops), so codegen materializes a real `_arr_float_uint_9` local
    backed by a Function-scope `OpVariable`. Before the fix, codegen
    also emitted `OpDecorate %_arr_float_uint_9 ArrayStride 4` for that
    array type -- which spirv-val rejects with VUID-StandaloneSpirv-None-10684
    because explicit-layout decorations are forbidden on the pointee
    types of Function-scope variables.
    """
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    n = 16
    a = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def k(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(n):
            m = qd.Matrix.zero(qd.f32, 3, 3)
            for r in range(3):
                for c in range(3):
                    m[r, c] = float(i * 9 + r * 3 + c)
            s = 0.0
            for r in range(3):
                for c in range(3):
                    s += m[r, c]
            a[i] = s

    k(a)
    qd.sync()
    a_np = a.to_numpy()
    for i in range(n):
        expected = sum(i * 9 + r * 3 + c for r in range(3) for c in range(3))
        assert a_np[i] == float(expected)

    _validate_dumped_spirv(tmp_path)


@test_utils.test(arch=[qd.vulkan], exclude=[_vk_on_mac], offline_cache=False)
def test_spirv_val_storage_buffer_array(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """ndarray binding -> StorageBuffer-backed Block struct wrapping a runtime array.

    This is the call path that *does* legitimately need ArrayStride on the
    array type (it's wrapped in a Block struct). Verifies the fix didn't
    accidentally drop the decoration where it's required.
    """
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    n = 16
    a = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def k(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(n):
            a[i] = float(i)

    k(a)
    qd.sync()
    a_np = a.to_numpy()
    for i in range(n):
        assert a_np[i] == float(i)

    _validate_dumped_spirv(tmp_path)
