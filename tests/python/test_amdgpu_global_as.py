"""Guards the AMDGPU addrspace(1) optimization. A functional test would still
pass if the flat_* -> global_* promotion silently regressed, just run slower, so
this asserts the IR shape instead: an ndarray-streaming kernel's device memory
traffic must lower to global_* rather than generic flat_*.
"""

import glob

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.amdgpu, print_kernel_amdgcn=True, offline_cache=False)
def test_amdgpu_global_as_promotes_flat_to_global(monkeypatch, tmp_path):
    # print_kernel_amdgcn writes .gcn dumps to the process CWD; isolate them.
    monkeypatch.chdir(tmp_path)

    import numpy as np

    n = 1024

    @qd.kernel
    def saxpy(
        x: qd.types.ndarray(dtype=qd.f32, ndim=1),
        y: qd.types.ndarray(dtype=qd.f32, ndim=1),
        a: qd.f32,
    ):
        for i in range(n):
            y[i] = a * x[i] + y[i]

    x = np.ones(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    saxpy(x, y, 2.0)

    # Functional sanity check.
    np.testing.assert_allclose(y, 2.0, rtol=1e-6)

    gcn_files = sorted(glob.glob(str(tmp_path / "quadrants_kernel_amdgcn_*.gcn")))
    assert gcn_files, "expected print_kernel_amdgcn to emit a .gcn dump, but none was found"
    asm = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in gcn_files)

    global_ops = asm.count("global_load") + asm.count("global_store")
    flat_ops = asm.count("flat_load") + asm.count("flat_store")

    assert global_ops > 0, (
        "expected global_load/global_store in the AMDGCN for an ndarray-streaming "
        "kernel (address-space-at-source promotion), but found none. This means the "
        "addrspace(1) tag did not reach codegen / InferAddressSpaces. ASM follows:\n" + asm
    )
    # Residual flat count is gfx-target / LLVM-version dependent, so assert the
    # shape (promotion happened) rather than an exact number.
    assert flat_ops <= 2 and flat_ops < global_ops, (
        f"expected flat_* traffic to be promoted to global_* (found flat_ops={flat_ops}, "
        f"global_ops={global_ops}); InferAddressSpaces likely did not receive the "
        f"addrspace(1) tag. ASM follows:\n" + asm
    )
