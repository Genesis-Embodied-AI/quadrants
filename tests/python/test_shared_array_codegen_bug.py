"""Regression test for shared-array name collision across tasks.

When two @qd.func each allocate a SharedArray of different sizes and are
inlined into the same @qd.kernel, the re_id pass can give both arrays the
same statement ID.  During LLVM linking the smaller declaration wins,
causing out-of-bounds shared-memory access (CUDA_ERROR_ILLEGAL_ADDRESS).
"""

import numpy as np

import quadrants as qd

from tests import test_utils

N = 128
BLOCK = 32
SMALL = 4
LARGE = 64


@qd.func
def _func_small(out: qd.template()):
    """Uses a small SharedArray (SMALL floats)."""
    qd.loop_config(block_dim=BLOCK)
    for i in range(N * BLOCK):
        tid = i % BLOCK
        bid = i // BLOCK
        s = qd.simt.block.SharedArray((SMALL,), qd.f32)
        if tid < SMALL:
            s[tid] = qd.f32(tid + 1)
        qd.simt.block.sync()
        if tid == 0:
            total = qd.f32(0.0)
            for k in range(SMALL):
                total += s[k]
            out[bid] = total


@qd.func
def _func_large(out: qd.template()):
    """Uses a large SharedArray (LARGE floats).

    Without the fix the linker keeps SMALL as the size for this array,
    so writes beyond index SMALL-1 corrupt memory or crash.
    """
    qd.loop_config(block_dim=BLOCK)
    for i in range(N * BLOCK):
        tid = i % BLOCK
        bid = i // BLOCK
        s = qd.simt.block.SharedArray((LARGE,), qd.f32)
        idx = tid
        while idx < LARGE:
            s[idx] = qd.f32(1.0)
            idx += BLOCK
        qd.simt.block.sync()
        if tid == 0:
            total = qd.f32(0.0)
            for k in range(LARGE):
                total += s[k]
            out[bid] = total


@test_utils.test(arch=[qd.cuda])
def test_shared_array_name_collision():
    out_small = qd.field(qd.f32, shape=(N,))
    out_large = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def kernel():
        _func_small(out_small)
        _func_large(out_large)

    kernel()

    assert np.allclose(out_small.to_numpy(), SMALL * (SMALL + 1) / 2)
    assert np.allclose(out_large.to_numpy(), LARGE)
