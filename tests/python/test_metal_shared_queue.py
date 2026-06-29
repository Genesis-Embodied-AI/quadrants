"""Tests for the external Metal command queue feature (``external_metal_command_queue``)."""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

pytestmark = pytest.mark.needs_torch


def _get_mps_command_queue() -> int:
    """Extract PyTorch MPS's MTLCommandQueue* via ``qd.interop``."""
    queue_ptr = qd.interop.get_mps_command_queue()
    if not queue_ptr:
        pytest.skip("qd.interop.get_mps_command_queue() returned 0")
    return queue_ptr


def _reinit_with_shared_queue(is_torch_queue=True):
    """Reset and re-init Quadrants with PyTorch MPS's shared command queue."""
    import torch

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS not available")
    queue_ptr = _get_mps_command_queue()
    qd.reset()
    qd.init(
        arch=qd.metal,
        external_metal_command_queue=queue_ptr,
        external_metal_command_queue_is_torch_queue=is_torch_queue,
    )
    return queue_ptr


@test_utils.test(arch=[qd.metal])
def test_init_with_external_queue():
    """Quadrants initializes successfully with an external Metal command queue."""
    queue_ptr = _reinit_with_shared_queue()
    assert qd.cfg.external_metal_command_queue == queue_ptr
    assert qd.cfg.external_metal_command_queue_is_torch_queue is True

    x = qd.field(dtype=qd.f32, shape=(16,))

    @qd.kernel
    def fill():
        for i in x:
            x[i] = 42.0

    fill()
    qd.sync()
    np.testing.assert_allclose(x.to_numpy(), np.full(16, 42.0))


@test_utils.test(arch=[qd.metal])
def test_init_without_external_queue():
    """Default init (no external queue) still works and config field is 0."""
    assert qd.cfg.external_metal_command_queue == 0
    assert qd.cfg.external_metal_command_queue_is_torch_queue is False

    x = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel
    def fill():
        for i in x:
            x[i] = 7.0

    fill()
    qd.sync()
    np.testing.assert_allclose(x.to_numpy(), np.full(4, 7.0))


@test_utils.test(arch=[qd.metal])
def test_zerocopy_with_shared_queue():
    """Zero-copy torch tensor is valid and on MPS device when using a shared queue."""
    import torch

    _reinit_with_shared_queue()

    x = qd.field(dtype=qd.f32, shape=(64,))

    @qd.kernel
    def write_val(v: qd.f32):
        for i in x:
            x[i] = v

    write_val(42.0)
    qd.sync()
    tc = x.to_torch(copy=False)
    assert tc.device.type == "mps"
    assert tc.shape == (64,)

    clone = tc.clone()
    torch.mps.synchronize()
    np.testing.assert_allclose(clone.cpu().numpy(), np.full(64, 42.0))


@test_utils.test(arch=[qd.metal])
def test_sync_skipped_with_shared_queue():
    """When using a shared queue, mps_sync_if_metal and _try_zerocopy_torch skip explicit sync."""
    from unittest.mock import MagicMock, patch

    import torch  # noqa: F401

    _reinit_with_shared_queue()

    x = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel
    def fill():
        for i in x:
            x[i] = 1.0

    fill()

    from quadrants.lang import impl

    runtime = impl.get_runtime()
    mock_sync = MagicMock()
    with patch.object(runtime, "sync", mock_sync):
        x.to_torch(copy=False)
        mock_sync.assert_not_called()

    with patch("torch.mps.synchronize") as mock_mps_sync:
        from quadrants.lang._metal_interop import mps_sync_if_metal

        mps_sync_if_metal()
        mock_mps_sync.assert_not_called()


@test_utils.test(arch=[qd.metal])
def test_kernel_writes_visible_without_explicit_sync():
    """Shared-queue mode: kernel writes are observable through a zero-copy view with no explicit cross-framework sync.

    Internal details: shared-queue + is_torch_queue commits Quadrants' cmdbuf at the end of every launch so the cmdbuf
    takes its FIFO slot in the shared MTLCommandQueue before any subsequent torch op enqueues; Metal's same-queue
    ordering then makes the torch readback transitively wait for in-flight Quadrants work.
    """
    _reinit_with_shared_queue()

    @qd.kernel
    def fill(out: qd.types.ndarray(ndim=1), value: float):
        for i in range(out.shape[0]):
            out[i] = value

    arr = qd.ndarray(qd.f32, shape=(4,))
    view = arr.to_torch(copy=False)
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        fill(arr, value)
        np.testing.assert_allclose(view.cpu().numpy(), np.full(4, value))


@test_utils.test(arch=[qd.metal])
def test_interop_get_mps_command_queue():
    """qd.interop.get_mps_command_queue() returns a non-zero pointer on MPS-capable machines."""
    import torch

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS not available")
    ptr = qd.interop.get_mps_command_queue()
    assert isinstance(ptr, int)
    assert ptr != 0


@test_utils.test(arch=[qd.metal])
def test_sync_preserved_with_non_torch_external_queue():
    """When external queue is provided but is_torch_queue=False, syncs still fire."""
    from unittest.mock import MagicMock, patch

    import torch  # noqa: F401

    _reinit_with_shared_queue(is_torch_queue=False)
    assert qd.cfg.external_metal_command_queue != 0
    assert qd.cfg.external_metal_command_queue_is_torch_queue is False

    x = qd.field(dtype=qd.f32, shape=(4,))

    @qd.kernel
    def fill():
        for i in x:
            x[i] = 1.0

    fill()

    from quadrants.lang import impl

    runtime = impl.get_runtime()
    original_sync = runtime.sync
    mock_sync = MagicMock(side_effect=original_sync)
    with patch.object(runtime, "sync", mock_sync):
        x.to_torch(copy=False)
        mock_sync.assert_called()

    with patch("torch.mps.synchronize") as mock_mps_sync:
        from quadrants.lang._metal_interop import mps_sync_if_metal

        mps_sync_if_metal()
        mock_mps_sync.assert_called()
