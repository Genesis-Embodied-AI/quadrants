"""Direct tensor access must preserve aliasing, not merely copy correct values back."""

import os

import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.needs_torch


def matching_gpu():
    arch = impl.current_cfg().arch
    torch_arch = qd.amdgpu if torch.version.hip else qd.cuda
    assert torch.cuda.is_available(), f"PyTorch cannot access a GPU for the selected Quadrants backend {arch}"
    assert arch == torch_arch, f"PyTorch GPU backend {torch_arch} does not match selected Quadrants backend {arch}"


@pytest.mark.parametrize("custom_stream", [False, True])
@pytest.mark.parametrize("with_grad", [False, True])
@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_torch_direct_aliases(monkeypatch, custom_stream, with_grad):
    """Catch separate buffers that break tensor or gradient sharing, even when copy-back gives correct values."""
    matching_gpu()
    # Initialize on CPU, then upload: some ROCm wheels support V520 copies but lack its GPU fill kernels.
    x = torch.full((16384,), 11.0).to("cuda:0").requires_grad_(with_grad)
    if with_grad:
        x.grad = torch.full(x.shape, 5.0, dtype=x.dtype).to(x.device)
    x_alias = x.detach()
    output = torch.empty_like(x)
    x_grad_alias = x.grad if with_grad else torch.empty_like(x)
    torch.cuda.synchronize(0)

    @qd.kernel
    def update(
        x: qd.types.ndarray(),
        x_alias: qd.types.ndarray(),
        x_grad_alias: qd.types.ndarray(),
        output: qd.types.ndarray(),
    ):
        for i in x:
            x[i] += 3
            if qd.static(with_grad):
                x.grad[i] += 2
        # Read through the aliases in a second task, after the writes finish.
        #
        # Reading x_alias outside the kernel is not enough: copy-back could update the original shared memory, making
        # x_alias show the correct value.
        #
        # Instead, pass both x and x_alias into the kernel. Write through x, then read through x_alias into output.
        # Separate staging buffers would hide the write from x_alias, so output would record the old value. Copy-back
        # afterward cannot repair that recorded value.
        for i in x_alias:
            output[i] = x_alias[i]
            if qd.static(with_grad):
                output[i] += x_grad_alias[i]

    # Save the original method before replacing Tensor.to; this stores the function without calling it.
    original_to = torch.Tensor.to

    def no_cpu_copy(self, *args, **kwargs):
        # Intercept Tensor.to calls and reject CPU transfers during kernel argument preparation.
        target = kwargs.get("device", args[0] if args else None)
        assert target != "cpu" and target != torch.device("cpu"), "Unexpected CPU staging"
        # Call the saved method with the tensor as self. Calling self.to here would re-enter this replacement.
        return original_to(self, *args, **kwargs)

    stream = qd.create_stream() if custom_stream else None
    try:
        # The Tensor.to replacement is scoped to this block; monkeypatch restores Tensor.to even if the call fails.
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "to", no_cpu_copy)
            update(x, x_alias, x_grad_alias, output, qd_stream=stream)
        if stream is not None:
            stream.synchronize()
        else:
            qd.sync()
        assert torch.equal(x.cpu(), torch.full((16384,), 14.0))
        assert torch.equal(output.cpu(), torch.full((16384,), 21.0 if with_grad else 14.0))
        if with_grad:
            assert torch.equal(x.grad.cpu(), torch.full((16384,), 7.0))
    finally:
        if stream is not None:
            stream.destroy()


@test_utils.test(arch=qd.amdgpu)
def test_torch_direct_allocates_gradient_on_custom_stream(monkeypatch):
    """Check that a newly created gradient is initialized before a custom Quadrants stream uses it."""
    matching_gpu()
    x = torch.full((16384,), 11.0).to("cuda:0").requires_grad_(True)
    torch.cuda.synchronize(0)
    assert x.grad is None

    if os.environ.get("QD_AMDGPU_V520") == "1":
        # The CI PyTorch wheel cannot run zeros_like on V520. Replace only its initialization with an asynchronous
        # upload; Quadrants must still allocate the missing gradient and wait for its initialization before use. Pinned
        # CPU memory permits an asynchronous copy. Keep it alive until the kernel and upload have finished.
        host_gradient = torch.zeros(x.shape, dtype=x.dtype, pin_memory=True)

        def zeros_like_by_upload(tensor):
            assert tensor is x
            return host_gradient.to(device=tensor.device, non_blocking=True)

        monkeypatch.setattr(torch, "zeros_like", zeros_like_by_upload)

    producer_stream = torch.cuda.Stream(device=0)
    original_synchronize = torch.cuda.Stream.synchronize
    synchronized_streams = []

    def record_synchronize(self):
        original_synchronize(self)
        synchronized_streams.append(self.cuda_stream)

    # A small initialization can finish before the kernel even without a wait. Check the wait explicitly as well.
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", record_synchronize)

    @qd.kernel
    def update(a: qd.types.ndarray()):
        for i in a:
            a.grad[i] += a[i]

    stream = qd.create_stream()
    try:
        # The adapter initializes the missing gradient on this non-default PyTorch stream after the caller's sync.
        with torch.cuda.stream(producer_stream):
            update(x, qd_stream=stream)
        stream.synchronize()
        assert producer_stream.cuda_stream in synchronized_streams
        assert torch.equal(x.grad.cpu(), torch.full((16384,), 11.0))
    finally:
        stream.destroy()


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_torch_mismatched_runtime_stages(monkeypatch):
    """Check that a mismatched PyTorch backend triggers CPU staging and copies the kernel result back."""
    matching_gpu()
    x = torch.full((32,), 11, dtype=torch.int32).to("cuda:0")
    torch.cuda.synchronize(0)
    # Save the original method before replacing Tensor.to; this stores the function without calling it.
    original_to = torch.Tensor.to
    copies = []

    def record_to(self, *args, **kwargs):
        # Record which tensors take the CPU fallback while still performing the real transfer.
        if kwargs.get("device") == "cpu":
            copies.append(self.data_ptr())
        # Call the saved method with the tensor as self. Calling self.to here would re-enter this replacement.
        return original_to(self, *args, **kwargs)

    @qd.kernel
    def update(a: qd.types.ndarray()):
        for i in a:
            a[i] += 3

    with monkeypatch.context() as patch:
        # Exercise runtime matching without requiring CUDA and HIP to coexist in one PyTorch build.
        patch.setattr(torch.version, "hip", None if torch.version.hip else "6.0")
        patch.setattr(torch.Tensor, "to", record_to)
        update(x)
    qd.sync()
    assert copies == [x.data_ptr()]
    assert torch.equal(x.cpu(), torch.full((32,), 14, dtype=torch.int32))


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_torch_other_gpu_device_stages(monkeypatch):
    """Check that a tensor on visible device 1 stages through CPU and receives the kernel result."""
    matching_gpu()
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two matching GPU devices")
    x = torch.full((32,), 11, dtype=torch.int32).to("cuda:1")
    torch.cuda.synchronize(1)
    # Save the original method before replacing Tensor.to; this stores the function without calling it.
    original_to = torch.Tensor.to
    copies = []

    def record_to(self, *args, **kwargs):
        # Record which tensors take the CPU fallback while still performing the real transfer.
        if kwargs.get("device") == "cpu":
            copies.append(self.data_ptr())
        # Call the saved method with the tensor as self. Calling self.to here would re-enter this replacement.
        return original_to(self, *args, **kwargs)

    @qd.kernel
    def update(a: qd.types.ndarray()):
        for i in a:
            a[i] += 3

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", record_to)
        update(x)
    qd.sync()
    assert copies == [x.data_ptr()]
    assert torch.equal(x.cpu(), torch.full((32,), 14, dtype=torch.int32))


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_torch_gradient_device_mismatch_rejected():
    """Prevent a CPU gradient pointer from being passed directly alongside a GPU tensor pointer."""
    matching_gpu()
    x = torch.full((32,), 11.0).to("cuda:0").requires_grad_(True)
    x.grad = torch.zeros(x.shape, dtype=x.dtype).to(x.device)
    torch.cuda.synchronize(0)
    # PyTorch rejects assigning a CPU tensor to x.grad directly, but permits replacing an existing gradient's data.
    x.grad.data = x.grad.cpu()
    assert x.device.type == "cuda" and x.grad.device.type == "cpu"

    @qd.kernel
    def update(x: qd.types.ndarray()):
        for i in x:
            x.grad[i] += x[i]

    with pytest.raises(ValueError, match="The gradient tensor must be on the same device as its tensor"):
        update(x)
