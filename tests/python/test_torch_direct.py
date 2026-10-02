"""Direct tensor access must preserve aliasing, not merely copy correct values back."""

import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.needs_torch


def matching_gpu():
    arch = impl.current_cfg().arch
    torch_arch = qd.amdgpu if torch.version.hip else qd.cuda
    if not torch.cuda.is_available() or arch != torch_arch:
        pytest.skip("Requires PyTorch and Quadrants on the same GPU backend")


@pytest.mark.parametrize("custom_stream", [False, True])
@pytest.mark.parametrize("with_grad", [False, True])
@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_torch_direct_aliases(monkeypatch, custom_stream, with_grad):
    matching_gpu()
    x = torch.full((16384,), 11.0, device="cuda:0", requires_grad=with_grad)
    if with_grad:
        x.grad = torch.full_like(x, 5.0)
    alias = x.detach()
    output = torch.empty_like(x)
    grad_alias = x.grad if with_grad else torch.empty_like(x)
    torch.cuda.synchronize(0)

    @qd.kernel
    def update(a: qd.types.ndarray(), b: qd.types.ndarray(), g: qd.types.ndarray(), out: qd.types.ndarray()):
        for i in a:
            a[i] += 3
            if qd.static(with_grad):
                a.grad[i] += 2
        # A second task reads through the aliases. Separate staging buffers would hide the preceding writes.
        for i in b:
            out[i] = b[i]
            if qd.static(with_grad):
                out[i] += g[i]

    original_to = torch.Tensor.to

    def no_cpu_copy(self, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        assert target != "cpu" and target != torch.device("cpu"), "Unexpected CPU staging"
        return original_to(self, *args, **kwargs)

    stream = qd.create_stream() if custom_stream else None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "to", no_cpu_copy)
            update(x, alias, grad_alias, output, qd_stream=stream)
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
def test_torch_direct_allocates_gradient_on_custom_stream():
    matching_gpu()
    x = torch.full((16384,), 11.0, device="cuda:0", requires_grad=True)
    torch.cuda.synchronize(0)
    assert x.grad is None

    @qd.kernel
    def update(a: qd.types.ndarray()):
        for i in a:
            a.grad[i] += a[i]

    stream = qd.create_stream()
    try:
        # The adapter initializes the missing gradient on this non-default PyTorch stream after the caller's sync.
        with torch.cuda.stream(torch.cuda.Stream(device=0)):
            update(x, qd_stream=stream)
        stream.synchronize()
        assert torch.equal(x.grad.cpu(), torch.full((16384,), 11.0))
    finally:
        stream.destroy()


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_torch_mismatched_runtime_stages(monkeypatch):
    matching_gpu()
    x = torch.full((32,), 11, dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize(0)
    original_to = torch.Tensor.to
    copies = []

    def record_to(self, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            copies.append(self.data_ptr())
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


@test_utils.test(arch=qd.amdgpu)
def test_torch_other_hip_device_stages(monkeypatch):
    matching_gpu()
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two HIP devices")
    x = torch.full((32,), 11, dtype=torch.int32, device="cuda:1")
    torch.cuda.synchronize(1)
    original_to = torch.Tensor.to
    copies = []

    def record_to(self, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            copies.append(self.data_ptr())
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
