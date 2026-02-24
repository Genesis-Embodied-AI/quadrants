import os
import sys

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.util import has_pytorch

from tests import test_utils

if has_pytorch():
    import torch


def is_v520_amdgpu():
    return os.environ.get("QD_AMDGPU_V520", None) == "1" and qd.cfg.arch == qd.amdgpu


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=qd.cuda)
def test_torch_cuda_context():
    device = torch.device("cuda:0")
    x = torch.tensor([2.0], requires_grad=True, device=device)
    assert torch._C._cuda_hasPrimaryContext(0)
    loss = x**2
    loss.backward()


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test()
def test_torch_ad():
    n = 32

    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def torch_kernel():
        for i in range(n):
            # Do whatever complex operations here
            y[n - i - 1] = x[i] * x[i]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            x.from_torch(inp)
            torch_kernel()
            outp = y.to_torch()
            return outp

        @staticmethod
        def backward(ctx, outp_grad):
            qd.ad.clear_all_gradients()
            y.grad.from_torch(outp_grad)
            torch_kernel.grad()
            inp_grad = x.grad.to_torch()
            return inp_grad

    sqr = Sqr.apply
    for i in range(10):
        X = torch.tensor(2 * np.ones((n,), dtype=np.float32), requires_grad=True)
        sqr(X).sum().backward()
        ret = X.grad.cpu().numpy()
        for j in range(n):
            assert ret[j] == 4


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@pytest.mark.skipif(sys.platform == "win32", reason="not working on Windows.")
@test_utils.test()
def test_torch_ad_gpu():
    if not torch.cuda.is_available():
        return

    if is_v520_amdgpu():
        pytest.skip(reason="cannot use torch .zero_like() on v520")

    device = torch.device("cuda:0")
    n = 32

    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def torch_kernel():
        for i in range(n):
            # Do whatever complex operations here
            y[n - i - 1] = x[i] * x[i]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            x.from_torch(inp)
            torch_kernel()
            outp = y.to_torch(device=device)
            return outp

        @staticmethod
        def backward(ctx, outp_grad):
            qd.ad.clear_all_gradients()
            y.grad.from_torch(outp_grad)
            torch_kernel.grad()
            inp_grad = x.grad.to_torch(device=device)
            return inp_grad

    sqr = Sqr.apply
    for i in range(10):
        X = torch.tensor(2 * np.ones((n,), dtype=np.float32), requires_grad=True, device=device)
        sqr(X).sum().backward()
        ret = X.grad.cpu().numpy()
        for j in range(n):
            assert ret[j] == 4
