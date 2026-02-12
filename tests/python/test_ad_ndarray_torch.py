import pytest

import quadrants as ti

from tests import test_utils

archs_support_ndarray_ad = [ti.cpu, ti.cuda]

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed. Skipping...", allow_module_level=True)


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_simple_demo():
    @test_utils.torch_op(output_shapes=[(1,)])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            a = 2.0
            for j in range(1):
                a += x[i] / 3
            y[0] += a

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    input = torch.rand(4, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, input)


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_ad_reduce():
    @test_utils.torch_op(output_shapes=[(1,)])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            y[0] += x[i] ** 2

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    input = torch.rand(4, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, input)


@pytest.mark.parametrize(
    "tifunc",
    [
        lambda x: x,
        lambda x: ti.abs(-x),
        lambda x: -x,
        lambda x: x * x,
        lambda x: x**2,
        lambda x: x * x * x,
        lambda x: x * x * x * x,
        lambda x: 0.4 * x * x - 3,
        lambda x: (x - 3) * (x - 1),
        lambda x: (x - 3) * (x - 1) + x * x,
        lambda x: ti.tanh(x),
        lambda x: ti.sin(x),
        lambda x: ti.cos(x),
        lambda x: ti.acos(x),
        lambda x: ti.asin(x),
        lambda x: 1 / x,
        lambda x: (x + 1) / (x - 1),
        lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
        lambda x: ti.sqrt(x),
        lambda x: ti.rsqrt(x),
        lambda x: ti.exp(x),
        lambda x: ti.log(x),
        lambda x: ti.min(x, 0),
        lambda x: ti.min(x, 1),
        lambda x: ti.min(0, x),
        lambda x: ti.min(1, x),
        lambda x: ti.max(x, 0),
        lambda x: ti.max(x, 1),
        lambda x: ti.max(0, x),
        lambda x: ti.max(1, x),
        lambda x: x % 3,
        lambda x: ti.atan2(0.4, x),
        lambda y: ti.atan2(y, 0.4),
        lambda x: 0.4**x,
        lambda y: y**0.4,
    ],
)
@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_poly(tifunc):
    s = (4,)

    @test_utils.torch_op(output_shapes=[s])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            y[i] = tifunc(x[i])

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    input = torch.rand(s, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, input)


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_ad_select():
    s = (4,)

    @test_utils.torch_op(output_shapes=[s])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray(), z: ti.types.ndarray()):
        for i in x:
            z[i] = ti.select(i % 2, x[i], y[i])

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    x = torch.rand(s, dtype=torch.double, device=device, requires_grad=True)
    y = torch.rand(s, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, [x, y])


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_mixed_with_torch():
    @test_utils.torch_op(output_shapes=[(1,)])
    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in a:
            p[0] += a[i] * 2

    N = 4
    a = torch.ones(N, requires_grad=True)
    b = a * 2
    c = compute_sum(b)
    c[0].sum().backward()

    for i in range(4):
        assert a.grad[i] == 4


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_tape_throw():
    N = 4

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in a:
            p[0] += a[i] * 2

    a = torch.ones(N, requires_grad=True)
    p = torch.ones(2, requires_grad=True)

    with pytest.raises(RuntimeError, match=r"he loss of `Tape` must be a tensor only contains one element"):
        with ti.ad.Tape(loss=p):
            compute_sum(a, p)

    b = ti.ndarray(ti.f32, shape=(N), needs_grad=True)
    q = ti.ndarray(ti.f32, shape=(2), needs_grad=True)

    with pytest.raises(RuntimeError, match=r"The loss of `Tape` must be an ndarray with only one element"):
        with ti.ad.Tape(loss=q):
            compute_sum(b, q)

    m = torch.ones(1, requires_grad=False)
    with pytest.raises(
        RuntimeError,
        match=r"Gradients of loss are not allocated, please set requires_grad=True for all tensors that are required by autodiff.",
    ):
        with ti.ad.Tape(loss=m):
            compute_sum(a, m)

    n = ti.ndarray(ti.f32, shape=(1), needs_grad=False)
    with pytest.raises(
        RuntimeError,
        match=r"Gradients of loss are not allocated, please set needs_grad=True for all ndarrays that are required by autodiff.",
    ):
        with ti.ad.Tape(loss=n):
            compute_sum(b, n)


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_tape_torch_tensor_grad_none():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            a = 2.0
            for j in range(N):
                a += x[i] / 3
            y[0] += a

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"

    a = torch.zeros((N,), device=device, requires_grad=True)
    loss = torch.zeros((1,), device=device, requires_grad=True)

    with ti.ad.Tape(loss=loss):
        test(a, loss)

    for i in range(N):
        assert a.grad[i] == 1.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_tensor_shape():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(N):
            a = 2.0
            for j in range(N):
                a += x[i] / x.shape[0]
            y[0] += a

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"

    a = torch.zeros((N,), device=device, requires_grad=True)
    loss = torch.zeros((1,), device=device, requires_grad=True)

    with ti.ad.Tape(loss=loss):
        test(a, loss)

    for i in range(N):
        assert a.grad[i] == 1.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_torch_needs_grad_false():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(needs_grad=False), y: ti.types.ndarray()):
        for i in range(N):
            a = 2.0
            for j in range(N):
                a += x[i] / x.shape[0]
            y[0] += a

    x = torch.rand((N,), dtype=torch.float, requires_grad=True)
    y = torch.rand((1,), dtype=torch.float, requires_grad=True)

    test(x, y)

    y.grad.fill_(1.0)
    test.grad(x, y)
    for i in range(N):
        assert x.grad[i] == 0.0
