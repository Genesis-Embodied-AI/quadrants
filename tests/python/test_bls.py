import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.bls)
def test_simple_1d():
    x, y = qd.field(qd.f32), qd.field(qd.f32)

    N = 64
    bs = 16

    qd.root.pointer(qd.i, N // bs).dense(qd.i, bs).place(x, y)

    @qd.kernel
    def populate():
        for i in range(N):
            x[i] = i

    @qd.kernel
    def copy():
        qd.block_local(x)
        for i in x:
            y[i] = x[i]

    populate()
    copy()

    for i in range(N):
        assert y[i] == i


@test_utils.test(require=qd.extension.bls)
def test_simple_2d():
    x, y = qd.field(qd.f32), qd.field(qd.f32)

    N = 16
    bs = 16

    qd.root.pointer(qd.ij, N // bs).dense(qd.ij, bs).place(x, y)

    @qd.kernel
    def populate():
        for i, j in qd.ndrange(N, N):
            x[i, j] = i - j

    @qd.kernel
    def copy():
        qd.block_local(x)
        for i, j in x:
            y[i, j] = x[i, j]

    populate()
    copy()

    for i in range(N):
        for j in range(N):
            assert y[i, j] == i - j


def _test_bls_stencil(*args, **kwargs):
    from .bls_test_template import bls_test_template

    bls_test_template(*args, **kwargs)


@test_utils.test(require=qd.extension.bls)
def test_gather_1d_trivial():
    # y[i] = x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((0,),))


@test_utils.test(require=qd.extension.bls)
def test_gather_1d():
    # y[i] = x[i - 1] + x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((-1,), (0,)))


@test_utils.test(require=qd.extension.bls)
def test_gather_2d():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=16, stencil=stencil)


@test_utils.test(require=qd.extension.bls)
def test_gather_2d_nonsquare():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=(4, 16), stencil=stencil)


@test_utils.test(require=qd.extension.bls)
def test_gather_3d():
    stencil = [(-1, -1, -1), (2, 0, 1)]
    _test_bls_stencil(3, 64, bs=(4, 8, 16), stencil=stencil)


@test_utils.test(require=qd.extension.bls)
def test_scatter_1d_trivial():
    # y[i] = x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((0,),), scatter=True)


@test_utils.test(require=qd.extension.bls)
def test_scatter_1d():
    _test_bls_stencil(
        1,
        128,
        bs=32,
        stencil=(
            (1,),
            (0,),
        ),
        scatter=True,
    )


@test_utils.test(require=qd.extension.bls)
def test_scatter_2d():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=16, stencil=stencil, scatter=True)


@test_utils.test(require=qd.extension.bls)
def test_multiple_inputs():
    x, y, z, w, w2 = (
        qd.field(qd.i32),
        qd.field(qd.i32),
        qd.field(qd.i32),
        qd.field(qd.i32),
        qd.field(qd.i32),
    )

    N = 128
    bs = 8

    qd.root.pointer(qd.ij, N // bs).dense(qd.ij, bs).place(x, y, z, w, w2)

    @qd.kernel
    def populate():
        for i, j in qd.ndrange((bs, N - bs), (bs, N - bs)):
            x[i, j] = i - j
            y[i, j] = i + j * j
            z[i, j] = i * i - j

    @qd.kernel
    def copy(bls: qd.template(), w: qd.template()):
        if qd.static(bls):
            qd.block_local(x, y, z)
        for i, j in x:
            w[i, j] = x[i, j - 2] + y[i + 2, j - 1] + y[i - 1, j] + z[i - 1, j] + z[i + 1, j]

    populate()
    copy(False, w2)
    copy(True, w)

    for i in range(N):
        for j in range(N):
            assert w[i, j] == w2[i, j]


@test_utils.test(require=qd.extension.bls)
def test_bls_large_block():
    n = 2**10
    block_size = 32
    stencil_length = 28  # uses 60 * 60 * 4B = 14.0625KiB shared memory

    a = qd.field(dtype=qd.f32)
    b = qd.field(dtype=qd.f32)
    block = qd.root.pointer(qd.ij, n // block_size)
    block.dense(qd.ij, block_size).place(a)
    block.dense(qd.ij, block_size).place(b)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=512)
        qd.block_local(a)
        for i, j in a:
            for k in range(stencil_length):
                b[i, j] += a[i + k, j]
                b[i, j] += a[i, j + k]

    foo()


# TODO: BLS on CPU
# TODO: BLS boundary out of bound
# TODO: BLS with TLS
