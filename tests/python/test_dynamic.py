import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dynamic():
    x = qd.field(qd.f32)
    n = 128

    qd.root.dynamic(qd.i, n, 32).place(x)

    @qd.kernel
    def func():
        pass

    for i in range(n):
        x[i] = i

    for i in range(n):
        assert x[i] == i


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dynamic2():
    x = qd.field(qd.f32)
    n = 128

    qd.root.dynamic(qd.i, n, 32).place(x)

    @qd.kernel
    def func():
        for i in range(n):
            x[i] = i

    func()

    for i in range(n):
        assert x[i] == i


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dynamic_matrix():
    x = qd.Matrix.field(2, 1, dtype=qd.i32)
    n = 8192

    qd.root.dynamic(qd.i, n, chunk_size=128).place(x)

    @qd.kernel
    def func():
        qd.loop_config(serialize=True)
        for i in range(n // 4):
            x[i * 4][1, 0] = i

    func()

    for i in range(n // 4):
        a = x[i * 4][1, 0]
        assert a == i
        if i + 1 < n // 4:
            b = x[i * 4 + 1][1, 0]
            assert b == 0


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append():
    x = qd.field(qd.i32)
    n = 128

    qd.root.dynamic(qd.i, n, 32).place(x)

    @qd.kernel
    def func():
        for i in range(n):
            qd.append(x.parent(), [], i)

    func()

    elements = []
    for i in range(n):
        elements.append(x[i])
    elements.sort()
    for i in range(n):
        assert elements[i] == i


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_length():
    x = qd.field(qd.i32)
    y = qd.field(qd.f32, shape=())
    n = 128

    qd.root.dynamic(qd.i, n, 32).place(x)

    @qd.kernel
    def func():
        for i in range(n):
            qd.append(x.parent(), [], i)

    func()

    @qd.kernel
    def get_len():
        y[None] = qd.length(x.parent(), [])

    get_len()

    assert y[None] == n


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append_ret_value():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    z = qd.field(qd.i32)
    n = 128

    qd.root.dynamic(qd.i, n, 32).place(x)
    qd.root.dynamic(qd.i, n, 32).place(y)
    qd.root.dynamic(qd.i, n, 32).place(z)

    @qd.kernel
    def func():
        for i in range(n):
            u = qd.append(x.parent(), [], i)
            y[u] = i + 1
            z[u] = i + 3

    func()

    for i in range(n):
        assert x[i] + 1 == y[i]
        assert x[i] + 3 == z[i]


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dense_dynamic():
    n = 128
    x = qd.field(qd.i32)
    l = qd.field(qd.i32, shape=n)

    qd.root.dense(qd.i, n).dynamic(qd.j, n, 8).place(x)

    @qd.kernel
    def func():
        qd.loop_config(serialize=True)
        for i in range(n):
            for j in range(n):
                qd.append(x.parent(), j, i)

        for i in range(n):
            l[i] = qd.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == n


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dense_dynamic_len():
    n = 128
    x = qd.field(qd.i32)
    l = qd.field(qd.i32, shape=n)

    qd.root.dense(qd.i, n).dynamic(qd.j, n, 32).place(x)

    @qd.kernel
    def func():
        for i in range(n):
            l[i] = qd.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == 0


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dynamic_activate():
    # record the lengths
    l = qd.field(qd.i32, 3)
    x = qd.field(qd.i32)
    xp = qd.root.dynamic(qd.i, 32, 32)
    xp.place(x)

    m = 5

    @qd.kernel
    def func():
        for i in range(m):
            qd.append(xp, [], i)
        l[0] = qd.length(xp, [])
        x[20] = 42
        l[1] = qd.length(xp, [])
        x[10] = 43
        l[2] = qd.length(xp, [])

    func()
    l = l.to_numpy()
    assert l[0] == m
    assert l[1] == 21
    assert l[2] == 21


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append_u8():
    x = qd.field(qd.u8)
    pixel = qd.root.dynamic(qd.j, 20)
    pixel.place(x)

    @qd.kernel
    def make_list():
        qd.loop_config(serialize=True)
        for i in range(20):
            x[()].append(i * i * i)

    make_list()

    for i in range(20):
        assert x[i] == i * i * i % 256


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append_u64():
    x = qd.field(qd.u64)
    pixel = qd.root.dynamic(qd.i, 20)
    pixel.place(x)

    @qd.kernel
    def make_list():
        qd.loop_config(serialize=True)
        for i in range(20):
            x[()].append(i * i * i * qd.u64(10000000000))

    make_list()

    for i in range(20):
        assert x[i] == i * i * i * 10000000000


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append_struct():
    struct = qd.types.struct(a=qd.u8, b=qd.u16, c=qd.u32, d=qd.u64)
    x = struct.field()
    pixel = qd.root.dense(qd.i, 10).dynamic(qd.j, 20, 5)
    pixel.place(x)

    @qd.kernel
    def make_list():
        for i in range(10):
            for j in range(20):
                x[i].append(
                    struct(
                        i * j * 10,
                        i * j * 10000,
                        i * j * 100000000,
                        i * j * qd.u64(10000000000),
                    )
                )

    make_list()

    for i in range(10):
        for j in range(20):
            assert x[i, j].a == i * j * 10 % 256
            assert x[i, j].b == i * j * 10000 % 65536
            assert x[i, j].c == i * j * 100000000 % 4294967296
            assert x[i, j].d == i * j * 10000000000


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append_matrix():
    mat = qd.types.matrix(n=2, m=2, dtype=qd.u8)
    f = mat.field()
    pixel = qd.root.dense(qd.i, 10).dynamic(qd.j, 20, 4)
    pixel.place(f)

    @qd.kernel
    def make_list():
        for i in range(10):
            for j in range(20):
                f[i].append(qd.Matrix([[i * j, i * j * 2], [i * j * 3, i * j * 4]], dt=qd.u8))

    make_list()

    for i in range(10):
        for j in range(20):
            for k in range(4):
                assert f[i, j][k // 2, k % 2] == i * j * (k + 1) % 256


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_append_matrix_in_struct():
    mat = qd.types.matrix(n=2, m=2, dtype=qd.u8)
    struct = qd.types.struct(a=qd.u64, b=mat, c=qd.u16)
    f = struct.field()
    pixel = qd.root.dense(qd.i, 10).dynamic(qd.j, 20, 4)
    pixel.place(f)

    @qd.kernel
    def make_list():
        for i in range(10):
            for j in range(20):
                f[i].append(
                    struct(
                        i * j * qd.u64(10**10),
                        qd.Matrix([[i * j, i * j * 2], [i * j * 3, i * j * 4]], dt=qd.u8),
                        i * j * 5000,
                    )
                )

    make_list()

    for i in range(10):
        for j in range(20):
            assert f[i, j].a == i * j * (10**10)
            for k in range(4):
                assert f[i, j].b[k // 2, k % 2] == i * j * (k + 1) % 256
            assert f[i, j].c == i * j * 5000 % 65536
