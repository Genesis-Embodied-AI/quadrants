import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse)
def test_basic():
    x = qd.field(qd.i32)
    c = qd.field(qd.i32)
    s = qd.field(qd.i32)

    bm = qd.root.bitmasked(qd.ij, (3, 6)).bitmasked(qd.i, 8)
    bm.place(x)
    qd.root.place(c, s)

    @qd.kernel
    def run():
        x[5, 1] = 2
        x[9, 4] = 20
        x[0, 3] = 20

    @qd.kernel
    def sum():
        for i, j in x:
            c[None] += qd.is_active(bm, [i, j])
            s[None] += x[i, j]

    run()
    sum()

    assert c[None] == 3
    assert s[None] == 42


@test_utils.test(require=qd.extension.sparse)
def test_bitmasked_then_dense():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.bitmasked(qd.i, n).dense(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[256] = 1
    x[257] = 1

    func()
    assert s[None] == 256


@test_utils.test(require=qd.extension.sparse)
def test_bitmasked_bitmasked():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.bitmasked(qd.i, n).bitmasked(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[256] = 1
    x[257] = 1

    func()
    assert s[None] == 4


@test_utils.test(require=qd.extension.sparse)
def test_huge_bitmasked():
    # Mainly for testing Metal listgen's grid-stride loop implementation.
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 1024

    qd.root.bitmasked(qd.i, n).bitmasked(qd.i, 2 * n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i in range(n * n * 2):
            if i % 32 == 0:
                x[i] = 1.0

    @qd.kernel
    def count():
        for i in x:
            s[None] += 1

    func()
    count()
    assert s[None] == (n * n * 2) // 32


@test_utils.test(require=qd.extension.sparse)
def test_bitmasked_listgen_bounded():
    # Mainly for testing Metal's listgen is bounded by the actual number of
    # elements possible for that SNode. Note that 1) SNode's size is padded
    # to POT, and 2) Metal ListManager's data size is not padded, we need to
    # make sure listgen doesn't go beyond ListManager's capacity.
    x = qd.field(qd.i32)
    c = qd.field(qd.i32)

    # A prime that is bit higher than 65536, which is Metal's maximum number of
    # threads for listgen.
    n = 80173

    qd.root.dense(qd.i, n).bitmasked(qd.i, 1).place(x)
    qd.root.place(c)

    @qd.kernel
    def func():
        for i in range(n):
            x[i] = 1

    @qd.kernel
    def count():
        for i in x:
            c[None] += 1

    func()
    count()
    assert c[None] == n


@test_utils.test(require=qd.extension.sparse)
def test_deactivate():
    # https://github.com/taichi-dev/quadrants/issues/778
    a = qd.field(qd.i32)
    a_a = qd.root.bitmasked(qd.i, 4)
    a_b = a_a.dense(qd.i, 4)
    a_b.place(a)
    c = qd.field(qd.i32)
    qd.root.place(c)

    @qd.kernel
    def run():
        a[0] = 123

    @qd.kernel
    def is_active():
        c[None] = qd.is_active(a_a, [0])

    @qd.kernel
    def deactivate():
        qd.deactivate(a_a, [0])

    run()
    is_active()
    assert c[None] == 1

    deactivate()
    is_active()
    assert c[None] == 0


@test_utils.test(require=qd.extension.sparse)
def test_sparsity_changes():
    x = qd.field(qd.i32)
    c = qd.field(qd.i32)
    s = qd.field(qd.i32)

    bm = qd.root.bitmasked(qd.i, 5).bitmasked(qd.i, 4)
    bm.place(x)
    qd.root.place(c, s)

    @qd.kernel
    def run():
        for i in x:
            s[None] += x[i]
            c[None] += 1

    # Only two elements of |x| are activated
    x[1] = 2
    x[8] = 20
    run()
    assert c[None] == 2
    assert s[None] == 22

    c[None] = 0
    s[None] = 0
    # Four elements are activated now
    x[7] = 15
    x[14] = 5

    run()
    assert c[None] == 4
    assert s[None] == 42


@test_utils.test(require=qd.extension.sparse)
def test_bitmasked_offset_child():
    x = qd.field(qd.i32)
    x2 = qd.field(qd.i32)
    y = qd.field(qd.i32)
    y2 = qd.field(qd.i32)
    y3 = qd.field(qd.i32)
    z = qd.field(qd.i32)
    s = qd.field(qd.i32, shape=())

    n = 16
    # Offset children:
    # * In |bm|'s cell: |bm2| has a non-zero offset
    # * In |bm2|'s cell: |z| has a non-zero offset
    # * We iterate over |z| to test the listgen handles offsets correctly
    bm = qd.root.bitmasked(qd.i, n)
    bm.dense(qd.i, 16).place(x, x2)
    bm2 = bm.bitmasked(qd.i, 4)

    bm2.dense(qd.i, 4).place(y, y2, y3)
    bm2.bitmasked(qd.i, 4).place(z)

    @qd.kernel
    def func():
        for _ in z:
            s[None] += 1

    z[0] = 1
    z[7] = 1
    z[42] = 1
    z[53] = 1
    z[88] = 1
    z[101] = 1
    z[233] = 1

    func()
    assert s[None] == 7


@test_utils.test(require=qd.extension.sparse)
def test_bitmasked_2d_power_of_two():
    some_val = qd.field(dtype=float)
    width, height = 10, 10
    total = width * height
    ptr = qd.root.bitmasked(qd.ij, (width, height))
    ptr.place(some_val)
    num_active = qd.field(dtype=int, shape=())

    @qd.kernel
    def init():
        num_active[None] = 0
        for x, y in qd.ndrange(width, height):
            some_val[x, y] = 5
            num_active[None] += 1

    @qd.kernel
    def run():
        num_active[None] = 0
        for x, y in some_val:
            num_active[None] += 1

    init()
    assert num_active[None] == total
    run()
    assert num_active[None] == total


@test_utils.test(require=qd.extension.sparse)
def test_root_deactivate():
    a = qd.field(qd.i32)
    a_a = qd.root.bitmasked(qd.i, 4)
    a_b = a_a.dense(qd.i, 4)
    a_b.place(a)
    c = qd.field(qd.i32)
    qd.root.place(c)

    @qd.kernel
    def run():
        a[0] = 123

    @qd.kernel
    def is_active():
        c[None] = qd.is_active(a_a, [0])

    run()
    is_active()
    assert c[None] == 1

    qd.root.deactivate_all()
    is_active()
    assert c[None] == 0
