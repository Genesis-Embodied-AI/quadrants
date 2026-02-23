import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_offload_with_cross_block_locals():
    ret = qd.field(qd.f32)

    qd.root.place(ret)

    @qd.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[None] = s

    ker()

    assert ret[None] == 45


@test_utils.test()
def test_offload_with_cross_block_locals2():
    ret = qd.field(qd.f32)

    qd.root.place(ret)

    @qd.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[None] = s
        s = ret[None] * 2
        for i in range(10):
            qd.atomic_add(ret[None], s)

    ker()

    assert ret[None] == 45 * 21


@test_utils.test()
def test_offload_with_cross_block_locals3():
    ret = qd.field(qd.f32, shape=())

    @qd.kernel
    def ker():
        s = 1
        t = s
        for i in range(10):
            s += i
        ret[None] = t

    ker()

    assert ret[None] == 1


@test_utils.test()
def test_offload_with_cross_block_locals4():
    ret = qd.field(qd.f32, shape=())

    @qd.kernel
    def ker():
        a = 1
        b = 0
        for i in range(10):
            b += a
        ret[None] = b

    ker()

    assert ret[None] == 10


@test_utils.test()
def test_offload_with_flexible_bounds():
    s = qd.field(qd.i32, shape=())
    lower = qd.field(qd.i32, shape=())
    upper = qd.field(qd.i32, shape=())

    @qd.kernel
    def ker():
        for i in range(lower[None], upper[None]):
            s[None] += i

    lower[None] = 10
    upper[None] = 20
    ker()

    assert s[None] == 29 * 10 // 2


@test_utils.test()
def test_offload_with_cross_block_globals():
    ret = qd.field(qd.f32)

    qd.root.place(ret)

    @qd.kernel
    def ker():
        ret[None] = 0
        for i in range(10):
            ret[None] += i
        ret[None] += 1

    ker()

    assert ret[None] == 46


@test_utils.test(exclude=qd.amdgpu)
def test_offload_with_cross_nested_for():
    @qd.kernel
    def run(a: qd.i32):
        b = a + 1
        for x in range(1):
            for i in range(b):
                print("OK")

    run(2)


@test_utils.test(exclude=qd.amdgpu)
def test_offload_with_cross_if_inside_for():
    @qd.kernel
    def run(a: qd.i32):
        b = a > 2
        for x in range(1):
            if b:
                print("OK")

    run(2)


@test_utils.test(exclude=qd.amdgpu)
def test_offload_with_save():
    a = qd.Vector.field(2, dtype=qd.f32, shape=1)
    b = qd.Vector.field(2, dtype=qd.f32, shape=1)
    c = qd.Vector.field(2, dtype=qd.f32, shape=1)

    @qd.kernel
    def test():
        a[0] = qd.Vector([1, 1])
        b[0] = qd.Vector([0, 0])
        c[0] = qd.Vector([0, 0])
        b[0] += a[0]  # b[0] = [1, 1]
        b[0] /= 2  # b[0] = [0.5, 0.5]
        for i in c:
            c[i] += b[0]  # c[0] = [0.5, 0.5]

    test()
    assert c[0][0] == 0.5
    assert c[0][1] == 0.5
