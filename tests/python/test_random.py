import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_random_float():
    for precision in [qd.f32, qd.f64]:
        qd.init()
        n = 1024
        x = qd.field(qd.f32, shape=(n, n))

        @qd.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    x[i, j] = qd.random(precision)

        fill()
        X = x.to_numpy()
        for i in range(1, 4):
            assert (X**i).mean() == test_utils.approx(1 / (i + 1), rel=1e-2)


@test_utils.test()
def test_random_int():
    for precision in [qd.i32, qd.i64]:
        qd.init()
        n = 1024
        x = qd.field(qd.f32, shape=(n, n))

        @qd.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    v = qd.random(precision)
                    if precision == qd.i32:
                        x[i, j] = (float(v) + float(2**31)) / float(2**32)
                    else:
                        x[i, j] = (float(v) + float(2**63)) / float(2**64)

        fill()
        X = x.to_numpy()
        for i in range(1, 4):
            assert (X**i).mean() == test_utils.approx(1 / (i + 1), rel=1e-2)


@test_utils.test()
def test_random_independent_product():
    n = 1024
    x = qd.field(qd.f32, shape=n * n)

    @qd.kernel
    def fill():
        for i in range(n * n):
            a = qd.random()
            b = qd.random()
            x[i] = a * b

    fill()
    X = x.to_numpy()
    for i in range(4):
        assert X.mean() == test_utils.approx(1 / 4, rel=1e-2)


@test_utils.test()
def test_random_2d_dist():
    n = 8192

    x = qd.Vector.field(2, dtype=qd.f32, shape=n)

    @qd.kernel
    def gen():
        for i in range(n):
            x[i] = qd.Vector([qd.random(), qd.random()])

    gen()

    X = x.to_numpy()
    counters = [0 for _ in range(4)]
    for i in range(n):
        c = int(X[i, 0] < 0.5) * 2 + int(X[i, 1] < 0.5)
        counters[c] += 1

    for c in range(4):
        assert counters[c] / n == test_utils.approx(1 / 4, rel=0.2)


@test_utils.test()
def test_random_seed_per_launch():
    n = 10
    x = qd.field(qd.f32, shape=n)

    @qd.kernel
    def gen(i: qd.i32):
        x[i] = qd.random()

    count = 0
    gen(0)
    for i in range(1, n):
        gen(i)
        count += 1 if x[i] == x[i - 1] else 0

    assert count <= n * 0.15


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.metal])
def test_random_seed_per_program():
    import numpy as np

    n = 10
    result = []
    for s in [0, 1]:
        qd.init(random_seed=s)
        x = qd.field(qd.f32, shape=n)

        @qd.kernel
        def gen():
            for i in x:
                x[i] = qd.random()

        gen()
        result.append(x.to_numpy())
        qd.reset()

    assert not np.allclose(result[0], result[1])


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_random_f64():
    """
    Tests the granularity of float64 random numbers.
    See https://github.com/taichi-dev/quadrants/issues/2251 for an explanation.
    """
    import numpy as np

    n = int(2**23)
    x = qd.field(qd.f64, shape=n)

    @qd.kernel
    def foo():
        for i in x:
            x[i] = qd.random(dtype=qd.f64)

    foo()
    frac, _ = np.modf(x.to_numpy() * 4294967296)
    assert np.max(frac) > 0


@test_utils.test()
def test_randn():
    """
    Tests the generation of Gaussian random numbers.
    """
    for precision in [qd.f32, qd.f64]:
        qd.init()
        n = 1024
        x = qd.field(qd.f32, shape=(n, n))

        @qd.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    x[i, j] = qd.randn(precision)

        fill()
        X = x.to_numpy()

        # https://en.wikipedia.org/wiki/Normal_distribution#Moments
        moments = [0.0, 1.0, 0.0, 3.0]
        for i in range(4):
            assert (X ** (i + 1)).mean() == test_utils.approx(moments[i], abs=3e-2)
