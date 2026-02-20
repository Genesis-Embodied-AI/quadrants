import time

import quadrants as qd


def template_fuse_dense_x2y2z(
    size=1024**3,
    repeat=10,
    first_n=100,
):
    x = qd.field(qd.i32, shape=(size,))
    y = qd.field(qd.i32, shape=(size,))
    z = qd.field(qd.i32, shape=(size,))
    first_n = min(first_n, size)

    @qd.kernel
    def x_to_y():
        for i in x:
            y[i] = x[i] + 1

    @qd.kernel
    def y_to_z():
        for i in x:
            z[i] = y[i] + 4

    def x_to_y_to_z():
        x_to_y()
        y_to_z()

    for i in range(first_n):
        x[i] = i * 10

    # Simply test
    for _ in range(repeat):
        t = time.time()
        x_to_y()
        qd.sync()
        print("x_to_y", time.time() - t)

    for _ in range(repeat):
        t = time.time()
        y_to_z()
        qd.sync()
        print("y_to_z", time.time() - t)

    for _ in range(repeat):
        t = time.time()
        x_to_y_to_z()
        qd.sync()
        print("fused x->y->z", time.time() - t)

    for i in range(first_n):
        assert x[i] == i * 10
        assert y[i] == x[i] + 1
        assert z[i] == x[i] + 5


def template_fuse_reduction(size=1024**3, repeat=10, first_n=100):
    x = qd.field(qd.i32, shape=(size,))
    first_n = min(first_n, size)

    @qd.kernel
    def reset():
        for i in range(first_n):
            x[i] = i * 10

    @qd.kernel
    def inc():
        for i in x:
            x[i] = x[i] + 1

    # Simply test
    reset()
    qd.sync()
    for _ in range(repeat):
        t = time.time()
        inc()
        qd.sync()
        print("single inc", time.time() - t)

    reset()
    qd.sync()
    t = time.time()
    for _ in range(repeat):
        inc()
    qd.sync()
    duration = time.time() - t
    print(f"fused {repeat} inc: total={duration} average={duration / repeat}")

    for i in range(first_n):
        assert x[i] == i * 10 + repeat
