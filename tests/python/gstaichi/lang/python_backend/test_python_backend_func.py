import quadrants as qd


def test_python_backend_func() -> None:
    qd.init(qd.python)

    @qd.func
    def foo2(a: qd.types.ndarray[qd.i32, 1]):
        a[0] = 5

    a = qd.ndarray(qd.i32, (10,))
    a[0] = 3
    foo2(a)
    assert a[0] == 5

    @qd.func
    def foo3(a: qd.types.ndarray[qd.i32, 1]):
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] += 1

    foo3(a)
    a.fill(3)
    foo3(a)
    assert a[2] == 4

    @qd.func
    def foo4(a: qd.types.ndarray[qd.i32, 1]):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] += 1

    a.fill(4)
    foo4(a)
    assert a[2] == 5
