import quadrants as qd


def test_python_backend_kernel() -> None:
    qd.init(qd.python)

    @qd.kernel
    def k1():
        print("hello")

    k1()

    @qd.kernel
    def foo2(a: qd.types.ndarray[qd.i32, 1]):
        print("hello")
        a[0] = 5

    print('about to create a')
    a = qd.ndarray(qd.i32, (10,))
    print('after a create')
    a[0] = 3
    print('a[0]', a[0])
    foo2(a)
    print('a[0]', a[0])

    @qd.kernel
    def foo3(a: qd.types.ndarray[qd.i32, 1]):
        print("hello")
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] += 1

    foo3(a)
    print('a', a)

    @qd.kernel
    def foo4(a: qd.types.ndarray[qd.i32, 1]):
        print("hello")
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] += 1

    foo4(a)
    print('a', a)

    @qd.kernel(fastcache=True)
    def k1b():
        print("hello")

    k1b()
