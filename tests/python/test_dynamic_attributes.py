import quadrants as qd

from tests import test_utils


def _test_dynamic_attributes(dt):
    n = 10
    x = qd.field(int)
    block = qd.root.dense(qd.i, n)
    pixel = block.dynamic(qd.j, n)
    pixel.place(x)

    y = qd.field(int)
    row = qd.root.dynamic(qd.i, n)
    row.place(y)

    @qd.kernel
    def test():
        # test append for depth 2 snode
        for i in range(n):
            for j in range(i):
                x[i].append(j)

        for i in range(n):
            assert x[i].length() == i
            for j in range(i):
                assert x[i, j] == j

        # test deactivate for depth 2 snode
        for i in range(n):
            x[i].deactivate()
            for j in range(n):
                assert x[i, j] == 0

        # test append for depth 1 snode in both two ways
        for j in range(n):
            y.append(j)
            assert y[j] == j

        # appending elements to fully active cells will take no effect
        y.deactivate()
        for j in range(n):
            qd.append(y.parent(), [], j * j)
            assert y[j] == j * j

        # test deactivate for depth 1 snode in both two ways
        y.deactivate()
        for j in range(n):
            assert y[j] == 0
            y[j] = j

        qd.deactivate(y.parent(), [])
        for j in range(n):
            assert y[j] == 0

    test()


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal], default_fp=qd.f32, debug=True)
def test_dynamic_attributes_f32():
    _test_dynamic_attributes(qd.f32)
