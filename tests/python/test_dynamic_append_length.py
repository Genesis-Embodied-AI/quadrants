import quadrants as qd

from tests import test_utils


def _test_dynamic_append_length(dt):
    x = qd.field(int)
    block = qd.root.dense(qd.i, 10)
    pixel = block.dynamic(qd.j, 10)
    pixel.place(x)

    @qd.kernel
    def test():
        for i in range(10):
            for j in range(i):
                x[i].append(j)
        for i in range(10):
            assert x[i].length() == i
            for j in range(i):
                assert x[i, j] == j

    test()


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal], default_fp=qd.f32, debug=True)
def test_dynamic_append_length_f32():
    _test_dynamic_append_length(qd.f32)
