import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_simplify_bug():
    @qd.kernel
    def foo() -> qd.types.vector(4, dtype=qd.i32):
        a = qd.Vector([0, 0, 0, 0])
        for i in range(5):
            for k in qd.static(range(4)):
                if i == 3:
                    a[k] = 1
        return a

    a = foo()

    assert (a == qd.Vector([1, 1, 1, 1])).all() == 1
