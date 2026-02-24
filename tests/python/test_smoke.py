import quadrants as qd

from tests.test_utils import test


@test()
def test_smoke() -> None:
    @qd.kernel
    def k1(a: qd.Template, b: qd.types.NDArray[qd.i32, 1]) -> None:
        a[0] += b[0]

    a = qd.field(qd.i32, (10,))
    b = qd.ndarray(qd.i32, (10,))
    a[0] = 3
    b[0] = 5
    k1(a, b)
    assert a[0] == 8
