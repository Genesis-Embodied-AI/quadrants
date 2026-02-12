import quadrants as ti

from tests.test_utils import test


@test()
def test_smoke() -> None:
    @ti.kernel
    def k1(a: ti.Template, b: ti.types.NDArray[ti.i32, 1]) -> None:
        a[0] += b[0]

    a = ti.field(ti.i32, (10,))
    b = ti.ndarray(ti.i32, (10,))
    a[0] = 3
    b[0] = 5
    k1(a, b)
    assert a[0] == 8
