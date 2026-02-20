import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_spirv_dual_return():
    """
    This is a bug that manifested on Mac Metal
    """
    n = 5

    @qd.kernel
    def k1() -> tuple[bool, bool]:
        a = False
        for i_b in qd.ndrange(1):
            a = True

        b = False
        for i_b in qd.ndrange(1):
            b = i_b > n
        return a, b

    assert k1()[0]  # should be True
