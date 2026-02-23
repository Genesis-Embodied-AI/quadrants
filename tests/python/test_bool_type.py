import quadrants as qd

from tests import test_utils


@test_utils.test(debug=True)
def test_bool_type_anno():
    @qd.func
    def f(x: bool) -> bool:
        return not x

    @qd.kernel
    def test():
        assert f(True) == False
        assert f(False) == True

    test()


@test_utils.test(debug=True)
def test_bool_type_conv():
    @qd.func
    def f(x: qd.u32) -> bool:
        return bool(x)

    @qd.kernel
    def test():
        assert f(1000) == True
        assert f(qd.u32(4_294_967_295)) == True

    test()
