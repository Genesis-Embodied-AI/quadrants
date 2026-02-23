import quadrants as qd

from tests import test_utils


@test_utils.test(qd.cpu)
def test_func_closure():
    def my_test():
        a = 32

        @qd.func
        def foo():
            qd.static_assert(a == 32)

        @qd.kernel
        def func():
            qd.static_assert(a == 32)
            foo()

        def dummy():
            func()

        func()
        dummy()
        return dummy, func

    dummy, func = my_test()
    func()
    dummy()
