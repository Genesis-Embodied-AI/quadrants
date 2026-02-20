import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

# TODO: these are not really tests...


@test_utils.test(arch=qd.cuda)
def test_do_nothing():
    @qd.kernel
    def test():
        for i in range(10):
            impl.call_internal("do_nothing")

    test()


@test_utils.test(arch=qd.cuda)
def test_active_mask():
    @qd.kernel
    def test():
        for i in range(48):
            if i % 2 == 0:
                impl.call_internal("test_active_mask")

    test()


@test_utils.test(arch=qd.cuda)
def test_shfl_down():
    @qd.kernel
    def test():
        for i in range(32):
            impl.call_internal("test_shfl")

    test()
