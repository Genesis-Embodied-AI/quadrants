import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_empty():
    @qd.kernel
    def func():
        pass

    func()


@test_utils.test()
def test_empty_args():
    @qd.kernel
    def func(x: qd.i32, arr: qd.types.ndarray()):
        pass

    import numpy as np

    func(42, np.arange(10, dtype=np.float32))
