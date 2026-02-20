import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_nested_kernel_error():
    @qd.kernel
    def B():
        pass

    @qd.kernel
    def A():
        B()

    with pytest.raises(qd.QuadrantsCompilationError):
        A()
