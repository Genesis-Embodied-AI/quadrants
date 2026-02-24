import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_name_error():
    with pytest.raises(qd.QuadrantsNameError, match='Name "a" is not defined'):

        @qd.kernel
        def foo():
            a + 1

        foo()
