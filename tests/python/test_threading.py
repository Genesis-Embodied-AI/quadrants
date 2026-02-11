import quadrants as ti
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_while():
    assert ti._lib.core.test_threading()
