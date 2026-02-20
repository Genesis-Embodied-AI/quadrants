import quadrants as qd
from quadrants._test_tools import ti_init_same_arch
from quadrants.lang._fast_caching import config_hasher

from tests import test_utils


@test_utils.test()
def test_config_hasher():
    assert qd.cfg is not None

    ti_init_same_arch()
    h_base = config_hasher.hash_compile_config()

    ti_init_same_arch()
    h_same = config_hasher.hash_compile_config()

    ti_init_same_arch(random_seed=123)
    h_diff = config_hasher.hash_compile_config()

    assert h_base == h_same
    assert h_base != h_diff
