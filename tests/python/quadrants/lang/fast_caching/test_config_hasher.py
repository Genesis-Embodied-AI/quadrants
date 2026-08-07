import quadrants as qd
from quadrants._test_tools import qd_init_same_arch
from quadrants.lang._fast_caching import config_hasher

from tests import test_utils


@test_utils.test()
def test_config_hasher():
    assert qd.cfg is not None

    qd_init_same_arch()
    h_base = config_hasher.hash_compile_config()

    qd_init_same_arch()
    h_same = config_hasher.hash_compile_config()

    qd_init_same_arch(random_seed=123)
    h_diff = config_hasher.hash_compile_config()

    assert h_base == h_same
    assert h_base != h_diff


@test_utils.test()
def test_config_hasher_normalizes_torch_mps_queue_address():
    qd_init_same_arch()
    assert qd.cfg is not None

    qd.cfg.external_metal_command_queue_is_torch_queue = True
    qd.cfg.external_metal_command_queue = 0x1234
    h_first_queue = config_hasher.hash_compile_config()

    qd.cfg.external_metal_command_queue = 0x5678
    h_second_queue = config_hasher.hash_compile_config()

    assert h_first_queue == h_second_queue

    # Arbitrary external queues may belong to a different Metal device, so
    # their addresses remain part of the identity.
    qd.cfg.external_metal_command_queue_is_torch_queue = False
    h_external_queue = config_hasher.hash_compile_config()
    qd.cfg.external_metal_command_queue = 0x9ABC
    h_other_external_queue = config_hasher.hash_compile_config()
    assert h_external_queue != h_other_external_queue

    # A missing queue also keeps its own identity, even if the companion flag
    # is accidentally set.
    qd.cfg.external_metal_command_queue_is_torch_queue = True
    qd.cfg.external_metal_command_queue = 0
    h_missing_queue = config_hasher.hash_compile_config()
    assert h_second_queue != h_missing_queue
