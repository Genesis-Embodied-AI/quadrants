import quadrants as qd
from quadrants._test_tools import qd_init_same_arch
from quadrants.lang import impl
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
def test_config_hasher_ignores_debug_and_runtime_only_keys():
    assert qd.cfg is not None

    qd_init_same_arch()
    h_base = config_hasher.hash_compile_config()

    # Only selects where IR dumps are written, and only when QD_DUMP_IR is set.
    qd.cfg.debug_dump_path = "/tmp/some-other-ir-dump-dir/"
    assert config_hasher.hash_compile_config() == h_base

    # Only sets a CUDA context limit at runtime.
    qd.cfg.cuda_stack_limit = 32768
    assert config_hasher.hash_compile_config() == h_base

    # A process-local queue handle, read live by the runtime. Two addresses, and no queue at all, all hash the same.
    qd.cfg.external_metal_command_queue = 0x1234
    assert config_hasher.hash_compile_config() == h_base
    qd.cfg.external_metal_command_queue = 0x5678
    assert config_hasher.hash_compile_config() == h_base
    qd.cfg.external_metal_command_queue = 0
    assert config_hasher.hash_compile_config() == h_base

    # A genuine compiler input still changes the hash, so the exclusions above are not masking everything.
    qd.cfg.fast_math = not qd.cfg.fast_math
    assert config_hasher.hash_compile_config() != h_base


@test_utils.test()
def test_hash_device_caps_is_stable_and_tagged():
    assert qd.cfg is not None

    qd_init_same_arch()
    h1 = config_hasher.hash_device_caps()
    h2 = config_hasher.hash_device_caps()

    # Deterministic for a fixed device, and non-empty.
    assert h1 == h2
    assert h1

    # Mixes in exactly the fingerprint the native offline-cache path folds into its own key, so the two caches agree
    # on device identity. The leading 'T' mirrors get_hashed_offline_cache_key (key must start with a letter).
    assert h1 == impl.get_runtime().prog.get_device_caps().hashed_key()
    assert h1.startswith("T")
