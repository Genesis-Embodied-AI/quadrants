from concurrent.futures import ThreadPoolExecutor

import pytest

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang._fast_caching import config_hasher, config_reads, src_hasher
from quadrants.lang._wrap_inspect import get_source_info_and_src

from tests.helpers import test_utils
from tests.helpers.runtime import qd_init_same_arch


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


@test_utils.test(arch=qd.cpu)
@pytest.mark.parametrize("source", ("current", "alias", "default", "worker"))
def test_random_seed_read_flag(monkeypatch, source):
    monkeypatch.setattr(config_reads, "random_seed_read", False)
    qd_init_same_arch(random_seed=11)
    assert not config_reads.random_seed_read
    alias = impl.current_cfg()
    alias.random_seed = 12
    assert not config_reads.random_seed_read

    if source == "current":
        assert qd.cfg.random_seed == 12
    elif source == "alias":
        assert getattr(alias, "random_seed") == 12
    elif source == "default":
        assert impl.default_cfg().random_seed == 11
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(getattr, alias, "random_seed").result() == 12

    assert config_reads.random_seed_read
    qd_init_same_arch(random_seed=13)
    assert config_reads.random_seed_read


@test_utils.test(arch=qd.cpu)
def test_seed_cache_key_variants(monkeypatch):
    monkeypatch.setattr(config_reads, "random_seed_read", False)
    monkeypatch.setenv("QD_RANDOM_SEED", "1")
    qd_init_same_arch()

    @qd.kernel
    def kernel():
        pass

    info, _ = get_source_info_and_src(kernel._primal.func)
    seed_free = src_hasher.make_source_config_key(info, include_random_seed=False)
    with_seed = src_hasher.make_source_config_key(info)
    assert seed_free != with_seed
    assert not config_reads.random_seed_read
    assert qd.cfg.random_seed == 1

    qd.cfg.random_seed = 2
    assert src_hasher.make_source_config_key(info, include_random_seed=False) == seed_free
    assert src_hasher.make_source_config_key(info) != with_seed


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
