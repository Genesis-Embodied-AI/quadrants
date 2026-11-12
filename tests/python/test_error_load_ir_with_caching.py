"""QD_LOAD_IR / QUADRANTS_LOAD_PTX read replacement IR / PTX from ``debug_dump_path``, which codegen only does for a
kernel it actually compiles. A cached kernel skips codegen, so the edited files would be ignored with no diagnostic.
``qd.init`` rejects the combination instead, for the backends that actually read each variable."""

import os
from contextlib import contextmanager

import pytest

import quadrants as qd
from quadrants.lang import misc

# qd.cpu is x64 or arm64, both of which go through the LLVM codegen that reads QD_LOAD_IR.
LLVM_ARCH = qd.cpu


@contextmanager
def env_vars(**overrides):
    """Set the given env vars for the duration of the block, restoring them afterwards. A value of None unsets."""
    previous = {name: os.environ.get(name) for name in overrides}

    def apply(values):
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    apply(overrides)
    try:
        yield
    finally:
        apply(previous)


class FakeCfg:
    """Stand-in for CompileConfig, so the arch matrix can be checked without that backend being available."""

    def __init__(self, arch, offline_cache=True):
        self.arch = arch
        self.offline_cache = offline_cache


# Checks through qd.init, on the host CPU arch, which is always available and always an LLVM arch.


def test_error_load_ir_with_offline_cache():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        with pytest.raises(ValueError, match="QD_LOAD_IR"):
            qd.init(arch=LLVM_ARCH, log_level="warn", offline_cache=True, src_ll_cache=False)


def test_error_load_ir_with_fastcache():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        with pytest.raises(ValueError, match="QD_LOAD_IR"):
            qd.init(arch=LLVM_ARCH, log_level="warn", offline_cache=False, src_ll_cache=True)


def test_no_error_load_ir_with_caching_disabled():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        qd.init(arch=LLVM_ARCH, log_level="warn", offline_cache=False, src_ll_cache=False)


def test_no_error_when_load_ir_is_zero():
    # QD_LOAD_IR goes through get_environ_config on the C++ side, which parses it as an int, so "0" is off.
    with env_vars(QD_LOAD_IR="0", QUADRANTS_LOAD_PTX=None):
        qd.init(arch=LLVM_ARCH, log_level="warn", offline_cache=True, src_ll_cache=True)


def test_no_error_without_load_env_vars():
    with env_vars(QD_LOAD_IR=None, QUADRANTS_LOAD_PTX=None):
        qd.init(arch=LLVM_ARCH, log_level="warn", offline_cache=True, src_ll_cache=True)


# Checks against the helper directly, so each arch can be exercised on any test machine. Going through qd.init would
# not work: adaptive_arch_select silently falls back to the CPU arch for an unavailable backend, which would turn a
# non-LLVM arch back into an LLVM one.


@pytest.mark.parametrize("arch", [qd.vulkan, qd.metal])
def test_no_error_load_ir_on_non_llvm_arch(arch):
    # Only the LLVM codegen reads QD_LOAD_IR, so the SPIR-V backends cannot consume it and must not be blocked.
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        misc._check_ir_load_envs_against_caching(FakeCfg(arch), src_ll_cache=True)


@pytest.mark.parametrize("arch", [qd.cuda, qd.amdgpu])
def test_error_load_ir_on_llvm_gpu_arch(arch):
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX=None):
        with pytest.raises(ValueError, match="QD_LOAD_IR"):
            misc._check_ir_load_envs_against_caching(FakeCfg(arch), src_ll_cache=True)


def test_no_error_when_load_ptx_is_zero():
    # QUADRANTS_LOAD_PTX goes through get_environ_config on the C++ side too, so "0" is off, as for QD_LOAD_IR.
    with env_vars(QD_LOAD_IR=None, QUADRANTS_LOAD_PTX="0"):
        misc._check_ir_load_envs_against_caching(FakeCfg(qd.cuda), src_ll_cache=True)


def test_error_load_ptx_on_cuda():
    with env_vars(QD_LOAD_IR=None, QUADRANTS_LOAD_PTX="1"):
        with pytest.raises(ValueError, match="QUADRANTS_LOAD_PTX"):
            misc._check_ir_load_envs_against_caching(FakeCfg(qd.cuda), src_ll_cache=False)


@pytest.mark.parametrize("arch", [qd.vulkan, qd.metal])
def test_no_error_load_ptx_on_non_cuda_arch(arch):
    # QUADRANTS_LOAD_PTX is only read by the CUDA JIT, so other backends must not be blocked by it.
    with env_vars(QD_LOAD_IR=None, QUADRANTS_LOAD_PTX="1"):
        misc._check_ir_load_envs_against_caching(FakeCfg(arch), src_ll_cache=True)


def test_no_error_on_cuda_when_caching_fully_disabled():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX="1"):
        misc._check_ir_load_envs_against_caching(FakeCfg(qd.cuda, offline_cache=False), src_ll_cache=False)


def test_error_names_both_variables_on_cuda():
    with env_vars(QD_LOAD_IR="1", QUADRANTS_LOAD_PTX="1"):
        with pytest.raises(ValueError, match="QD_LOAD_IR and QUADRANTS_LOAD_PTX"):
            misc._check_ir_load_envs_against_caching(FakeCfg(qd.cuda), src_ll_cache=True)
