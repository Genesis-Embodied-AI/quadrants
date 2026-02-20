import pathlib

import quadrants as qd
from quadrants._test_tools import ti_init_same_arch
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_cache_primitive_args():
    @qd.data_oriented
    class StructStaticConfig:
        flag: bool = True

    @qd.kernel
    def fun(static_args: qd.template(), constant: qd.template(), value: qd.types.ndarray()):
        if qd.static(static_args.flag):
            if qd.static(constant > 0):
                value[None] = value[None] + 1
            else:
                assert "Invalid 'constant' branch"
        else:
            assert "Invalid 'static_args.flag' branch"

    assert len(fun._primal.compiled_kernel_data_by_key) == 0
    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 0

    static_args = StructStaticConfig()
    constant = 1234567890
    value = qd.ndarray(qd.i32, shape=())
    value[None] = 1

    fun(static_args, constant, value)
    assert value[None] == 2
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1

    fun(static_args, 1234567890, value)
    assert value[None] == 3
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1

    static_args_2 = StructStaticConfig()
    assert id(static_args) != id(static_args_2)
    fun(static_args_2, constant, value)
    assert value[None] == 4
    assert len(fun._primal.compiled_kernel_data_by_key) == 2
    assert len(fun._primal.mapper._mapping_cache) == 2
    assert len(fun._primal.mapper._mapping_cache_tracker) == 2
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 2
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 2


@test_utils.test(arch=get_host_arch_list())
def test_cache_multi_entry_static():
    @qd.kernel
    def fun(flag: qd.template(), value: qd.template()):
        if qd.static(flag):
            value[None] = value[None] + 1
        else:
            value[None] = value[None] - 1

    assert len(fun._primal.compiled_kernel_data_by_key) == 0
    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 0

    value = qd.field(qd.i32, shape=())
    value[None] = 1

    fun(True, value)
    assert value[None] == 2
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1

    fun(True, value)
    assert value[None] == 3
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1

    fun(False, value)
    assert value[None] == 2
    assert len(fun._primal.compiled_kernel_data_by_key) == 2
    assert len(fun._primal.mapper._mapping_cache) == 2
    assert len(fun._primal.mapper._mapping_cache_tracker) == 2
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 2
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 2


@test_utils.test(arch=get_host_arch_list())
def test_cache_fields_only():
    @qd.kernel
    def fun(flag: qd.template(), value: qd.template()):
        if qd.static(flag):
            value[None] = value[None] + 1
        else:
            assert "Invalid 'static_args.flag_1' branch"

    assert len(fun._primal.compiled_kernel_data_by_key) == 0
    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 0

    flag = True
    value = qd.field(qd.i32, shape=())
    value[None] = 1

    fun(flag, value)
    assert value[None] == 2
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1

    fun(flag, value)
    assert value[None] == 3
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1


@test_utils.test(arch=get_host_arch_list())
def test_cache_ndarray_only():
    @qd.kernel
    def fun(value: qd.types.ndarray()):
        value[None] = value[None] + 1

    assert len(fun._primal.compiled_kernel_data_by_key) == 0
    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 0
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 0

    value = qd.ndarray(qd.i32, shape=())
    value[None] = 1

    fun(value)
    assert value[None] == 2
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1

    fun(value)
    assert value[None] == 3
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache) == 1
    assert len(fun._primal.launch_context_buffer_cache._launch_ctx_cache_tracker) == 1


@test_utils.test(arch=get_host_arch_list())
def test_fastcache(tmp_path: pathlib.Path, monkeypatch):
    launch_kernel_orig = qd.lang.kernel_impl.Kernel.launch_kernel

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    is_valid = False

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args):
        nonlocal is_valid
        is_valid = True
        assert compiled_kernel_data is None
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    @qd.kernel(fastcache=True)
    def fun(value: qd.types.ndarray(), offset: qd.template()):
        value[None] = value[None] + offset

    assert len(fun._primal.compiled_kernel_data_by_key) == 0

    value = qd.ndarray(qd.i32, shape=())
    value[None] = 1

    assert not is_valid
    fun(value, 3)
    assert is_valid
    assert value[None] == 4
    assert len(fun._primal.compiled_kernel_data_by_key) == 1

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    is_valid = False

    def launch_kernel(self, key, t_kernel, compiled_kernel_data, *args):
        nonlocal is_valid
        is_valid = True
        assert compiled_kernel_data is not None
        return launch_kernel_orig(self, key, t_kernel, compiled_kernel_data, *args)

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.launch_kernel", launch_kernel)

    value = qd.ndarray(qd.i32, shape=())
    value[None] = 1

    assert not is_valid
    fun(value, 3)
    assert is_valid
    assert value[None] == 4
    assert len(fun._primal.compiled_kernel_data_by_key) == 1
