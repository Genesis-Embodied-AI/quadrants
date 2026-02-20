import importlib
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Callable

import pydantic
import pytest

import quadrants as qd
from quadrants._test_tools import ti_init_same_arch
from quadrants.lang import _wrap_inspect
from quadrants.lang._fast_caching import function_hasher, src_hasher
from quadrants.lang._wrap_inspect import get_source_info_and_src

from tests import test_utils

TEST_RAN = "test ran"
RET_SUCCESS = 42


@test_utils.test()
def test_src_hasher_create_cache_key_vary_config() -> None:
    @qd.kernel
    def f1() -> None:
        pass

    # for some reason, print_ir_dbg_info is being set to True after a bit
    # so we are forcing it to false each initialization for now
    ti_init_same_arch(print_ir_dbg_info=False)
    kernel_info, _src = get_source_info_and_src(f1.fn)
    cache_key_base = src_hasher.create_cache_key(False, kernel_info, [], [])

    ti_init_same_arch(print_ir_dbg_info=False)
    kernel_info, _src = get_source_info_and_src(f1.fn)
    cache_key_same = src_hasher.create_cache_key(False, kernel_info, [], [])

    ti_init_same_arch(print_ir_dbg_info=False, random_seed=123)
    kernel_info, _src = get_source_info_and_src(f1.fn)
    cache_key_diff = src_hasher.create_cache_key(False, kernel_info, [], [])

    assert cache_key_base == cache_key_same
    assert cache_key_same != cache_key_diff


@test_utils.test()
def test_src_hasher_validate_hashed_function_infos(monkeypatch, tmp_path: pathlib.Path, temporary_module) -> None:
    test_files_path = pathlib.Path("tests/python/quadrants/lang/fast_caching/test_files")
    monkeypatch.syspath_prepend(tmp_path)

    def setup_folder(filename: str) -> None:
        shutil.copy2(test_files_path / filename, tmp_path / "child_diff.py")

    setup_folder("child_diff_base.py")
    mod = temporary_module("child_diff")

    def get_fileinfos(functions: list[Callable]) -> list[_wrap_inspect.FunctionSourceInfo]:
        fileinfos = []
        for f in functions:
            file_info, _src = _wrap_inspect.get_source_info_and_src(f)
            fileinfos.append(file_info)
        return fileinfos

    fileinfos = get_fileinfos([mod.f1.fn, mod.f2.fn])
    hashed_fileinfos = function_hasher.hash_functions(fileinfos)
    assert function_hasher.validate_hashed_function_infos(hashed_fileinfos)

    setup_folder("child_diff_same.py")
    assert function_hasher.validate_hashed_function_infos(hashed_fileinfos)

    setup_folder("child_diff_diff.py")
    assert not function_hasher.validate_hashed_function_infos(hashed_fileinfos)

    setup_folder("child_diff_same.py")
    assert function_hasher.validate_hashed_function_infos(hashed_fileinfos)


@test_utils.test()
def test_src_hasher_store_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, temporary_module) -> None:
    test_files_path = pathlib.Path("tests/python/quadrants/lang/fast_caching/test_files")

    offline_cache_path = tmp_path / "cache"
    temp_import_path = tmp_path / "temp_import"
    temp_import_path.mkdir(exist_ok=True)

    ti_init_same_arch(offline_cache_file_path=str(offline_cache_path))

    monkeypatch.syspath_prepend(temp_import_path)

    def setup_folder(filename: str) -> None:
        shutil.copy2(test_files_path / filename, temp_import_path / "child_diff_test_src_hasher_store_validate.py")

    def get_fileinfos(functions: list[Callable]) -> list[_wrap_inspect.FunctionSourceInfo]:
        fileinfos = []
        for f in functions:
            file_info, _src = _wrap_inspect.get_source_info_and_src(f)
            fileinfos.append(file_info)
        return fileinfos

    setup_folder("child_diff_base.py")
    mod = temporary_module("child_diff_test_src_hasher_store_validate")
    kernel_info = get_fileinfos([mod.f1.fn])[0]
    fileinfos = get_fileinfos([mod.f1.fn, mod.f2.fn])
    fast_cache_key = src_hasher.create_cache_key(False, kernel_info, [], [])

    assert fast_cache_key is not None

    kernel_cache_key = "I'm a kernel cache key"

    load_res = src_hasher.load(fast_cache_key)
    assert load_res[0] is None and load_res[1] is None

    some_used_vars = {"fee", "fi", "fo"}
    src_hasher.store(kernel_cache_key, fast_cache_key, fileinfos, some_used_vars)

    def assert_loaded(cache_key: str) -> None:
        res = src_hasher.load(cache_key)
        assert res[0] is not None and res[1] is not None

    def assert_not_loaded(cache_key: str) -> None:
        res = src_hasher.load(cache_key)
        assert res[0] is None and res[1] is None

    assert_loaded(fast_cache_key)

    setup_folder("child_diff_same.py")
    assert_loaded(fast_cache_key)

    setup_folder("child_diff_diff.py")
    assert_not_loaded(fast_cache_key)

    setup_folder("child_diff_same.py")
    assert_loaded(fast_cache_key)

    assert_not_loaded("abcdefg")

    assert src_hasher.load(fast_cache_key)[0] == some_used_vars


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=qd.cpu)
@pytest.mark.parametrize(
    "print_non_pure",
    [
        None,
        False,
        True,
    ],
)
def test_src_hasher_print_non_pure(tmp_path: pathlib.Path, print_non_pure: bool | None, capfd) -> None:
    """
    Test qd.init parameter print_non_pure, which should print non pure functions when enabled
    """
    if print_non_pure:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True, print_non_pure=print_non_pure)
    else:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.pure
    @qd.kernel
    def k1_pure() -> None:
        pass

    k1_pure()
    out, _err = capfd.readouterr()
    output_contains_not_pure = "[NOT_PURE]" in out
    assert not output_contains_not_pure

    @qd.kernel
    def not_pure_k1() -> None:
        pass

    not_pure_k1()
    out, _err = capfd.readouterr()
    output_contains_not_pure = "[NOT_PURE]" in out
    if output_contains_not_pure:
        assert not_pure_k1.__name__ in out
    if print_non_pure is None:
        assert not output_contains_not_pure
    else:
        assert output_contains_not_pure == print_non_pure


class VaryKernelFuncKernelArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    module_file_path: str
    module_name: str


def src_hasher_vary_kernel_func_child(args: list[str]) -> None:
    args_obj: VaryKernelFuncKernelArgs = VaryKernelFuncKernelArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=True,
    )
    sys.path.append(args_obj.module_file_path)
    mod = importlib.import_module(args_obj.module_name)
    info, _src = _wrap_inspect.get_source_info_and_src(mod.f1.fn)
    cache_key = src_hasher.create_cache_key(False, info, [], [])
    print(f"CACHE_KEY={cache_key}")

    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_src_hasher_vary_kernel_func(tmp_path: pathlib.Path) -> None:
    test_files_path = pathlib.Path("tests/python/quadrants/lang/fast_caching/test_files")

    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    module_file_path = tmp_path / "module"
    module_file_path.mkdir()

    def get_cache_key(name: str) -> str | None:
        target_mod = "target"
        shutil.copy(test_files_path / f"{name}.py", module_file_path / f"{target_mod}.py")
        args_obj = VaryKernelFuncKernelArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path / "cache"),
            module_file_path=str(module_file_path),
            module_name=target_mod,
        )
        args_json = VaryKernelFuncKernelArgs.model_dump_json(args_obj)
        cmd_line = [sys.executable, __file__, src_hasher_vary_kernel_func_child.__name__, args_json]
        print("cmd_line", " ".join(cmd_line))
        proc = subprocess.run(
            cmd_line,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != RET_SUCCESS:
            print(" ".join(cmd_line))
            print(proc.stdout)  # needs to do this to see error messages
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS
        cache_key = None
        for line in proc.stdout.split("\n"):
            print("line", line)
            if line.startswith("CACHE_KEY="):
                cache_key = line.strip().partition("=")[2]
        assert cache_key
        return cache_key

    key_base = get_cache_key("f1_base")
    key_same = get_cache_key("f1_same")
    key_diff = get_cache_key("f1_diff")

    assert key_base is not None
    assert key_base != ""
    assert key_base == key_same
    assert key_base != key_diff


# The following lines are critical for subprocess-using tests to work.
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
