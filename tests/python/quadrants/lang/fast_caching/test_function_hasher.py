import pathlib
import shutil
from typing import Callable

import pytest

from quadrants.lang import _wrap_inspect
from quadrants.lang._fast_caching import function_hasher
from quadrants.lang._fast_caching.fast_caching_types import HashedFunctionSourceInfo

from tests import test_utils


@test_utils.test()
def test_read_file(tmp_path: pathlib.Path) -> None:
    out_filepath = tmp_path / "somefile.txt"
    with open(out_filepath, "w") as f:
        f.write(
            """0
1
2
3
4
5
"""
        )
    info = _wrap_inspect.FunctionSourceInfo(
        function_name="foo", filepath=str(out_filepath), start_lineno=1, end_lineno=3
    )
    src = function_hasher._read_file(info)
    assert (
        "".join(src)
        == """1
2
3
"""
    )


@test_utils.test()
@pytest.mark.xfail()
def test_read_file_with_arrows(monkeypatch, tmp_path: pathlib.Path) -> None:
    """
    This test doesn't actually reproduce the 0xe28094 bug with the hyphen, but anyway
    """
    out_filepath = tmp_path / "somefile.txt"
    with open(out_filepath, "w") as f:
        f.write(
            """0
1 →
2 😂
3 🙌 —
4 🙃
5 🔥
"""
        )
    info = _wrap_inspect.FunctionSourceInfo(
        function_name="foo", filepath=str(out_filepath), start_lineno=1, end_lineno=3
    )
    src = function_hasher._read_file(info)
    assert (
        "".join(src)
        == """1 →
2 😂
3 🙌 —
"""
    )


@test_utils.test()
def test_function_hasher_hash_kernel(monkeypatch, temporary_module) -> None:
    test_files_path = "tests/python/quadrants/lang/fast_caching/test_files"
    monkeypatch.syspath_prepend(test_files_path)

    def get_hash(name: str) -> _wrap_inspect.FunctionSourceInfo:
        mod = temporary_module(name)
        info, _src = _wrap_inspect.get_source_info_and_src(mod.f1.fn)
        hash = function_hasher.hash_kernel(info)
        return hash

    f1_base_hash = get_hash("f1_base")
    f1_same_hash = get_hash("f1_same")
    f1_diff_hash = get_hash("f1_diff")

    assert f1_base_hash is not None and f1_base_hash != ""
    assert f1_base_hash == f1_same_hash
    assert f1_base_hash != f1_diff_hash


@test_utils.test()
def test_function_hasher_hash_functions(monkeypatch, temporary_module) -> None:
    test_files_path = "tests/python/quadrants/lang/fast_caching/test_files"
    monkeypatch.syspath_prepend(test_files_path)

    def get_infos(name: str) -> list[HashedFunctionSourceInfo]:
        mod = temporary_module(name)
        info, _src = _wrap_inspect.get_source_info_and_src(mod.f1.fn)
        hashed_infos = function_hasher.hash_functions([info])
        return hashed_infos

    f1_base_hashed_infos = get_infos("f1_base")
    f1_same_hashed_infos = get_infos("f1_same")
    f1_diff_hashed_infos = get_infos("f1_diff")

    assert len(f1_base_hashed_infos) == len(f1_same_hashed_infos) == len(f1_diff_hashed_infos) == 1
    assert f1_base_hashed_infos[0].hash is not None and f1_base_hashed_infos[0].hash != ""
    assert f1_base_hashed_infos[0].hash == f1_same_hashed_infos[0].hash
    assert f1_base_hashed_infos[0].hash != f1_diff_hashed_infos[0].hash


@test_utils.test()
def test_function_hasher_validate_hashed_function_infos(monkeypatch, tmp_path: pathlib.Path, temporary_module) -> None:
    test_files_path = pathlib.Path("tests/python/quadrants/lang/fast_caching/test_files")
    monkeypatch.syspath_prepend(str(tmp_path))

    name = "child_diff"

    def setup_folder(filename: str) -> None:
        shutil.copy2(test_files_path / filename, tmp_path / f"{name}.py")

    setup_folder("child_diff_base.py")
    mod = temporary_module(name)

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
