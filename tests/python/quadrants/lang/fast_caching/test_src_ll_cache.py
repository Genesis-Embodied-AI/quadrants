import importlib
import os
import pathlib
import subprocess
import sys

import pydantic
import pytest

import quadrants as qd
import quadrants.lang
from quadrants._test_tools import ti_init_same_arch
from quadrants.lang._kernel_types import SrcLlCacheObservations

from tests import test_utils

TEST_RAN = "test ran"
RET_SUCCESS = 42


@test_utils.test()
def test_src_ll_cache1(tmp_path: pathlib.Path) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.kernel
    def no_pure() -> None:
        pass

    no_pure()
    assert no_pure._primal is not None
    assert not no_pure._primal.src_ll_cache_observations.cache_key_generated

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.kernel(fastcache=True)
    def has_pure() -> None:
        pass

    has_pure()
    assert has_pure._primal is not None
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert not has_pure._primal.src_ll_cache_observations.cache_validated
    assert not has_pure._primal.src_ll_cache_observations.cache_loaded
    assert has_pure._primal.src_ll_cache_observations.cache_stored
    assert has_pure._primal._last_compiled_kernel_data is not None

    last_compiled_kernel_data_str = None
    if quadrants.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda]:
        # we only support _last_compiled_kernel_data on cpu and cuda
        # and it only changes anything on cuda anyway, because it affects the PTX
        # cache
        last_compiled_kernel_data_str = has_pure._primal._last_compiled_kernel_data._debug_dump_to_string()
        assert last_compiled_kernel_data_str is not None and last_compiled_kernel_data_str != ""

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert has_pure._primal.src_ll_cache_observations.cache_validated
    assert has_pure._primal.src_ll_cache_observations.cache_loaded
    if quadrants.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda]:
        assert has_pure._primal._last_compiled_kernel_data._debug_dump_to_string() == last_compiled_kernel_data_str


@test_utils.test()
def test_src_ll_cache_with_corruption(tmp_path: pathlib.Path) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.pure
    @qd.kernel
    def has_pure() -> None:
        pass

    has_pure()
    assert has_pure._primal is not None
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert not has_pure._primal.src_ll_cache_observations.cache_validated
    assert not has_pure._primal.src_ll_cache_observations.cache_loaded
    assert has_pure._primal.src_ll_cache_observations.cache_stored
    assert has_pure._primal._last_compiled_kernel_data is not None

    # reset observations
    has_pure._primal.src_ll_cache_observations = SrcLlCacheObservations()
    assert not has_pure._primal.src_ll_cache_observations.cache_key_generated

    last_compiled_kernel_data_str = None
    if quadrants.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda]:
        # we only support _last_compiled_kernel_data on cpu and cuda
        # and it only changes anything on cuda anyway, because it affects the PTX
        # cache
        last_compiled_kernel_data_str = has_pure._primal._last_compiled_kernel_data._debug_dump_to_string()
        assert last_compiled_kernel_data_str is not None and last_compiled_kernel_data_str != ""

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    # corrupt the cache files
    for file in tmp_path.glob("python_side_cache/*"):
        print("file", file)
        with open(file, "wb") as f:
            f.write(b"\x00\x0a\xe2\xff\xfe\x80\x99JUNK")
        os.system(f"hexdump -C {file}")

    # check cache doesnt crash
    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert not has_pure._primal.src_ll_cache_observations.cache_validated
    assert not has_pure._primal.src_ll_cache_observations.cache_loaded
    has_pure._primal.src_ll_cache_observations = SrcLlCacheObservations()
    if quadrants.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda]:
        assert has_pure._primal._last_compiled_kernel_data._debug_dump_to_string() == last_compiled_kernel_data_str

    # check cache works again
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert has_pure._primal.src_ll_cache_observations.cache_validated
    assert has_pure._primal.src_ll_cache_observations.cache_loaded
    has_pure._primal.src_ll_cache_observations = SrcLlCacheObservations()
    if quadrants.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda]:
        assert has_pure._primal._last_compiled_kernel_data._debug_dump_to_string() == last_compiled_kernel_data_str


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_src_ll_cache_arg_warnings(tmp_path: pathlib.Path, capfd) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    class RandomClass:
        pass

    @qd.pure
    @qd.kernel
    def k1(foo: qd.Template) -> None:
        pass

    k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][PARAM_INVALID]" in err
    assert RandomClass.__name__ in err
    assert "[FASTCACHE][INVALID_FUNC]" in err
    assert k1.__name__ in err

    @qd.kernel
    def not_pure_k1(foo: qd.Template) -> None:
        pass

    not_pure_k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][PARAM_INVALID]" not in err
    assert RandomClass.__name__ not in err
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert k1.__name__ not in err


@test_utils.test()
def test_src_ll_cache_repeat_after_load(tmp_path: pathlib.Path) -> None:
    """
    Check that repeatedly calling kernel actually works, c.f. was doing
    no-op for a bit.
    """
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.pure
    @qd.kernel
    def has_pure(a: qd.types.NDArray[qd.i32, 1]) -> None:
        a[0] += 1

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = qd.ndarray(qd.i32, (10,))
    a[0] = 5
    for i in range(3):
        has_pure(a)
        assert a[0] == 6 + i

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = qd.ndarray(qd.i32, (10,))
    a[0] = 5
    for i in range(3):
        has_pure(a)
        assert a[0] == 6 + i


@pytest.mark.parametrize("src_ll_cache", [None, False, True])
@test_utils.test()
def test_src_ll_cache_flag(tmp_path: pathlib.Path, src_ll_cache: bool) -> None:
    """
    Test qd.init(src_ll_cache) flag
    """
    if src_ll_cache:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), src_ll_cache=src_ll_cache)
    else:
        ti_init_same_arch()

    @qd.pure
    @qd.kernel
    def k1() -> None:
        pass

    k1()
    cache_used = k1._primal.src_ll_cache_observations.cache_key_generated
    if src_ll_cache:
        assert cache_used == src_ll_cache
    else:
        assert cache_used  # default


class TemplateParamsKernelArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    a: int
    src_ll_cache: bool


def src_ll_cache_template_params_child(args: list[str]) -> None:
    args_obj = TemplateParamsKernelArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=args_obj.src_ll_cache,
    )

    @qd.pure
    @qd.kernel
    def k1(a: qd.template(), output: qd.types.NDArray[qd.i32, 1]) -> None:
        output[0] = a

    output = qd.ndarray(qd.i32, (10,))
    k1(args_obj.a, output)
    assert output[0] == args_obj.a
    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@pytest.mark.parametrize("src_ll_cache", [False, True])
@test_utils.test()
def test_src_ll_cache_template_params(tmp_path: pathlib.Path, src_ll_cache: bool) -> None:
    """
    template primitive kernel params should be in the cache key
    """
    arch = qd.lang.impl.current_cfg().arch.name

    def create_args(a: int) -> str:
        obj = TemplateParamsKernelArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path),
            src_ll_cache=src_ll_cache,
            a=a,
        )
        json = TemplateParamsKernelArgs.model_dump_json(obj)
        return json

    env = os.environ
    env["PYTHONPATH"] = "."
    for a in [3, 4]:
        proc = subprocess.run(
            [sys.executable, __file__, src_ll_cache_template_params_child.__name__, create_args(a)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != RET_SUCCESS:
            print(proc.stdout)  # needs to do this to see error messages
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS


class HasReturnKernelArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    src_ll_cache: bool
    return_something: bool
    expect_used_src_ll_cache: bool
    expect_src_ll_cache_hit: bool


def src_ll_cache_has_return_child(args: list[str]) -> None:
    args_obj = HasReturnKernelArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=args_obj.src_ll_cache,
    )

    @qd.pure
    @qd.kernel
    def k1(a: qd.i32, output: qd.types.NDArray[qd.i32, 1]) -> bool:
        output[0] = a
        if qd.static(args_obj.return_something):
            return True

    output = qd.ndarray(qd.i32, (10,))
    if args_obj.return_something:
        assert k1(3, output)
        # Sanity check that the kernel actually ran, and did something.
        assert output[0] == 3
        assert k1._primal.src_ll_cache_observations.cache_key_generated == args_obj.expect_used_src_ll_cache
        assert k1._primal.src_ll_cache_observations.cache_loaded == args_obj.expect_src_ll_cache_hit
        assert k1._primal.src_ll_cache_observations.cache_validated == args_obj.expect_src_ll_cache_hit
    else:
        # Even though we only check when not loading from the cache
        # we won't ever be able to load from the cache, since it will have failed
        # to cache the first time. By induction, it will always raise.
        with pytest.raises(
            qd.QuadrantsSyntaxError, match="Kernel has a return type but does not have a return statement"
        ):
            k1(3, output)
    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@pytest.mark.parametrize("return_something", [False, True])
@pytest.mark.parametrize("src_ll_cache", [False, True])
@test_utils.test()
def test_src_ll_cache_has_return(tmp_path: pathlib.Path, src_ll_cache: bool, return_something: bool) -> None:
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    # need to test what happens when loading from fast cache, so run several runs
    # - first iteration stores to cache
    # - second and third will load from cache
    for it in range(3):
        args_obj = HasReturnKernelArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path),
            src_ll_cache=src_ll_cache,
            return_something=return_something,
            expect_used_src_ll_cache=src_ll_cache,
            expect_src_ll_cache_hit=src_ll_cache and it > 0,
        )
        args_json = HasReturnKernelArgs.model_dump_json(args_obj)
        cmd_line = [sys.executable, __file__, src_ll_cache_has_return_child.__name__, args_json]
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


@test_utils.test()
def test_src_ll_cache_self_arg_checked(tmp_path: pathlib.Path) -> None:
    """
    Check that modifiying primtiive values in a data oriented object does result
    in the kernel correctly recompiling to reflect those new values, even with pure on.
    """
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.data_oriented
    class MyDataOrientedChild:
        def __init__(self) -> None:
            self.b = 10

    @qd.data_oriented
    class MyDataOriented:
        def __init__(self) -> None:
            self.a = 3
            self.child = MyDataOrientedChild()

        @qd.pure
        @qd.kernel
        def k1(self) -> tuple[qd.i32, qd.i32]:
            return self.a, self.child.b

    my_do = MyDataOriented()

    # weirdly, if I don't use the name to get the arch, then on Mac github CI, the value of
    # arch can change during the below execcution 🤔
    # TODO: figure out why this is happening, and/or remove arch from python config object (replace
    # with arch_name and arch_idx for example)
    arch = getattr(qd, qd.lang.impl.current_cfg().arch.name)

    # need to initialize up front, in order that config hash doesn't change when we re-init later
    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 5
    my_do.child.b = 20
    assert tuple(my_do.k1()) == (5, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert not my_do.k1._primal.src_ll_cache_observations.cache_validated

    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 5
    assert tuple(my_do.k1()) == (5, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert my_do.k1._primal.src_ll_cache_observations.cache_validated

    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 7
    assert tuple(my_do.k1()) == (7, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert not my_do.k1._primal.src_ll_cache_observations.cache_validated

    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 7
    assert tuple(my_do.k1()) == (7, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert my_do.k1._primal.src_ll_cache_observations.cache_validated

    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.child.b = 30
    assert tuple(my_do.k1()) == (7, 30)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert not my_do.k1._primal.src_ll_cache_observations.cache_validated

    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.child.b = 30
    assert tuple(my_do.k1()) == (7, 30)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert my_do.k1._primal.src_ll_cache_observations.cache_validated


class ModifySubFuncKernelArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    module_file_path: str
    module_name: str
    expected_val: int
    expect_loaded_from_fastcache: bool


def src_ll_cache_modify_sub_func_child(args: list[str]) -> None:
    args_obj: ModifySubFuncKernelArgs = ModifySubFuncKernelArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=True,
    )

    sys.path.append(args_obj.module_file_path)
    mod = importlib.import_module(args_obj.module_name)

    a = qd.ndarray(qd.i32, (10,))
    mod.k1(a)
    assert a[0] == args_obj.expected_val
    assert mod.k1._primal.src_ll_cache_observations.cache_loaded == args_obj.expect_loaded_from_fastcache

    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@test_utils.test()
def test_src_ll_cache_modify_sub_func(tmp_path: pathlib.Path) -> None:
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    kernels_src = """
import quadrants as qd

@qd.kernel(fastcache=True)
def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
    f1(a)

@qd.func
def f1(a: qd.types.NDArray[qd.i32, 1]) -> None:
    a[0] = {val}
"""

    module_file_path = tmp_path / "module"
    module_file_path.mkdir()
    file_path = module_file_path / "foo.py"
    # Note: it's VERY important that the first two values are different,
    # and the last two values are the SAME
    # We had a bug as follows:
    # - first value => ran correclty, saved to c++ + python cache
    # - second value => detects cache invalid, so
    #   - compiles from fresh
    #   - gets correct results,
    #   - attempts to save out
    #   - importantly, ONLY saved to python cache, not c++ cache
    # - if the third value is differnet again, it detects the cache is invalid,
    #   and compiles from fresh again, and it passes
    # - however, if however the third value matches the second value:
    #   - the cache key matches hte previous value
    #   - the python validation passes (since we didnt change the underlying kernel in any way, sicne last time)
    #   - however, the c++ saved kernel, in the cache, still contains the 123 kernel
    #   - => so the assert fails, demonstrating the bug
    for val, expect_loaded_from_fastcache in [(123, False), (222, False), (222, True)]:
        rendered_kernels = kernels_src.format(val=val)
        file_path.write_text(rendered_kernels)
        args_obj = ModifySubFuncKernelArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path / "cache"),
            module_file_path=str(module_file_path),
            module_name="foo",
            expected_val=val,
            expect_loaded_from_fastcache=expect_loaded_from_fastcache,
        )
        args_json = HasReturnKernelArgs.model_dump_json(args_obj)
        cmd_line = [sys.executable, __file__, src_ll_cache_modify_sub_func_child.__name__, args_json]
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


@test_utils.test()
def test_src_ll_cache_dupe_kernels(tmp_path: pathlib.Path) -> None:
    use_fast_cache = True
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name

    qd.init(arch=getattr(qd, arch), src_ll_cache=True, offline_cache=True, offline_cache_file_path=str(tmp_path))

    @qd.func
    def f1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        a[0] = 123

    @qd.kernel(fastcache=use_fast_cache)
    def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        f1(a)

    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert a[0] == 123
    assert not k1._primal.src_ll_cache_observations.cache_loaded

    qd.init(arch=getattr(qd, arch), src_ll_cache=True, offline_cache=True, offline_cache_file_path=str(tmp_path))
    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert a[0] == 123
    assert k1._primal.src_ll_cache_observations.cache_loaded

    qd.init(arch=getattr(qd, arch), src_ll_cache=True, offline_cache=True, offline_cache_file_path=str(tmp_path))

    @qd.func
    def f1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        a[0] = 222

    @qd.kernel(fastcache=use_fast_cache)
    def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        f1(a)

    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert not k1._primal.src_ll_cache_observations.cache_loaded
    assert a[0] == 222

    qd.init(arch=getattr(qd, arch), src_ll_cache=True, offline_cache=True, offline_cache_file_path=str(tmp_path))
    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert k1._primal.src_ll_cache_observations.cache_loaded
    assert a[0] == 222


# The following lines are critical for subprocess-using tests to work. If they are missing, the tests will
# incorrectly pass, without doing anything.
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
