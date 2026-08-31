import importlib
import os
import pathlib
import subprocess
import sys

import pydantic
import pytest

import quadrants as qd

_KERNEL_COVERAGE = os.environ.get("QD_KERNEL_COVERAGE") == "1"
import quadrants.lang
from quadrants._test_tools import qd_init_same_arch
from quadrants.lang._kernel_types import SrcLlCacheObservations

from tests import test_utils

TEST_RAN = "test ran"
RET_SUCCESS = 42


@test_utils.test()
def test_src_ll_cache1(tmp_path: pathlib.Path) -> None:
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.kernel
    def no_pure() -> None:
        pass

    no_pure()
    assert no_pure._primal is not None
    assert not no_pure._primal.src_ll_cache_observations.cache_key_generated

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

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

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert has_pure._primal.src_ll_cache_observations.cache_validated
    assert has_pure._primal.src_ll_cache_observations.cache_loaded
    if quadrants.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda]:
        assert has_pure._primal._last_compiled_kernel_data._debug_dump_to_string() == last_compiled_kernel_data_str


@pytest.mark.skipif(
    _KERNEL_COVERAGE,
    reason="Coverage probes change LLVM IR addresses after reinit, breaking recompile comparison",
)
@test_utils.test()
def test_src_ll_cache_with_corruption(tmp_path: pathlib.Path) -> None:
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

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

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
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
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
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
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    class RandomClass:
        pass

    @qd.pure
    @qd.kernel
    def k1(foo: qd.Template) -> None:
        pass

    k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    # An unrecognised type at a kernel-read path fails fastcache loudly: ``[UNKNOWN_TYPE]`` names the type and
    # ``[INVALID_FUNC]`` reports the disabled cache. See ``args_hasher.py::_fail_unknown_type``.
    assert "[FASTCACHE][UNKNOWN_TYPE]" in err
    assert RandomClass.__name__ in err
    assert "[FASTCACHE][INVALID_FUNC]" in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err

    @qd.kernel
    def not_pure_k1(foo: qd.Template) -> None:
        pass

    not_pure_k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    # Without ``@qd.pure``, fastcache is not active at all, so none of its diagnostics should fire.
    assert "[FASTCACHE][UNKNOWN_TYPE]" not in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert k1.__name__ not in err


@test_utils.test()
def test_src_ll_cache_repeat_after_load(tmp_path: pathlib.Path) -> None:
    """
    Check that repeatedly calling kernel actually works, c.f. was doing
    no-op for a bit.
    """
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.pure
    @qd.kernel
    def has_pure(a: qd.types.NDArray[qd.i32, 1]) -> None:
        a[0] += 1

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = qd.ndarray(qd.i32, (10,))
    a[0] = 5
    for i in range(3):
        has_pure(a)
        assert a[0] == 6 + i

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
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
        qd_init_same_arch(offline_cache_file_path=str(tmp_path), src_ll_cache=src_ll_cache)
    else:
        qd_init_same_arch()

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
    def k1(a: qd.i32, output: qd.types.NDArray[qd.i32, 1], return_something: qd.Template) -> bool:
        output[0] = a
        if qd.static(return_something):
            return True

    output = qd.ndarray(qd.i32, (10,))
    if args_obj.return_something:
        assert k1(3, output, args_obj.return_something)
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
            k1(3, output, args_obj.return_something)
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
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

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


@test_utils.test()
def test_src_ll_cache_needs_grad_distinguishes_args_hash(tmp_path: pathlib.Path) -> None:
    """Pin: the narrow args_hash must fold in ``needs_grad`` for every ndarray leaf, or two scenes differing only by
    ``requires_grad`` collide on the L2 key.

    ``insert_ndarray_param`` bakes grad-presence into the compiled parameter slot, while the launch path picks
    ``_QD_ARRAY`` vs ``_QD_ARRAY_WITH_GRAD`` off ``v.grad is not None``. Bind a with-grad ndarray to a slot declared
    without one and the primal pointer lands at the wrong offset: wrong results or OOB.

    This is the Genesis ``kernel_init_link_fields`` shape at minimum size - a frozen dataclass of two ndarrays, a
    kernel that writes the second - run across two ``qd.init`` cycles sharing a cache directory.
    """
    import dataclasses

    import numpy as np

    arch = getattr(qd, qd.lang.impl.current_cfg().arch.name)
    N = 4

    @dataclasses.dataclass(frozen=True)
    class State:
        a: qd.types.NDArray[qd.f32, 1]
        b: qd.types.NDArray[qd.f32, 1]

    @qd.pure
    @qd.kernel
    def write_b(s: State) -> None:
        for i in range(N):
            s.b[i] = qd.cast(i + 1, qd.f32) * 7.0

    # Cold run, needs_grad=False: the stored artifact declares ``s.b``'s slot without a grad pointer.
    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    a1 = qd.ndarray(qd.f32, shape=(N,))
    b1 = qd.ndarray(qd.f32, shape=(N,))
    state1 = State(a=a1, b=b1)
    write_b(state1)
    assert write_b._primal.src_ll_cache_observations.cache_key_generated
    assert not write_b._primal.src_ll_cache_observations.cache_loaded
    expected = np.array([7, 14, 21, 28], dtype=np.float32)
    np.testing.assert_allclose(b1.to_numpy(), expected)

    # Hot run, needs_grad=True: this must miss L2. Were the args_hash to collide with the run above, the launch
    # would route ``b2`` through ``_QD_ARRAY_WITH_GRAD`` against a slot compiled as plain ``_QD_ARRAY``.
    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    a2 = qd.ndarray(qd.f32, shape=(N,), needs_grad=True)
    b2 = qd.ndarray(qd.f32, shape=(N,), needs_grad=True)
    state2 = State(a=a2, b=b2)
    write_b(state2)
    assert not write_b._primal.src_ll_cache_observations.cache_loaded, (
        "fastcache hit between needs_grad=False (cold) and needs_grad=True (hot) - narrow args_hash is "
        "missing needs_grad, the without-grad artifact will be launched against with-grad ndarrays"
    )
    np.testing.assert_allclose(b2.to_numpy(), expected)
    # A misaligned param struct would smear primal data into the (unwritten) grad slot.
    np.testing.assert_allclose(b2.grad.to_numpy(), np.zeros(N, dtype=np.float32))


@test_utils.test()
def test_src_ll_cache_hit_predeclare_struct_ndarrays_pruned(tmp_path: pathlib.Path) -> None:
    """Pin ``_predeclare_struct_ndarrays`` on the fastcache-hit path, where pass 0 is skipped and the ``id(nd)``-keyed
    used-ndarray set is therefore empty: registration has to be gated on the cached flat-name set instead, so that the
    same ndarray set is registered as by the compile that produced the artifact.

    Registering every reachable ndarray instead scrambles the arg-slot bindings, and the write lands in ``state.a``
    (first in insertion order) rather than ``state.b``. Both the cold and hot paths run here via ``qd.reset()``.
    """
    import numpy as np  # local import keeps the test module's top-level deps unchanged

    arch = getattr(qd, qd.lang.impl.current_cfg().arch.name)
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self) -> None:
            self.a = qd.ndarray(qd.i32, shape=(N,))
            self.b = qd.ndarray(qd.i32, shape=(N,))
            self.c = qd.ndarray(qd.i32, shape=(N,))

    @qd.pure
    @qd.kernel
    def write_b(s: qd.template()) -> None:
        for i in range(N):
            s.b[i] = (i + 1) * 17

    # Cold: populates the fastcache, flat-name set included.
    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    state = State()
    write_b(state)
    assert write_b._primal.src_ll_cache_observations.cache_key_generated
    assert not write_b._primal.src_ll_cache_observations.cache_loaded
    np.testing.assert_array_equal(state.b.to_numpy(), np.array([17, 34, 51, 68], dtype=np.int32))
    np.testing.assert_array_equal(state.a.to_numpy(), np.zeros(N, dtype=np.int32))
    np.testing.assert_array_equal(state.c.to_numpy(), np.zeros(N, dtype=np.int32))

    # Hot: the cache-hit path, which skips pass 0.
    qd.reset()
    qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    state = State()
    write_b(state)
    assert write_b._primal.src_ll_cache_observations.cache_loaded, "expected a fastcache hit on the second run"
    np.testing.assert_array_equal(state.b.to_numpy(), np.array([17, 34, 51, 68], dtype=np.int32))
    np.testing.assert_array_equal(state.a.to_numpy(), np.zeros(N, dtype=np.int32))
    np.testing.assert_array_equal(state.c.to_numpy(), np.zeros(N, dtype=np.int32))


@test_utils.test()
def test_src_ll_cache_pruning_union_across_static_branches(tmp_path: pathlib.Path) -> None:
    """Pin: the L1 pruning entry is shared by every specialization of a kernel source, so it must hold the *union* of
    the paths they read - ``qd.static`` makes those paths specialization-dependent.

    With only the first specialization's paths (``flag=True``, reading ``s.flag`` and ``s.x``), the sibling
    specialization (``flag=False``, reading ``s.y``) derives an L2 key that ignores ``s.y``, and a later process whose
    ``s.y`` has a different dtype is served the artifact compiled for the old one - f32 bits land in an i32 ndarray
    (``1073741824`` instead of ``2``).

    Run 4 pins the other side: growing the union must converge, not recompile on every launch.
    """
    import numpy as np  # local import keeps the test module's top-level deps unchanged

    arch = getattr(qd, qd.lang.impl.current_cfg().arch.name)
    N = 4

    @qd.data_oriented
    class State:
        def __init__(self, flag, x, y) -> None:
            self.flag = flag
            self.x = x
            self.y = y

    @qd.pure
    @qd.kernel
    def write_branch(s: qd.template()) -> None:
        if qd.static(s.flag):
            for i in range(N):
                s.x[i] = 1
        else:
            for i in range(N):
                s.y[i] = 2

    def run(flag: bool, y_dtype):
        qd.reset()
        qd.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
        x = qd.ndarray(qd.f32, shape=(N,))
        y = qd.ndarray(y_dtype, shape=(N,))
        write_branch(State(flag, x, y))
        return write_branch._primal.src_ll_cache_observations.cache_loaded, y.to_numpy()

    # Run 1: L1 records ``s.flag`` + ``s.x``, not ``s.y``.
    loaded, _ = run(True, qd.f32)
    assert not loaded

    # Run 2: this specialization reads ``s.y``, so the union has to grow before its L2 key is derived.
    loaded, y_values = run(False, qd.f32)
    assert not loaded
    np.testing.assert_array_equal(y_values, np.full(N, 2, dtype=np.float32))

    loaded, y_values = run(False, qd.i32)
    assert not loaded, (
        "fastcache hit after a dtype change on s.y - the L2 key was narrowed by a pruning set that omits s.y, "
        "so the artifact compiled for f32 is being launched against an i32 ndarray"
    )
    np.testing.assert_array_equal(y_values, np.full(N, 2, dtype=np.int32))

    loaded, y_values = run(False, qd.i32)
    assert loaded, "expected a fastcache hit once the pruning union stopped growing"
    np.testing.assert_array_equal(y_values, np.full(N, 2, dtype=np.int32))


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
