"""
Tests for fastcache warning suppression when Fields arrive via qd.Tensor annotations.

Cases:
1. Struct with qd.Tensor fields, all populated with Fields → no warning
2. Struct with qd.Tensor fields, all populated with Ndarrays → no warning (fastcache succeeds)
3. Struct with qd.Tensor fields, mix of Fields and Ndarrays → no warning
4. Non-pure kernel with Fields → no warning (fastcache not attempted)
5. Field passed directly to qd.Tensor kernel param → no warning
6. Field passed directly to qd.Template kernel param → warning fires
7. Struct with qd.Template-annotated field containing a Field → warning fires
"""

import dataclasses
import sys

import pytest

import quadrants as qd
from quadrants._test_tools import qd_init_same_arch

from tests import test_utils


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_no_warning_for_field_via_tensor(tmp_path, capfd):
    """Struct whose qd.Tensor fields contain Fields — fastcache silently skipped."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @dataclasses.dataclass(frozen=True)
    class S:
        a: qd.Tensor = None
        b: qd.Tensor = None

    s = S(a=qd.field(qd.f32, shape=(4,)), b=qd.field(qd.i32, shape=(2,)))

    @qd.pure
    @qd.kernel
    def k(x: S):
        pass

    capfd.readouterr()
    k(s)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_no_warning_for_ndarray_via_tensor(tmp_path, capfd):
    """Struct whose qd.Tensor fields contain Ndarrays — fastcache succeeds, no warning."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @dataclasses.dataclass(frozen=True)
    class S:
        a: qd.Tensor = None
        b: qd.Tensor = None

    s = S(a=qd.ndarray(qd.f32, shape=(4,)), b=qd.ndarray(qd.i32, shape=(2,)))

    @qd.pure
    @qd.kernel
    def k(x: S):
        pass

    capfd.readouterr()
    k(s)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_no_warning_for_mixed_field_ndarray(tmp_path, capfd):
    """Struct with qd.Tensor fields containing a mix of Field and Ndarray — still silently skipped."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @dataclasses.dataclass(frozen=True)
    class S:
        a: qd.Tensor = None
        b: qd.Tensor = None

    s = S(a=qd.field(qd.f32, shape=(4,)), b=qd.ndarray(qd.i32, shape=(2,)))

    @qd.pure
    @qd.kernel
    def k(x: S):
        pass

    capfd.readouterr()
    k(s)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_no_warning_non_pure_kernel(tmp_path, capfd):
    """Non-pure kernel with Field args — fastcache not attempted, no warning."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @dataclasses.dataclass(frozen=True)
    class S:
        a: qd.Tensor = None

    s = S(a=qd.field(qd.f32, shape=(4,)))

    @qd.kernel
    def k(x: S):
        pass

    capfd.readouterr()
    k(s)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_no_warning_top_level_tensor(tmp_path, capfd):
    """Field passed directly to a kernel parameter annotated as qd.Tensor — no warning."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.pure
    @qd.kernel
    def k(x: qd.Tensor):
        pass

    f = qd.field(qd.f32, shape=(4,))
    capfd.readouterr()
    k(f)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert "[FASTCACHE][PARAM_INVALID]" not in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_warn_top_level_template(tmp_path, capfd):
    """Field passed directly to a kernel parameter annotated as qd.Template — warning should fire."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @qd.pure
    @qd.kernel
    def k(x: qd.Template):
        pass

    f = qd.field(qd.f32, shape=(4,))
    capfd.readouterr()
    k(f)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_fastcache_field_warnings_warn_struct_template_field(tmp_path, capfd):
    """Struct with qd.Template-annotated field containing a Field - warning should fire when the field is
    actually read by the kernel.

    Pruning-driven narrowing of args hashing only walks members the kernel reads; an unused dataclass field cannot
    affect kernel codegen so it's correctly omitted from the hash (and from the Field-disables-fastcache check). For
    the warning path to fire, the kernel must reference the Field - that matches the user-visible contract that
    fastcache fails iff a "live" Field argument prevents safe parametrisation.
    """
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @dataclasses.dataclass(frozen=True)
    class S:
        a: qd.Template = None

    f = qd.field(qd.f32, shape=(4,))
    s = S(a=f)

    @qd.pure
    @qd.kernel
    def k(x: S):
        x.a[0] = 1

    capfd.readouterr()
    k(s)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" in err
