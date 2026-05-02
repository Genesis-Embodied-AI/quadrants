"""
Tests for fastcache warning suppression when Fields arrive via qd.Tensor annotations.

Cases:
1. Struct with qd.Tensor fields, all populated with Fields → no warning (FIELD_VIA_TENSOR)
2. Struct with qd.Tensor fields, all populated with Ndarrays → no warning (fastcache succeeds)
3. Struct with qd.Tensor fields, mix of Fields and Ndarrays → no warning (FIELD_VIA_TENSOR)
4. Struct with non-Tensor annotation, populated with Field → warning fires
5. Unknown type passed directly to a pure kernel → warning fires
6. Non-pure kernel with Fields → no warning (fastcache not attempted)
"""
import dataclasses
import sys

import pytest

import quadrants as qd
from quadrants._test_tools import qd_init_same_arch

from tests import test_utils


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_no_warning_for_field_via_tensor(tmp_path, capfd):
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
def test_no_warning_for_ndarray_via_tensor(tmp_path, capfd):
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
def test_no_warning_for_mixed_field_ndarray_via_tensor(tmp_path, capfd):
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
def test_warning_for_field_in_non_tensor_annotation(tmp_path, capfd):
    """Struct with a ScalarField field annotated as float (not qd.Tensor) — warning should fire."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @dataclasses.dataclass(frozen=True)
    class S:
        a: float = None

    f = qd.field(qd.f32, shape=(4,))
    s = S(a=f)

    @qd.pure
    @qd.kernel
    def k(x: S):
        pass

    capfd.readouterr()
    k(s)
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][INVALID_FUNC]" in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_warning_for_unknown_type_in_pure_kernel(tmp_path, capfd):
    """Passing an unknown type to a pure kernel — warning should fire (existing behavior)."""
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    class UnknownType:
        pass

    @qd.pure
    @qd.kernel
    def k(x: qd.Template):
        pass

    capfd.readouterr()
    k(UnknownType())
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][PARAM_INVALID]" in err
    assert "[FASTCACHE][INVALID_FUNC]" in err


@test_utils.test(arch=qd.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_no_warning_for_non_pure_kernel_with_fields(tmp_path, capfd):
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
