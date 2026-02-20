import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], debug=True)
def test_proper_typecheck():
    @qd.kernel
    def test():
        impl.call_internal("test_internal_func_args", 1.0, 2.0, 3)

    test()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], debug=True)
def test_type_mismatch():
    @qd.kernel
    def test():
        impl.call_internal("test_internal_func_args", 1, 2, 3)

    with pytest.raises(qd.QuadrantsTypeError, match="expected f32 for argument #1, but got i32"):
        test()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], debug=True)
def test_arg_length_mismatch():
    @qd.kernel
    def test():
        impl.call_internal("test_internal_func_args", 1.0)

    with pytest.raises(qd.QuadrantsTypeError, match="1 arguments were passed in but expected 3"):
        test()


# re-xyr: Consider adding tests for TyVarMismatch, TraitMismatch once we have signatures that can raise such errors
