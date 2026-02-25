import os
import shutil
import tempfile

import pytest

import quadrants as qd
from quadrants.lang.util import has_clangpp

from tests import test_utils


@pytest.mark.skipif(not has_clangpp(), reason="Clang not installed.")
@test_utils.test(arch=[qd.x64, qd.cuda])
def test_source_builder_from_source():
    source_bc = """
    extern "C" {
        void add_and_mul(float *a, float *b, float *c, float *d, int *e) {
            *c = (*a) + (*b);
            *d = (*a) * (*b);
            *e = int((*a) * (*b) + (*a));
        }
        void pow_int(int *a, int *b, int *c) {
            int ret = 1;
            for (int i = 0; i < (*b); i++)
                ret = ret * (*a);
            *c = ret;
        }
    }
    """
    sb_bc = qd.lang.source_builder.SourceBuilder.from_source(source_bc)

    @qd.kernel
    def func_bc() -> qd.i32:
        a = 2.0
        b = 3.0
        c = 0.0
        d = 0.0
        e = 3
        sb_bc.add_and_mul(a, b, c, d, e)
        p = 0
        c_plus_d = int(c + d)
        sb_bc.pow_int(c_plus_d, e, p)
        return p

    assert func_bc() == 11**8


@pytest.mark.skipif(not has_clangpp(), reason="Clang not installed.")
@test_utils.test(arch=[qd.x64, qd.cuda])
def test_source_builder_from_file():
    source_code = """
    extern "C" {
        void add_and_mul(float *a, float *b, float *c, float *d, int *e) {
            *c = (*a) + (*b);
            *d = (*a) * (*b);
            *e = int((*a) * (*b) + (*a));
        }
        void pow_int(int *a, int *b, int *c) {
            int ret = 1;
            for (int i = 0; i < (*b); i++)
                ret = ret * (*a);
            *c = ret;
        }
    }
    """

    td = tempfile.mkdtemp()
    fn = os.path.join(td, "source.cpp")
    with open(fn, "w") as f:
        f.write(source_code)
    sb_bc = qd.lang.source_builder.SourceBuilder.from_file(fn)

    @qd.kernel
    def func_bc() -> qd.i32:
        a = 2.0
        b = 3.0
        c = 0.0
        d = 0.0
        e = 3
        sb_bc.add_and_mul(a, b, c, d, e)
        p = 0
        c_plus_d = int(c + d)
        sb_bc.pow_int(c_plus_d, e, p)
        return p

    assert func_bc() == 11**8

    shutil.rmtree(td)
