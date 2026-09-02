import dataclasses
import gc
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest

import quadrants as qd
from quadrants.lang._kernel_types import KernelBatchedArgType
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.impl import Kernel, QuadrantsSyntaxError

from tests import test_utils


@pytest.fixture
def qd_type(use_ndarray: bool) -> Any:
    if use_ndarray:
        return qd.ndarray
    return qd.field


@pytest.fixture
def qd_annotation(use_ndarray: bool) -> Any:
    class QdTemplateBuilder:
        """
        Allows qd_annotation[qd.i32, 2] to be legal
        """

        def __getitem__(self, _):
            return qd.Template

    if use_ndarray:
        return qd.types.ndarray
    return QdTemplateBuilder()


@test_utils.test()
def test_ndarray_struct_kwargs():
    gc.collect()
    gc.collect()

    a = qd.ndarray(qd.i32, shape=(55,))
    b = qd.ndarray(qd.i32, shape=(57,))
    c = qd.ndarray(qd.i32, shape=(211,))
    d = qd.ndarray(qd.i32, shape=(223,))
    e = qd.ndarray(qd.i32, shape=(227,))

    @dataclass
    class MyStruct:
        a: qd.types.NDArray[qd.i32, 1]
        b: qd.types.NDArray[qd.i32, 1]
        c: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def s4(a: qd.types.NDArray[qd.i32, 1], b: qd.types.NDArray[qd.i32, 1]) -> None:
        # note: no used py dataclass parameters
        a[1] += 888
        b[2] += 999

    @qd.func
    def s3(z3: qd.types.NDArray[qd.i32, 1], my_struct3: MyStruct, bar3: qd.types.NDArray[qd.i32, 1]) -> None:
        # used py dataclass variables:
        # __qd_my_struct3__qd_a
        # __qd_my_struct3__qd_b
        # __qd_my_struct3__qd_c
        z3[25] += 90
        my_struct3.a[47] += 42
        my_struct3.b[49] += 43
        my_struct3.c[43] += 44
        bar3[113] += 125
        s4(my_struct3.a, my_struct3.b)

    @qd.func
    def s2(z3: qd.types.NDArray[qd.i32, 1], my_struct3: MyStruct, bar3: qd.types.NDArray[qd.i32, 1]) -> None:
        # used py dataclass variables:
        # __qd_my_struct3__qd_a
        # __qd_my_struct3__qd_b
        # __qd_my_struct3__qd_c
        z3[24] += 89
        my_struct3.a[46] += 32
        my_struct3.b[48] += 33
        my_struct3.c[42] += 34
        bar3[112] += 125
        s3(z3=z3, my_struct3=my_struct3, bar3=bar3)

    @qd.func
    def s1(z2: qd.types.NDArray[qd.i32, 1], my_struct2: MyStruct, bar2: qd.types.NDArray[qd.i32, 1]) -> None:
        # used py dataclass variables:
        # __qd_my_struct2__qd_a
        # __qd_my_struct2__qd_b
        # __qd_my_struct2__qd_c
        z2[22] += 88
        my_struct2.a[45] += 22
        my_struct2.b[47] += 23
        my_struct2.c[41] += 24
        bar2[111] += 123
        s2(z3=z2, my_struct3=my_struct2, bar3=bar2)

    @qd.kernel
    def k1(z: qd.types.NDArray[qd.i32, 1], my_struct: MyStruct, bar: qd.types.NDArray[qd.i32, 1]) -> None:
        # used py dataclass variables:
        # __qd_my_struct__qd_a
        # __qd_my_struct__qd_b
        # __qd_my_struct__qd_c
        z[33] += 2
        my_struct.a[35] += 3
        my_struct.b[37] += 5
        my_struct.c[51] += 17
        bar[222] = 41
        s1(z2=z, my_struct2=my_struct, bar2=bar)

    my_struct = MyStruct(a=a, b=b, c=c)
    k1(z=d, my_struct=my_struct, bar=e)
    assert d[33] == 2
    assert a[35] == 3
    assert b[37] == 5
    assert c[51] == 17

    assert d[22] == 88
    assert a[45] == 22
    assert b[47] == 23
    assert c[41] == 24
    assert e[111] == 123

    assert d[24] == 89
    assert a[46] == 32
    assert b[48] == 33
    assert c[42] == 34
    assert e[112] == 125

    assert d[25] == 90
    assert a[47] == 42
    assert b[49] == 43
    assert c[43] == 44
    assert e[113] == 125

    assert a[1] == 888
    assert b[2] == 999


@test_utils.test()
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_ndarray_struct(qd_type: Any, qd_annotation: Any) -> None:
    gc.collect()
    gc.collect()
    a = qd_type(qd.i32, shape=(55,))
    b = qd_type(qd.i32, shape=(57, 23))
    c = qd_type(qd.i32, shape=(211, 34, 25))
    d = qd_type(qd.i32, shape=(223,))
    e = qd_type(qd.i32, shape=(227,))

    @dataclass
    class MyStruct:
        a: qd_annotation[qd.i32, 1]
        b: qd_annotation[qd.i32, 2]
        c: qd_annotation[qd.i32, 3]

    @qd.func
    def s3(z3: qd_annotation[qd.i32, 1], my_struct3: MyStruct, bar3: qd_annotation[qd.i32, 1]) -> None:
        # stores
        z3[25] += 90
        my_struct3.a[47] += 42
        my_struct3.b[49, 0] += 43
        my_struct3.c[43, 0, 0] += 44
        bar3[113] += 125

        # loads
        bar3[16] = z3[1]
        my_struct3.a[17] = z3[1]
        my_struct3.b[18, 0] = my_struct3.a[3]
        my_struct3.c[19, 0, 0] = my_struct3.b[18, 0]
        z3[20] = my_struct3.c[5, 0, 0]

    @qd.func
    def s2(z3: qd_annotation[qd.i32, 1], my_struct3: MyStruct, bar3: qd_annotation[qd.i32, 1]) -> None:
        # stores
        z3[24] += 89
        my_struct3.a[46] += 32
        my_struct3.b[48, 0] += 33
        my_struct3.c[42, 0, 0] += 34
        bar3[112] += 125
        s3(z3, my_struct3, bar3)

    @qd.func
    def s1(z2: qd_annotation[qd.i32, 1], my_struct2: MyStruct, bar2: qd_annotation[qd.i32, 1]) -> None:
        # stores
        z2[22] += 88
        my_struct2.a[45] += 22
        my_struct2.b[47, 0] += 23
        my_struct2.c[41, 0, 0] += 24
        bar2[111] += 123
        s2(z2, my_struct2, bar2)

    @qd.kernel
    def k1(z: qd_annotation[qd.i32, 1], my_struct: MyStruct, bar: qd_annotation[qd.i32, 1]) -> None:
        # stores
        z[33] += 2
        my_struct.a[35] += 3
        my_struct.b[37, 0] += 5
        my_struct.c[51, 0, 0] += 17
        bar[222] = 41

        # loads
        bar[6] = z[1]
        my_struct.a[7] = z[1]
        my_struct.b[8, 0] = my_struct.a[3]
        my_struct.c[9, 0, 0] = my_struct.b[8, 0]
        z[10] = my_struct.c[5, 0, 0]
        s1(z, my_struct, bar)

    d[1] = 11
    a[3] = 12
    b[2, 0] = 13
    c[5, 0, 0] = 14
    e[4] = 15

    my_struct = MyStruct(a=a, b=b, c=c)
    k1(d, my_struct, e)
    # store tests k1
    assert d[33] == 2
    assert a[35] == 3
    assert b[37, 0] == 5
    assert c[51, 0, 0] == 17

    # from load tests, k1
    assert e[6] == 11
    assert a[7] == 11
    assert b[8, 0] == 12
    assert c[9, 0, 0] == 12
    assert d[10] == 14

    assert d[22] == 88
    assert a[45] == 22
    assert b[47, 0] == 23
    assert c[41, 0, 0] == 24
    assert e[111] == 123

    assert d[24] == 89
    assert a[46] == 32
    assert b[48, 0] == 33
    assert c[42, 0, 0] == 34
    assert e[112] == 125

    # s3 stores
    assert d[25] == 90
    assert a[47] == 42
    assert b[49, 0] == 43
    assert c[43, 0, 0] == 44
    assert e[113] == 125

    # s3 loads
    assert e[16] == 11
    assert a[17] == 11
    assert b[18, 0] == 12
    assert c[19, 0, 0] == 12
    assert d[20] == 14


@test_utils.test()
def test_ndarray_struct_diverse_params():
    gc.collect()
    gc.collect()

    a = qd.ndarray(qd.i32, shape=(55,))
    b = qd.ndarray(qd.i32, shape=(57,))
    c = qd.ndarray(qd.i32, shape=(211,))
    z_param = qd.ndarray(qd.i32, shape=(223,))
    bar_param = qd.ndarray(qd.i32, shape=(227,))

    field1 = qd.field(qd.i32, shape=(300,))

    @dataclass
    class MyStructAB:
        a: qd.types.NDArray[qd.i32, 1]
        b: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class MyStructC:
        c: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def s2(
        my_struct_ab3: MyStructAB,
        z3: qd.types.NDArray[qd.i32, 1],
        fieldparam1_3: qd.Template,
        my_struct_c3: MyStructC,
        bar3: qd.types.NDArray[qd.i32, 1],
    ) -> None:
        # stores
        z3[24] += 89
        my_struct_ab3.a[46] += 32
        my_struct_ab3.b[48] += 33
        my_struct_c3.c[42] += 34
        bar3[112] += 125
        fieldparam1_3[4] = 69

    @qd.func
    def s1(
        z2: qd.types.NDArray[qd.i32, 1],
        my_struct_c2: MyStructC,
        my_struct_ab2: MyStructAB,
        fieldparam1_2: qd.Template,
        bar2: qd.types.NDArray[qd.i32, 1],
    ) -> None:
        # stores
        z2[22] += 88
        my_struct_ab2.a[45] += 22
        my_struct_ab2.b[47] += 23
        my_struct_c2.c[41] += 24
        bar2[111] += 123
        fieldparam1_2[3] = 68

        s2(my_struct_ab2, z2, fieldparam1_2, my_struct_c2, bar2)

    @qd.kernel
    def k1(
        z: qd.types.NDArray[qd.i32, 1],
        my_struct_ab: MyStructAB,
        bar: qd.types.NDArray[qd.i32, 1],
        my_struct_c: MyStructC,
        fieldparam1: qd.Template,
    ) -> None:
        # stores
        z[33] += 2
        my_struct_ab.a[35] += 3
        my_struct_ab.b[37] += 5
        my_struct_c.c[51] += 17
        bar[222] = 41
        fieldparam1[2] = 67

        # loads
        bar[6] = z[1]
        my_struct_ab.a[7] = z[1]
        my_struct_ab.b[8] = my_struct_ab.a[3]
        my_struct_c.c[9] = my_struct_ab.b[8]
        z[10] = my_struct_c.c[5]
        bar[7] = fieldparam1[3]

        s1(z, my_struct_c, my_struct_ab, fieldparam1, bar)

    z_param[1] = 11
    a[3] = 12
    b[2] = 13
    c[5] = 14
    bar_param[4] = 15
    field1[3] = 16

    my_struct_ab_param = MyStructAB(a=a, b=b)
    my_struct_c_param = MyStructC(c=c)
    k1(z_param, my_struct_ab_param, bar_param, my_struct_c_param, field1)
    # store tests k1
    assert z_param[33] == 2
    assert a[35] == 3
    assert b[37] == 5
    assert c[51] == 17
    assert bar_param[222] == 41
    assert field1[2] == 67

    # from load tests, k1
    assert bar_param[6] == 11
    assert a[7] == 11
    assert b[8] == 12
    assert c[9] == 12
    assert z_param[10] == 14
    assert bar_param[7] == 16

    # s1
    assert z_param[22] == 88
    assert a[45] == 22
    assert b[47] == 23
    assert c[41] == 24
    assert bar_param[111] == 123
    assert field1[3] == 68

    # s2
    assert z_param[24] == 89
    assert a[46] == 32
    assert b[48] == 33
    assert c[42] == 34
    assert bar_param[112] == 125
    assert field1[4] == 69


@test_utils.test()
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_ndarray_struct_primitives(qd_type: Any, qd_annotation: Any) -> None:
    gc.collect()
    gc.collect()

    a = qd_type(qd.i32, shape=(55,))
    b = qd_type(qd.i32, shape=(57,))
    c = qd_type(qd.i32, shape=(211,))
    z_param = qd_type(qd.i32, shape=(223,))
    bar_param = qd_type(qd.i32, shape=(227,))

    @dataclass
    class MyStructAB:
        p3: qd.i32
        a: qd_annotation[qd.i32, 1]
        p1: qd.i32
        p2: qd.i32

    @dataclass
    class MyStructC:
        c: qd_annotation[qd.i32, 1]

    @qd.kernel
    def k1(
        z: qd_annotation[qd.i32, 1],
        my_struct_ab: MyStructAB,
        bar: qd_annotation[qd.i32, 1],
        my_struct_c: MyStructC,
    ) -> None:
        my_struct_ab.a[36] += my_struct_ab.p1
        my_struct_ab.a[37] += my_struct_ab.p2
        my_struct_ab.a[38] += my_struct_ab.p3

    my_struct_ab_param = MyStructAB(a=a, p1=119, p2=123, p3=345)
    my_struct_c_param = MyStructC(c=c)
    k1(z_param, my_struct_ab_param, bar_param, my_struct_c_param)
    assert a[36] == 119
    assert a[37] == 123
    assert a[38] == 345


@test_utils.test()
def test_ndarray_struct_nested_ndarray():
    a = qd.ndarray(qd.i32, shape=(101,))
    b = qd.ndarray(qd.i32, shape=(57,))
    c = qd.ndarray(qd.i32, shape=(211,))
    d = qd.ndarray(qd.i32, shape=(211,))
    e = qd.ndarray(qd.i32, shape=(251,))
    f = qd.ndarray(qd.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: qd.types.NDArray[qd.i32, 1]
        f: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class MyStructCD:
        c: qd.types.NDArray[qd.i32, 1]
        d: qd.types.NDArray[qd.i32, 1]
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: qd.types.NDArray[qd.i32, 1]
        b: qd.types.NDArray[qd.i32, 1]
        struct_cd: MyStructCD

    @qd.func
    def f3(
        my_struct_ab3: MyStructAB,
    ) -> None:
        my_struct_ab3.a[47] += 23
        my_struct_ab3.b[42] += 25
        my_struct_ab3.struct_cd.c[51] += 33
        my_struct_ab3.struct_cd.d[57] += 43
        my_struct_ab3.struct_cd.struct_ef.e[52] += 34
        my_struct_ab3.struct_cd.struct_ef.f[58] += 44

        my_struct_ab3.a[50] = my_struct_ab3.a.shape[0]
        my_struct_ab3.a[51] = my_struct_ab3.struct_cd.c.shape[0]
        my_struct_ab3.a[52] = my_struct_ab3.struct_cd.struct_ef.e.shape[0]

    @qd.func
    def f2(
        my_struct_ab2: MyStructAB,
    ) -> None:
        my_struct_ab2.a[27] += 13
        my_struct_ab2.b[22] += 15
        my_struct_ab2.struct_cd.c[31] += 23
        my_struct_ab2.struct_cd.d[37] += 33
        my_struct_ab2.struct_cd.struct_ef.e[32] += 24
        my_struct_ab2.struct_cd.struct_ef.f[38] += 34
        f3(my_struct_ab2)
        my_struct_ab2.a[60] = my_struct_ab2.a.shape[0]
        my_struct_ab2.a[61] = my_struct_ab2.struct_cd.c.shape[0]
        my_struct_ab2.a[62] = my_struct_ab2.struct_cd.struct_ef.e.shape[0]

    @qd.kernel
    def k1(
        my_struct_ab: MyStructAB,
    ) -> None:
        my_struct_ab.a[7] += 3
        my_struct_ab.b[2] += 5
        my_struct_ab.struct_cd.c[11] += 13
        my_struct_ab.struct_cd.d[17] += 23
        my_struct_ab.struct_cd.struct_ef.e[12] += 14
        my_struct_ab.struct_cd.struct_ef.f[18] += 24
        f2(my_struct_ab)
        my_struct_ab.a[70] = my_struct_ab.a.shape[0]
        my_struct_ab.a[71] = my_struct_ab.struct_cd.c.shape[0]
        my_struct_ab.a[72] = my_struct_ab.struct_cd.struct_ef.e.shape[0]

    my_struct_ef_param = MyStructEF(e=e, f=f)
    my_struct_cd_param = MyStructCD(c=c, d=d, struct_ef=my_struct_ef_param)
    my_struct_ab_param = MyStructAB(a=a, b=b, struct_cd=my_struct_cd_param)
    k1(my_struct_ab_param)

    assert a[7] == 3
    assert b[2] == 5
    assert c[11] == 13
    assert d[17] == 23
    assert e[12] == 14
    assert f[18] == 24

    assert a[27] == 13
    assert b[22] == 15
    assert c[31] == 23
    assert d[37] == 33
    assert e[32] == 24
    assert f[38] == 34

    assert a[47] == 23
    assert b[42] == 25
    assert c[51] == 33
    assert d[57] == 43
    assert e[52] == 34
    assert f[58] == 44

    # shapes
    assert a[50] == 101
    assert a[51] == 211
    assert a[52] == 251

    assert a[60] == 101
    assert a[61] == 211
    assert a[62] == 251

    assert a[70] == 101
    assert a[71] == 211
    assert a[72] == 251


@test_utils.test()
def test_field_struct_nested_field() -> None:
    a = qd.field(qd.i32, shape=(55,))
    b = qd.field(qd.i32, shape=(57,))
    c = qd.field(qd.i32, shape=(211,))
    d = qd.field(qd.i32, shape=(211,))
    e = qd.field(qd.i32, shape=(251,))
    f = qd.field(qd.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: qd.Template
        f: qd.Template

    @dataclass
    class MyStructCD:
        c: qd.Template
        d: qd.Template
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: qd.Template
        b: qd.Template
        struct_cd: MyStructCD

    @qd.func
    def f3(
        my_struct_ab3: MyStructAB,
    ) -> None:
        my_struct_ab3.a[47] += 23
        my_struct_ab3.b[42] += 25
        my_struct_ab3.struct_cd.c[51] += 33
        my_struct_ab3.struct_cd.d[57] += 43
        my_struct_ab3.struct_cd.struct_ef.e[52] += 34
        my_struct_ab3.struct_cd.struct_ef.f[58] += 44
        my_struct_ab3.a[50] = my_struct_ab3.a.shape[0]
        my_struct_ab3.a[51] = my_struct_ab3.struct_cd.c.shape[0]
        my_struct_ab3.a[52] = my_struct_ab3.struct_cd.struct_ef.e.shape[0]

    @qd.func
    def f2(
        my_struct_ab2: MyStructAB,
    ) -> None:
        my_struct_ab2.a[27] += 13
        my_struct_ab2.b[22] += 15
        my_struct_ab2.struct_cd.c[31] += 23
        my_struct_ab2.struct_cd.d[37] += 33
        my_struct_ab2.struct_cd.struct_ef.e[32] += 24
        my_struct_ab2.struct_cd.struct_ef.f[38] += 34
        f3(my_struct_ab2)
        my_struct_ab2.a[60] = my_struct_ab2.a.shape[0]
        my_struct_ab2.a[61] = my_struct_ab2.struct_cd.c.shape[0]
        my_struct_ab2.a[62] = my_struct_ab2.struct_cd.struct_ef.e.shape[0]

    @qd.kernel
    def k1(
        my_struct_ab: MyStructAB,
    ) -> None:
        my_struct_ab.a[7] += 3
        my_struct_ab.b[2] += 5
        my_struct_ab.struct_cd.c[11] += 13
        my_struct_ab.struct_cd.d[17] += 23
        my_struct_ab.struct_cd.struct_ef.e[12] += 14
        my_struct_ab.struct_cd.struct_ef.f[18] += 24
        f2(my_struct_ab)
        my_struct_ab.a[70] = my_struct_ab.a.shape[0]
        my_struct_ab.a[71] = my_struct_ab.struct_cd.c.shape[0]
        my_struct_ab.a[72] = my_struct_ab.struct_cd.struct_ef.e.shape[0]

    my_struct_ef_param = MyStructEF(e=e, f=f)
    my_struct_cd_param = MyStructCD(c=c, d=d, struct_ef=my_struct_ef_param)
    my_struct_ab_param = MyStructAB(a=a, b=b, struct_cd=my_struct_cd_param)
    k1(my_struct_ab_param)

    assert a[7] == 3
    assert b[2] == 5
    assert c[11] == 13
    assert d[17] == 23
    assert e[12] == 14
    assert f[18] == 24

    assert a[27] == 13
    assert b[22] == 15
    assert c[31] == 23
    assert d[37] == 33
    assert e[32] == 24
    assert f[38] == 34

    assert a[47] == 23
    assert b[42] == 25
    assert c[51] == 33
    assert d[57] == 43
    assert e[52] == 34
    assert f[58] == 44

    # shapes
    assert a[50] == 55
    assert a[51] == 211
    assert a[52] == 251

    assert a[60] == 55
    assert a[61] == 211
    assert a[62] == 251

    assert a[70] == 55
    assert a[71] == 211
    assert a[72] == 251


@test_utils.test()
def test_field_struct_nested_field_kwargs() -> None:
    a = qd.field(qd.i32, shape=(55,))
    b = qd.field(qd.i32, shape=(57,))
    c = qd.field(qd.i32, shape=(211,))
    d = qd.field(qd.i32, shape=(211,))
    e = qd.field(qd.i32, shape=(251,))
    f = qd.field(qd.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: qd.Template
        f: qd.Template

    @dataclass
    class MyStructCD:
        c: qd.Template
        d: qd.Template
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: qd.Template
        b: qd.Template
        struct_cd: MyStructCD

    @qd.func
    def f3(
        my_struct_ab3: MyStructAB,
    ) -> None:
        my_struct_ab3.a[47] += 23
        my_struct_ab3.b[42] += 25
        my_struct_ab3.struct_cd.c[51] += 33
        my_struct_ab3.struct_cd.d[57] += 43
        my_struct_ab3.struct_cd.struct_ef.e[52] += 34
        my_struct_ab3.struct_cd.struct_ef.f[58] += 44
        my_struct_ab3.a[50] = my_struct_ab3.a.shape[0]
        my_struct_ab3.a[51] = my_struct_ab3.struct_cd.c.shape[0]
        my_struct_ab3.a[52] = my_struct_ab3.struct_cd.struct_ef.e.shape[0]

    @qd.func
    def f2(
        my_struct_ab2: MyStructAB,
    ) -> None:
        my_struct_ab2.a[27] += 13
        my_struct_ab2.b[22] += 15
        my_struct_ab2.struct_cd.c[31] += 23
        my_struct_ab2.struct_cd.d[37] += 33
        my_struct_ab2.struct_cd.struct_ef.e[32] += 24
        my_struct_ab2.struct_cd.struct_ef.f[38] += 34
        f3(my_struct_ab3=my_struct_ab2)
        my_struct_ab2.a[60] = my_struct_ab2.a.shape[0]
        my_struct_ab2.a[61] = my_struct_ab2.struct_cd.c.shape[0]
        my_struct_ab2.a[62] = my_struct_ab2.struct_cd.struct_ef.e.shape[0]

    @qd.kernel
    def k1(
        my_struct_ab: MyStructAB,
    ) -> None:
        my_struct_ab.a[7] += 3
        my_struct_ab.b[2] += 5
        my_struct_ab.struct_cd.c[11] += 13
        my_struct_ab.struct_cd.d[17] += 23
        my_struct_ab.struct_cd.struct_ef.e[12] += 14
        my_struct_ab.struct_cd.struct_ef.f[18] += 24
        f2(my_struct_ab2=my_struct_ab)
        my_struct_ab.a[70] = my_struct_ab.a.shape[0]
        my_struct_ab.a[71] = my_struct_ab.struct_cd.c.shape[0]
        my_struct_ab.a[72] = my_struct_ab.struct_cd.struct_ef.e.shape[0]

    my_struct_ef_param = MyStructEF(e=e, f=f)
    my_struct_cd_param = MyStructCD(c=c, d=d, struct_ef=my_struct_ef_param)
    my_struct_ab_param = MyStructAB(a=a, b=b, struct_cd=my_struct_cd_param)
    k1(my_struct_ab=my_struct_ab_param)

    assert a[7] == 3
    assert b[2] == 5
    assert c[11] == 13
    assert d[17] == 23
    assert e[12] == 14
    assert f[18] == 24

    assert a[27] == 13
    assert b[22] == 15
    assert c[31] == 23
    assert d[37] == 33
    assert e[32] == 24
    assert f[38] == 34

    assert a[47] == 23
    assert b[42] == 25
    assert c[51] == 33
    assert d[57] == 43
    assert e[52] == 34
    assert f[58] == 44

    # shapes
    assert a[50] == 55
    assert a[51] == 211
    assert a[52] == 251

    assert a[60] == 55
    assert a[61] == 211
    assert a[62] == 251

    assert a[70] == 55
    assert a[71] == 211
    assert a[72] == 251


@test_utils.test()
def test_ndarray_struct_multiple_child_structs_ndarray():
    a = qd.ndarray(qd.i32, shape=(55,))
    b = qd.ndarray(qd.i32, shape=(57,))
    c = qd.ndarray(qd.i32, shape=(211,))
    d = qd.ndarray(qd.i32, shape=(211,))
    e = qd.ndarray(qd.i32, shape=(251,))
    f = qd.ndarray(qd.i32, shape=(251,))

    d11 = qd.ndarray(qd.i32, shape=(251,))
    d12 = qd.ndarray(qd.i32, shape=(251,))
    d21 = qd.ndarray(qd.i32, shape=(251,))
    d22 = qd.ndarray(qd.i32, shape=(251,))
    d31 = qd.ndarray(qd.i32, shape=(251,))
    d32 = qd.ndarray(qd.i32, shape=(251,))

    @dataclass
    class D1:
        d11: qd.types.NDArray[qd.i32, 1]
        d12: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class D2:
        d21: qd.types.NDArray[qd.i32, 1]
        d22: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class D3:
        d31: qd.types.NDArray[qd.i32, 1]
        d32: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class C1:
        a: qd.types.NDArray[qd.i32, 1]
        d1: D1
        d2: D2
        d3: D3
        b: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class C2:
        c: qd.types.NDArray[qd.i32, 1]
        d: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class C3:
        e: qd.types.NDArray[qd.i32, 1]
        f: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class P1:
        c1: C1
        c2: C2
        c3: C3

    @qd.kernel
    def k1(p1: P1) -> None:
        p1.c1.a[0] = 22
        p1.c1.b[0] = 33
        p1.c2.c[0] = 44
        p1.c2.d[0] = 55
        p1.c3.e[0] = 66
        p1.c3.f[0] = 77

    d1 = D1(d11=d11, d12=d12)
    d2 = D2(d21=d21, d22=d22)
    d3 = D3(d31=d31, d32=d32)
    c1 = C1(a=a, b=b, d1=d1, d2=d2, d3=d3)
    c2 = C2(c=c, d=d)
    c3 = C3(e=e, f=f)
    p1 = P1(c1=c1, c2=c2, c3=c3)
    k1(p1)
    assert a[0] == 22
    assert b[0] == 33
    assert c[0] == 44
    assert d[0] == 55
    assert e[0] == 66
    assert f[0] == 77


@test_utils.test()
def test_ndarray_struct_multiple_child_structs_field():
    a = qd.field(qd.i32, shape=(55,))
    b = qd.field(qd.i32, shape=(57,))
    c = qd.field(qd.i32, shape=(211,))
    d = qd.field(qd.i32, shape=(211,))
    e = qd.field(qd.i32, shape=(251,))
    f = qd.field(qd.i32, shape=(251,))

    @dataclass
    class C1:
        a: qd.Template
        b: qd.Template

    @dataclass
    class C2:
        c: qd.Template
        d: qd.Template

    @dataclass
    class C3:
        e: qd.Template
        f: qd.Template

    @dataclass
    class P1:
        c1: C1
        c2: C2
        c3: C3

    @qd.kernel
    def k1(p1: P1) -> None:
        p1.c1.a[0] = 22
        p1.c1.b[0] = 33
        p1.c2.c[0] = 44
        p1.c2.d[0] = 55
        p1.c3.e[0] = 66
        p1.c3.f[0] = 77

    c1 = C1(a=a, b=b)
    c2 = C2(c=c, d=d)
    c3 = C3(e=e, f=f)
    p1 = P1(c1=c1, c2=c2, c3=c3)
    k1(p1)
    assert a[0] == 22
    assert b[0] == 33
    assert c[0] == 44
    assert d[0] == 55
    assert e[0] == 66
    assert f[0] == 77


# --- Sub-struct passing: f(s.child) where the child is itself a dataclass ---


class _TemplateBuilder:
    """Makes ``qd.Template`` subscriptable so ``_TemplateBuilder()[qd.i32, 1]`` returns ``qd.Template`` (matching the
    shape of ``qd.types.ndarray[qd.i32, 1]``). Lets a single test body work for both NDArray-annotated and
    Template-annotated leaves."""

    def __getitem__(self, _):
        return qd.Template


# (leaf-value factory, leaf-annotation builder). NDArray and qd.Tensor share the same annotation form (qd.Tensor
# wrappers are unwrapped to bare impls at arg-binding time, so the annotation stays NDArray).
_SUBSTRUCT_LEAF_KINDS = [
    pytest.param(qd.ndarray, qd.types.ndarray, id="ndarray"),
    pytest.param(qd.field, _TemplateBuilder(), id="field"),
    pytest.param(
        lambda dtype, shape: qd.tensor(dtype, shape=shape, backend=qd.Backend.NDARRAY),
        qd.types.ndarray,
        id="tensor",
    ),
]


@test_utils.test()
@pytest.mark.parametrize("qd_make,qd_anno", _SUBSTRUCT_LEAF_KINDS)
def test_substruct_passed_to_func(qd_make: Any, qd_anno: Any, request) -> None:
    """``f(s.struct_cd)`` where the kernel arg is a nested dataclass and the callee is typed with the child dataclass.
    Mirrors test_ndarray_struct_nested_ndarray but passes the sub-struct (not the whole struct) into the inner func.
    Parametrized across the three supported leaf kinds (raw ndarray, field/template, qd.Tensor wrapper).
    Every dataclass also carries one ``extra`` leaf that no kernel/func ever reads, so pruning is checked end-to-end
    via ``kernel_args_count_by_type``."""
    a = qd_make(qd.i32, shape=(101,))
    b = qd_make(qd.i32, shape=(57,))
    c = qd_make(qd.i32, shape=(211,))
    d = qd_make(qd.i32, shape=(211,))
    e = qd_make(qd.i32, shape=(251,))
    f = qd_make(qd.i32, shape=(251,))
    extra_ab = qd_make(qd.i32, shape=(8,))
    extra_cd = qd_make(qd.i32, shape=(8,))
    extra_ef = qd_make(qd.i32, shape=(8,))

    @dataclass
    class MyStructEF:
        e: qd_anno[qd.i32, 1]
        f: qd_anno[qd.i32, 1]
        extra: qd_anno[qd.i32, 1]

    @dataclass
    class MyStructCD:
        c: qd_anno[qd.i32, 1]
        d: qd_anno[qd.i32, 1]
        extra: qd_anno[qd.i32, 1]
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: qd_anno[qd.i32, 1]
        b: qd_anno[qd.i32, 1]
        extra: qd_anno[qd.i32, 1]
        struct_cd: MyStructCD

    @qd.func
    def fef(struct_ef: MyStructEF) -> None:
        struct_ef.e[12] += 14
        struct_ef.f[18] += 24

    @qd.func
    def fcd(struct_cd: MyStructCD) -> None:
        struct_cd.c[11] += 13
        struct_cd.d[17] += 23
        fef(struct_cd.struct_ef)

    @qd.kernel
    def k1(my_struct_ab: MyStructAB) -> None:
        my_struct_ab.a[7] += 3
        my_struct_ab.b[2] += 5
        fcd(my_struct_ab.struct_cd)
        fef(my_struct_ab.struct_cd.struct_ef)

    s = MyStructAB(
        a=a,
        b=b,
        extra=extra_ab,
        struct_cd=MyStructCD(
            c=c,
            d=d,
            extra=extra_cd,
            struct_ef=MyStructEF(e=e, f=f, extra=extra_ef),
        ),
    )
    k1(s)

    assert a[7] == 3
    assert b[2] == 5
    assert c[11] == 13
    assert d[17] == 23
    assert e[12] == 28
    assert f[18] == 48

    # The three ``extra`` leaves must be pruned. ndarray/tensor leaves show up in QD_ARRAY; field leaves are template
    # globals and never appear as kernel args at all (QD_ARRAY == 0 across the board).
    leaf_kind = request.node.callspec.id.split("-")[0]
    expected_qd_arrays = 6 if leaf_kind in ("ndarray", "tensor") else 0
    k1_primal: Kernel = k1._primal
    assert k1_primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == expected_qd_arrays


@test_utils.test()
def test_substruct_passed_to_func_kwargs() -> None:
    """``f(child=s.struct_cd)`` — kwargs at the sub-struct call site. ``a``, ``extra_ab``, ``extra_cd`` are all
    unread and must be pruned out of the compiled kernel arg list."""
    c = qd.ndarray(qd.i32, shape=(8,))
    d = qd.ndarray(qd.i32, shape=(8,))

    @dataclass
    class CD:
        c: qd.types.NDArray[qd.i32, 1]
        d: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class AB:
        a: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]
        cd: CD

    @qd.func
    def fcd(cd: CD) -> None:
        cd.c[0] += 7
        cd.d[0] += 9

    @qd.kernel
    def k(s: AB) -> None:
        fcd(cd=s.cd)

    a = qd.ndarray(qd.i32, shape=(8,))
    extra_ab = qd.ndarray(qd.i32, shape=(8,))
    extra_cd = qd.ndarray(qd.i32, shape=(8,))
    k(AB(a=a, extra=extra_ab, cd=CD(c=c, d=d, extra=extra_cd)))

    assert c[0] == 7
    assert d[0] == 9
    k_primal: Kernel = k._primal
    assert k_primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_substruct_pruning() -> None:
    """When the callee uses only one of the sub-struct's leaves, the unused leaves must be pruned from the kernel's
    compiled argument list. Exercises pruning across the sub-struct boundary."""
    c = qd.ndarray(qd.i32, shape=(8,))
    d = qd.ndarray(qd.i32, shape=(8,))
    a = qd.ndarray(qd.i32, shape=(8,))

    @dataclass
    class CD:
        c: qd.types.NDArray[qd.i32, 1]
        d: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class AB:
        a: qd.types.NDArray[qd.i32, 1]
        cd: CD

    @qd.func
    def fcd(cd: CD) -> None:
        cd.c[0] += 5

    @qd.kernel
    def k(s: AB) -> None:
        fcd(s.cd)

    k(AB(a=a, cd=CD(c=c, d=d)))

    assert c[0] == 5
    assert d[0] == 0
    assert a[0] == 0

    k_primal: Kernel = k._primal
    kernel_args_count_by_type = k_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1


@test_utils.test()
def test_substruct_inside_func() -> None:
    """The sub-struct call site is inside a ``qd.func`` body (not directly in the kernel).
    Exercises ``_transform_as_func``'s intermediate sentinel binding. ``a``, ``extra_a``, ``extra_c`` are all unread
    and must be pruned out of the compiled kernel arg list (only ``c`` survives)."""
    c = qd.ndarray(qd.i32, shape=(8,))

    @dataclass
    class C:
        c: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class A:
        a: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]
        child: C

    @qd.func
    def f2(child: C) -> None:
        child.c[0] += 11

    @qd.func
    def f1(s: A) -> None:
        f2(s.child)

    @qd.kernel
    def k(s: A) -> None:
        f1(s)

    a = qd.ndarray(qd.i32, shape=(8,))
    extra_a = qd.ndarray(qd.i32, shape=(8,))
    extra_c = qd.ndarray(qd.i32, shape=(8,))
    k(A(a=a, extra=extra_a, child=C(c=c, extra=extra_c)))

    assert c[0] == 11
    k_primal: Kernel = k._primal
    assert k_primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1


@test_utils.test()
def test_substruct_scalar_leaf() -> None:
    """Sub-struct contains scalar (int) fields, mixed with an ndarray sibling. ``extra_ab`` is an unused ndarray on
    the outer struct and must be pruned out of the kernel arg list (only ``out`` survives as QD_ARRAY)."""
    out = qd.ndarray(qd.i32, shape=(8,))
    extra_ab = qd.ndarray(qd.i32, shape=(8,))

    @dataclass
    class CD:
        c: int
        d: int

    @dataclass
    class AB:
        out: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]
        cd: CD

    @qd.func
    def add_pair(cd: CD, out: qd.types.NDArray[qd.i32, 1]) -> None:
        out[0] = cd.c + cd.d

    @qd.kernel
    def k(s: AB) -> None:
        add_pair(s.cd, s.out)

    k(AB(out=out, extra=extra_ab, cd=CD(c=3, d=4)))

    assert out[0] == 7
    k_primal: Kernel = k._primal
    assert k_primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1


@test_utils.test()
def test_substruct_deep_nesting() -> None:
    """Three levels of dataclass nesting (L0 -> L1 -> L2 -> L3) combined with a three-level func-call chain
    (kernel -> touch_l1 -> touch_l2 -> touch_l3). Each layer writes its own leaf and forwards its inner sub-struct to
    the next callee. The kernel also exercises a direct 3-deep attribute access ``s.inner.inner.inner`` to make sure
    multi-level call-site flattening works straight from the kernel body, not just via intermediate funcs.
    Every level also carries an unused ``extra`` leaf to confirm pruning works at every depth — only the 4 ``leaf``
    fields must survive in the compiled kernel arg list."""
    n0 = qd.ndarray(qd.i32, shape=(4,))
    n1 = qd.ndarray(qd.i32, shape=(4,))
    n2 = qd.ndarray(qd.i32, shape=(4,))
    n3 = qd.ndarray(qd.i32, shape=(4,))
    x0 = qd.ndarray(qd.i32, shape=(4,))
    x1 = qd.ndarray(qd.i32, shape=(4,))
    x2 = qd.ndarray(qd.i32, shape=(4,))
    x3 = qd.ndarray(qd.i32, shape=(4,))

    @dataclass
    class L3:
        leaf: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class L2:
        leaf: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]
        inner: L3

    @dataclass
    class L1:
        leaf: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]
        inner: L2

    @dataclass
    class L0:
        leaf: qd.types.NDArray[qd.i32, 1]
        extra: qd.types.NDArray[qd.i32, 1]
        inner: L1

    @qd.func
    def touch_l3(s: L3) -> None:
        s.leaf[0] += 1

    @qd.func
    def touch_l2(s: L2) -> None:
        s.leaf[0] += 10
        touch_l3(s.inner)

    @qd.func
    def touch_l1(s: L1) -> None:
        s.leaf[0] += 100
        touch_l2(s.inner)

    @qd.kernel
    def k(s: L0) -> None:
        s.leaf[0] += 1000
        touch_l1(s.inner)
        touch_l3(s.inner.inner.inner)

    s = L0(
        leaf=n0,
        extra=x0,
        inner=L1(
            leaf=n1,
            extra=x1,
            inner=L2(
                leaf=n2,
                extra=x2,
                inner=L3(leaf=n3, extra=x3),
            ),
        ),
    )
    k(s)

    assert n0[0] == 1000
    assert n1[0] == 100
    assert n2[0] == 10
    assert n3[0] == 1 + 1
    k_primal: Kernel = k._primal
    assert k_primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@pytest.mark.parametrize("use_slots", [False, True])
@test_utils.test()
def test_template_mapper_cache(use_slots, monkeypatch):
    # Mock '_extract_arg' to track the number of (recursive) calls
    counter = 0
    _extract_arg_orig = qd.lang._template_mapper_hotpath._extract_arg

    def _extract_arg(*args, **kwargs):
        nonlocal counter
        counter += 1
        return _extract_arg_orig(*args, **kwargs)

    monkeypatch.setattr("quadrants.lang._template_mapper_hotpath._extract_arg", _extract_arg)

    @dataclass(frozen=True, slots=use_slots)
    class MyStruct:
        value: qd.types.ndarray()
        placeholder: qd.i32

    @qd.kernel
    def my_kernel(my_struct_1d: MyStruct, my_struct_2d: MyStruct) -> None:
        for i in qd.ndrange(my_struct_1d.value.shape[0]):
            my_struct_1d.value[i] += 1
        for i, j in qd.ndrange(my_struct_2d.value.shape[0], my_struct_2d.value.shape[1]):
            my_struct_2d.value[i, j] += 1

    num_fields = len(fields(MyStruct))
    value = qd.ndarray(qd.i32, shape=(1,))
    value.fill(0)
    placeholder = 0
    my_struct_1d = MyStruct(value=value, placeholder=placeholder)
    value = qd.ndarray(qd.f32, shape=(1, 2))
    value.fill(0.0)
    my_struct_2d = MyStruct(value=value, placeholder=placeholder)

    my_kernel(my_struct_1d, my_struct_2d)
    assert counter == 2 * num_fields
    assert my_struct_1d.value[0] == 1
    assert my_struct_2d.value[0, 0] == 1.0
    assert my_struct_2d.value[0, 1] == 1.0

    counter = 0
    my_kernel(my_struct_1d, my_struct_2d)
    if use_slots:
        # template mapper caching mechanism is disabled for dataclasses that enable slots
        assert counter == 2 * num_fields
    else:
        assert counter == 0
    assert my_struct_1d.value[0] == 2
    assert my_struct_2d.value[0, 0] == 2.0
    assert my_struct_2d.value[0, 1] == 2.0


@test_utils.test()
def test_print_used_parameters():
    @dataclasses.dataclass
    class MyDataclass:
        used1: qd.types.NDArray[qd.i32, 1]
        used2: qd.types.NDArray[qd.i32, 1]
        used3: qd.types.NDArray[qd.i32, 1]
        an_int: qd.i32
        not_used_int: qd.i32
        not_used: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def f1(md: MyDataclass) -> None:
        md.used3[0] = 123
        md.used3[1] = md.an_int

    @qd.kernel
    def k1(md: MyDataclass, trigger_static: qd.Template) -> None:
        md.used1[0] = 222
        md.used1[1] = md.used2[0]
        f1(md)
        if qd.static(trigger_static):
            md.used1[2] = 444

    u1 = qd.ndarray(qd.i32, (10,))
    u2 = qd.ndarray(qd.i32, (10,))
    u3 = qd.ndarray(qd.i32, (10,))
    nu1 = qd.ndarray(qd.i32, (10,))
    md = MyDataclass(used1=u1, used2=u2, used3=u3, not_used=nu1, an_int=555, not_used_int=888)

    u2[0] = 333
    k1(md, False)
    assert u1[0] == 222
    assert u3[0] == 123
    assert u1[1] == 333
    assert u1[2] == 0
    kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3
    assert kernel_args_count_by_type[KernelBatchedArgType.INT] == 1

    u1[0] = 0
    u1[1] = 0
    u1[2] = 0
    u3[0] = 0
    u2[0] = 333
    k1(md, True)
    assert u1[0] == 222
    assert u3[0] == 123
    assert u1[1] == 333
    assert u1[2] == 444
    kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3
    assert kernel_args_count_by_type[KernelBatchedArgType.INT] == 1


@test_utils.test()
def test_prune_used_parameters1():
    @dataclasses.dataclass
    class Nested1:
        n1: qd.types.NDArray[qd.i32, 1]
        n1u: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass
    class MyDataclass1:
        used1: qd.types.NDArray[qd.i32, 1]
        used2: qd.types.NDArray[qd.i32, 1]
        used3: qd.types.NDArray[qd.i32, 1]
        not_used: qd.types.NDArray[qd.i32, 1]
        nested1: Nested1

    @dataclasses.dataclass
    class MyDataclass2:
        used1: qd.types.NDArray[qd.i32, 1]
        used2: qd.types.NDArray[qd.i32, 1]
        used3: qd.types.NDArray[qd.i32, 1]
        not_used: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def f1(md1: MyDataclass1, md2: MyDataclass2) -> None:
        md1.used3[0] = 123
        md2.used1[5] = 555
        md2.used2[5] = 444
        md2.used3[5] = 333
        md1.nested1.n1[0] = 777

    @qd.kernel
    def k1(md1: MyDataclass1, md2: MyDataclass2, trigger_static: qd.Template) -> None:
        md1.used1[0] = 222
        md1.used1[1] = md1.used2[0]
        f1(md1, md2)
        if qd.static(trigger_static):
            md1.used1[2] = 444

    u1 = qd.ndarray(qd.i32, (10,))
    u2 = qd.ndarray(qd.i32, (10,))
    u3 = qd.ndarray(qd.i32, (10,))
    n1 = qd.ndarray(qd.i32, (10,))
    nu1 = qd.ndarray(qd.i32, (10,))
    n1u = qd.ndarray(qd.i32, (10,))
    nested1 = Nested1(n1=n1, n1u=n1u)
    md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1, nested1=nested1)

    u1b = qd.ndarray(qd.i32, (10,))
    u2b = qd.ndarray(qd.i32, (10,))
    u3b = qd.ndarray(qd.i32, (10,))
    nu1b = qd.ndarray(qd.i32, (10,))
    md2 = MyDataclass2(used1=u1b, used2=u2b, used3=u3b, not_used=nu1b)

    u2[0] = 333
    k1(md1, md2, False)
    assert u1[0] == 222
    assert u3[0] == 123
    assert u1[1] == 333
    assert u1b[5] == 555
    assert n1[0] == 777
    assert u1[2] == 0

    u1[0] = 0
    u1[1] = 0
    u1[2] = 0
    u3[0] = 0
    u2[0] = 333
    u1b[5] = 0
    n1[0] == 0
    k1(md1, md2, True)
    assert u1[0] == 222
    assert u3[0] == 123
    assert u1[1] == 333
    assert u1[2] == 444
    assert u1b[5] == 555
    assert n1[0] == 777


@test_utils.test()
def test_prune_used_parameters2():
    @dataclasses.dataclass
    class MyDataclass1:
        used1: qd.types.NDArray[qd.i32, 1]
        used2: qd.types.NDArray[qd.i32, 1]
        used3: qd.types.NDArray[qd.i32, 1]
        not_used: qd.types.NDArray[qd.i32, 1]

    @dataclasses.dataclass
    class MyDataclass2:
        used1: qd.types.NDArray[qd.i32, 1]
        used2: qd.types.NDArray[qd.i32, 1]
        used3: qd.types.NDArray[qd.i32, 1]
        not_used: qd.types.NDArray[qd.i32, 1]

    @qd.func
    def f2(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
        md1.used1[0] = 111
        md1.used2[0] = 222
        md1.used3[0] = 123
        md2.used1[0] = 555
        md2.used2[0] = 444
        md2.used3[0] = 333

    @qd.func
    def f1(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
        f2(i_b, md1=md1, md2=md2)

    @qd.kernel
    def k1(envs_idx: qd.types.NDArray[qd.i32, 1], md1: MyDataclass1, md2: MyDataclass2) -> None:
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            f1(i_b, md1=md1, md2=md2)

    envs_idx = qd.ndarray(qd.i32, (10,))

    u1 = qd.ndarray(qd.i32, (10,))
    u2 = qd.ndarray(qd.i32, (10,))
    u3 = qd.ndarray(qd.i32, (10,))
    nu1 = qd.ndarray(qd.i32, (10,))
    md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1)

    u1b = qd.ndarray(qd.i32, (10,))
    u2b = qd.ndarray(qd.i32, (10,))
    u3b = qd.ndarray(qd.i32, (10,))
    nu1b = qd.ndarray(qd.i32, (10,))
    md2 = MyDataclass2(used1=u1b, used2=u2b, used3=u3b, not_used=nu1b)

    k1(envs_idx, md1=md1, md2=md2)
    assert u1[0] == 111
    assert u2[0] == 222
    assert u3[0] == 123
    assert u1b[0] == 555
    assert u2b[0] == 444
    assert u3b[0] == 333

    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    print(sorted(list(k1_primal.used_py_dataclass_parameters_by_key_enforcing[k1_primal._last_launch_key])))
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 7  # +1 for envs_idx


@test_utils.test()
def test_prune_used_parameters_fastcache1(tmp_path: Path):
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class Nested1:
            n1: qd.types.NDArray[qd.i32, 1]
            n1u: qd.types.NDArray[qd.i32, 1]

        @dataclasses.dataclass
        class MyDataclass1:
            used1: qd.types.NDArray[qd.i32, 1]
            used2: qd.types.NDArray[qd.i32, 1]
            used3: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]
            nested1: Nested1

        @dataclasses.dataclass
        class MyDataclass2:
            used1: qd.types.NDArray[qd.i32, 1]
            used2: qd.types.NDArray[qd.i32, 1]
            used3: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def f1(md1: MyDataclass1, md2: MyDataclass2) -> None:
            # used:
            # __qd_md1__qd_used3
            # __qd_md2__qd_used1
            # __qd_md2__qd_used2
            # __qd_md2__qd_used3
            # __qd_md1__qd_nested1__qd_n1
            md1.used3[0] = 123
            md2.used1[5] = 555
            md2.used2[5] = 444
            md2.used3[5] = 333
            md1.nested1.n1[0] = 777

        @qd.kernel(fastcache=True)
        def k1(md1: MyDataclass1, md2: MyDataclass2, trigger_static: qd.Template) -> None:
            # used:
            # __qd_md1__qd_used1
            # __qd_md1__qd_used2
            # __qd_md1__qd_used3
            # __qd_md2__qd_used1
            # __qd_md2__qd_used2
            # __qd_md2__qd_used3
            # __qd_md1__qd_nested1__qd_n1
            md1.used1[0] = 222
            md1.used1[1] = md1.used2[0]
            f1(md1, md2)
            if qd.static(trigger_static):
                md1.used1[2] = 444

        u1 = qd.ndarray(qd.i32, (10,))
        u2 = qd.ndarray(qd.i32, (10,))
        u3 = qd.ndarray(qd.i32, (10,))
        n1 = qd.ndarray(qd.i32, (10,))
        nu1 = qd.ndarray(qd.i32, (10,))
        n1u = qd.ndarray(qd.i32, (10,))
        nested1 = Nested1(n1=n1, n1u=n1u)
        md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1, nested1=nested1)

        u1b = qd.ndarray(qd.i32, (10,))
        u2b = qd.ndarray(qd.i32, (10,))
        u3b = qd.ndarray(qd.i32, (10,))
        nu1b = qd.ndarray(qd.i32, (10,))
        md2 = MyDataclass2(used1=u1b, used2=u2b, used3=u3b, not_used=nu1b)

        u2[0] = 333
        k1(md1, md2, False)
        assert u1[0] == 222
        assert u3[0] == 123
        assert u1[1] == 333
        assert u1[2] == 0
        assert u1b[5] == 555
        assert n1[0] == 777
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 7
        assert kernel_args_count_by_type[KernelBatchedArgType.INT] == 0

        u1[0] = 0
        u1[1] = 0
        u1[2] = 0
        u3[0] = 0
        u2[0] = 333
        u1b[5] = 0
        n1[0] == 0
        k1(md1, md2, True)
        assert u1[0] == 222
        assert u3[0] == 123
        assert u1[1] == 333
        assert u1[2] == 444
        assert u1b[5] == 555
        assert n1[0] == 777
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 7
        assert kernel_args_count_by_type[KernelBatchedArgType.INT] == 0


@test_utils.test()
def test_prune_used_parameters_fastcache2(tmp_path: Path):
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass1:
            used1: qd.types.NDArray[qd.i32, 1]
            used2: qd.types.NDArray[qd.i32, 1]
            used3: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]
            not_used2: qd.types.NDArray[qd.i32, 1]

        @dataclasses.dataclass
        class MyDataclass2:
            used1: qd.types.NDArray[qd.i32, 1]
            used2: qd.types.NDArray[qd.i32, 1]
            used3: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]
            not_used2: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def f2(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            md1.used1[0] = 111
            md1.used2[0] = 222
            md1.used3[0] = 123
            md2.used1[0] = 555
            md2.used2[0] = 444
            md2.used3[0] = 333

        @qd.func
        def f1(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            f2(i_b, md1=md1, md2=md2)

        @qd.kernel(fastcache=True)
        def k1(envs_idx: qd.types.NDArray[qd.i32, 1], md1: MyDataclass1, md2: MyDataclass2) -> None:
            for i_b_ in range(envs_idx.shape[0]):
                i_b = envs_idx[i_b_]
                f1(i_b, md1=md1, md2=md2)

        envs_idx = qd.ndarray(qd.i32, (10,))

        u1 = qd.ndarray(qd.i32, (10,))
        u2 = qd.ndarray(qd.i32, (10,))
        u3 = qd.ndarray(qd.i32, (10,))
        nu1 = qd.ndarray(qd.i32, (10,))
        nu2 = qd.ndarray(qd.i32, (10,))
        md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1, not_used2=nu2)

        u1b = qd.ndarray(qd.i32, (10,))
        u2b = qd.ndarray(qd.i32, (10,))
        u3b = qd.ndarray(qd.i32, (10,))
        nu1b = qd.ndarray(qd.i32, (10,))
        nu2b = qd.ndarray(qd.i32, (10,))
        md2 = MyDataclass2(used1=u1b, used2=u2b, used3=u3b, not_used=nu1b, not_used2=nu2b)

        k1(envs_idx, md1=md1, md2=md2)
        assert u1[0] == 111
        assert u2[0] == 222
        assert u3[0] == 123
        assert u1b[0] == 555
        assert u2b[0] == 444
        assert u3b[0] == 333

        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        # remember to add 1 for envs_idx
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 7
        assert kernel_args_count_by_type[KernelBatchedArgType.INT] == 0


@test_utils.test()
def test_prune_used_parameters_fastcache_no_used(tmp_path: Path):
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass1:
            not_used1: qd.types.NDArray[qd.i32, 1]
            not_used2: qd.types.NDArray[qd.i32, 1]

        @dataclasses.dataclass
        class MyDataclass2:
            not_used1: qd.types.NDArray[qd.i32, 1]
            not_used2: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def f2(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            pass

        @qd.func
        def f1(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            f2(i_b, md1, md2=md2)

        @qd.kernel(fastcache=True)
        def k1(envs_idx: qd.types.NDArray[qd.i32, 1], md1: MyDataclass1, md2: MyDataclass2) -> None:
            for i_b_ in range(envs_idx.shape[0]):
                i_b = envs_idx[i_b_]
                f1(i_b, md1, md2=md2)

        envs_idx = qd.ndarray(qd.i32, (10,))

        nu1 = qd.ndarray(qd.i32, (10,))
        nu2 = qd.ndarray(qd.i32, (10,))
        md1 = MyDataclass1(not_used1=nu1, not_used2=nu2)

        nu1b = qd.ndarray(qd.i32, (10,))
        nu2b = qd.ndarray(qd.i32, (10,))
        md2 = MyDataclass2(not_used1=nu1b, not_used2=nu2b)

        k1(envs_idx, md1, md2=md2)


@test_utils.test()
def test_prune_used_parameters_fastcache_dead_static_branch(tmp_path: Path):
    # inner() reads md.deep only inside a dead qd.static branch, so md.deep is marked used under inner's
    # (instantiation-shared) func id via the qd.static(True) instantiation alone. Two separate caller paths
    # forward md to inner: path_dead (qd.static(False)) is walked before path_live (qd.static(True)), so the
    # single-shot used-set copy at path_dead's call misses md.deep. Without cross-call fixpoint propagation
    # the enforcing pass still forwards md.deep from path_dead, which never bound it -> compile failure.
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass:
            base: qd.types.NDArray[qd.i32, 1]
            deep: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def inner(md: MyDataclass, use_deep: qd.template()) -> None:
            md.base[0] = 1
            if qd.static(use_deep):
                md.deep[0] = 99

        @qd.func
        def path_dead(md: MyDataclass) -> None:
            inner(md, False)

        @qd.func
        def path_live(md: MyDataclass) -> None:
            inner(md, True)

        @qd.kernel(fastcache=True)
        def k1(md: MyDataclass) -> None:
            path_dead(md)
            path_live(md)

        base = qd.ndarray(qd.i32, (4,))
        deep = qd.ndarray(qd.i32, (4,))
        not_used = qd.ndarray(qd.i32, (4,))
        md = MyDataclass(base=base, deep=deep, not_used=not_used)

        k1(md)
        assert base[0] == 1
        assert deep[0] == 99
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_prune_used_parameters_fastcache_dead_static_branch_reversed_order(tmp_path: Path):
    # Same setup as the flat dead-static-branch case, but the kernel walks path_live (qd.static(True)) before
    # path_dead (qd.static(False)). In this order inner's used-set already contains md.deep by the time
    # path_dead's call is recorded, so this case compiles even without the fix. It is included as a regression
    # guard: a correct fix must stay order-independent, so both walk orders must keep passing.
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass:
            base: qd.types.NDArray[qd.i32, 1]
            deep: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def inner(md: MyDataclass, use_deep: qd.template()) -> None:
            md.base[0] = 1
            if qd.static(use_deep):
                md.deep[0] = 99

        @qd.func
        def path_dead(md: MyDataclass) -> None:
            inner(md, False)

        @qd.func
        def path_live(md: MyDataclass) -> None:
            inner(md, True)

        @qd.kernel(fastcache=True)
        def k1(md: MyDataclass) -> None:
            path_live(md)
            path_dead(md)

        base = qd.ndarray(qd.i32, (4,))
        deep = qd.ndarray(qd.i32, (4,))
        not_used = qd.ndarray(qd.i32, (4,))
        md = MyDataclass(base=base, deep=deep, not_used=not_used)

        k1(md)
        assert base[0] == 1
        assert deep[0] == 99
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_prune_used_parameters_fastcache_dead_static_branch_kwargs(tmp_path: Path):
    # Same dead-static-branch bug as the flat case, but md is forwarded by keyword (md=md) at every call site
    # instead of positionally. Keyword-forwarded dataclasses are pruned in _expand_Call_dataclass_kwargs (gated
    # on the callee used-set), a different path from the positional filter_call_args, so this confirms the used-
    # set fixpoint also closes over keyword-forwarded edges. use_deep stays positional so the template value is
    # not passed by keyword (an unrelated concern).
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass:
            base: qd.types.NDArray[qd.i32, 1]
            deep: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def inner(use_deep: qd.template(), md: MyDataclass) -> None:
            md.base[0] = 1
            if qd.static(use_deep):
                md.deep[0] = 99

        @qd.func
        def path_dead(md: MyDataclass) -> None:
            inner(False, md=md)

        @qd.func
        def path_live(md: MyDataclass) -> None:
            inner(True, md=md)

        @qd.kernel(fastcache=True)
        def k1(md: MyDataclass) -> None:
            path_dead(md=md)
            path_live(md=md)

        base = qd.ndarray(qd.i32, (4,))
        deep = qd.ndarray(qd.i32, (4,))
        not_used = qd.ndarray(qd.i32, (4,))
        md = MyDataclass(base=base, deep=deep, not_used=not_used)

        k1(md)
        assert base[0] == 1
        assert deep[0] == 99
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_prune_used_parameters_fastcache_dead_static_branch_nested(tmp_path: Path):
    # Same dead-static-branch forwarding bug as the flat case, but the forwarded field lives three dataclass
    # levels deep (top.mid.leaf.deep), confirming the fix closes the used set across arbitrary nesting depth.
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class Leaf:
            deep: qd.types.NDArray[qd.i32, 1]
            not_used: qd.types.NDArray[qd.i32, 1]

        @dataclasses.dataclass
        class Mid:
            not_used: qd.types.NDArray[qd.i32, 1]
            leaf: Leaf

        @dataclasses.dataclass
        class Top:
            base: qd.types.NDArray[qd.i32, 1]
            mid: Mid

        @qd.func
        def inner(top: Top, use_deep: qd.template()) -> None:
            top.base[0] = 1
            if qd.static(use_deep):
                top.mid.leaf.deep[0] = 99

        @qd.func
        def path_dead(top: Top) -> None:
            inner(top, False)

        @qd.func
        def path_live(top: Top) -> None:
            inner(top, True)

        @qd.kernel(fastcache=True)
        def k1(top: Top) -> None:
            path_dead(top)
            path_live(top)

        base = qd.ndarray(qd.i32, (4,))
        deep = qd.ndarray(qd.i32, (4,))
        mid_not_used = qd.ndarray(qd.i32, (4,))
        leaf_not_used = qd.ndarray(qd.i32, (4,))
        top = Top(base=base, mid=Mid(not_used=mid_not_used, leaf=Leaf(deep=deep, not_used=leaf_not_used)))

        k1(top)
        assert base[0] == 1
        assert deep[0] == 99
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_prune_used_parameters_fastcache_forward_same_name_swapped_slots(tmp_path: Path):
    # inner() reads only its first struct (a); b is entirely unused. caller1 and caller2 declare identically
    # named parameters and forward them into swapped inner slots, so the flat name __qd_md__qd_x binds inner's
    # used slot a in one caller and its unused slot b in the other. Keyed by callee alone, the mapping from a
    # caller argument name to its callee parameter lets the second caller overwrite the first, so the enforcing
    # pass prunes the field the first caller needs -> a Missing argument failure and a write to the wrong struct.
    # Keying the mapping by (caller, callee) keeps the two call sites independent.
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass:
            x: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def inner(a: MyDataclass, b: MyDataclass) -> None:
            a.x[0] = 42

        @qd.func
        def caller1(md: MyDataclass, other: MyDataclass) -> None:
            inner(md, other)

        @qd.func
        def caller2(md: MyDataclass, other: MyDataclass) -> None:
            inner(other, md)

        @qd.kernel(fastcache=True)
        def k1(p: MyDataclass, q: MyDataclass) -> None:
            caller1(p, q)
            caller2(p, q)

        p_x = qd.ndarray(qd.i32, (4,))
        q_x = qd.ndarray(qd.i32, (4,))
        k1(MyDataclass(x=p_x), MyDataclass(x=q_x))
        assert p_x[0] == 42
        assert q_x[0] == 42
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_prune_used_parameters_fastcache_forward_same_name_swapped_slots_same_caller(tmp_path: Path):
    # Same swapped-slot forwarding as the cross-caller case, but both call sites live in a single caller: md
    # binds inner's used slot a on the first line and its unused slot b on the second. The forwarding map is
    # keyed per call site (source position), so the two calls stay independent; a map shared for the whole
    # (caller, callee) pair would let the second line overwrite the first and prune the field the first needs.
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass:
            x: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def inner(a: MyDataclass, b: MyDataclass) -> None:
            a.x[0] = 42

        @qd.func
        def caller(md: MyDataclass, other: MyDataclass) -> None:
            inner(md, other)
            inner(other, md)

        @qd.kernel(fastcache=True)
        def k1(p: MyDataclass, q: MyDataclass) -> None:
            caller(p, q)

        p_x = qd.ndarray(qd.i32, (4,))
        q_x = qd.ndarray(qd.i32, (4,))
        k1(MyDataclass(x=p_x), MyDataclass(x=q_x))
        assert p_x[0] == 42
        assert q_x[0] == 42
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2


@test_utils.test()
def test_prune_used_parameters_fastcache_forward_same_name_same_call_two_slots(tmp_path: Path):
    # A single call site forwards the SAME dataclass into two positional slots: inner(md, md). inner reads
    # only its first struct (a); b is entirely unused, so the expanded call carries the same flat name in both
    # slots. A positional map keyed by the caller flat name lets slot b overwrite slot a, so the enforcing pass
    # prunes the field slot a needs -> a Missing argument failure. Keying the map by slot index keeps the two
    # occurrences independent.
    arch_name = qd.lang.impl.current_cfg().arch.name
    for _it in range(3):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass:
            x: qd.types.NDArray[qd.i32, 1]

        @qd.func
        def inner(a: MyDataclass, b: MyDataclass) -> None:
            a.x[0] = 42

        @qd.func
        def caller(md: MyDataclass) -> None:
            inner(md, md)

        @qd.kernel(fastcache=True)
        def k1(p: MyDataclass) -> None:
            caller(p)

        p_x = qd.ndarray(qd.i32, (4,))
        k1(MyDataclass(x=p_x))
        assert p_x[0] == 42
        kernel_args_count_by_type = k1._primal.launch_stats.kernel_args_count_by_type
        assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1


@test_utils.test()
def test_pruning_with_keyword_rename() -> None:
    @dataclasses.dataclass
    class MyStruct:
        used: qd.types.NDArray[qd.f32, 2]
        not_used: qd.types.NDArray[qd.f32, 2]

    def create_struct():
        my_struct_outside = MyStruct(
            used=qd.ndarray(dtype=qd.f32, shape=(1, 1)), not_used=qd.ndarray(dtype=qd.f32, shape=(1, 1))
        )
        return my_struct_outside

    @qd.func
    def f1(new_struct_name: MyStruct):
        new_struct_name.used[0, 0] = 100

    @qd.kernel
    def k1(my_struct: MyStruct):
        f1(new_struct_name=my_struct)

    my_struct_outside = create_struct()
    k1(my_struct=my_struct_outside)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1
    assert my_struct_outside.used[0, 0] == 100
    assert my_struct_outside.not_used[0, 0] == 0


@test_utils.test()
def test_pruning_with_arg_rename() -> None:
    @dataclasses.dataclass
    class MyStruct:
        used: qd.types.NDArray[qd.f32, 2]
        not_used: qd.types.NDArray[qd.f32, 2]

    def create_struct():
        return MyStruct(used=qd.ndarray(dtype=qd.f32, shape=(1, 1)), not_used=qd.ndarray(dtype=qd.f32, shape=(1, 1)))

    @qd.func
    def f1(new_struct_name: MyStruct):
        new_struct_name.used[0, 0] = 100

    @qd.kernel
    def k1(my_struct: MyStruct):
        f1(my_struct)

    my_struct = create_struct()
    k1(my_struct=my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1
    assert my_struct.used[0, 0] == 100
    assert my_struct.not_used[0, 0] == 0

    my_struct = create_struct()
    k1(my_struct=my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1
    assert my_struct.used[0, 0] == 100
    assert my_struct.not_used[0, 0] == 0


@test_utils.test()
def test_pruning_with_arg_kwargs_rename() -> None:
    @dataclasses.dataclass
    class MyStruct:
        used: qd.types.NDArray[qd.f32, 2]
        not_used: qd.types.NDArray[qd.f32, 2]

    def create_structs():
        my_struct1 = MyStruct(
            used=qd.ndarray(dtype=qd.f32, shape=(1, 1)), not_used=qd.ndarray(dtype=qd.f32, shape=(1, 1))
        )
        my_struct2 = MyStruct(
            used=qd.ndarray(dtype=qd.f32, shape=(1, 1)), not_used=qd.ndarray(dtype=qd.f32, shape=(1, 1))
        )
        my_struct3 = MyStruct(
            used=qd.ndarray(dtype=qd.f32, shape=(1, 1)), not_used=qd.ndarray(dtype=qd.f32, shape=(1, 1))
        )
        my_struct4 = MyStruct(
            used=qd.ndarray(dtype=qd.f32, shape=(1, 1)), not_used=qd.ndarray(dtype=qd.f32, shape=(1, 1))
        )
        return my_struct1, my_struct2, my_struct3, my_struct4

    @qd.func
    def g1(struc3_g1: MyStruct):
        # should be used:
        # struc3_g1.used
        struc3_g1.used[0, 0] = 102

    @qd.func
    def f2(a3: qd.i32, struct_f2: MyStruct, b3: qd.i32, d3: qd.i32, struct2_f2: MyStruct, c3: qd.i32):
        # should be used:
        # struct_f2.used
        # struct2_f2.useds
        struct_f2.used[0, 0] = 100
        struct2_f2.used[0, 0] = 101

    @qd.func
    def f1(a2: qd.i32, struct_f1: MyStruct, b2: qd.i32, d2: qd.i32, struct2_f1: MyStruct, c2: qd.i32):
        # should be used:
        # struct_f1.used
        # struct2_f1.used
        f2(a2, struct_f1, b2, d3=d2, struct2_f2=struct2_f1, c3=c2)

    @qd.kernel
    def k1(
        a: qd.i32,
        struct1_k1: MyStruct,
        b: qd.i32,
        d: qd.i32,
        struct2_k1: MyStruct,
        c: qd.i32,
        struct3_k1: MyStruct,
        struct4_k1: MyStruct,
    ):
        # should be used:
        # struct1_k1.used
        # struct2_k1.used
        f1(a, struct1_k1, b, d2=d, struct2_f1=struct2_k1, c2=c)
        # should be used:
        # struct3_k1.used
        g1(struct3_k1)
        # should be used:
        # struct4_k1.used
        g1(struct4_k1)

    # should be used:
    # my_struct1.used
    # my_struct2.used
    # my_struct3.used
    # my_struct4.used
    s1, s2, s3, s4 = create_structs()
    k1(1, s1, 2, d=5, struct2_k1=s2, c=3, struct3_k1=s3, struct4_k1=s4)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4
    assert s1.used[0, 0] == 100
    assert s2.used[0, 0] == 101
    assert s3.used[0, 0] == 102
    assert s4.used[0, 0] == 102

    assert s1.not_used[0, 0] == 0
    assert s2.not_used[0, 0] == 0
    assert s3.not_used[0, 0] == 0
    assert s4.not_used[0, 0] == 0

    s1, s2, s3, s4 = create_structs()
    k1(1, s1, 2, d=5, struct2_k1=s2, c=3, struct3_k1=s3, struct4_k1=s4)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4

    assert s1.used[0, 0] == 100
    assert s2.used[0, 0] == 101
    assert s3.used[0, 0] == 102
    assert s4.used[0, 0] == 102

    assert s1.not_used[0, 0] == 0
    assert s2.not_used[0, 0] == 0
    assert s3.not_used[0, 0] == 0
    assert s4.not_used[0, 0] == 0


@pytest.mark.xfail(reason="calling sub functions with different templated values seems unsupported currently")
@test_utils.test()
def test_pruning_with_recursive_func() -> None:
    @dataclasses.dataclass
    class MyStruct:
        a: qd.types.NDArray[qd.f32, 2]
        b: qd.types.NDArray[qd.f32, 2]
        c: qd.types.NDArray[qd.f32, 2]
        d: qd.types.NDArray[qd.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            a=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            b=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            c=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            d=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f1(depth: qd.template(), struc_f1: MyStruct):
        if qd.static(depth) == 0:
            struc_f1.a[0, 0] = 100
            f1(1, struc_f1)
        elif qd.static(depth) == 1:
            struc_f1.b[0, 0] = 101
            f1(2, struc_f1)
        elif qd.static(depth) == 2:
            struc_f1.c[0, 0] = 102
            f1(2, struc_f1)

    @qd.kernel
    def k1(struct_k1: MyStruct):
        f1(0, struct_k1)

    my_struct = create_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3
    assert my_struct.a[0, 0] == 100
    assert my_struct.b[0, 0] == 101
    assert my_struct.c[0, 0] == 102

    my_struct = create_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3
    assert my_struct.a[0, 0] == 100
    assert my_struct.b[0, 0] == 101
    assert my_struct.c[0, 0] == 102


@test_utils.test()
def test_pruning_reuse_func_diff_kernel_parameters() -> None:
    """
    In this test, any vertical call stack doesn't ever
    contain the same function more than once.
    However, the same function might be present in multiple
    child calls of a function.
    We assume however that the same py dataclass members will be used
    in both calls.s
    """

    @dataclasses.dataclass
    class MyStruct:
        _f3: qd.types.NDArray[qd.f32, 2]
        _f2b: qd.types.NDArray[qd.f32, 2]
        _f2a: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _k1: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            _f3=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2b=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2a=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f3(struc_f3: MyStruct):
        struc_f3._f3[0, 0] = 104
        f2b(struc_f3)

    @qd.func
    def f2b(struc_f2b: MyStruct):
        struc_f2b._f2b[0, 0] = 103

    @qd.func
    def f2a(struc_f2a: MyStruct):
        struc_f2a._f2a[0, 0] = 102
        f2b(struc_f2a)

    @qd.func
    def f1(struc_f1: MyStruct):
        struc_f1._f1[0, 0] = 101
        f2a(struc_f1)
        f3(struc_f1)

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(struct_k1)

    my_struct = create_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 5
    assert my_struct._f1[0, 0] == 101
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f2a[0, 0] == 102
    assert my_struct._f2b[0, 0] == 103
    assert my_struct._f3[0, 0] == 104

    my_struct = create_struct()
    k1(my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 5
    assert my_struct._f1[0, 0] == 101
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f2a[0, 0] == 102
    assert my_struct._f2b[0, 0] == 103
    assert my_struct._f3[0, 0] == 104


@test_utils.test()
def test_pruning_reuse_func_same_kernel_call_l1() -> None:
    @dataclasses.dataclass
    class MyStruct:
        _f1b: qd.types.NDArray[qd.f32, 2]
        _f1a: qd.types.NDArray[qd.f32, 2]
        _k1: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            _f1b=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1a=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f1(flag: qd.template(), struc_f1: MyStruct):
        if qd.static(flag):
            struc_f1._f1a[0, 0] = 101
        else:
            struc_f1._f1b[0, 0] = 102

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(False, struct_k1)
        f1(True, struct_k1)

    my_struct = create_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1a[0, 0] == 101
    assert my_struct._f1b[0, 0] == 102

    my_struct = create_struct()
    k1(my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1a[0, 0] == 101
    assert my_struct._f1b[0, 0] == 102


@test_utils.test()
def test_pruning_reuse_func_same_kernel_call_l2() -> None:
    @dataclasses.dataclass
    class MyStruct:
        _f2b: qd.types.NDArray[qd.f32, 2]
        _f2a: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _k1: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            _f2b=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2a=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f2(flag: qd.template(), struc_f2: MyStruct):
        if qd.static(flag):
            struc_f2._f2a[0, 0] = 102
        else:
            struc_f2._f2b[0, 0] = 103

    @qd.func
    def f1(struct_f1: MyStruct):
        struct_f1._f1[0, 0] = 101
        f2(False, struct_f1)
        f2(True, struct_f1)

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(struct_k1)

    my_struct = create_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1[0, 0] == 101
    assert my_struct._f2a[0, 0] == 102
    assert my_struct._f2b[0, 0] == 103

    my_struct = create_struct()
    k1(my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1[0, 0] == 101
    assert my_struct._f2a[0, 0] == 102
    assert my_struct._f2b[0, 0] == 103


@test_utils.test()
def test_pruning_reuse_func_across_kernels() -> None:
    """
    In this test, the same function can be used in different kernels,
    but with *different* used members
    """

    @dataclasses.dataclass
    class MyStruct:
        _k1: qd.types.NDArray[qd.f32, 2]
        _k2: qd.types.NDArray[qd.f32, 2]
        _f1_no_flag: qd.types.NDArray[qd.f32, 2]
        _f1_with_flag: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _k2=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1_no_flag=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1_with_flag=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f1(flag: qd.template(), struct_f1: MyStruct):
        if qd.static(flag):
            struct_f1._f1_with_flag[0, 0] = 102
        else:
            struct_f1._f1_no_flag[0, 0] = 103

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 101
        f1(False, struct_k1)

    @qd.kernel
    def k2(struct_k2: MyStruct):
        struct_k2._k2[0, 0] = 100
        f1(True, struct_k2)

    my_struct = make_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert my_struct._k1[0, 0] == 101
    assert my_struct._f1_with_flag[0, 0] == 0
    assert my_struct._f1_no_flag[0, 0] == 103

    my_struct = make_struct()
    k2(my_struct)
    k2_primal: Kernel = k2._primal
    kernel_args_count_by_type = k2_primal.launch_stats.kernel_args_count_by_type
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert my_struct._k2[0, 0] == 100
    assert my_struct._f1_with_flag[0, 0] == 102
    assert my_struct._f1_no_flag[0, 0] == 0


@test_utils.test()
def test_pruning_reuse_func_same_kernel_diff_call() -> None:
    """
    In this test, the same function can be used in different calls to the same kernel,
    but with *different* used members
    """

    @dataclasses.dataclass
    class MyStruct:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1_no_flag: qd.types.NDArray[qd.f32, 2]
        _f1_with_flag: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1_no_flag=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1_with_flag=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f1(flag: qd.template(), struct_f1: MyStruct):
        if qd.static(flag):
            struct_f1._f1_with_flag[0, 0] = 101
        else:
            struct_f1._f1_no_flag[0, 0] = 102

    @qd.kernel
    def k1(flag: qd.Template, struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(flag, struct_k1)

    my_struct = make_struct()
    k1(False, my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1_no_flag[0, 0] == 102
    assert my_struct._f1_with_flag[0, 0] == 0
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert sorted(list(k1_primal.used_py_dataclass_parameters_by_key_enforcing[k1_primal._last_launch_key])) == [
        "__qd_struct_k1",
        "__qd_struct_k1__qd__f1_no_flag",
        "__qd_struct_k1__qd__k1",
    ]

    my_struct = make_struct()
    k1(False, my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1_no_flag[0, 0] == 102
    assert my_struct._f1_with_flag[0, 0] == 0
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert sorted(list(k1_primal.used_py_dataclass_parameters_by_key_enforcing[k1_primal._last_launch_key])) == [
        "__qd_struct_k1",
        "__qd_struct_k1__qd__f1_no_flag",
        "__qd_struct_k1__qd__k1",
    ]

    my_struct = make_struct()
    k1(True, my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1_no_flag[0, 0] == 0
    assert my_struct._f1_with_flag[0, 0] == 101
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert sorted(list(k1_primal.used_py_dataclass_parameters_by_key_enforcing[k1_primal._last_launch_key])) == [
        "__qd_struct_k1",
        "__qd_struct_k1__qd__f1_with_flag",
        "__qd_struct_k1__qd__k1",
    ]

    my_struct = make_struct()
    k1(False, my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1_no_flag[0, 0] == 102
    assert my_struct._f1_with_flag[0, 0] == 0
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert sorted(list(k1_primal.used_py_dataclass_parameters_by_key_enforcing[k1_primal._last_launch_key])) == [
        "__qd_struct_k1",
        "__qd_struct_k1__qd__f1_no_flag",
        "__qd_struct_k1__qd__k1",
    ]

    my_struct = make_struct()
    k1(True, my_struct)
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1_no_flag[0, 0] == 0
    assert my_struct._f1_with_flag[0, 0] == 101
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 2
    assert sorted(list(k1_primal.used_py_dataclass_parameters_by_key_enforcing[k1_primal._last_launch_key])) == [
        "__qd_struct_k1",
        "__qd_struct_k1__qd__f1_with_flag",
        "__qd_struct_k1__qd__k1",
    ]


@test_utils.test()
def test_pruning_kwargs_same_param_names_diff_names() -> None:
    """
    In this test, we call functions from one parent, passing the same struct
    with same name, and with different name
    """

    @dataclasses.dataclass
    class MyStruct:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _f2a: qd.types.NDArray[qd.f32, 2]
        _f2b: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2a=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2b=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f2a(struct_f2a: MyStruct):
        struct_f2a._f2a[0, 0] += 3

    @qd.func
    def f2b(struct_f2b: MyStruct):
        struct_f2b._f2b[0, 0] += 5

    @qd.func
    def f1(struct_f1: MyStruct):
        struct_f1._f1[0, 0] = 101
        f2a(struct_f2a=struct_f1)
        f2a(struct_f2a=struct_f1)
        f2b(struct_f2b=struct_f1)

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(struct_f1=struct_k1)

    my_struct = make_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1[0, 0] == 101
    assert my_struct._f2a[0, 0] == 6
    assert my_struct._f2b[0, 0] == 5
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@pytest.mark.xfail(reason="cannot use * when calling qd.func")
@test_utils.test()
def test_pruning_func_return_star_to_another() -> None:
    """
    Using the tuple return from one fucntion as the args to
    another
    """

    @qd.func
    def return_params(a: qd.i32):
        return a + 1, a + 5

    @qd.func
    def f2(t: qd.types.NDArray[qd.i32, 1], a: qd.i32, b: qd.i32) -> None:
        t[0] = a
        t[1] = b

    @qd.kernel
    def k1(t: qd.types.NDArray[qd.i32, 1], a: qd.i32) -> None:
        f2(t, *return_params(a))

    t = qd.ndarray(qd.i32, (10,))
    k1(t, 3)
    assert t[0] == 4
    assert t[0] == 8


@pytest.mark.xfail(reason="cannot use * when calling qd.func")
@test_utils.test()
def test_pruning_func_return_star_to_another_two_step() -> None:
    """
    Using the tuple return from one fucntion as the args to
    another
    """

    @qd.func
    def return_params(a: qd.i32):
        return a + 1, a + 5

    @qd.func
    def f2(t: qd.types.NDArray[qd.i32, 1], a: qd.i32, b: qd.i32) -> None:
        t[0] = a
        t[1] = b

    @qd.kernel
    def k1(t: qd.types.NDArray[qd.i32, 1], a: qd.i32) -> None:
        res = return_params(a)
        f2(t, *res)

    t = qd.ndarray(qd.i32, (10,))
    k1(t, 3)
    assert t[0] == 4
    assert t[0] == 8


@test_utils.test()
def test_pruning_func_return_star_to_another_explicit_vars() -> None:
    """
    Using the tuple return from one fucntion as the args to
    another
    """

    @qd.func
    def return_params(a: qd.i32):
        return a + 1, a + 5

    @qd.func
    def f2(t: qd.types.NDArray[qd.i32, 1], a: qd.i32, b: qd.i32) -> None:
        t[0] = a
        t[1] = b

    @qd.kernel
    def k1(t: qd.types.NDArray[qd.i32, 1], a: qd.i32) -> None:
        b, c = return_params(a)
        f2(t, b, c)

    t = qd.ndarray(qd.i32, (10,))
    k1(t, 3)
    assert t[0] == 4
    assert t[1] == 8


@test_utils.test()
def test_pruning_pass_element_of_tensor_of_dataclass() -> None:
    vec3 = qd.types.vector(3, qd.f32)

    @dataclasses.dataclass
    class MyStruct:
        _unused0: qd.types.NDArray[vec3, 2]
        _k1: qd.types.NDArray[vec3, 2]
        _unused0b: qd.types.NDArray[vec3, 2]
        _f1: qd.types.NDArray[vec3, 2]
        _unused1: qd.types.NDArray[vec3, 2]
        _in: qd.types.NDArray[vec3, 2]
        _unused2: qd.types.NDArray[vec3, 2]
        _out: qd.types.NDArray[vec3, 2]
        _unused3: qd.types.NDArray[vec3, 2]

    def make_struct():
        my_struct = MyStruct(
            _unused0=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _k1=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _unused0b=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _f1=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _unused1=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _in=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _unused2=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _out=qd.ndarray(dtype=vec3, shape=(1, 1)),
            _unused3=qd.ndarray(dtype=vec3, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f2(_in: vec3) -> vec3:
        return _in + 5.0

    @qd.func
    def f1(struct_f1: MyStruct):
        struct_f1._f1[0, 0] = 101
        struct_f1._out[0, 0] = f2(struct_f1._in[0, 0])

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(struct_f1=struct_k1)

    my_struct = make_struct()
    k1(my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0][0] == 100
    assert my_struct._f1[0, 0][0] == 101
    assert my_struct._out[0, 0][0] == 5
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@test_utils.test()
def test_pruning_kwargs_swap_order() -> None:
    """
    In this test, we call into a kwargs function with the kwargs in a different
    order than in the child function declaration; and different number of params
    in each struct
    """

    @dataclasses.dataclass
    class MyStruct1:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _unused1: qd.types.NDArray[qd.f32, 2]
        _unused2: qd.types.NDArray[qd.f32, 2]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused2=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct1, my_struct2

    @qd.func
    def f1(struct1_f1: MyStruct1, struct2_f1: MyStruct2):
        struct1_f1._f1[0, 0] = 102
        struct2_f1._f1[0, 0] = 103

    @qd.kernel
    def k1(struct1_k1: MyStruct1, struct2_k1: MyStruct2):
        struct1_k1._k1[0, 0] = 100
        struct2_k1._k1[0, 0] = 101
        f1(struct2_f1=struct2_k1, struct1_f1=struct1_k1)

    my_struct1, my_struct2 = make_structs()
    k1(my_struct1, my_struct2)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct1._k1[0, 0] == 100
    assert my_struct2._k1[0, 0] == 101
    assert my_struct1._f1[0, 0] == 102
    assert my_struct2._f1[0, 0] == 103
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@test_utils.test()
def test_pruning_kwargs_swap_order_bound_callable() -> None:
    """
    In this test, we call into a kwargs function with the kwargs in a different
    order than in the child function declaration; and different number of params
    in each struct.

    Compared to test_pruning_kwargs_swap_order, we use a data oriented object, with
    the function on that
    """

    @dataclasses.dataclass
    class MyStruct1:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _unused1: qd.types.NDArray[qd.f32, 2]
        _unused2: qd.types.NDArray[qd.f32, 2]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused2=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct1, my_struct2

    @qd.data_oriented
    class MyDataOriented:
        def __init__(self) -> None: ...

        @qd.func
        def f1(self, struct1_f1: MyStruct1, struct2_f1: MyStruct2):
            struct1_f1._f1[0, 0] = 102
            struct2_f1._f1[0, 0] = 103

    @qd.kernel
    def k1(my_data_oriented: qd.Template, struct1_k1: MyStruct1, struct2_k1: MyStruct2):
        struct1_k1._k1[0, 0] = 100
        struct2_k1._k1[0, 0] = 101
        my_data_oriented.f1(struct2_f1=struct2_k1, struct1_f1=struct1_k1)

    my_struct1, my_struct2 = make_structs()
    my_data_oriented = MyDataOriented()
    k1(my_data_oriented, my_struct1, my_struct2)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct1._k1[0, 0] == 100
    assert my_struct2._k1[0, 0] == 101
    assert my_struct1._f1[0, 0] == 102
    assert my_struct2._f1[0, 0] == 103
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@test_utils.test()
def test_pruning_bound_callable_args() -> None:
    @dataclasses.dataclass
    class MyStruct1:
        _k1: qd.types.NDArray[qd.f32, 1]
        _f1: qd.types.NDArray[qd.f32, 2]
        _unused1: qd.types.NDArray[qd.f32, 4]
        _unused2: qd.types.NDArray[qd.f32, 4]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: qd.types.NDArray[qd.f32, 1]
        _f1: qd.types.NDArray[qd.f32, 3]
        _unused: qd.types.NDArray[qd.f32, 4]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused1=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1, 1)),
            _unused2=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1, 1)),
        )
        return my_struct1, my_struct2

    @qd.data_oriented
    class MyDataOriented:
        def __init__(self) -> None: ...

        @qd.func
        def f1(self, struct1_f1: MyStruct1, struct2_f1: MyStruct2):
            struct1_f1._f1[0, 0] = 102
            struct2_f1._f1[0, 0, 0] = 103

    @qd.kernel
    def k1(my_data_oriented: qd.Template, struct1_k1: MyStruct1, struct2_k1: MyStruct2):
        struct1_k1._k1[0] = 100
        struct2_k1._k1[0] = 101
        my_data_oriented.f1(struct1_k1, struct2_k1)

    my_struct1, my_struct2 = make_structs()
    my_data_oriented = MyDataOriented()
    k1(my_data_oriented, my_struct1, my_struct2)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct1._k1[0] == 100
    assert my_struct2._k1[0] == 101
    assert my_struct1._f1[0, 0] == 102
    assert my_struct2._f1[0, 0, 0] == 103
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@test_utils.test()
def test_pruning_bound_callable_kwargs() -> None:
    @dataclasses.dataclass
    class MyStruct1:
        _k1: qd.types.NDArray[qd.f32, 1]
        _f1: qd.types.NDArray[qd.f32, 2]
        _unused1: qd.types.NDArray[qd.f32, 4]
        _unused2: qd.types.NDArray[qd.f32, 4]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: qd.types.NDArray[qd.f32, 1]
        _f1: qd.types.NDArray[qd.f32, 3]
        _unused: qd.types.NDArray[qd.f32, 4]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused1=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1, 1)),
            _unused2=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1, 1, 1)),
        )
        return my_struct1, my_struct2

    @qd.data_oriented
    class MyDataOriented:
        def __init__(self) -> None: ...

        @qd.func
        def f1(self, struct1_f1: MyStruct1, struct2_f1: MyStruct2):
            struct1_f1._f1[0, 0] = 102
            struct2_f1._f1[0, 0, 0] = 103

    @qd.kernel
    def k1(my_data_oriented: qd.Template, struct1_k1: MyStruct1, struct2_k1: MyStruct2):
        struct1_k1._k1[0] = 100
        struct2_k1._k1[0] = 101
        my_data_oriented.f1(struct1_f1=struct1_k1, struct2_f1=struct2_k1)

    my_struct1, my_struct2 = make_structs()
    my_data_oriented = MyDataOriented()
    k1(my_data_oriented=my_data_oriented, struct1_k1=my_struct1, struct2_k1=my_struct2)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct1._k1[0] == 100
    assert my_struct2._k1[0] == 101
    assert my_struct1._f1[0, 0] == 102
    assert my_struct2._f1[0, 0, 0] == 103
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 4


@test_utils.test()
def test_pruning_star_args() -> None:
    """
    Designed to test
    https://github.com/Genesis-Embodied-AI/Genesis/blob/2d98bbb786e94b3f6c4e7171c87b4ff31ff3ccdf/tests/test_utils.py#L103
    scenario
    """

    @qd.func
    def f1(a: qd.types.NDArray[qd.i32, 1], b: qd.i32, c: qd.i32):
        a[0] = b
        a[1] = c

    @qd.kernel
    def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        f1(a, *star_args)

    star_args = [3, 5]

    a = qd.ndarray(qd.i32, (10,))
    k1(a)
    assert a[0] == 3
    assert a[1] == 5


@test_utils.test()
def test_pruning_star_args_error_not_at_end_another_arg() -> None:
    @qd.func
    def f1(a: qd.types.NDArray[qd.i32, 1], b: qd.i32, c: qd.i32, d: qd.i32):
        a[0] = b
        a[1] = c

    @qd.kernel
    def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        f1(a, *star_args, 3)

    star_args = [3, 5]

    a = qd.ndarray(qd.i32, (10,))
    with pytest.raises(QuadrantsSyntaxError) as e:
        k1(a)
    assert "STARNOTLAST" in e.value.args[0]


@test_utils.test()
def test_pruning_star_args_error_not_at_end_kwargs() -> None:
    @qd.func
    def f1(a: qd.types.NDArray[qd.i32, 1], b: qd.i32, c: qd.i32, d: qd.i32):
        a[0] = b
        a[1] = c

    @qd.kernel
    def k1(a: qd.types.NDArray[qd.i32, 1]) -> None:
        f1(a, *star_args, d=3)

    star_args = [3, 5]

    a = qd.ndarray(qd.i32, (10,))
    with pytest.raises(QuadrantsSyntaxError) as e:
        k1(a)
    assert "STARNOTLAST" in e.value.args[0]


@test_utils.test()
def test_pruning_iterate_function() -> None:
    """
    Designed to test
    https://github.com/Genesis-Embodied-AI/Genesis/blob/6d344d0d4c46b7c9de98442bc4d09f9f9bfa541b/genesis/engine/couplers/sap_coupler.py#L631
    """

    @dataclasses.dataclass
    class MyStruct:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _f2: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f1(struct: MyStruct):
        struct._f1[0, 0] = 101

    @qd.func
    def f2(struct: MyStruct):
        struct._f2[0, 0] = 102

    functions = [f1, f2]

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        for fn in qd.static(functions):
            fn(struct=struct_k1)

    my_struct = make_struct()
    k1(struct_k1=my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1[0, 0] == 101
    assert my_struct._f2[0, 0] == 102
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3


@test_utils.test()
def test_pruning_iterate_function_no_iterate() -> None:
    @dataclasses.dataclass
    class MyStruct:
        _k1: qd.types.NDArray[qd.f32, 2]
        _f1: qd.types.NDArray[qd.f32, 2]
        _f2: qd.types.NDArray[qd.f32, 2]
        _unused: qd.types.NDArray[qd.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f1=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _f2=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
            _unused=qd.ndarray(dtype=qd.f32, shape=(1, 1)),
        )
        return my_struct

    @qd.func
    def f1(struct: MyStruct):
        struct._f1[0, 0] = 101

    @qd.func
    def f2(struct: MyStruct):
        struct._f2[0, 0] = 102

    @qd.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        f1(struct=struct_k1)
        f2(struct=struct_k1)

    my_struct = make_struct()
    k1(struct_k1=my_struct)
    k1_primal: Kernel = k1._primal
    kernel_args_count_by_type = k1_primal.launch_stats.kernel_args_count_by_type
    assert not k1_primal.launch_observations.found_kernel_in_materialize_cache
    assert my_struct._k1[0, 0] == 100
    assert my_struct._f1[0, 0] == 101
    assert my_struct._f2[0, 0] == 102
    assert kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 3


@test_utils.test()
def test_dataclass_with_template_emits_deprecation_warning():
    """A frozen ``@dataclasses.dataclass`` passed into a ``qd.Template``-annotated kernel parameter must emit a
    ``DeprecationWarning`` at materialize time. The pattern was never an intentional Quadrants pattern (the template
    walker happens to handle dataclass-shaped objects, but the supported annotation is the dataclass type itself).
    The warning is emitted from ``Kernel.materialize`` after the cache-hit early return, so it fires once per
    (kernel, spec-key) and stays off the steady-state launch hot path. See ``compound_types.md`` Overview."""

    @dataclass(frozen=True)
    class Foo:
        x: object = None

    x = qd.ndarray(qd.f32, shape=(4,))
    f = Foo(x=x)

    @qd.kernel
    def run(foo: qd.Template):
        for i in range(4):
            foo.x[i] = float(i)

    with pytest.warns(DeprecationWarning, match="qd.Template-annotated kernel parameter"):
        run(f)


@test_utils.test()
def test_data_oriented_with_template_does_not_emit_deprecation_warning():
    """The canonical ``@qd.data_oriented`` + ``qd.Template`` path must NOT emit the deprecation warning. The
    materialize-side check excludes ``@qd.data_oriented`` instances via ``is_data_oriented(val)`` because doubly-decorated
    objects (``@qd.data_oriented`` over ``@dataclasses.dataclass``) are a legitimate pattern routed through the
    data-oriented path."""

    @qd.data_oriented
    class Foo:
        def __init__(self, x):
            self.x = x

    x = qd.ndarray(qd.f32, shape=(4,))
    f = Foo(x=x)

    @qd.kernel
    def run(foo: qd.Template):
        for i in range(4):
            foo.x[i] = float(i)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(f)
    matching = [w for w in caught if issubclass(w.category, DeprecationWarning) and "qd.Template" in str(w.message)]
    assert matching == [], f"unexpected DeprecationWarning(s): {[str(w.message) for w in matching]}"


@test_utils.test()
def test_typed_dataclass_does_not_emit_deprecation_warning():
    """The recommended path — frozen ``@dataclasses.dataclass`` passed as a typed kernel parameter (the dataclass
    type itself) — must NOT emit the deprecation warning. Only the ``qd.Template`` outer-annotation path is
    deprecated; the typed-dataclass flatten-to-args path is the supported pattern."""

    @dataclass(frozen=True)
    class Foo:
        x: qd.types.NDArray[qd.f32, 1]

    x = qd.ndarray(qd.f32, shape=(4,))
    f = Foo(x=x)

    @qd.kernel
    def run(foo: Foo):
        for i in range(4):
            foo.x[i] = float(i)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(f)
    matching = [w for w in caught if issubclass(w.category, DeprecationWarning) and "qd.Template" in str(w.message)]
    assert matching == [], f"unexpected DeprecationWarning(s): {[str(w.message) for w in matching]}"


@test_utils.test()
def test_kernel_accepts_subclass_of_annotated_dataclass_param():
    @dataclass
    class QdSafe:
        x: qd.types.NDArray[qd.i32, 1]

    @dataclass
    class SubDataclass(QdSafe):
        y: str

    class SubPyClass(QdSafe):
        y: str | None

    x_desired = 100

    @qd.kernel
    def mykernel(dat: QdSafe) -> None:
        for i in range(dat.x.shape[0]):
            dat.x[i] = x_desired

    y_desired = "i love quadrants"

    dat = SubDataclass(x=qd.ndarray(qd.i32, (8,)), y=y_desired)
    mykernel(dat)
    assert dat.x[0] == x_desired
    assert dat.x[4] == x_desired
    assert dat.x[7] == x_desired
    assert dat.y == y_desired

    pydat = SubPyClass(x=qd.ndarray(qd.i32, (8,)))
    pydat.y = y_desired
    mykernel(pydat)
    assert pydat.x[1] == x_desired
    assert pydat.x[5] == x_desired
    assert pydat.x[6] == x_desired
    assert pydat.y == y_desired


@test_utils.test()
def test_kernel_accepts_subclass_of_annotated_frozen_dataclass_param():
    @dataclass(frozen=True)
    class QdSafe:
        x: int

    @dataclass(frozen=True)
    class SubDataclass(QdSafe):
        y: set[int]

    class SubPyClass(QdSafe):
        y: set[int] | None

    @qd.func
    def read_x(dat: QdSafe) -> int:
        return dat.x

    @qd.kernel
    def mykernel(dat: QdSafe) -> int:
        return read_x(dat)

    y_desired = {1, 2, 3, 4, 5, 6, 7, 8}
    x_desired = 12345678

    dat = SubDataclass(x=x_desired, y=y_desired)
    assert mykernel(dat) == x_desired
    assert dat.y is y_desired

    pydat = SubPyClass(x=x_desired)
    object.__setattr__(pydat, "y", y_desired)
    assert mykernel(pydat) == x_desired
    assert pydat.y is y_desired


@test_utils.test()
def test_kernel_accepts_subclass_with_unsupported_final_app_field():
    """A subclass may carry an application-only ``Final`` field of a type Quadrants cannot bake (e.g. ``Final[list]``).
    Only the annotated base's fields reach the kernel, so the extra ``Final`` field must be ignored, not validated
    (validating it would reject the supported subclass pattern)."""
    from typing import Final

    import numpy as np

    @dataclass(frozen=True)
    class Base:
        x: qd.types.NDArray[qd.i32, 1]

    @dataclass(frozen=True)
    class Sub(Base):
        extra: Final[list]  # unbakeable as a Quadrants Final field, but the kernel (annotated Base) never sees it

    @qd.kernel
    def fill(dat: Base):
        for i in range(4):
            dat.x[i] = 7

    x = qd.ndarray(qd.i32, shape=(4,))
    sub = Sub(x=x, extra=[1, 2, 3])
    fill(sub)  # must not raise while validating the ignored ``extra`` field
    np.testing.assert_array_equal(x.to_numpy(), [7, 7, 7, 7])


@test_utils.test()
def test_kernel_accepts_data_oriented_subclass_of_dataclass_param():
    # A @qd.data_oriented class subclassing a dataclasses.dataclass is still a subclass of the annotated type, so it
    # should dispatch through the dataclass path: the kernel reads only the base's declared fields via getattr.
    @dataclass
    class QdSafe:
        x: qd.types.NDArray[qd.i32, 1]

    @qd.data_oriented
    class DataOrientedSub(QdSafe):
        pass

    @qd.kernel
    def mykernel(dat: QdSafe) -> None:
        for i in range(4):
            dat.x[i] = 7

    sub = DataOrientedSub(x=qd.ndarray(qd.i32, shape=(4,)))
    mykernel(sub)
    assert sub.x[0] == 7
    assert sub.x[3] == 7


@test_utils.test()
def test_kernel_disallows_unassignable_dataclass():
    @dataclass
    class Expected:
        x: int

    @dataclass
    class Unrelated:
        x: int

    @qd.kernel
    def mykernel(_: Expected) -> None: ...

    other = Unrelated(x=0)
    with pytest.raises(QuadrantsRuntimeTypeError):
        mykernel(other)


@test_utils.test()
def test_frozen_dataclass_passed_to_multiple_ancestor_annotations():
    @dataclass(frozen=True)
    class QdSafe1:
        x1: qd.types.NDArray[qd.i32, 1]

    @dataclass(frozen=True)
    class QdSafe2:
        x2: qd.types.NDArray[qd.i32, 1]

    @dataclass(frozen=True)
    class Sub(QdSafe1, QdSafe2): ...

    @qd.kernel
    def mykernel1(dat: QdSafe1) -> None:
        dat.x1[0] = 1

    @qd.kernel
    def mykernel2(dat: QdSafe2) -> None:
        dat.x2[0] = 2

    sub = Sub(x1=qd.ndarray(qd.i32, shape=(4,)), x2=qd.ndarray(qd.i32, shape=(4,)))
    # mykernel1 and mykernel2 might use different caching mechanisms! Well, we hope that this is fine and still works.
    mykernel1(sub)
    mykernel2(sub)
    assert sub.x1[0] == 1
    assert sub.x2[0] == 2


# ---------------------------------------------------------------------------
# POC: ``typing.Final[T]`` fields on frozen dataclasses => compile-time templates.
#
# Design goal (see perso_hugh/doc/final_dataclass_templates.md): let users mark selected fields of a plain frozen
# ``@dataclasses.dataclass`` config as ``typing.Final[T]`` to signal that the field's value must be baked as a
# compile-time constant. This replaces the pre-existing ``@qd.data_oriented`` + ``qd.Template`` pattern for static
# configs - where the same effect required opting the whole class into the data_oriented machinery and paying the
# associated per-launch overhead (see PR #705 discussion of ``_arg_disposition`` / ``TemplateMapper.lookup``).
#
# Semantics:
# - ``config.field`` accessed inside a kernel body resolves at AST-build time to the actual Python value baked into
#   ``config``. ``qd.static(config.field)`` therefore works, as does ``if qd.static(config.flag): ...`` dead-branch
#   elimination.
# - The value of every ``Final[T]`` field is folded into the template mapper's spec key; two configs that differ on
#   any Final field compile distinct kernels.
# - Final fields are NOT declared as runtime scalar kernel args and are NOT pushed into the launch context.
# - Fields without ``Final`` retain the current typed-dataclass behavior (declared as runtime scalar / ndarray kernel
#   args and passed at launch time). Mixing Final and non-Final fields in the same dataclass is supported.
# ---------------------------------------------------------------------------


@test_utils.test()
def test_final_field_bakes_as_compile_time_constant_via_qd_static():
    """Baseline POC: ``qd.static(config.dt)`` on a frozen dataclass with ``dt: Final[float]`` compiles and produces
    the expected numeric result. The important part is that the kernel compiles at all - ``qd.static`` fails today
    on a non-Final dataclass field because the field lowers to a runtime scalar ``Expr`` (see
    ``QuadrantsCompilationError: Invalid data type typing.Final[int]`` before this change).

    ``qd.static(config.dt)`` is materialised into a named local ``dt_const`` before the kernel's inner-loop assign to
    avoid ``build_Assign``'s ``is_static_assign`` check, which rejects ``out[i] = qd.static(...)`` with "Static assign
    cannot be used on elements in arrays". Reading the same Python value from a bound local is unaffected - it is
    still the baked constant."""
    from typing import Final

    @dataclass(frozen=True)
    class SimConfig:
        dt: Final[float]
        enable_gravity: Final[bool]

    @qd.kernel
    def integrate(config: SimConfig, positions: qd.types.NDArray[qd.f32, 1]):
        dt_const = qd.static(config.dt)
        for i in positions:
            positions[i] += dt_const
            if qd.static(config.enable_gravity):
                positions[i] -= 9.8 * dt_const

    cfg = SimConfig(dt=0.5, enable_gravity=True)
    x = qd.ndarray(qd.f32, shape=(4,))
    integrate(cfg, x)
    # 0.5 - 9.8 * 0.5 = -4.4
    for i in range(4):
        assert abs(x[i] - (-4.4)) < 1e-4


@test_utils.test()
def test_final_field_value_change_triggers_recompilation():
    """Two ``SimConfig`` instances with different Final-field values must compile as distinct kernels - the Final
    field's value has to be part of the template mapper spec key, otherwise the second launch would reuse the
    first's baked-in constant and produce the wrong output."""
    from typing import Final

    @dataclass(frozen=True)
    class SimConfig:
        offset: Final[int]

    @qd.kernel
    def bump(config: SimConfig, x: qd.types.NDArray[qd.i32, 1]):
        v = qd.static(config.offset)
        for i in x:
            x[i] += v

    x = qd.ndarray(qd.i32, shape=(3,))
    bump(SimConfig(offset=7), x)
    bump(SimConfig(offset=100), x)
    for i in range(3):
        assert x[i] == 107


@test_utils.test()
def test_final_and_non_final_fields_mix():
    """Non-Final fields retain the current runtime-scalar-arg behavior; Final fields flow through the template path.
    Both must coexist in the same dataclass."""
    from typing import Final

    @dataclass(frozen=True)
    class Params:
        scale: Final[float]  # compile-time constant
        bias: float  # runtime scalar kernel arg

    @qd.kernel
    def apply(p: Params, x: qd.types.NDArray[qd.f32, 1]):
        s = qd.static(p.scale)
        for i in x:
            x[i] = x[i] * s + p.bias

    x = qd.ndarray(qd.f32, shape=(3,))
    for i in range(3):
        x[i] = float(i)
    apply(Params(scale=2.0, bias=1.0), x)
    # scale=2.0 baked; bias=1.0 passed at launch.
    for i in range(3):
        assert abs(x[i] - (float(i) * 2.0 + 1.0)) < 1e-5


@test_utils.test()
def test_final_field_identical_values_share_compiled_kernel():
    """Kernel caching correctness: two ``SimConfig`` instances with the same Final-field value must reuse the same
    compiled kernel. Two consecutive launches with equal Final values keep ``template_mapper.mapping`` size at 1;
    a third launch with a different value grows it to 2. Guards against accidentally hashing the *instance* rather
    than the *value* in the spec key."""
    from typing import Final

    @dataclass(frozen=True)
    class SimConfig:
        offset: Final[int]

    @qd.kernel
    def bump(config: SimConfig, x: qd.types.NDArray[qd.i32, 1]):
        v = qd.static(config.offset)
        for i in x:
            x[i] += v

    x = qd.ndarray(qd.i32, shape=(3,))
    bump(SimConfig(offset=5), x)
    assert len(bump._primal.mapper.mapping) == 1
    bump(SimConfig(offset=5), x)  # same value, different instance
    assert len(bump._primal.mapper.mapping) == 1, "identical Final values must share a compiled kernel"
    bump(SimConfig(offset=100), x)  # different value
    assert len(bump._primal.mapper.mapping) == 2, "distinct Final values must recompile"


@test_utils.test()
def test_final_field_propagates_through_qd_func_call():
    """``@qd.func`` invoked from a kernel with a dataclass argument containing a ``Final[T]`` field must see the
    baked value on its side too. The caller's flat name resolves to a Python value via ``build_Name``, then
    ``_transform_func_arg`` sees ``annotation=Final[T]`` and binds the value directly - if the func body used
    ``impl.expr_init_func(data)`` instead, ``cfg.scale`` would arrive as an ``Expr`` and ``qd.static`` would fail
    with ``Input to qd.static must be compile-time constants``."""
    from typing import Final

    @dataclass(frozen=True)
    class Cfg:
        scale: Final[float]

    @qd.func
    def scale_by(cfg: Cfg, x: qd.types.NDArray[qd.f32, 1], i: qd.i32):
        x[i] = x[i] * qd.static(cfg.scale)

    @qd.kernel
    def apply(cfg: Cfg, x: qd.types.NDArray[qd.f32, 1]):
        for i in x:
            scale_by(cfg, x, i)

    x = qd.ndarray(qd.f32, shape=(4,))
    for i in range(4):
        x[i] = float(i)
    apply(Cfg(scale=3.0), x)
    for i in range(4):
        assert abs(x[i] - float(i) * 3.0) < 1e-5


@test_utils.test()
def test_final_field_on_nested_dataclass():
    """Final fields work at any depth: the top-level kernel arg is a dataclass whose field is another dataclass,
    whose field is ``Final[T]``. The recursive ``_transform_kernel_arg`` walker threads the sub-instance's runtime
    value through so ``getattr(sub, final_field_name)`` reads the baked value."""
    from typing import Final

    @dataclass(frozen=True)
    class Inner:
        scale: Final[float]

    @dataclass(frozen=True)
    class Outer:
        inner: Inner
        bias: Final[float]

    @qd.kernel
    def apply(outer: Outer, x: qd.types.NDArray[qd.f32, 1]):
        s = qd.static(outer.inner.scale)
        b = qd.static(outer.bias)
        for i in x:
            x[i] = x[i] * s + b

    x = qd.ndarray(qd.f32, shape=(3,))
    for i in range(3):
        x[i] = float(i)
    apply(Outer(inner=Inner(scale=2.0), bias=1.0), x)
    for i in range(3):
        assert abs(x[i] - (float(i) * 2.0 + 1.0)) < 1e-5


@test_utils.test()
def test_final_field_with_ndarray_sibling():
    """Final scalar fields alongside ``ndarray`` fields in the same frozen dataclass. The ndarray field still flows
    as an ndarray kernel arg (runtime); the Final scalar bakes as compile-time."""
    from typing import Final

    @dataclass(frozen=True)
    class State:
        n: Final[int]
        buf: qd.types.NDArray[qd.i32, 1]

    @qd.kernel
    def fill(state: State):
        n_const = qd.static(state.n)
        for i in range(n_const):
            state.buf[i] = n_const - i

    buf = qd.ndarray(qd.i32, shape=(5,))
    fill(State(n=5, buf=buf))
    for i in range(5):
        assert buf[i] == 5 - i


@test_utils.test()
def test_final_field_value_is_part_of_offline_fastcache_key(tmp_path: Path):
    """A Final field's value must be part of the *offline* fastcache key, not just the in-process template mapper spec
    key. Regression test for a soundness bug in the original implementation: because ``dataclass_to_repr`` only
    appended a field's value to the cache key when that field carried ``FIELD_METADATA_CACHE_VALUE`` metadata, a kernel
    compiled with ``offset=7`` baked in was loaded from the offline cache in a later process for a config carrying
    ``offset=100``, silently returning 7.

    Uses two separate ``qd.init`` cycles sharing one ``offline_cache_file_path`` so the second launch genuinely hits
    the persisted cache - the same structure as ``test_prune_used_parameters_fastcache_dead_static_branch``."""
    from typing import Final

    @dataclass(frozen=True)
    class Cfg:
        offset: Final[int]

    arch_name = qd.lang.impl.current_cfg().arch.name
    for offset_value in (7, 100):
        qd.init(arch=getattr(qd, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @qd.kernel(fastcache=True)
        def bump(config: Cfg, x: qd.types.NDArray[qd.i32, 1]):
            v = qd.static(config.offset)
            for i in x:
                x[i] = v

        x = qd.ndarray(qd.i32, shape=(3,))
        bump(Cfg(offset=offset_value), x)
        assert x[0] == offset_value, (
            f"expected {offset_value}, got {x[0]} - a kernel with a different Final value was reused from the "
            f"offline fastcache"
        )


@test_utils.test()
def test_final_field_on_non_frozen_dataclass_is_rejected():
    """A ``Final`` field's value is baked into compiled code, so its carrier must not be reassignable. A plain
    ``@dataclasses.dataclass`` (``eq=True``, non-frozen) sets ``__hash__ = None`` and is rejected with an actionable
    message. ``frozen=True`` and ``unsafe_hash=True`` are both accepted."""
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    @dataclass  # not frozen
    class Mutable:
        x: Final[int]

    with pytest.raises(TypeError, match="but is not frozen"):
        final_field_names(Mutable)

    # ``eq=False`` inherits ``object.__hash__``, so the codebase-wide ``__hash__ is not None`` frozen proxy reads this
    # plain mutable class as frozen. It must still be rejected: ``config.x = 9`` on such a class raises nothing, and
    # the ``id(arg)``-keyed lookup cache would then keep using the kernel baked with the old value.
    @dataclass(eq=False)
    class EqFalse:
        x: Final[int]

    assert EqFalse.__hash__ is not None
    with pytest.raises(TypeError, match="but is not frozen"):
        final_field_names(EqFalse)

    @dataclass(frozen=True)
    class Frozen:
        x: Final[int]

    assert final_field_names(Frozen) == {"x"}

    @dataclass(unsafe_hash=True)
    class UnsafeHash:
        x: Final[int]

    assert final_field_names(UnsafeHash) == {"x"}


@test_utils.test()
def test_mutable_ancestor_of_nested_final_field_is_rejected():
    """Every dataclass on the path down to a ``Final`` leaf must be non-reassignable, not just its direct carrier.

    A frozen ``Inner`` holding ``n: Final[int]`` protects ``inner.n``, but a mutable ``Outer`` holding ``child: Inner``
    still allows ``outer.child = Inner(n=9)``. ``TemplateMapper.lookup`` memoises the specialisation on the top-level
    ``id(arg)``, so re-launching with that same ``outer`` would silently reuse the kernel compiled with the previous
    child's baked constant and produce results for the old value. Reject the mutable ancestor instead."""
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    @dataclass(frozen=True)
    class Inner:
        n: Final[int]

    @dataclass  # mutable ancestor: `outer.child` can be rebound
    class MutableOuter:
        child: Inner

    # The direct-carrier check cannot catch this: MutableOuter declares no Final field of its own.
    assert not any(f.name == "n" for f in dataclasses.fields(MutableOuter))
    with pytest.raises(TypeError, match=r"is not frozen but reaches the ``Final`` field MutableOuter\.child\.n"):
        final_field_names(MutableOuter)

    # Reported at launch rather than silently returning stale results.
    @qd.kernel
    def k(o: MutableOuter, out: qd.types.NDArray[qd.i32, 1]):
        v = qd.static(o.child.n)
        for i in out:
            out[i] = v

    with pytest.raises(Exception, match="is not frozen but reaches"):
        k(MutableOuter(child=Inner(n=1)), qd.ndarray(qd.i32, shape=(2,)))

    # Deeper nesting names the innermost mutable carrier, which is the one that needs changing.
    @dataclass
    class MutableMid:
        child: Inner

    @dataclass(frozen=True)
    class FrozenTop:
        mid: MutableMid

    with pytest.raises(
        TypeError, match=r"MutableMid is not frozen but reaches the ``Final`` field MutableMid\.child\.n"
    ):
        final_field_names(MutableMid)

    # A frozen top level does not launder a mutable class further down: the spec-key walk validates each nested
    # dataclass type it descends into, so the error still surfaces on the first launch rather than silently baking a
    # value that ``top.mid.child = Inner(n=9)`` could change.
    @qd.kernel
    def k_deep(t: FrozenTop, out: qd.types.NDArray[qd.i32, 1]):
        v = qd.static(t.mid.child.n)
        for i in out:
            out[i] = v

    with pytest.raises(Exception, match="MutableMid is not frozen but reaches"):
        k_deep(FrozenTop(mid=MutableMid(child=Inner(n=1))), qd.ndarray(qd.i32, shape=(2,)))

    # Frozen all the way down is fine, and so is the explicit ``unsafe_hash`` opt-out.
    @dataclass(frozen=True)
    class FrozenOuter:
        child: Inner

    assert final_field_names(FrozenOuter) == frozenset()

    @dataclass(unsafe_hash=True)
    class UnsafeHashOuter:
        child: Inner

    assert final_field_names(UnsafeHashOuter) == frozenset()

    # A mutable dataclass that reaches no Final field at all keeps its pre-existing behaviour: not our business.
    @dataclass
    class NoFinalInner:
        n: int

    @dataclass
    class MutableNoFinalOuter:
        child: NoFinalInner

    assert final_field_names(MutableNoFinalOuter) == frozenset()


@test_utils.test()
def test_nested_final_field_recompiles_per_frozen_carrier_instance():
    """The nested-Final path must specialise per value, not just per top-level object identity.

    Complements the rejection test above: with every carrier frozen, distinct nested ``Final`` values must produce
    distinct kernels rather than reusing the first one compiled."""
    from typing import Final

    @dataclass(frozen=True)
    class Inner:
        n: Final[int]

    @dataclass(frozen=True)
    class Outer:
        child: Inner
        scale: int

    @qd.kernel
    def k(o: Outer, out: qd.types.NDArray[qd.i32, 1]):
        v = qd.static(o.child.n)
        for i in out:
            out[i] = v * o.scale

    out = qd.ndarray(qd.i32, shape=(2,))
    # Held in a list rather than passed as temporaries: the launch-context cache is keyed on the argument ids, and a
    # temporary carrier is freed straight after its launch, so the next one can be allocated at the same address and
    # collide with the previous entry. That is orthogonal to Final (it applies to any dataclass carrying a runtime
    # scalar), but it would make the ``scale`` assertion below flaky.
    configs = [Outer(child=Inner(n=3), scale=2), Outer(child=Inner(n=5), scale=2), Outer(child=Inner(n=5), scale=4)]

    k(configs[0], out)
    assert out[0] == 6
    k(configs[1], out)
    assert out[0] == 10, f"nested Final value change did not recompile: got {out[0]}, expected 10"
    # ``scale`` is an ordinary runtime field, so it must not add a specialisation.
    k(configs[2], out)
    assert out[0] == 20


@test_utils.test()
def test_final_field_rejects_non_bakeable_inner_types():
    """``Final[T]`` only accepts a ``T`` that is meaningful as a compile-time literal and hashes / reprs by value
    stably across processes: ``bool`` / ``int`` / ``float`` / ``str`` and ``enum.Enum`` subclasses. Arrays, structs,
    nested dataclasses and arbitrary objects are rejected with a tailored remediation hint rather than silently
    miscompiling."""
    import enum
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    @dataclass(frozen=True)
    class Leaf:
        a: Final[int]

    @dataclass(frozen=True)
    class FinalNestedDataclass:
        inner: Final[Leaf]

    with pytest.raises(TypeError, match="nested dataclasses are walked structurally"):
        final_field_names(FinalNestedDataclass)

    @dataclass(frozen=True)
    class FinalNdarrayType:
        buf: Final[qd.types.ndarray_type.NdarrayType]

    with pytest.raises(TypeError, match="arrays are runtime data"):
        final_field_names(FinalNdarrayType)

    @dataclass(frozen=True)
    class FinalTemplate:
        x: Final[qd.Template]

    with pytest.raises(TypeError, match="is redundant"):
        final_field_names(FinalTemplate)

    @dataclass(frozen=True)
    class FinalObject:
        x: Final[object]

    with pytest.raises(TypeError, match="supports T in"):
        final_field_names(FinalObject)

    # Accepted: the scalar set plus any Enum subclass.
    class Mode(enum.IntEnum):
        A = 0
        B = 1

    @dataclass(frozen=True)
    class AllGood:
        a: Final[int]
        b: Final[float]
        c: Final[bool]
        d: Final[str]
        e: Final[Mode]
        f: int  # ordinary runtime field

    assert final_field_names(AllGood) == {"a", "b", "c", "d", "e"}


@test_utils.test()
def test_final_field_int_annotation_holding_intenum_value():
    """Genesis's static-config classes declare ``ccd_algorithm: int`` / ``integrator: int`` etc. but store ``IntEnum``
    members in them. Baking such a value must work: an ``IntEnum`` member is a valid literal, compares and does
    arithmetic as an ``int``, and ``repr``s stably for the cache key. Validation is on the declared annotation, so
    ``Final[int]`` holding an ``IntEnum`` is accepted without an ``isinstance`` check on the hot path."""
    import enum
    from typing import Final

    class CcdAlgorithm(enum.IntEnum):
        MPR = 0
        MJ_MPR = 1
        GJK = 2

    @dataclass(frozen=True)
    class StaticConfig:
        ccd_algorithm: Final[int]
        enable_collision: Final[bool]

    @qd.kernel
    def run(cfg: StaticConfig, out: qd.types.NDArray[qd.i32, 1]):
        algo = qd.static(cfg.ccd_algorithm)
        for i in out:
            if qd.static(cfg.enable_collision):
                out[i] = algo
            else:
                out[i] = -1

    out = qd.ndarray(qd.i32, shape=(2,))
    run(StaticConfig(ccd_algorithm=CcdAlgorithm.GJK, enable_collision=True), out)
    assert out[0] == 2 and out[1] == 2
    run(StaticConfig(ccd_algorithm=CcdAlgorithm.MPR, enable_collision=False), out)
    assert out[0] == -1 and out[1] == -1


@test_utils.test()
def test_bare_final_annotation_is_rejected():
    """A bare ``Final`` (no ``[T]``) must be rejected up front with a message naming the fix.

    ``typing.get_origin(typing.Final)`` is ``None``, so the bare spelling does not look like a Final annotation and
    would otherwise be treated as an ordinary runtime field - reaching ``decl_scalar_arg`` and dying in ``cook_dtype``
    with ``ValueError: Invalid data type typing.Final``. That is precisely the confusing failure this feature removes
    for ``Final[T]``, so it must not be how the unsupported spelling behaves."""
    import typing

    from quadrants.lang._final_dataclass_fields import (
        final_field_names,
        is_final_annotation,
    )

    # The bare form genuinely is not a subscripted Final...
    assert is_final_annotation(typing.Final) is False
    assert is_final_annotation(typing.Final[int]) is True

    # ...but it is still rejected rather than silently lowered to a runtime arg.
    @dataclass(frozen=True)
    class BareFinal:
        x: typing.Final

    with pytest.raises(TypeError, match="bare ``typing.Final`` is not supported"):
        final_field_names(BareFinal)

    # And end-to-end, so the user sees that error instead of a cook_dtype failure.
    @qd.kernel
    def k(cfg: BareFinal, out: qd.types.NDArray[qd.i32, 1]):
        for i in out:
            out[i] = 1

    with pytest.raises(Exception, match="bare ``typing.Final`` is not supported"):
        k(BareFinal(x=5), qd.ndarray(qd.i32, shape=(2,)))


@test_utils.test()
def test_final_float_field_honors_raise_on_templated_floats():
    """``raise_on_templated_floats`` exists to stop float values from driving kernel specialisation, since each
    distinct value compiles another kernel. A ``Final[float]`` field does exactly that, so it must honour the setting
    the same way a ``qd.template()`` float does.

    Note this is deliberately *stricter* than the ``@qd.data_oriented`` pattern ``Final`` replaces: a float member of
    a data_oriented template arg currently bypasses the guard, because ``_extract_arg``'s data_oriented branch returns
    ``weakref.ref(arg)`` before reaching the float check. That gap is pre-existing and out of scope here; the point of
    this test is that the new annotation does not inherit it."""
    from typing import Final

    arch_name = qd.lang.impl.current_cfg().arch.name

    @dataclass(frozen=True)
    class Cfg:
        dt: Final[float]
        n: Final[int]

    def run(cfg):
        @qd.kernel
        def k(config: Cfg, out: qd.types.NDArray[qd.i32, 1]):
            v = qd.static(config.n)
            for i in out:
                out[i] = v

        k(cfg, qd.ndarray(qd.i32, shape=(2,)))

    qd.init(arch=getattr(qd, arch_name), raise_on_templated_floats=True)
    with pytest.raises(ValueError, match="Floats not allowed as templated types"):
        run(Cfg(dt=0.5, n=3))


@test_utils.test()
def test_final_numpy_float_field_honors_raise_on_templated_floats():
    """A ``Final[float]`` field can be launched with a NumPy floating scalar or a ``float`` subclass, both of which
    ``final_scalar_key`` specialises on exactly like a builtin ``float``. So ``raise_on_templated_floats`` must
    reject those too, not only a builtin one - otherwise the option's guarantee (no float value drives kernel
    specialisation) is bypassed."""
    from typing import Final

    import numpy as np

    arch_name = qd.lang.impl.current_cfg().arch.name

    class Meters(float):
        pass

    @dataclass(frozen=True)
    class Cfg:
        dt: Final[float]
        n: Final[int]

    def run(cfg):
        @qd.kernel
        def k(config: Cfg, out: qd.types.NDArray[qd.i32, 1]):
            v = qd.static(config.n)
            for i in out:
                out[i] = v

        k(cfg, qd.ndarray(qd.i32, shape=(2,)))

    qd.init(arch=getattr(qd, arch_name), raise_on_templated_floats=True)
    for bad_value in (np.float32(0.5), Meters(0.5)):
        with pytest.raises(ValueError, match="Floats not allowed as templated types"):
            run(Cfg(dt=bad_value, n=3))

    # Default setting: the same config compiles fine, and the Final int still specialises.
    qd.init(arch=getattr(qd, arch_name))
    run(Cfg(dt=0.5, n=3))


@test_utils.test()
def test_final_float_cached_spec_key_revalidated_after_reinit():
    """A frozen config whose subtree bakes a ``Final`` value is never spec-key-cached on its instance (see
    ``subtree_has_final_fields``): it recomputes each launch so validation-sensitive guards re-run. Here the
    ``Final[float]`` guard must fire when the same instance is reused after re-initialising with
    ``raise_on_templated_floats`` on - a stale cached key would otherwise silently keep specialising on the float."""
    from typing import Final

    arch_name = qd.lang.impl.current_cfg().arch.name

    @dataclass(frozen=True)
    class Cfg:
        dt: Final[float]
        n: Final[int]

    def run(cfg):
        @qd.kernel
        def k(config: Cfg, out: qd.types.NDArray[qd.i32, 1]):
            v = qd.static(config.n)
            for i in out:
                out[i] = v

        k(cfg, qd.ndarray(qd.i32, shape=(2,)))

    cfg = Cfg(dt=0.5, n=3)

    # First launch with the option OFF: compiles fine. A Final-bearing config is deliberately never cached, so the
    # instance carries no ``_qd_spec_key`` to short-circuit (and thus bypass a guard) on later launches.
    qd.init(arch=getattr(qd, arch_name))
    run(cfg)
    assert not hasattr(cfg, "_qd_spec_key")  # Final-bearing config is never cached; every launch recomputes+revalidates

    # Re-initialise with the option ON and reuse the SAME instance: the recompute must re-run the guard.
    qd.init(arch=getattr(qd, arch_name), raise_on_templated_floats=True)
    with pytest.raises(ValueError, match="Floats not allowed as templated types"):
        run(cfg)

    # Turning the option back off serves the still-valid cached key again (the key value was never setting-dependent).
    qd.init(arch=getattr(qd, arch_name))
    run(cfg)


@test_utils.test()
def test_final_unsafe_hash_ordinary_field_reread_each_launch():
    """A ``Final``-bearing ``unsafe_hash=True`` config is re-read on every launch: its per-instance caches (spec key,
    offline repr, mapper, *and* the frozen unwrapped-value cache ``_qd_dc_unwrapped``) are all disabled, so mutating an
    ordinary (runtime) field between launches sends the new value rather than a stale first-launch snapshot."""
    from typing import Final

    @dataclass(unsafe_hash=True)
    class Cfg:
        scale: Final[int]  # baked compile-time constant
        n: int  # ordinary runtime field, mutable via unsafe_hash

    @qd.kernel
    def k(config: Cfg, out: qd.types.NDArray[qd.i32, 1]):
        s = qd.static(config.scale)
        for i in out:
            out[i] = config.n * s

    cfg = Cfg(scale=10, n=3)
    out = qd.ndarray(qd.i32, shape=(2,))
    k(cfg, out)
    assert out.to_numpy()[0] == 30  # 3 * 10

    cfg.n = 7  # mutate an ordinary field on the same instance (same Final -> same specialization)
    k(cfg, out)
    assert out.to_numpy()[0] == 70  # 7 * 10 - the new value, not a stale 30 from a cached unwrap
    assert not hasattr(cfg, "_qd_dc_unwrapped")  # Final-bearing configs never cache unwrapped field values


@test_utils.test()
def test_final_field_string_annotation_is_rejected():
    """``from __future__ import annotations`` (or any explicit string annotation) leaves ``field.type`` as an
    unresolved string, so Quadrants cannot see the ``Final`` and would silently lower the field as a *runtime* kernel
    argument - a field the user believes is a compile-time constant. Rather than half-support it, raise."""
    from quadrants.lang._final_dataclass_fields import final_field_names

    @dataclass(frozen=True)
    class StringAnnotated:
        x: "Final[int]"

    with pytest.raises(TypeError, match="unresolved string"):
        final_field_names(StringAnnotated)

    # A non-Final string annotation stays on the pre-existing path (unchanged behavior, no error from us).
    @dataclass(frozen=True)
    class PlainStringAnnotated:
        x: "int"

    assert final_field_names(PlainStringAnnotated) == frozenset()


@test_utils.test()
def test_final_field_aliased_string_annotation_is_rejected():
    """A substring test for the literal name ``Final`` misses an *aliased* spelling: with
    ``from __future__ import annotations`` and ``from typing import Final as F``, ``x: F[int]`` is stored as the string
    ``"F[int]"``. ``final_field_names`` resolves the class's hints (which sees the alias in the module globals) so the
    field is recognised as ``Final`` and rejected, rather than silently lowered as an ordinary runtime argument."""
    import sys
    import types
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    # A real module whose globals carry the ``Final as F`` alias, so ``typing.get_type_hints`` can resolve ``"F[int]"``
    # (a function-local class would not expose the alias to the resolver).
    mod = types.ModuleType("_qd_final_alias_test_mod")
    mod.F = Final
    mod.dc = dataclasses
    sys.modules[mod.__name__] = mod
    try:
        exec("@dc.dataclass(frozen=True)\nclass Cfg:\n    x: 'F[int]'\n", mod.__dict__)
        with pytest.raises(TypeError, match="unresolved string"):
            final_field_names(mod.Cfg)
    finally:
        sys.modules.pop(mod.__name__, None)


@test_utils.test()
def test_final_aliased_string_detected_despite_unresolvable_sibling():
    """Whole-class ``typing.get_type_hints`` is all-or-nothing: an unrelated field whose annotation cannot be resolved
    (e.g. a ``TYPE_CHECKING``-only import) makes it fail for the *entire* class, so the fallback substring test would
    then miss an aliased ``Final`` on a sibling and silently lower it as a runtime arg. Per-field resolution must still
    catch it."""
    import sys
    import types
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    mod = types.ModuleType("_qd_final_alias_unresolvable_mod")
    mod.F = Final
    mod.dc = dataclasses
    sys.modules[mod.__name__] = mod
    try:
        # ``y``'s annotation names a type never defined in the module, so class-wide ``get_type_hints`` raises; the
        # aliased ``Final`` on ``x`` must still be seen.
        exec(
            "@dc.dataclass(frozen=True)\nclass Cfg:\n    x: 'F[int]'\n    y: 'OnlyUnderTypeChecking'\n",
            mod.__dict__,
        )
        with pytest.raises(TypeError, match="unresolved string"):
            final_field_names(mod.Cfg)
    finally:
        sys.modules.pop(mod.__name__, None)


@test_utils.test()
def test_final_scalar_key_distinguishes_signed_zero_and_nan_payloads():
    """``final_scalar_key`` encodes floats by their IEEE-754 bits so values that Python conflates stay distinct
    in both the in-process spec key and the on-disk fastcache key: ``-0.0`` vs ``0.0`` (equal under ``==`` with
    equal hashes) and NaNs differing only in sign/payload (all rendered ``"nan"`` by ``str``). Non-float values
    pass through unchanged."""
    import math
    import struct

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    # Signed zero: Python conflates these; the bit encoding must not.
    assert 0.0 == -0.0 and hash(0.0) == hash(-0.0)
    assert final_scalar_key(0.0) != final_scalar_key(-0.0)

    # NaN sign/payload: ``str`` renders them all ``"nan"``; the bit encoding keeps them distinct.
    payload_nan = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000ABC))[0]
    nan_keys = {final_scalar_key(math.nan), final_scalar_key(-math.nan), final_scalar_key(payload_nan)}
    assert len(nan_keys) == 3, "distinct NaN bit patterns must produce distinct keys"

    # Equal floats produce equal keys, so kernel reuse stays correct.
    assert final_scalar_key(1.5) == final_scalar_key(1.5)

    # A ``Final[float]`` field may legally receive an ``int`` at runtime, so the encoded float is tagged: its bits
    # must not collide with a bare int equal to those bits (both bake different constants).
    one_as_bits = struct.unpack("<Q", struct.pack("<d", 1.0))[0]  # == 4607182418800017408
    assert final_scalar_key(1.0) != final_scalar_key(one_as_bits)

    # NumPy scalars hit the same conflations and may legally land in a ``Final[float]`` field, so they are encoded
    # too: signed zero stays distinct and different float widths must not alias.
    import numpy as np

    assert final_scalar_key(np.float32(0.0)) != final_scalar_key(np.float32(-0.0))
    assert final_scalar_key(np.float64(0.0)) != final_scalar_key(np.float64(-0.0))
    assert final_scalar_key(np.float32(1.0)) != final_scalar_key(np.float64(1.0))

    # A *subclass* of a NumPy float takes the ``np.floating`` branch (which precedes the generic one), so it must
    # run the same state/behavior rejection there: an exact NumPy scalar is a pure value, but a subclass carrying
    # per-instance state (or class-level behavior) is not captured by dtype+bytes and must be rejected.
    try:

        class TaggedNPFloat(np.float64):
            def __new__(cls, v, unit):
                obj = super().__new__(cls, v)
                obj.unit = unit
                return obj

    except TypeError:
        pass  # some NumPy builds forbid subclassing this scalar; nothing to test then
    else:
        with pytest.raises(TypeError, match="extra per-instance state"):
            final_scalar_key(TaggedNPFloat(1.0, "m"))

    # ``IntEnum`` / ``StrEnum`` members are ``==`` (with equal hashes) to their bare scalar value and to same-valued
    # members of other enum classes; keying on class + member identity must keep all of these distinct.
    import enum

    class ModeA(enum.IntEnum):
        X = 0

    class ModeB(enum.IntEnum):
        Y = 0

    assert final_scalar_key(ModeA.X) != final_scalar_key(0)  # enum vs bare int
    assert final_scalar_key(ModeA.X) != final_scalar_key(ModeB.Y)  # same value, different enum class
    assert final_scalar_key(ModeA.X) == final_scalar_key(ModeA.X)  # same member is stable

    # Unnamed ``IntFlag`` composites have ``name is None``, so distinct bitmasks must be kept apart by their value.
    class Perm(enum.IntFlag):
        R = 1
        W = 2
        X = 4

    assert final_scalar_key(Perm.R | Perm.W) != final_scalar_key(Perm.R | Perm.X)  # values 3 vs 5 (name may be None)
    assert final_scalar_key(Perm.R | Perm.W) == final_scalar_key(Perm.R | Perm.W)  # stable

    # Inverting a flag member caches the value-derived ``_inverted_`` on the member (CPython >=3.11). That is enum
    # bookkeeping, not user state, so a member that has been inverted anywhere must still be accepted and key the
    # same as before the inversion.
    key_before = final_scalar_key(Perm.R)
    _ = ~Perm.R  # populates ``Perm.R._inverted_`` on 3.11+
    assert final_scalar_key(Perm.R) == key_before  # not rejected, and the cache does not perturb the key

    # A ``float`` subclass must be bit-encoded (signed zero stays distinct) yet tagged with its own type.
    class Meters(float):
        pass

    assert final_scalar_key(Meters(0.0)) != final_scalar_key(Meters(-0.0))  # signed zero within the subclass
    assert final_scalar_key(Meters(1.0)) != final_scalar_key(1.0)  # subclass not confused with a plain float

    # But a primitive subclass carrying extra observable per-instance state cannot be captured by value alone, so it
    # is rejected rather than silently sharing a specialization with a different-state instance of equal value.
    class TaggedFloat(float):
        def __new__(cls, v, unit):
            obj = super().__new__(cls, v)
            obj.unit = unit
            return obj

    class TaggedInt(int):
        def __new__(cls, v, unit):
            obj = super().__new__(cls, v)
            obj.unit = unit
            return obj

    with pytest.raises(TypeError, match="extra per-instance state"):
        final_scalar_key(TaggedFloat(1.0, "m"))
    with pytest.raises(TypeError, match="extra per-instance state"):
        final_scalar_key(TaggedInt(1, "m"))

    # A subclass need not carry *instance* state to be unkeyable: ``module``/``qualname`` cannot tell apart two
    # distinct classes a factory builds under the same name, so class-level behavior/state (a property/method/class
    # var a kernel could read) is rejected too - else ``UnitFloat("m")(1.0)`` and ``UnitFloat("ft")(1.0)`` would
    # share a specialization while ``qd.static(cfg.x.unit == "m")`` observes different results.
    def _unit_float_cls(unit):
        class UnitFloat(float):
            @property
            def unit(self):
                return unit

        return UnitFloat

    cls_m, cls_ft = _unit_float_cls("m"), _unit_float_cls("ft")
    assert type(cls_m(1.0)).__qualname__ == type(cls_ft(1.0)).__qualname__  # same qualname, distinct classes
    with pytest.raises(TypeError, match="observable class-level behavior/state"):
        final_scalar_key(cls_m(1.0))

    class ScaledFloat(float):  # a class variable is observable class-level state as well
        scale = 2

    with pytest.raises(TypeError, match="observable class-level behavior/state"):
        final_scalar_key(ScaledFloat(1.0))

    # Enum members follow the same rule: user-defined per-member state (attributes set in ``__init__``) is rejected,
    # while plain members and unnamed ``IntFlag`` composites (name/value bookkeeping only) are accepted above.
    class StatefulMode(enum.Enum):
        A = (1, "m")

        def __init__(self, code, unit):
            self.code = code
            self.unit = unit

    with pytest.raises(TypeError, match="user-defined per-member state"):
        final_scalar_key(StatefulMode.A)

    # Per-member state stored in a ``__slots__`` slot never appears in ``__dict__``, so the rejection must inspect
    # populated slots too (otherwise the slot stays observable by kernel code while the Final key ignores it).
    class SlottedMode(enum.Enum):
        __slots__ = ("unit",)
        A = 1

        def __init__(self, _v):
            self.unit = "m"

    assert not {k for k in vars(SlottedMode.A) if not (k.startswith("__") and k.endswith("__"))} - {
        "_name_",
        "_value_",
        "_sort_order_",
    }  # __dict__ carries only enum bookkeeping; the state lives in the ``unit`` slot
    assert SlottedMode.A.unit == "m"
    with pytest.raises(TypeError, match="user-defined per-member state"):
        final_scalar_key(SlottedMode.A)

    # State stashed under a *dunder-looking* name is still observable (``cfg.mode.__unit__``), so the allowlist is
    # exact rather than "skip every dunder" - only the known enum bookkeeping dunders (e.g. ``__objclass__``) pass.
    class DunderStateMode(enum.Enum):
        A = 1

        def __init__(self, _v):
            self.__unit__ = "m"

    assert DunderStateMode.A.__unit__ == "m"  # a genuine dunder-named per-member attribute
    with pytest.raises(TypeError, match="user-defined per-member state"):
        final_scalar_key(DunderStateMode.A)

    # Class-level behavior on the enum class is unkeyable, like for primitive subclasses: a factory can build
    # same-named enum classes whose ``label`` property closes over different strings, colliding while
    # ``qd.static(cfg.mode.label == "x")`` differs. A plain member is accepted; one on a class with a user
    # property/method/class var is rejected.
    def _labeled_enum(label):
        class Local(enum.IntEnum):
            A = 1

            @property
            def label(self):
                return label

        return Local

    lab_x, lab_y = _labeled_enum("x"), _labeled_enum("y")
    assert type(lab_x.A).__qualname__ == type(lab_y.A).__qualname__  # same qualname, distinct classes/behavior
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(lab_x.A)

    # An overridden operator dunder on the enum class is observable behavior too (``qd.static(cfg.mode == 1)``),
    # and the enum machinery never injects e.g. ``__eq__`` into a user enum's own class dict, so it is rejected -
    # unlike the structural ``__new__`` / ``__doc__`` the machinery does inject (which must stay accepted).
    class EqEnum(enum.IntEnum):
        A = 1

        def __eq__(self, other):
            return True

        def __hash__(self):
            return 0

    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(EqEnum.A)

    # An enum can also inherit observable behavior/state from a *non-enum* mixin (``class Mode(Labels, enum.Enum)``).
    # ``cfg.mode.label`` is then observable at compile time but absent from the key (which records only
    # class/name/value), so mutating ``Labels.label`` after compilation would reuse a stale kernel - and two
    # same-named factory mixins could define ``label`` differently. The class-behavior scan must inspect user mixins,
    # not only ``Enum`` bases, so such an enum is rejected while a plain enum (with only a builtin data-type mixin like
    # ``int``) is still accepted.
    class Labels:
        label = "x"

    class MixinMode(Labels, enum.Enum):
        A = 1
        B = 2

    assert MixinMode.A.label == "x"  # observable via cfg.mode.label, not captured by the key
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(MixinMode.A)

    # A user-defined enum *sunder hook* (``_missing_`` on any version; ``_repr_html_`` on 3.13+) is observable
    # behavior on the baked member, but a blanket "skip every ``_x_`` name" would mistake it for enum bookkeeping.
    # Only machinery-generated names are exempt (``_ENUM_GENERATED_CLASS_ATTRS``, computed from the running Python),
    # so a user hook is rejected while a plain enum - whose only ``_x_`` names are machinery bookkeeping - is accepted.
    class MissingHook(enum.Enum):
        A = 1
        B = 2

        @classmethod
        def _missing_(cls, value):
            return cls.A

    assert "_missing_" in vars(MissingHook)  # the user hook lands in the class's own dict, not merely inherited
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(MissingHook.A)

    # Enum classes rebuilt by a local factory share module+qualname and can have same-named members with different
    # values; the key includes the value so those stay distinct.
    def _make_int_enum(v):
        class Local(enum.IntEnum):
            A = v

        return Local

    e1, e2 = _make_int_enum(1), _make_int_enum(2)
    assert type(e1.A).__qualname__ == type(e2.A).__qualname__  # identical qualname across the two factory enums
    assert final_scalar_key(e1.A) != final_scalar_key(e2.A)  # but different values -> distinct keys

    # The member value is itself routed through ``final_scalar_key``, so two factory members named ``A`` whose raw
    # values are ``True`` vs ``1`` (``==`` with equal hashes) stay distinct rather than collapsing to one key.
    def _make_enum(v):
        class Local(enum.Enum):
            A = v

        return Local

    et, eo = _make_enum(True), _make_enum(1)
    assert type(et.A).__qualname__ == type(eo.A).__qualname__
    assert et.A.value == eo.A.value and hash(et.A.value) == hash(eo.A.value)  # True == 1, equal hashes
    assert final_scalar_key(et.A) != final_scalar_key(eo.A)

    # Even with identical name AND value, two dynamically recreated (behavior-free) classes have genuinely distinct
    # members - for a plain ``Enum``, ``First.A != Second.A`` - so a kernel branching on ``cfg.mode == First.A``
    # needs distinct specializations. ``module``/``qualname``/name/value are all identical, so the key adds a per-class
    # identity token (a ``_ClassRef`` in-process, a non-recyclable ``_dynamic_class_serial`` offline) for such
    # non-uniquely-identifiable classes to keep them apart.
    first, second = _make_enum(1), _make_enum(1)
    assert type(first.A).__qualname__ == type(second.A).__qualname__ and first.A.value == second.A.value
    assert first.A != second.A  # plain Enum uses identity equality: distinct class objects -> distinct members
    assert final_scalar_key(first.A) != final_scalar_key(second.A)  # ...kept distinct via the per-class identity token
    assert final_scalar_key(first.A) == final_scalar_key(first.A)  # stable for the same class

    # A *behavior-free* primitive subclass is accepted and keyed by its true value via the base slot, staying
    # distinct from a plain ``int`` and producing a faithful, process-stable offline (string) key.
    class Grams(int):
        pass

    assert final_scalar_key(Grams(1)) != final_scalar_key(Grams(2))
    assert final_scalar_key(Grams(1)) != final_scalar_key(1)  # subclass distinct from a plain int
    assert str(final_scalar_key(Grams(1))) != str(final_scalar_key(Grams(2)))  # faithful offline string

    # Like recreated enums, two behavior-free primitive subclasses built by a local factory share module+qualname
    # and value yet are distinct class objects (``cfg.x.__class__ is First`` is observable at compile time), so the
    # key adds a per-class identity token for such non-uniquely-identifiable classes to keep an instance of the second
    # from reusing the first's specialization. This applies to ``int``/``str`` subclasses (generic branch) and
    # ``float`` subclasses (dedicated branch) alike.
    def _make_int_subclass():
        class Local(int):
            pass

        return Local

    ci1, ci2 = _make_int_subclass(), _make_int_subclass()
    assert ci1 is not ci2 and ci1.__qualname__ == ci2.__qualname__  # distinct classes, identical qualname
    assert final_scalar_key(ci1(1)) != final_scalar_key(ci2(1))  # kept distinct via the per-class identity token
    assert final_scalar_key(ci1(1)) == final_scalar_key(ci1(1))  # stable for the same class

    def _make_float_subclass():
        class Local(float):
            pass

        return Local

    cf1, cf2 = _make_float_subclass(), _make_float_subclass()
    assert cf1 is not cf2 and cf1.__qualname__ == cf2.__qualname__
    assert final_scalar_key(cf1(1.0)) != final_scalar_key(cf2(1.0))  # kept distinct via the per-class identity token

    # But a subclass that overrides an observable dunder (repr / a conversion / an operator) is rejected: two
    # same-named factory subclasses could override it differently, sharing ``module``/``qualname`` while
    # a kernel observes the difference via ``repr(cfg.x)`` / ``int(cfg.x)`` / ``cfg.x == 1``.
    class OddRepr(int):
        def __repr__(self):
            return "odd"

    class ConstInt(int):
        def __int__(self):
            return 0

        def __index__(self):
            return 0

    class WeirdEq(int):
        def __eq__(self, other):
            return True

        def __hash__(self):
            return 0

    for bad in (OddRepr(1), ConstInt(1), WeirdEq(1)):
        with pytest.raises(TypeError, match="observable class-level behavior/state"):
            final_scalar_key(bad)

    # Remaining scalars are type-tagged: Python conflates value-equal but distinct-typed constants (True == 1 ==
    # np.int64(1), with equal hashes), and they bake observably different Python constants, so they must not alias.
    assert final_scalar_key(True) != final_scalar_key(1)
    assert final_scalar_key(1) != final_scalar_key(np.int64(1))
    assert final_scalar_key(7) != final_scalar_key("7")
    # ...but equal values of the same type stay equal, so legitimate kernel reuse is preserved.
    assert final_scalar_key(7) == final_scalar_key(7)
    assert final_scalar_key("abc") == final_scalar_key("abc")
    assert final_scalar_key(True) == final_scalar_key(True)

    # Annotations are not enforced at runtime: an arbitrary object (whose ``__eq__``/``__hash__`` we cannot trust to
    # capture all observable state) is rejected rather than keyed by identity/value.
    class Arbitrary:
        def __init__(self, tag):
            self.tag = tag

        def __eq__(self, other):
            return isinstance(other, Arbitrary)

        def __hash__(self):
            return 0

    with pytest.raises(TypeError, match="not a supported compile-time constant"):
        final_scalar_key(Arbitrary("a"))

    # An enum whose member value is mutable/unsupported (e.g. a ``list``) is rejected too, since the value is routed
    # through ``final_scalar_key`` and a mutable value could silently change under the cached spec key.
    class ListValued(enum.Enum):
        A = [1, 2, 3]

    with pytest.raises(TypeError, match="not a supported compile-time constant"):
        final_scalar_key(ListValued.A)


@test_utils.test()
def test_final_scalar_key_live_preserves_class_identity_across_module_rebind():
    """The in-process spec key (``live=True``) must distinguish two *distinct* class objects even when they
    transiently share ``module``/``qualname`` - e.g. a module-level enum or primitive subclass redefined by a module
    reload, each resolvable to its own definition while bound. In the offline (``live=False``) form
    ``final_scalar_key`` keys a resolvable class by ``module``/``qualname`` alone (process-stable, so it cannot tell
    the rebinding apart - a safe cross-process strategy, never a wrong reuse), but the in-process key keys on
    ``id(cls)`` so a second launch with the fresh class does not reuse the kernel baked for the first
    (``cfg.x.__class__ is SavedOldClass`` is observable at compile time)."""
    import sys
    import types

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    mod_name = "qd_reload_identity_probe"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    try:
        # Bind a class so it is uniquely resolvable via ``module``/``qualname`` (offline id component is None).
        old = type("Foo", (int,), {"__module__": mod_name, "__qualname__": "Foo"})
        mod.Foo = old
        old_offline = final_scalar_key(old(1))
        old_live = final_scalar_key(old(1), live=True)

        # "Reload": rebind the same name to a fresh, equally resolvable class object.
        new = type("Foo", (int,), {"__module__": mod_name, "__qualname__": "Foo"})
        mod.Foo = new
        new_offline = final_scalar_key(new(1))
        new_live = final_scalar_key(new(1), live=True)

        assert old is not new
        # Offline: both resolvable while bound, so both carry a None id component and collide. That is the safe,
        # process-stable strategy - it only fails to distinguish a (rare) reload, and never causes a wrong reuse.
        assert old_offline == new_offline
        # In-process: ``id(cls)`` keeps the two distinct classes apart.
        assert old_live != new_live
        # Same class -> stable in-process key across launches, so legitimate kernel reuse is preserved.
        assert final_scalar_key(old(1), live=True) == old_live
    finally:
        del sys.modules[mod_name]


@test_utils.test()
def test_final_scalar_key_live_uses_object_identity_not_class_equality():
    """The class-identity key component (``_subclass_identity``/``_ClassRef``) keys by *object identity*, not the class
    object's ``==``, so a metaclass that makes two distinct classes ``==`` with equal hashes cannot collapse their
    keys - ``cfg.x.__class__ is First`` is observable, so distinct classes must key apart, and the component also
    retains the class as a strong ref (pinning its ``id``). This identity behavior is defense-in-depth: such a
    metaclass carries observable equality behavior, so ``final_scalar_key`` itself now *rejects* the value."""
    from quadrants.lang import _final_dataclass_fields as _fdf

    class EqMeta(type):
        # Distinct classes with the same qualname compare equal with equal hashes (a pathological metaclass).
        def __eq__(cls, other):
            return isinstance(other, EqMeta) and cls.__qualname__ == other.__qualname__

        def __hash__(cls):
            return hash(cls.__qualname__)

    def _make():
        class Local(int, metaclass=EqMeta):
            pass

        return Local

    first, second = _make(), _make()
    assert first is not second and first == second and hash(first) == hash(second)  # metaclass forces class ==

    # The identity component keys the two distinct classes apart (by ``id``/``is``), is stable/hashable for one class,
    # and retains the class object as a strong ref - all immune to the metaclass's ``==``/``hash``.
    id_first = _fdf._subclass_identity(first, live=True)
    assert id_first != _fdf._subclass_identity(second, live=True)  # not collapsed by the metaclass ``==``
    assert id_first == _fdf._subclass_identity(first, live=True)
    assert hash(id_first) == hash(_fdf._subclass_identity(first, live=True))

    def _flat(x):
        if isinstance(x, tuple):
            for e in x:
                yield from _flat(e)
        elif isinstance(x, _fdf._ClassRef):
            yield x.cls
        else:
            yield x

    assert any(e is first for e in _flat(id_first))  # class object retained (strong ref) in the identity component

    # But a metaclass ``__eq__``/``__hash__`` is observable (``qd.static(cfg.x.__class__ == Expected)``) and the key
    # cannot capture a later mutation of the state it consults, so the value itself is rejected.
    with pytest.raises(TypeError, match="metaclass defines observable"):
        _fdf.final_scalar_key(first(1))


@test_utils.test()
def test_final_scalar_key_offline_dynamic_class_is_process_unique():
    """The per-class serial that distinguishes dynamic classes offline (``_dynamic_class_serial``) is only
    process-local and restarts from zero in every process. So the *offline* (cross-process) key for a non-resolvable
    (locally/dynamically created) class also embeds a per-process nonce: its serialized string is unique to this
    process, guaranteeing a dynamic class is a cross-process cache miss (never a wrong reuse of a kernel baked for a
    distinct class in another worker). The *in-process* (``live=True``) key is process-local anyway, so it stays
    nonce-free."""
    from quadrants.lang import _final_dataclass_fields as fdf

    def _make_local_int():
        class Local(int):
            pass

        return Local

    loc = _make_local_int()
    assert fdf._PROCESS_NONCE in str(fdf.final_scalar_key(loc(1)))  # dynamic-class offline key -> process-unique
    assert fdf._PROCESS_NONCE not in str(fdf.final_scalar_key(loc(1), live=True))  # live key never carries the nonce
    # Distinct dynamic classes still separate within this process (same nonce, different serial); the class is stable.
    other = _make_local_int()
    assert fdf.final_scalar_key(loc(1)) != fdf.final_scalar_key(other(1))
    assert fdf.final_scalar_key(loc(1)) == fdf.final_scalar_key(loc(1))


@test_utils.test()
def test_final_offline_dynamic_class_serial_is_monotonic_not_recyclable():
    """The *offline* key for a locally/dynamically created class must survive ``id(cls)`` recycling. Once ``qd.reset()``
    drops the live ``_ClassRef`` that pinned such a class, the class can be collected while its on-disk fastcache
    artifact remains, and CPython can hand its freed address to the next same-qualified factory class - an ``id``-based
    offline key would then serialize identically and load the dead class's kernel, even though ``cfg.x.__class__ is
    First`` is observable at compile time. The offline component uses a monotonic, never-reused serial instead: a
    distinct class object always keys apart, the serial is stable for a class's lifetime, and a collected class frees
    its (weak) registry slot without pinning it and without ever having its serial reissued."""
    import gc
    import weakref as _weakref

    from quadrants.lang import _final_dataclass_fields as fdf

    def _make():
        class Local(int):
            pass

        return Local

    a = _make()
    sa = fdf._dynamic_class_serial(a)
    assert fdf._dynamic_class_serial(a) == sa  # stable for the class's lifetime

    b = _make()
    sb = fdf._dynamic_class_serial(b)
    assert sb > sa  # a distinct class object never reuses an earlier serial (monotonic counter)

    # A collected class frees its weak registry slot (the offline key pins nothing, unlike the live ``_ClassRef``),
    # and the counter only advances - so the next class, even were it allocated at ``a``'s recycled address, draws a
    # fresh, higher serial rather than colliding.
    ref = _weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None  # the serial registry did not pin the collected class
    c = _make()
    assert fdf._dynamic_class_serial(c) > sb  # still monotonic after a collection; never a recycled serial


@test_utils.test()
def test_final_scalar_key_live_hashable_under_unhashable_metaclass():
    """A metaclass may set ``__hash__ = None`` (making the *class* itself unhashable). The class-identity key component
    retains the class via a ``_ClassRef`` token that hashes/compares by object identity, so the whole key stays
    hashable - a mapper's ``self.mapping[key]`` would otherwise raise ``TypeError``. This is defense-in-depth: such a
    metaclass carries observable behavior, so ``final_scalar_key`` itself rejects the value."""
    from quadrants.lang import _final_dataclass_fields as _fdf

    class Unhashable(type):
        __hash__ = None  # the classes this metaclass produces are themselves unhashable

        def __eq__(cls, other):
            return cls is other

    def _make():
        class Local(int, metaclass=Unhashable):
            pass

        return Local

    a, b = _make(), _make()
    with pytest.raises(TypeError):
        hash(a)  # the class object is itself unhashable...

    # ...yet the identity component (via ``_ClassRef.object.__hash__``) stays hashable, stable, and usable as a dict
    # key, and keeps distinct classes apart.
    ka = _fdf._subclass_identity(a, live=True)
    assert hash(ka) == hash(_fdf._subclass_identity(a, live=True))
    assert {ka: 1}[_fdf._subclass_identity(a, live=True)] == 1
    assert ka != _fdf._subclass_identity(b, live=True)

    # But the value itself is rejected: the metaclass carries observable ``__hash__``/``__eq__`` behavior.
    with pytest.raises(TypeError, match="metaclass defines observable"):
        _fdf.final_scalar_key(a(1))


@test_utils.test()
def test_final_plan_cache_keyed_by_type_identity():
    """The Final-field plan/path caches key on type *identity* (``id``), not the type object, so a metaclass that
    makes two distinct dataclass types compare equal with equal hashes cannot make one reuse the other's Final schema.
    A plain ``dict[type, ...]`` would return the first type's plan for the second - baking a runtime field, or lowering
    a real ``Final`` field as an ordinary one and failing compilation, depending on lookup order."""
    import dataclasses as dcs
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    class DcMeta(type):
        def __eq__(cls, other):
            return isinstance(other, DcMeta)  # any two such classes compare equal...

        def __hash__(cls):
            return 0  # ...with equal hashes, so a plain dict would merge them

    @dcs.dataclass(frozen=True)
    class HasFinal(metaclass=DcMeta):
        x: Final[int]

    @dcs.dataclass(frozen=True)
    class NoFinal(metaclass=DcMeta):
        x: int

    assert HasFinal == NoFinal and hash(HasFinal) == hash(NoFinal)  # metaclass forces type-level equality
    # Each type must resolve to its OWN plan, regardless of caching order (identity keys keep them separate).
    assert final_field_names(HasFinal) == frozenset({"x"})
    assert final_field_names(NoFinal) == frozenset()


@test_utils.test()
def test_final_first_final_path_tracks_visited_by_identity():
    """The recursive mutable-ancestor walk tracks visited dataclass types by ``id``, not by equality. A metaclass that
    makes a nested inner type compare equal to its mutable outer type must not make the inner look "already visited"
    (which would return ``None`` early and let the mutable outer - which could be rebound, changing a baked value -
    slip past the rejection)."""
    import dataclasses as dcs
    from typing import Final

    from quadrants.lang._final_dataclass_fields import final_field_names

    class DcMeta(type):
        def __eq__(cls, other):
            return isinstance(other, DcMeta)  # any two such types compare equal (incl. inner == outer)...

        def __hash__(cls):
            return 0  # ...with equal hashes

    @dcs.dataclass(frozen=True)
    class Inner(metaclass=DcMeta):
        x: Final[int]

    @dcs.dataclass  # NOT frozen: a mutable ancestor of the Final leaf ``Inner.x``
    class Outer(metaclass=DcMeta):
        inner: Inner

    assert Inner == Outer and hash(Inner) == hash(Outer)  # metaclass makes the inner "equal" to the outer
    with pytest.raises(TypeError, match="not frozen"):
        final_field_names(Outer)  # mutable ancestor of a Final leaf must still be rejected


@test_utils.test()
def test_final_enum_rejects_observable_metaclass_state():
    """An enum whose *metaclass* (a custom ``EnumMeta`` subclass) carries observable class-level state/behavior is
    rejected: a kernel can read ``cfg.mode.__class__.label`` (which resolves to ``type(Mode).label``), and the key -
    keyed on the enum class's ``module``/``qualname``/member - does not capture it, so two same-named factory
    metaclasses (or a mutated one) would select the same specialization. A plain enum (framework metaclass) is fine."""
    import enum as en

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class LabeledMeta(en.EnumMeta):
        label = "x"  # observable via ``Mode.label`` / ``cfg.mode.__class__.label``, absent from the key

    class Mode(en.Enum, metaclass=LabeledMeta):
        A = 1
        B = 2

    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Mode.A)

    class Plain(en.Enum):  # metaclass is the framework ``EnumMeta`` -> not inspected -> accepted
        A = 1
        B = 2

    final_scalar_key(Plain.A)  # does not raise

    class Weekday(en.IntEnum):  # IntEnum still uses the framework metaclass -> accepted
        MON = 0

    final_scalar_key(Weekday.MON)  # does not raise


@test_utils.test()
def test_final_is_baked_base_type_ignores_spoofed_module():
    """``_is_baked_base_type`` identifies NumPy scalar bases by type nature (a static ``np.generic`` subclass), not by
    the mutable ``__module__`` string. A user subclass that spoofs ``__module__ = "numpy"`` is still a heap type, so it
    must not masquerade as a trusted base: its state/behavior is inspected (and rejected) and it is keyed by class
    identity like any other user subclass."""
    from quadrants.lang._final_dataclass_fields import (
        _is_baked_base_type,
        final_scalar_key,
    )

    class BehaviorSubclass(int):
        def __eq__(self, other):  # observable class-level behavior a kernel could read (``cfg.x == 1``)
            return True

        __hash__ = int.__hash__

    BehaviorSubclass.__module__ = "numpy"  # spoof - must not buy trusted-base treatment
    assert _is_baked_base_type(BehaviorSubclass) is False  # heap type, so not a baked base despite the module string
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(BehaviorSubclass(1))

    class FreeSubclass(int):  # behavior-free, also spoofing the module
        pass

    FreeSubclass.__module__ = "numpy"
    assert _is_baked_base_type(FreeSubclass) is False  # still not a base; keyed by identity in the scalar branch
    final_scalar_key(FreeSubclass(1))  # accepted (does not raise)


@test_utils.test()
def test_final_primitive_subclass_rejects_observable_metaclass_state():
    """A primitive subclass whose *metaclass* carries observable state/behavior is rejected: a kernel can read
    ``cfg.x.__class__.label`` (which resolves to ``type(cls).label``), which the subclass-MRO walk never sees and the
    key - keyed on the subclass ``module``/``qualname`` - does not capture, so mutating it (or two factory metaclasses
    differing) would reuse a stale specialization. Even a metaclass ``__eq__`` / ``__hash__`` is rejected: identity-safe
    keying stops dict collisions but the operator is still observable via ``qd.static(cfg.x.__class__ == Expected)``.
    A metaclass that is a bare ``type`` subclass (no observable attrs) stays accepted."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class UnitMeta(type):
        label = "m"  # observable via ``cfg.x.__class__.label``, absent from the key

    class Unit(int, metaclass=UnitMeta):
        pass

    with pytest.raises(TypeError, match="metaclass defines observable"):
        final_scalar_key(Unit(1))

    class Plain(int):  # plain metaclass (``type``) -> accepted
        pass

    final_scalar_key(Plain(1))  # does not raise

    class BareMeta(type):  # a custom metaclass with no observable attrs of its own -> accepted
        pass

    class Bare(int, metaclass=BareMeta):
        pass

    final_scalar_key(Bare(1))  # does not raise

    class EqMeta(type):  # metaclass equality behavior is observable (``cfg.x.__class__ == Expected``) -> rejected
        def __eq__(cls, other):
            return cls is other

        def __hash__(cls):
            return id(cls)

    class Tagged(int, metaclass=EqMeta):
        pass

    with pytest.raises(TypeError, match="metaclass defines observable"):
        final_scalar_key(Tagged(1))


@test_utils.test()
def test_final_float_signed_zero_keys_distinct_kernels():
    """``-0.0`` and ``0.0`` are equal under Python ``==``/``hash`` but name different baked constants (the sign
    bit is observable). Encoding Final floats by their IEEE bits keeps them as distinct entries in the template
    mapper spec key, so each value compiles its own kernel instead of the second launch reusing the first's."""
    from typing import Final

    @dataclass(frozen=True)
    class Cfg:
        z: Final[float]

    @qd.kernel
    def write_z(config: Cfg, out: qd.types.NDArray[qd.f32, 1]):
        v = qd.static(config.z)
        for i in out:
            out[i] = v

    out = qd.ndarray(qd.f32, shape=(1,))
    write_z(Cfg(z=0.0), out)
    assert len(write_z._primal.mapper.mapping) == 1
    write_z(Cfg(z=-0.0), out)  # equal to 0.0 under ==/hash, but a distinct baked constant
    assert len(write_z._primal.mapper.mapping) == 2, "-0.0 and 0.0 must not share a compiled kernel"


@test_utils.test()
def test_final_key_field_name_does_not_shadow_internal_spec_key_cache():
    """A frozen dataclass may legitimately declare a field named ``_key``. Any internal per-instance attribute lives in
    the reserved ``_qd_`` namespace (here ``_qd_spec_key``) precisely so a user ``_key`` field can never be mistaken for
    it. This config also bakes a ``Final`` value, so it recomputes its spec key each launch (never caching) - the
    distinct ``Final`` values must therefore drive distinct kernels regardless of the equal user ``_key`` fields."""
    from typing import Final

    @dataclass(frozen=True)
    class Config:
        _key: int  # user field whose name used to collide with the internal cache attribute
        value: Final[int]

    @qd.kernel
    def bump(config: Config, out: qd.types.NDArray[qd.i32, 1]):
        v = qd.static(config.value)
        for i in out:
            out[i] = v

    out = qd.ndarray(qd.i32, shape=(1,))
    bump(Config(_key=0, value=1), out)
    assert out[0] == 1
    assert len(bump._primal.mapper.mapping) == 1
    bump(Config(_key=0, value=2), out)  # same user _key, different Final value
    assert out[0] == 2, "distinct Final values must not share a compiled kernel despite equal user ``_key`` fields"
    assert len(bump._primal.mapper.mapping) == 2


@test_utils.test()
def test_final_subtree_has_final_fields_predicate():
    """``subtree_has_final_fields`` gates the per-instance spec-key / offline-repr caches. It must report a ``Final``
    field anywhere in the *transitive* dataclass subtree, so a Final-free dataclass keeps caching (untouched hot path)
    while any Final-bearing one - even one reaching a ``Final`` leaf only through a nested dataclass - is forced to
    recompute and revalidate each launch."""
    from typing import Final

    from quadrants.lang._final_dataclass_fields import subtree_has_final_fields as shf

    @dataclass(frozen=True)
    class PlainLeaf:
        a: int

    @dataclass(frozen=True)
    class DirectFinal:
        x: Final[int]

    @dataclass(frozen=True)
    class Inner:
        n: Final[int]

    @dataclass(frozen=True)
    class OuterViaInner:  # no own Final field; reaches one only through the nested ``Inner``
        child: Inner

    @dataclass(frozen=True)
    class OuterPlain:  # no Final anywhere in the subtree
        leaf: PlainLeaf

    assert not shf(PlainLeaf)
    assert shf(DirectFinal)
    assert shf(Inner)
    assert shf(OuterViaInner), "a Final leaf nested under a Final-free ancestor must still disable the ancestor's cache"
    assert not shf(OuterPlain)


@test_utils.test()
def test_final_offline_repr_not_cached_on_final_bearing_config():
    """The offline fastcache repr is cached on a frozen instance as ``_qd_dc_repr`` - but never on a Final-bearing one.
    A cached repr could not notice a ``Final`` value's class turning behaviorful between launches (e.g. an enum whose
    ``__eq__`` is monkey-patched), so such a config recomputes each launch to re-run ``final_scalar_key`` validation.
    A Final-free frozen dataclass still caches, keeping the offline hot path intact."""
    from typing import Final

    from quadrants.lang._fast_caching.args_hasher import dataclass_to_repr

    @dataclass(frozen=True)
    class PlainCfg:
        a: int

    @dataclass(frozen=True)
    class FinalCfg:
        x: Final[int]

    plain = PlainCfg(a=1)
    dataclass_to_repr(False, (), plain)
    assert hasattr(plain, "_qd_dc_repr")  # Final-free frozen dataclass caches its repr (result or the NONE sentinel)

    fcfg = FinalCfg(x=1)
    dataclass_to_repr(False, (), fcfg)
    assert not hasattr(fcfg, "_qd_dc_repr")  # Final-bearing config is never repr-cached; it recomputes+revalidates


@test_utils.test()
def test_final_str_field_does_not_disable_offline_fastcache():
    """A ``Final[str]`` field must not disable the *offline* fastcache. ``stringify_obj_type`` has no case for a bare
    ``str`` (it fails and logs an UNKNOWN_TYPE warning), so routing the Final field's value through it would make
    ``dataclass_to_repr`` fail and force a recompile in every process for an explicitly supported field type. Final
    fields are serialized directly via ``final_scalar_key``, which yields a non-None, value-distinguishing repr; the
    neighbouring non-Final field confirms the ordinary type-only path still works alongside it."""
    from typing import Final

    from quadrants.lang._fast_caching.args_hasher import dataclass_to_repr

    @dataclass(frozen=True)
    class Cfg:
        name: Final[str]
        scale: int

    r_a = dataclass_to_repr(False, (), Cfg(name="a", scale=1))
    r_a2 = dataclass_to_repr(False, (), Cfg(name="a", scale=1))
    r_b = dataclass_to_repr(False, (), Cfg(name="b", scale=1))
    assert r_a is not None, "a Final[str] field must not disable the offline fastcache"
    assert r_a == r_a2, "equal Final[str] values must produce equal offline keys"
    assert r_a != r_b, "distinct Final[str] values must produce distinct offline keys"


@test_utils.test()
def test_subclass_extra_field_offline_fastcache_uses_annotated_field_set():
    """A subclass passed where a *base* dataclass is annotated must fast-cache against the base's field set. Without an
    ``annotated_type``, ``dataclass_to_repr`` hashes the runtime type, so a subclass adding a non-fastcacheable field
    (``list``) fails and disables the offline cache. Passing the base restricts hashing to its fields: the subclass
    then hashes identically to a plain base instance, and the extra field cannot affect the key."""
    from typing import Final

    from quadrants.lang._fast_caching.args_hasher import (
        _FAIL_FASTCACHE,
        dataclass_to_repr,
    )

    @dataclass(frozen=True)
    class Base:
        scale: Final[int]  # baked -> its value drives the offline key

    @dataclass(frozen=True)
    class Sub(Base):
        name: list  # extra application-only field; not fastcacheable on its own and never seen by the kernel

    base_repr = dataclass_to_repr(False, (), Base(scale=1))
    assert base_repr is not None and "scale" in base_repr

    # Runtime-type hashing (no annotation) trips on the ``list`` field and disables the offline cache.
    assert dataclass_to_repr(False, (), Sub(scale=1, name=["a"])) is _FAIL_FASTCACHE

    # Hashing against the annotated base drops the extra field, so the subclass hashes identically to its base.
    assert dataclass_to_repr(False, (), Sub(scale=1, name=["a"]), Base) == base_repr
    assert (
        dataclass_to_repr(False, (), Sub(scale=1, name=["b"]), Base) == base_repr
    ), "the extra (non-annotated) field must not affect the offline key"
    # The base's own field still drives the key: a different ``Final`` value splits it.
    assert (
        dataclass_to_repr(False, (), Sub(scale=2, name=["a"]), Base) != base_repr
    ), "a baked base field must still split the offline key by value"


@test_utils.test()
def test_final_exact_baked_type_membership_is_identity_not_equality():
    """A ``Final`` scalar recognises an *exact* builtin (``bool``/``int``/``float``/``str``) by class *identity*, never
    ``==``. A subclass whose metaclass makes the class compare equal to a builtin (``X == int``) must not be mistaken
    for an exact builtin: equality-based membership would skip subclass/metaclass validation and canonicalize ``X`` as
    a builtin, collapsing two distinct same-qualified factory classes onto one live key while ``cfg.x.__class__ is
    First`` is observable. Such a metaclass carries observable equality behavior, so the value is rejected outright."""
    from quadrants.lang._final_dataclass_fields import (
        _is_exact_baked_type,
        final_scalar_key,
    )

    class EqIntMeta(type):
        # Make the class compare (and hash) as the builtin ``int`` - a hostile spoof of exact-builtin-ness.
        def __eq__(cls, other):
            return other is int or cls is other

        def __hash__(cls):
            return hash(int)

    class Spoof(int, metaclass=EqIntMeta):
        pass

    assert Spoof == int  # equality is spoofed...
    assert not _is_exact_baked_type(Spoof)  # ...but identity membership is not fooled
    assert _is_exact_baked_type(int) and _is_exact_baked_type(bool)  # exact builtins still recognised
    # A metaclass ``__eq__``/``__hash__`` is observable, so the value is rejected rather than keyed as an exact int.
    with pytest.raises(TypeError, match="metaclass defines observable"):
        final_scalar_key(Spoof(1))


@test_utils.test()
def test_final_enum_rejects_user_override_of_generated_hook():
    """The enum machinery *copies* ``enum.Enum._generate_next_value_`` into every subclass's own dict, so the name
    alone appears even on a plain enum. A user override (defined or monkey-patched later) is a distinct object a kernel
    could observe through the baked member, and mutating it would leave the member's class/name/value key unchanged -
    so it must be rejected. A clean enum (still the inherited default) is accepted - across every enum kind and Python
    version: CPython 3.12+ copies the inherited hook as a *fresh* ``staticmethod`` wrapper, so acceptance must compare
    the unwrapped function identity rather than the wrapper's."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Clean(enum.Enum):
        A = 1
        B = 2

    class CleanInt(enum.IntEnum):
        A = 1

    class CleanFlag(enum.IntFlag):
        A = 1

    final_scalar_key(Clean.A)  # inherited ``_generate_next_value_`` default -> accepted (plain Enum)
    final_scalar_key(CleanInt.A)  # ... IntEnum
    final_scalar_key(CleanFlag.A)  # ... IntFlag (all three would fail on 3.12+ under a wrapper-identity check)

    class Overridden(enum.Enum):
        def _generate_next_value_(name, start, count, last_values):  # observable custom auto-value policy
            return name

        A = enum.auto()
        B = enum.auto()

    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Overridden.A)

    # Monkey-patching the hook onto a previously-clean enum must flip it from accepted to rejected on the next key.
    class Patched(enum.Enum):
        A = 1

    final_scalar_key(Patched.A)  # accepted while it holds the inherited default
    Patched._generate_next_value_ = staticmethod(lambda name, start, count, last_values: name)
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Patched.A)


@test_utils.test()
def test_final_enum_rejects_user_override_of_mixin_dependent_hook():
    """A user override of a machinery hook whose default is *mix-in dependent* must also be rejected - not just
    ``_generate_next_value_`` (whose default is a verbatim base copy). ``_value_repr_`` is the example Codex flagged: a
    plain ``enum.Enum`` sets it to ``None`` while an ``int`` mix-in sets it to ``int``'s repr, so a nearest-base
    identity test cannot judge it (the mix-in value is a copy of no *enum* base). Overriding it is observable as
    ``cfg.mode._value_repr_`` yet leaves the member's class/name/value key unchanged, so it must flip an accepted enum
    to rejected. Acceptance is decided against a member-free rebuild of the enum's own bases, so a legitimate mix-in
    default - including a direct ``class M(int, enum.Enum)`` - is *not* mistaken for a user override."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class DirectIntMixin(int, enum.Enum):  # a hand-written IntEnum: its _value_repr_/_new_member_ are int's, not Enum's
        A = 1
        B = 2

    final_scalar_key(DirectIntMixin.A)  # a legitimate mix-in default must not be flagged as a user override

    class Patched(enum.Enum):
        A = 1
        B = 2

    final_scalar_key(Patched.A)  # accepted while it holds the machinery default
    Patched._value_repr_ = staticmethod(lambda value: "custom")  # observable as cfg.mode._value_repr_
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Patched.A)


@test_utils.test()
def test_final_enum_rejects_user_added_enum_valued_class_attribute():
    """Skipping every enum-valued class attribute would let a user-added one that is *not* a machinery member/alias
    through - e.g. ``Mode.X = Mode.A``, later reassignable to ``Mode.B``. That is observable as
    ``cfg.mode.__class__.X`` yet absent from the member's class/name/value key, so relaunching a Final config after
    rebinding ``X`` would silently reuse the stale specialization. Only names present under their own key in
    ``_member_map_`` (canonical members and aliases, whose identity the key captures) are exempt; a user-added
    attribute - even one holding a real member - is rejected. Machinery sunders that merely *hold* an enum value
    (``IntFlag._boundary_`` is a ``FlagBoundary`` member) must stay exempt, so an ordinary ``IntFlag`` is unaffected."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class WithAlias(enum.Enum):
        A = 1
        B = 2
        ALIAS = 1  # a machinery-defined alias for A - keyed by member identity, must stay accepted

    final_scalar_key(WithAlias.A)  # a plain enum carrying a canonical alias is accepted
    final_scalar_key(WithAlias.ALIAS)  # the alias resolves to A and is likewise accepted

    class Flags(enum.IntFlag):  # carries the enum-valued machinery sunder ``_boundary_`` (a FlagBoundary member)
        A = 1
        B = 2

    final_scalar_key(Flags.A)  # a plain IntFlag must not be rejected for its ``_boundary_`` bookkeeping

    class Mode(enum.Enum):
        A = 1
        B = 2

    final_scalar_key(Mode.A)  # clean, accepted
    Mode.X = Mode.A  # a user-added enum-valued class attribute (not an official member), observable and unkeyed
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Mode.A)


@test_utils.test()
def test_final_enum_rejects_cross_kind_machinery_sunder():
    """Whether a sunder/dunder in an enum's own dict is machinery bookkeeping (exempt) or a user hook (rejected) is
    judged against a member-free rebuild of *that enum's* bases/metaclass - not the union of names generated across
    every enum kind. Otherwise a name the machinery emits only for one kind - ``_boundary_``, generated for
    ``Flag``/``IntFlag`` - would wrongly exempt a user-added attribute of the same name on an unrelated enum:
    ``Mode._boundary_ = 1`` on a plain ``enum.Enum`` is observable as ``cfg.mode.__class__._boundary_`` yet leaves the
    member's class/name/value key unchanged, so rebinding it (``1`` -> ``2``) would silently reuse the stale
    specialization. A plain ``IntFlag`` (whose ``_boundary_`` is genuine machinery) stays accepted; overriding that
    ``_boundary_`` to a different value is itself observable and must flip it to rejected."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Mode(enum.Enum):
        A = 1
        B = 2

    final_scalar_key(Mode.A)  # clean, accepted
    Mode._boundary_ = 1  # a name generated only for Flag/IntFlag, added by a user to a plain Enum (not this shape)
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Mode.A)

    if hasattr(enum, "IntFlag") and hasattr(enum, "FlagBoundary"):

        class Flags(enum.IntFlag):
            A = 1
            B = 2

        final_scalar_key(Flags.A)  # ``_boundary_`` here is genuine machinery for this kind -> accepted
        default_boundary = Flags._boundary_
        other = next(b for b in enum.FlagBoundary if b is not default_boundary)
        Flags._boundary_ = other  # override the machinery value -> observable as cfg.mode.__class__._boundary_
        with pytest.raises(TypeError, match="observable class-level behavior"):
            final_scalar_key(Flags.A)


@test_utils.test()
def test_final_primitive_subclass_rejects_mixin_after_base():
    """The observable-behavior scan on a baked-primitive subclass must inspect *every* user class on the MRO, not stop
    at the first primitive base. ``class Unit(int, Labels)`` has MRO ``(Unit, int, Labels, object)``, so a scan that
    broke at ``int`` would never see ``Labels.label`` - observable as ``cfg.x.__class__.label`` yet absent from the
    key (which identifies the subclass only by module/qualname). A behavior-free subclass is still accepted."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Labels:
        label = "kg"

    class Unit(int, Labels):
        pass

    assert [c.__name__ for c in Unit.__mro__] == ["Unit", "int", "Labels", "object"]
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_scalar_key(Unit(1))

    class Plain(int):  # no mixin, no observable attrs -> accepted (keyed by class identity)
        pass

    final_scalar_key(Plain(1))


@test_utils.test()
def test_final_enum_rejects_state_on_unselected_member():
    """Per-member state is rejected for *any* member, not only the selected one. With ``cfg.mode = Mode.A``
    (state-free) but a sibling carrying imperatively-assigned state (``Mode.B.unit = "m"``), the state is still
    observable as ``cfg.mode.__class__.B.unit`` while the member-map key records only each member's value - so it must
    be rejected."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Mode(enum.Enum):
        A = 1
        B = 2

    final_scalar_key(Mode.A)  # clean, accepted
    Mode.B.unit = "m"  # state on a *sibling* of the selected member
    assert not Mode.A.__dict__.get("unit")  # the selected member itself stays state-free
    with pytest.raises(TypeError, match="per-member state"):
        final_scalar_key(Mode.A)


@test_utils.test()
def test_final_enum_kind_in_offline_key():
    """Redefining an enum's kind (``enum.Enum`` -> ``enum.IntEnum``) while keeping module/qualname/names/values flips
    compile-time behavior like ``cfg.mode == 1`` (false vs true), so the *offline* fastcache key must differ.
    In-process the class-identity token already separates the two class objects; this covers the cross-process key,
    where a resolvable enum's identity is only its module/qualname. Same-kind definitions still key identically."""
    import sys
    import types

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    def offline_key(base_name):
        mod = types.ModuleType("qd_enum_kind_probe")
        sys.modules[mod.__name__] = mod
        try:
            exec(f"import enum\nclass Mode(enum.{base_name}):\n    A = 1\n    B = 2\n", mod.__dict__)
            return str(final_scalar_key(mod.Mode.A, live=False))
        finally:
            sys.modules.pop(mod.__name__, None)

    assert offline_key("Enum") != offline_key("IntEnum")
    assert offline_key("Enum") == offline_key("Enum")  # same kind, same module/qualname -> stable reuse


@test_utils.test()
def test_final_scalar_subclass_base_kind_in_offline_key():
    """Symmetric to the enum-kind case, for a behavior-free scalar subclass: redefining ``class Unit(int)`` with a
    different primitive/NumPy base (``np.int64``, or a different width) keeps the resolvable module/qualname and
    canonicalizes ``Unit(1)`` to the same integer, so the offline key would collide - yet ``cfg.x.__class__.__mro__[1]
    is int`` changes. The base-kind component of ``_subclass_identity`` must separate them; same-base definitions stay
    identical."""
    import sys
    import types

    import numpy as np

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    def offline_key(base_expr):
        mod = types.ModuleType("qd_scalar_kind_probe")
        mod.np = np
        sys.modules[mod.__name__] = mod
        try:
            exec(f"class Unit({base_expr}):\n    pass\n", mod.__dict__)
            return str(final_scalar_key(mod.Unit(1), live=False))
        finally:
            sys.modules.pop(mod.__name__, None)

    assert offline_key("int") != offline_key("np.int64")  # int vs NumPy integer base
    assert offline_key("np.int32") != offline_key("np.int64")  # different NumPy integer widths
    assert offline_key("int") == offline_key("int")  # same base -> stable reuse


@test_utils.test()
def test_final_subclass_metaclass_kind_in_offline_key():
    """The class kind in ``_subclass_identity`` also covers the *metaclass*: a resolvable behavior-free subclass moved
    from the default metaclass to an empty custom one (``metaclass=EmptyMeta``) keeps module/qualname/base/canonical
    but ``cfg.x.__class__.__class__ is type`` flips, so the offline key must differ. An empty metaclass stays accepted
    (no observable behavior); only the key identity changes."""
    import sys
    import types

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    def offline_key(use_meta):
        mod = types.ModuleType("qd_metaclass_kind_probe")
        sys.modules[mod.__name__] = mod
        try:
            src = (
                "class EmptyMeta(type):\n    pass\nclass Unit(int, metaclass=EmptyMeta):\n    pass\n"
                if use_meta
                else "class Unit(int):\n    pass\n"
            )
            exec(src, mod.__dict__)
            return str(final_scalar_key(mod.Unit(1), live=False))
        finally:
            sys.modules.pop(mod.__name__, None)

    assert offline_key(False) != offline_key(True)  # default metaclass vs empty custom metaclass
    assert offline_key(True) == offline_key(True)  # same custom metaclass -> stable reuse


@test_utils.test()
def test_final_subclass_dynamic_metaclass_identity_in_offline_key():
    """A metaclass entry in the class kind also carries a dynamic-identity component: a *resolvable* subclass whose
    metaclass is a factory-built ``<locals>`` type shares module/qualname/structural values with every sibling
    metaclass, yet ``cfg.x.__class__.__class__ is ExpectedMeta`` is observable. Two builds with distinct metaclasses
    must not share an offline key, else fastcache could load a kernel baked for the other metaclass."""
    import sys
    import types

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    def make_meta():
        class M(type):  # <locals> -> a distinct object each call, same qualname
            pass

        return M

    def offline_key():
        meta = make_meta()
        mod = types.ModuleType("qd_dynamic_metaclass_probe")
        sys.modules[mod.__name__] = mod
        try:
            mod.Meta = meta  # the subclass below is module-level (resolvable); only its metaclass is dynamic
            exec("class Unit(int, metaclass=Meta):\n    pass\n", mod.__dict__)
            return str(final_scalar_key(mod.Unit(1), live=False))
        finally:
            sys.modules.pop(mod.__name__, None)

    assert offline_key() != offline_key()  # distinct <locals> metaclasses key apart despite a resolvable subclass


@test_utils.test()
def test_final_subclass_structural_identified_object_rejected():
    """A Python class or callable bound into a keyed structural attr (here ``__doc__``) is *rejected*, not keyed: its
    mutable behavior (``__code__``/``__defaults__``/a class dict) is readable at compile time
    (``cfg.x.__class__.__doc__.__defaults__``) yet uncapturable by the key, and it is not among the sources the source
    cache validates, so keying it by identity/qualname alone could reuse a stale specialization after its body
    changes. Rejection also covers such an object nested in a container structural value."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    def helper(a=1):  # a callable whose __defaults__ a kernel could read at compile time
        return a

    Unit.__doc__ = helper
    with pytest.raises(TypeError, match="mutable behavior"):
        final_scalar_key(Unit(1), live=True)
    with pytest.raises(TypeError, match="mutable behavior"):
        final_scalar_key(Unit(1), live=False)

    class Payload:  # a class bound as a structural value
        pass

    Unit.__doc__ = Payload
    with pytest.raises(TypeError, match="mutable behavior"):
        final_scalar_key(Unit(1), live=False)

    Unit.__doc__ = "plain"  # reset so the next rejection is attributable to the nested value, not this attr
    Unit.__slots__ = (helper,)  # nested inside a container value -> rejected recursively
    with pytest.raises(TypeError, match="mutable behavior"):
        final_scalar_key(Unit(1), live=True)


@test_utils.test()
def test_final_subclass_structural_float_bits_in_key():
    """A keyed structural attr holding a float is encoded by exact IEEE bits, so ``0.0`` and ``-0.0`` (equal with equal
    hashes, yet distinguishable at compile time via ``float.hex()``/``math.copysign``) do not share a key."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    Unit.__doc__ = 0.0
    live_before = final_scalar_key(Unit(1), live=True)
    offline_before = str(final_scalar_key(Unit(1), live=False))
    Unit.__doc__ = -0.0  # == 0.0 with an equal hash, but a distinct bit pattern
    assert final_scalar_key(Unit(1), live=True) != live_before
    assert str(final_scalar_key(Unit(1), live=False)) != offline_before


@test_utils.test()
def test_final_subclass_docstring_in_key():
    """``__doc__`` is the one structural class attr that is user-writable and readable at compile time
    (``cfg.x.__class__.__doc__``), so mutating it between launches must change both the in-process and offline keys
    rather than reuse a stale specialization. It is *keyed*, not rejected - documented classes stay legal."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    live_before = final_scalar_key(Unit(1), live=True)
    offline_before = str(final_scalar_key(Unit(1), live=False))
    Unit.__doc__ = "mutated between launches"
    assert final_scalar_key(Unit(1), live=True) != live_before  # in-process key flips (same class object)
    assert str(final_scalar_key(Unit(1), live=False)) != offline_before  # offline string flips too

    class Mode(enum.Enum):
        A = 1
        B = 2

    enum_before = final_scalar_key(Mode.A, live=True)
    Mode.__doc__ = "changed"
    assert final_scalar_key(Mode.A, live=True) != enum_before


@test_utils.test()
def test_final_subclass_slots_in_key():
    """``__slots__`` is likewise structural yet readable at compile time (``cfg.x.__class__.__slots__ == ()``).
    Reassigning it after class creation only rebinds the attribute (no new descriptors -> no state to reject), so the
    observed value changes with the state-scan none the wiser; the key must fold it in so a stale specialization is not
    reused."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        __slots__ = ()

    live_before = final_scalar_key(Unit(1), live=True)
    offline_before = str(final_scalar_key(Unit(1), live=False))
    Unit.__slots__ = ("relabeled",)  # rebinds the observable value without adding a real slot descriptor
    assert final_scalar_key(Unit(1), live=True) != live_before
    assert str(final_scalar_key(Unit(1), live=False)) != offline_before

    # Container type is retained: () and [] compare unequal, so they must not collapse to the same key.
    class Empty(int):
        __slots__ = ()

    empty_tuple = final_scalar_key(Empty(1), live=True)
    Empty.__slots__ = []
    assert final_scalar_key(Empty(1), live=True) != empty_tuple

    # Absent __slots__ must stay distinct from an explicit ``__slots__ = None`` ('__slots__' in cls.__dict__ differs).
    class Absent(int):
        pass

    absent_key = final_scalar_key(Absent(1), live=True)
    Absent.__slots__ = None
    assert final_scalar_key(Absent(1), live=True) != absent_key

    class Mode(enum.Enum):
        A = 1
        B = 2

    enum_before = final_scalar_key(Mode.A, live=True)
    Mode.__slots__ = ("relabeled",)
    assert final_scalar_key(Mode.A, live=True) != enum_before


@test_utils.test()
def test_final_subclass_weakref_in_key():
    """``__weakref__`` is structural-exempt yet user-rebindable and observable (``cfg.x.__class__.__weakref__``); the
    baked scalar/enum subclasses do not carry it in their own dict, so binding it is observable class state the scan
    ignores. It is keyed (auto-generated slot descriptors reduce to an address-free token so offline keys stay stable;
    a user value like ``None`` is kept), not rejected."""
    import enum

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    live_before = final_scalar_key(Unit(1), live=True)
    offline_before = str(final_scalar_key(Unit(1), live=False))
    Unit.__weakref__ = None  # rebinding an exempt machinery attr to an observable value
    assert final_scalar_key(Unit(1), live=True) != live_before
    assert str(final_scalar_key(Unit(1), live=False)) != offline_before

    class Mode(enum.Enum):
        A = 1
        B = 2

    enum_before = final_scalar_key(Mode.A, live=True)
    Mode.__weakref__ = None
    assert final_scalar_key(Mode.A, live=True) != enum_before


@test_utils.test()
def test_final_subclass_name_in_key():
    """``__name__`` is a mutable ``type`` getset (not in ``vars(cls)``, so the behavior scan misses it) that is
    independently reassignable from ``__qualname__`` and observable as ``cfg.x.__class__.__name__``, so renaming a
    class between launches must change the key."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    live_before = final_scalar_key(Unit(1), live=True)
    offline_before = str(final_scalar_key(Unit(1), live=False))
    Unit.__name__ = "Renamed"  # __qualname__ still ends in ".Unit", so only a __name__-aware key separates them
    assert final_scalar_key(Unit(1), live=True) != live_before
    assert str(final_scalar_key(Unit(1), live=False)) != offline_before


@test_utils.test()
def test_final_subclass_structural_dict_value_is_lossless():
    """A structural attr rebound to a ``dict`` must be serialized faithfully: two distinct mappings cannot collapse to
    one token (a compile-time branch can tell them apart). A value we cannot represent faithfully is rejected, not
    silently collapsed."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    Unit.__slots__ = {"a": 1}
    key_a = final_scalar_key(Unit(1), live=True)
    Unit.__slots__ = {"b": 2}
    assert final_scalar_key(Unit(1), live=True) != key_a  # distinct dicts -> distinct keys, no lossy collapse

    class Opaque(int):
        pass

    Opaque.__slots__ = object()  # not plain data and not a descriptor/callable -> unrepresentable
    with pytest.raises(TypeError, match="cannot be keyed faithfully"):
        final_scalar_key(Opaque(1), live=True)


@test_utils.test()
def test_final_subclass_313_metadata_in_key():
    """3.13 adds ``__firstlineno__`` / ``__static_attributes__`` to a class dict; both are writable and readable at
    compile time (``cfg.x.__class__.__firstlineno__``), so mutating either - or redefining a class at a different source
    line - must change the key. On older runtimes they are absent (``_ATTR_ABSENT``) and this is a no-op."""
    import sys

    if sys.version_info < (3, 13):
        return

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    live_before = final_scalar_key(Unit(1), live=True)
    offline_before = str(final_scalar_key(Unit(1), live=False))
    Unit.__firstlineno__ = 424242
    assert final_scalar_key(Unit(1), live=True) != live_before
    assert str(final_scalar_key(Unit(1), live=False)) != offline_before

    class Other(int):
        pass

    static_before = final_scalar_key(Other(1), live=True)
    Other.__static_attributes__ = ("zzz",)
    assert final_scalar_key(Other(1), live=True) != static_before


@test_utils.test()
def test_final_subclass_structural_scalar_subclass_state_rejected():
    """A scalar subclass bound in a keyed structural attr routes through ``final_scalar_key``, not a bare-value token:
    per-instance state (``cfg.x.__class__.__doc__.label``) is rejected, and two distinct stateless subclasses sharing a
    qualname/value key apart instead of collapsing."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class Unit(int):
        pass

    class Labelled(float):
        pass

    stateful = Labelled(1.0)
    stateful.label = "a"  # per-instance state a kernel could read; the bare value would drop it
    Unit.__doc__ = stateful
    with pytest.raises(TypeError, match="per-instance state"):
        final_scalar_key(Unit(1), live=True)

    def make_plain():
        class Plain(float):
            pass

        return Plain

    class UnitA(int):
        pass

    class UnitB(int):
        pass

    UnitA.__doc__ = make_plain()(1.0)  # distinct <locals> classes, same qualname/value
    UnitB.__doc__ = make_plain()(1.0)
    assert final_scalar_key(UnitA(1), live=True) != final_scalar_key(UnitB(1), live=True)


@test_utils.test()
def test_final_subclass_structural_container_subclass_rejected():
    """A container *subclass* in a keyed structural attr can add per-instance state / a distinct identity that keying by
    its elements alone would drop, so it is rejected (exact ``tuple``/``list``/``set``/``dict`` stay legal)."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class MyList(list):
        pass

    class Unit(int):
        pass

    Unit.__slots__ = MyList()
    with pytest.raises(TypeError, match="builtin subclass"):
        final_scalar_key(Unit(1), live=True)


@test_utils.test()
def test_final_subclass_structural_foreign_descriptor_identity():
    """A *foreign* descriptor bound in a keyed structural attr (``Unit.__weakref__ = A.__weakref__``) must carry its
    owner's identity: its address-based ``repr`` is unkeyable and its own ``__qualname__`` is just the slot name, so
    without the owner two distinct descriptors collapse to one token and a compile-time ``... is A.__weakref__`` branch
    would reuse the wrong specialization."""
    from quadrants.lang._final_dataclass_fields import final_scalar_key

    class OwnerA:
        pass

    class OwnerB:
        pass

    class Unit(int):
        pass

    Unit.__weakref__ = OwnerA.__dict__["__weakref__"]
    live_a = final_scalar_key(Unit(1), live=True)
    offline_a = str(final_scalar_key(Unit(1), live=False))
    Unit.__weakref__ = OwnerB.__dict__["__weakref__"]  # distinct owner -> distinct qualname
    assert final_scalar_key(Unit(1), live=True) != live_a
    assert str(final_scalar_key(Unit(1), live=False)) != offline_a

    def make_owner():  # two owners sharing a qualname but distinct identity
        class Owner:
            pass

        return Owner

    Unit.__weakref__ = make_owner().__dict__["__weakref__"]
    live_c = final_scalar_key(Unit(1), live=True)
    Unit.__weakref__ = make_owner().__dict__["__weakref__"]
    assert final_scalar_key(Unit(1), live=True) != live_c  # separated by the owner's identity component


@test_utils.test()
def test_final_subclass_dynamic_serial_keyed_by_identity():
    """The offline serial registry (backing every dynamic object's identity token - a subclass, its metaclass, or a
    descriptor owner) keys by object *identity*, not ``__eq__``: a metaclass that makes distinct ``<locals>`` classes
    compare equal with equal hashes must not alias them to one serial (a ``WeakKeyDictionary`` would), or a
    ``fastcache`` offline key could load code baked for a different object. Each serial is stable per object and
    distinct across objects."""
    from quadrants.lang._final_dataclass_fields import _dynamic_class_serial

    class Meta(type):
        def __eq__(cls, other):
            return isinstance(other, Meta)

        def __hash__(cls):
            return 0

    def make():
        class Local(metaclass=Meta):
            pass

        return Local

    first, second = make(), make()
    assert first == second and hash(first) == hash(second) and first is not second  # equality-aliased, distinct id
    serial_first = _dynamic_class_serial(first)
    assert _dynamic_class_serial(first) == serial_first  # stable for the same object
    assert _dynamic_class_serial(second) != serial_first  # distinct object -> distinct serial despite == aliasing


@test_utils.test()
def test_final_enum_key_incorporates_full_member_map():
    """A baked ``Final`` enum member keys on the *entire* member map, not only the selected member: a kernel can read a
    sibling at compile time (``cfg.mode.__class__.OTHER.value``), so changing another member's value must invalidate
    the key. This matters most for a *resolvable* (module-level) enum, whose class-identity component is ``None`` (keyed
    by ``module``/``qualname``): without the member map, redefining the module's enum with a different sibling value -
    in-process, or in a separate ``fastcache`` process that otherwise only hashes kernel source - would reuse the stale
    specialization. Identical definitions still key identically (legitimate reuse preserved), and an unsupported/mutable
    sibling value is rejected even when the selected member is a plain scalar."""
    import enum
    import sys
    import types

    from quadrants.lang._final_dataclass_fields import final_scalar_key

    mod_name = "qd_member_map_probe"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    try:

        def define(b_value):
            class Mode(enum.Enum):
                A = 1
                B = b_value

            Mode.__module__ = mod_name
            Mode.__qualname__ = "Mode"
            mod.Mode = Mode  # resolvable via module+qualname -> None class-identity component (member map is the guard)
            return Mode

        key_b2 = final_scalar_key(define(2).A)
        assert final_scalar_key(define(2).A) == key_b2  # identical resolvable definition -> identical key (reuse kept)
        changed = define(3)  # a changed *sibling* value, selected member A and class name unchanged
        assert final_scalar_key(changed.A) != key_b2  # in-process key flips
        assert str(final_scalar_key(changed.A)) != str(key_b2)  # and the offline (fastcache) string flips too

        class Bad(enum.Enum):
            A = 1
            B = [1, 2]  # a mutable, unkeyable sibling - observable as cfg.mode.__class__.B.value

        with pytest.raises(TypeError, match="not a supported"):
            final_scalar_key(Bad.A)
    finally:
        del sys.modules[mod_name]


@test_utils.test()
def test_final_outer_mapping_cache_disabled_for_final_bearing_arg():
    """``TemplateMapper.lookup`` keeps an instance-keyed ``(count, key)`` cache that returns the prior result for the
    same live argument *before* calling ``extract()``. For a Final-bearing argument that would bypass
    ``final_scalar_key``'s per-launch revalidation - so a launch could reuse the specialization compiled before a
    ``Final`` value's class was monkey-patched behaviorful. This cache is therefore disabled for any mapper carrying a
    Final-bearing argument: it stores nothing and revalidates every launch. A Final-free mapper still caches, so the
    common hot path is untouched."""
    import enum
    from typing import Final

    from quadrants.lang._template_mapper import TemplateMapper
    from quadrants.lang.kernel_arguments import ArgMetadata

    @dataclass(frozen=True)
    class Plain:
        a: int

    plain_mapper = TemplateMapper([ArgMetadata(Plain, "p")], [])
    p = Plain(a=1)
    plain_mapper.lookup(False, (p,))
    plain_mapper.lookup(False, (p,))
    assert plain_mapper._mapping_cache, "a Final-free mapper must keep its instance-keyed (count, key) cache"

    class Mode(enum.Enum):
        A = 0
        B = 1

    @dataclass(frozen=True)
    class Cfg:
        mode: Final[Mode]

    final_mapper = TemplateMapper([ArgMetadata(Cfg, "cfg")], [])
    cfg = Cfg(mode=Mode.A)
    final_mapper.lookup(False, (cfg,))  # clean enum accepted on the first launch
    assert not final_mapper._mapping_cache, "a Final-bearing mapper must not populate the instance-keyed cache"

    # Monkey-patch the enum class behaviorful; the SAME live instance must now be re-rejected, proving the second
    # lookup re-ran extract()+validation rather than returning a cached key.
    Mode._generate_next_value_ = staticmethod(lambda name, start, count, last_values: name)
    with pytest.raises(TypeError, match="observable class-level behavior"):
        final_mapper.lookup(False, (cfg,))
