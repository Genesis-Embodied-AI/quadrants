import dataclasses
import gc
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest

import quadrants as ti
from quadrants.lang._kernel_types import KernelBatchedArgType
from quadrants.lang.impl import Kernel, QuadrantsSyntaxError

from tests import test_utils


@pytest.fixture
def ti_type(use_ndarray: bool) -> Any:
    if use_ndarray:
        return ti.ndarray
    return ti.field


@pytest.fixture
def ti_annotation(use_ndarray: bool) -> Any:
    class TiTemplateBuilder:
        """
        Allows ti_annotation[ti.i32, 2] to be legal
        """

        def __getitem__(self, _):
            return ti.Template

    if use_ndarray:
        return ti.types.ndarray
    return TiTemplateBuilder()


@test_utils.test()
def test_ndarray_struct_kwargs():
    gc.collect()
    gc.collect()

    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    d = ti.ndarray(ti.i32, shape=(223,))
    e = ti.ndarray(ti.i32, shape=(227,))

    @dataclass
    class MyStruct:
        a: ti.types.NDArray[ti.i32, 1]
        b: ti.types.NDArray[ti.i32, 1]
        c: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def s4(a: ti.types.NDArray[ti.i32, 1], b: ti.types.NDArray[ti.i32, 1]) -> None:
        # note: no used py dataclass parameters
        a[1] += 888
        b[2] += 999

    @ti.func
    def s3(z3: ti.types.NDArray[ti.i32, 1], my_struct3: MyStruct, bar3: ti.types.NDArray[ti.i32, 1]) -> None:
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

    @ti.func
    def s2(z3: ti.types.NDArray[ti.i32, 1], my_struct3: MyStruct, bar3: ti.types.NDArray[ti.i32, 1]) -> None:
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

    @ti.func
    def s1(z2: ti.types.NDArray[ti.i32, 1], my_struct2: MyStruct, bar2: ti.types.NDArray[ti.i32, 1]) -> None:
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

    @ti.kernel
    def k1(z: ti.types.NDArray[ti.i32, 1], my_struct: MyStruct, bar: ti.types.NDArray[ti.i32, 1]) -> None:
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
def test_ndarray_struct(ti_type: Any, ti_annotation: Any) -> None:
    gc.collect()
    gc.collect()
    a = ti_type(ti.i32, shape=(55,))
    b = ti_type(ti.i32, shape=(57, 23))
    c = ti_type(ti.i32, shape=(211, 34, 25))
    d = ti_type(ti.i32, shape=(223,))
    e = ti_type(ti.i32, shape=(227,))

    @dataclass
    class MyStruct:
        a: ti_annotation[ti.i32, 1]
        b: ti_annotation[ti.i32, 2]
        c: ti_annotation[ti.i32, 3]

    @ti.func
    def s3(z3: ti_annotation[ti.i32, 1], my_struct3: MyStruct, bar3: ti_annotation[ti.i32, 1]) -> None:
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

    @ti.func
    def s2(z3: ti_annotation[ti.i32, 1], my_struct3: MyStruct, bar3: ti_annotation[ti.i32, 1]) -> None:
        # stores
        z3[24] += 89
        my_struct3.a[46] += 32
        my_struct3.b[48, 0] += 33
        my_struct3.c[42, 0, 0] += 34
        bar3[112] += 125
        s3(z3, my_struct3, bar3)

    @ti.func
    def s1(z2: ti_annotation[ti.i32, 1], my_struct2: MyStruct, bar2: ti_annotation[ti.i32, 1]) -> None:
        # stores
        z2[22] += 88
        my_struct2.a[45] += 22
        my_struct2.b[47, 0] += 23
        my_struct2.c[41, 0, 0] += 24
        bar2[111] += 123
        s2(z2, my_struct2, bar2)

    @ti.kernel
    def k1(z: ti_annotation[ti.i32, 1], my_struct: MyStruct, bar: ti_annotation[ti.i32, 1]) -> None:
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

    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    z_param = ti.ndarray(ti.i32, shape=(223,))
    bar_param = ti.ndarray(ti.i32, shape=(227,))

    field1 = ti.field(ti.i32, shape=(300,))

    @dataclass
    class MyStructAB:
        a: ti.types.NDArray[ti.i32, 1]
        b: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class MyStructC:
        c: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def s2(
        my_struct_ab3: MyStructAB,
        z3: ti.types.NDArray[ti.i32, 1],
        fieldparam1_3: ti.Template,
        my_struct_c3: MyStructC,
        bar3: ti.types.NDArray[ti.i32, 1],
    ) -> None:
        # stores
        z3[24] += 89
        my_struct_ab3.a[46] += 32
        my_struct_ab3.b[48] += 33
        my_struct_c3.c[42] += 34
        bar3[112] += 125
        fieldparam1_3[4] = 69

    @ti.func
    def s1(
        z2: ti.types.NDArray[ti.i32, 1],
        my_struct_c2: MyStructC,
        my_struct_ab2: MyStructAB,
        fieldparam1_2: ti.Template,
        bar2: ti.types.NDArray[ti.i32, 1],
    ) -> None:
        # stores
        z2[22] += 88
        my_struct_ab2.a[45] += 22
        my_struct_ab2.b[47] += 23
        my_struct_c2.c[41] += 24
        bar2[111] += 123
        fieldparam1_2[3] = 68

        s2(my_struct_ab2, z2, fieldparam1_2, my_struct_c2, bar2)

    @ti.kernel
    def k1(
        z: ti.types.NDArray[ti.i32, 1],
        my_struct_ab: MyStructAB,
        bar: ti.types.NDArray[ti.i32, 1],
        my_struct_c: MyStructC,
        fieldparam1: ti.Template,
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
def test_ndarray_struct_primitives(ti_type: Any, ti_annotation: Any) -> None:
    gc.collect()
    gc.collect()

    a = ti_type(ti.i32, shape=(55,))
    b = ti_type(ti.i32, shape=(57,))
    c = ti_type(ti.i32, shape=(211,))
    z_param = ti_type(ti.i32, shape=(223,))
    bar_param = ti_type(ti.i32, shape=(227,))

    @dataclass
    class MyStructAB:
        p3: ti.i32
        a: ti_annotation[ti.i32, 1]
        p1: ti.i32
        p2: ti.i32

    @dataclass
    class MyStructC:
        c: ti_annotation[ti.i32, 1]

    @ti.kernel
    def k1(
        z: ti_annotation[ti.i32, 1],
        my_struct_ab: MyStructAB,
        bar: ti_annotation[ti.i32, 1],
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
    a = ti.ndarray(ti.i32, shape=(101,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    d = ti.ndarray(ti.i32, shape=(211,))
    e = ti.ndarray(ti.i32, shape=(251,))
    f = ti.ndarray(ti.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: ti.types.NDArray[ti.i32, 1]
        f: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class MyStructCD:
        c: ti.types.NDArray[ti.i32, 1]
        d: ti.types.NDArray[ti.i32, 1]
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: ti.types.NDArray[ti.i32, 1]
        b: ti.types.NDArray[ti.i32, 1]
        struct_cd: MyStructCD

    @ti.func
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

    @ti.func
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

    @ti.kernel
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
    a = ti.field(ti.i32, shape=(55,))
    b = ti.field(ti.i32, shape=(57,))
    c = ti.field(ti.i32, shape=(211,))
    d = ti.field(ti.i32, shape=(211,))
    e = ti.field(ti.i32, shape=(251,))
    f = ti.field(ti.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: ti.Template
        f: ti.Template

    @dataclass
    class MyStructCD:
        c: ti.Template
        d: ti.Template
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: ti.Template
        b: ti.Template
        struct_cd: MyStructCD

    @ti.func
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

    @ti.func
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

    @ti.kernel
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
    a = ti.field(ti.i32, shape=(55,))
    b = ti.field(ti.i32, shape=(57,))
    c = ti.field(ti.i32, shape=(211,))
    d = ti.field(ti.i32, shape=(211,))
    e = ti.field(ti.i32, shape=(251,))
    f = ti.field(ti.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: ti.Template
        f: ti.Template

    @dataclass
    class MyStructCD:
        c: ti.Template
        d: ti.Template
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: ti.Template
        b: ti.Template
        struct_cd: MyStructCD

    @ti.func
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

    @ti.func
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

    @ti.kernel
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
    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    d = ti.ndarray(ti.i32, shape=(211,))
    e = ti.ndarray(ti.i32, shape=(251,))
    f = ti.ndarray(ti.i32, shape=(251,))

    d11 = ti.ndarray(ti.i32, shape=(251,))
    d12 = ti.ndarray(ti.i32, shape=(251,))
    d21 = ti.ndarray(ti.i32, shape=(251,))
    d22 = ti.ndarray(ti.i32, shape=(251,))
    d31 = ti.ndarray(ti.i32, shape=(251,))
    d32 = ti.ndarray(ti.i32, shape=(251,))

    @dataclass
    class D1:
        d11: ti.types.NDArray[ti.i32, 1]
        d12: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class D2:
        d21: ti.types.NDArray[ti.i32, 1]
        d22: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class D3:
        d31: ti.types.NDArray[ti.i32, 1]
        d32: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class C1:
        a: ti.types.NDArray[ti.i32, 1]
        d1: D1
        d2: D2
        d3: D3
        b: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class C2:
        c: ti.types.NDArray[ti.i32, 1]
        d: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class C3:
        e: ti.types.NDArray[ti.i32, 1]
        f: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class P1:
        c1: C1
        c2: C2
        c3: C3

    @ti.kernel
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
    a = ti.field(ti.i32, shape=(55,))
    b = ti.field(ti.i32, shape=(57,))
    c = ti.field(ti.i32, shape=(211,))
    d = ti.field(ti.i32, shape=(211,))
    e = ti.field(ti.i32, shape=(251,))
    f = ti.field(ti.i32, shape=(251,))

    @dataclass
    class C1:
        a: ti.Template
        b: ti.Template

    @dataclass
    class C2:
        c: ti.Template
        d: ti.Template

    @dataclass
    class C3:
        e: ti.Template
        f: ti.Template

    @dataclass
    class P1:
        c1: C1
        c2: C2
        c3: C3

    @ti.kernel
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


@pytest.mark.parametrize("use_slots", [False, True])
@test_utils.test()
def test_template_mapper_cache(use_slots, monkeypatch):
    # Mock '_extract_arg' to track the number of (recursive) calls
    counter = 0
    _extract_arg_orig = ti.lang._template_mapper_hotpath._extract_arg

    def _extract_arg(*args, **kwargs):
        nonlocal counter
        counter += 1
        return _extract_arg_orig(*args, **kwargs)

    monkeypatch.setattr("quadrants.lang._template_mapper_hotpath._extract_arg", _extract_arg)

    @dataclass(frozen=True, slots=use_slots)
    class MyStruct:
        value: ti.types.ndarray()
        placeholder: ti.i32

    @ti.kernel
    def my_kernel(my_struct_1d: MyStruct, my_struct_2d: MyStruct) -> None:
        for i in ti.ndrange(my_struct_1d.value.shape[0]):
            my_struct_1d.value[i] += 1
        for i, j in ti.ndrange(my_struct_2d.value.shape[0], my_struct_2d.value.shape[1]):
            my_struct_2d.value[i, j] += 1

    num_fields = len(fields(MyStruct))
    value = ti.ndarray(ti.i32, shape=(1,))
    value.fill(0)
    placeholder = 0
    my_struct_1d = MyStruct(value=value, placeholder=placeholder)
    value = ti.ndarray(ti.f32, shape=(1, 2))
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
        used1: ti.types.NDArray[ti.i32, 1]
        used2: ti.types.NDArray[ti.i32, 1]
        used3: ti.types.NDArray[ti.i32, 1]
        an_int: ti.i32
        not_used_int: ti.i32
        not_used: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def f1(md: MyDataclass) -> None:
        md.used3[0] = 123
        md.used3[1] = md.an_int

    @ti.kernel
    def k1(md: MyDataclass, trigger_static: ti.Template) -> None:
        md.used1[0] = 222
        md.used1[1] = md.used2[0]
        f1(md)
        if ti.static(trigger_static):
            md.used1[2] = 444

    u1 = ti.ndarray(ti.i32, (10,))
    u2 = ti.ndarray(ti.i32, (10,))
    u3 = ti.ndarray(ti.i32, (10,))
    nu1 = ti.ndarray(ti.i32, (10,))
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
        n1: ti.types.NDArray[ti.i32, 1]
        n1u: ti.types.NDArray[ti.i32, 1]

    @dataclasses.dataclass
    class MyDataclass1:
        used1: ti.types.NDArray[ti.i32, 1]
        used2: ti.types.NDArray[ti.i32, 1]
        used3: ti.types.NDArray[ti.i32, 1]
        not_used: ti.types.NDArray[ti.i32, 1]
        nested1: Nested1

    @dataclasses.dataclass
    class MyDataclass2:
        used1: ti.types.NDArray[ti.i32, 1]
        used2: ti.types.NDArray[ti.i32, 1]
        used3: ti.types.NDArray[ti.i32, 1]
        not_used: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def f1(md1: MyDataclass1, md2: MyDataclass2) -> None:
        md1.used3[0] = 123
        md2.used1[5] = 555
        md2.used2[5] = 444
        md2.used3[5] = 333
        md1.nested1.n1[0] = 777

    @ti.kernel
    def k1(md1: MyDataclass1, md2: MyDataclass2, trigger_static: ti.Template) -> None:
        md1.used1[0] = 222
        md1.used1[1] = md1.used2[0]
        f1(md1, md2)
        if ti.static(trigger_static):
            md1.used1[2] = 444

    u1 = ti.ndarray(ti.i32, (10,))
    u2 = ti.ndarray(ti.i32, (10,))
    u3 = ti.ndarray(ti.i32, (10,))
    n1 = ti.ndarray(ti.i32, (10,))
    nu1 = ti.ndarray(ti.i32, (10,))
    n1u = ti.ndarray(ti.i32, (10,))
    nested1 = Nested1(n1=n1, n1u=n1u)
    md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1, nested1=nested1)

    u1b = ti.ndarray(ti.i32, (10,))
    u2b = ti.ndarray(ti.i32, (10,))
    u3b = ti.ndarray(ti.i32, (10,))
    nu1b = ti.ndarray(ti.i32, (10,))
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
        used1: ti.types.NDArray[ti.i32, 1]
        used2: ti.types.NDArray[ti.i32, 1]
        used3: ti.types.NDArray[ti.i32, 1]
        not_used: ti.types.NDArray[ti.i32, 1]

    @dataclasses.dataclass
    class MyDataclass2:
        used1: ti.types.NDArray[ti.i32, 1]
        used2: ti.types.NDArray[ti.i32, 1]
        used3: ti.types.NDArray[ti.i32, 1]
        not_used: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def f2(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
        md1.used1[0] = 111
        md1.used2[0] = 222
        md1.used3[0] = 123
        md2.used1[0] = 555
        md2.used2[0] = 444
        md2.used3[0] = 333

    @ti.func
    def f1(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
        f2(i_b, md1=md1, md2=md2)

    @ti.kernel
    def k1(envs_idx: ti.types.NDArray[ti.i32, 1], md1: MyDataclass1, md2: MyDataclass2) -> None:
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            f1(i_b, md1=md1, md2=md2)

    envs_idx = ti.ndarray(ti.i32, (10,))

    u1 = ti.ndarray(ti.i32, (10,))
    u2 = ti.ndarray(ti.i32, (10,))
    u3 = ti.ndarray(ti.i32, (10,))
    nu1 = ti.ndarray(ti.i32, (10,))
    md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1)

    u1b = ti.ndarray(ti.i32, (10,))
    u2b = ti.ndarray(ti.i32, (10,))
    u3b = ti.ndarray(ti.i32, (10,))
    nu1b = ti.ndarray(ti.i32, (10,))
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
    arch_name = ti.lang.impl.current_cfg().arch.name
    for _it in range(3):
        ti.init(arch=getattr(ti, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class Nested1:
            n1: ti.types.NDArray[ti.i32, 1]
            n1u: ti.types.NDArray[ti.i32, 1]

        @dataclasses.dataclass
        class MyDataclass1:
            used1: ti.types.NDArray[ti.i32, 1]
            used2: ti.types.NDArray[ti.i32, 1]
            used3: ti.types.NDArray[ti.i32, 1]
            not_used: ti.types.NDArray[ti.i32, 1]
            nested1: Nested1

        @dataclasses.dataclass
        class MyDataclass2:
            used1: ti.types.NDArray[ti.i32, 1]
            used2: ti.types.NDArray[ti.i32, 1]
            used3: ti.types.NDArray[ti.i32, 1]
            not_used: ti.types.NDArray[ti.i32, 1]

        @ti.func
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

        @ti.kernel(fastcache=True)
        def k1(md1: MyDataclass1, md2: MyDataclass2, trigger_static: ti.Template) -> None:
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
            if ti.static(trigger_static):
                md1.used1[2] = 444

        u1 = ti.ndarray(ti.i32, (10,))
        u2 = ti.ndarray(ti.i32, (10,))
        u3 = ti.ndarray(ti.i32, (10,))
        n1 = ti.ndarray(ti.i32, (10,))
        nu1 = ti.ndarray(ti.i32, (10,))
        n1u = ti.ndarray(ti.i32, (10,))
        nested1 = Nested1(n1=n1, n1u=n1u)
        md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1, nested1=nested1)

        u1b = ti.ndarray(ti.i32, (10,))
        u2b = ti.ndarray(ti.i32, (10,))
        u3b = ti.ndarray(ti.i32, (10,))
        nu1b = ti.ndarray(ti.i32, (10,))
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
    arch_name = ti.lang.impl.current_cfg().arch.name
    for _it in range(3):
        ti.init(arch=getattr(ti, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass1:
            used1: ti.types.NDArray[ti.i32, 1]
            used2: ti.types.NDArray[ti.i32, 1]
            used3: ti.types.NDArray[ti.i32, 1]
            not_used: ti.types.NDArray[ti.i32, 1]
            not_used2: ti.types.NDArray[ti.i32, 1]

        @dataclasses.dataclass
        class MyDataclass2:
            used1: ti.types.NDArray[ti.i32, 1]
            used2: ti.types.NDArray[ti.i32, 1]
            used3: ti.types.NDArray[ti.i32, 1]
            not_used: ti.types.NDArray[ti.i32, 1]
            not_used2: ti.types.NDArray[ti.i32, 1]

        @ti.func
        def f2(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            md1.used1[0] = 111
            md1.used2[0] = 222
            md1.used3[0] = 123
            md2.used1[0] = 555
            md2.used2[0] = 444
            md2.used3[0] = 333

        @ti.func
        def f1(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            f2(i_b, md1=md1, md2=md2)

        @ti.kernel(fastcache=True)
        def k1(envs_idx: ti.types.NDArray[ti.i32, 1], md1: MyDataclass1, md2: MyDataclass2) -> None:
            for i_b_ in range(envs_idx.shape[0]):
                i_b = envs_idx[i_b_]
                f1(i_b, md1=md1, md2=md2)

        envs_idx = ti.ndarray(ti.i32, (10,))

        u1 = ti.ndarray(ti.i32, (10,))
        u2 = ti.ndarray(ti.i32, (10,))
        u3 = ti.ndarray(ti.i32, (10,))
        nu1 = ti.ndarray(ti.i32, (10,))
        nu2 = ti.ndarray(ti.i32, (10,))
        md1 = MyDataclass1(used1=u1, used2=u2, used3=u3, not_used=nu1, not_used2=nu2)

        u1b = ti.ndarray(ti.i32, (10,))
        u2b = ti.ndarray(ti.i32, (10,))
        u3b = ti.ndarray(ti.i32, (10,))
        nu1b = ti.ndarray(ti.i32, (10,))
        nu2b = ti.ndarray(ti.i32, (10,))
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
    arch_name = ti.lang.impl.current_cfg().arch.name
    for _it in range(3):
        ti.init(arch=getattr(ti, arch_name), offline_cache_file_path=str(tmp_path), offline_cache=True)

        @dataclasses.dataclass
        class MyDataclass1:
            not_used1: ti.types.NDArray[ti.i32, 1]
            not_used2: ti.types.NDArray[ti.i32, 1]

        @dataclasses.dataclass
        class MyDataclass2:
            not_used1: ti.types.NDArray[ti.i32, 1]
            not_used2: ti.types.NDArray[ti.i32, 1]

        @ti.func
        def f2(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            pass

        @ti.func
        def f1(i_b, md1: MyDataclass1, md2: MyDataclass2) -> None:
            f2(i_b, md1, md2=md2)

        @ti.kernel(fastcache=True)
        def k1(envs_idx: ti.types.NDArray[ti.i32, 1], md1: MyDataclass1, md2: MyDataclass2) -> None:
            for i_b_ in range(envs_idx.shape[0]):
                i_b = envs_idx[i_b_]
                f1(i_b, md1, md2=md2)

        envs_idx = ti.ndarray(ti.i32, (10,))

        nu1 = ti.ndarray(ti.i32, (10,))
        nu2 = ti.ndarray(ti.i32, (10,))
        md1 = MyDataclass1(not_used1=nu1, not_used2=nu2)

        nu1b = ti.ndarray(ti.i32, (10,))
        nu2b = ti.ndarray(ti.i32, (10,))
        md2 = MyDataclass2(not_used1=nu1b, not_used2=nu2b)

        k1(envs_idx, md1, md2=md2)


@test_utils.test()
def test_pruning_with_keyword_rename() -> None:
    @dataclasses.dataclass
    class MyStruct:
        used: ti.types.NDArray[ti.f32, 2]
        not_used: ti.types.NDArray[ti.f32, 2]

    def create_struct():
        my_struct_outside = MyStruct(
            used=ti.ndarray(dtype=ti.f32, shape=(1, 1)), not_used=ti.ndarray(dtype=ti.f32, shape=(1, 1))
        )
        return my_struct_outside

    @ti.func
    def f1(new_struct_name: MyStruct):
        new_struct_name.used[0, 0] = 100

    @ti.kernel
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
        used: ti.types.NDArray[ti.f32, 2]
        not_used: ti.types.NDArray[ti.f32, 2]

    def create_struct():
        return MyStruct(used=ti.ndarray(dtype=ti.f32, shape=(1, 1)), not_used=ti.ndarray(dtype=ti.f32, shape=(1, 1)))

    @ti.func
    def f1(new_struct_name: MyStruct):
        new_struct_name.used[0, 0] = 100

    @ti.kernel
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
        used: ti.types.NDArray[ti.f32, 2]
        not_used: ti.types.NDArray[ti.f32, 2]

    def create_structs():
        my_struct1 = MyStruct(
            used=ti.ndarray(dtype=ti.f32, shape=(1, 1)), not_used=ti.ndarray(dtype=ti.f32, shape=(1, 1))
        )
        my_struct2 = MyStruct(
            used=ti.ndarray(dtype=ti.f32, shape=(1, 1)), not_used=ti.ndarray(dtype=ti.f32, shape=(1, 1))
        )
        my_struct3 = MyStruct(
            used=ti.ndarray(dtype=ti.f32, shape=(1, 1)), not_used=ti.ndarray(dtype=ti.f32, shape=(1, 1))
        )
        my_struct4 = MyStruct(
            used=ti.ndarray(dtype=ti.f32, shape=(1, 1)), not_used=ti.ndarray(dtype=ti.f32, shape=(1, 1))
        )
        return my_struct1, my_struct2, my_struct3, my_struct4

    @ti.func
    def g1(struc3_g1: MyStruct):
        # should be used:
        # struc3_g1.used
        struc3_g1.used[0, 0] = 102

    @ti.func
    def f2(a3: ti.i32, struct_f2: MyStruct, b3: ti.i32, d3: ti.i32, struct2_f2: MyStruct, c3: ti.i32):
        # should be used:
        # struct_f2.used
        # struct2_f2.useds
        struct_f2.used[0, 0] = 100
        struct2_f2.used[0, 0] = 101

    @ti.func
    def f1(a2: ti.i32, struct_f1: MyStruct, b2: ti.i32, d2: ti.i32, struct2_f1: MyStruct, c2: ti.i32):
        # should be used:
        # struct_f1.used
        # struct2_f1.used
        f2(a2, struct_f1, b2, d3=d2, struct2_f2=struct2_f1, c3=c2)

    @ti.kernel
    def k1(
        a: ti.i32,
        struct1_k1: MyStruct,
        b: ti.i32,
        d: ti.i32,
        struct2_k1: MyStruct,
        c: ti.i32,
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
        a: ti.types.NDArray[ti.f32, 2]
        b: ti.types.NDArray[ti.f32, 2]
        c: ti.types.NDArray[ti.f32, 2]
        d: ti.types.NDArray[ti.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            a=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            b=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            c=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            d=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f1(depth: ti.template(), struc_f1: MyStruct):
        if ti.static(depth) == 0:
            struc_f1.a[0, 0] = 100
            f1(1, struc_f1)
        elif ti.static(depth) == 1:
            struc_f1.b[0, 0] = 101
            f1(2, struc_f1)
        elif ti.static(depth) == 2:
            struc_f1.c[0, 0] = 102
            f1(2, struc_f1)

    @ti.kernel
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
        _f3: ti.types.NDArray[ti.f32, 2]
        _f2b: ti.types.NDArray[ti.f32, 2]
        _f2a: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _k1: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            _f3=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2b=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2a=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f3(struc_f3: MyStruct):
        struc_f3._f3[0, 0] = 104
        f2b(struc_f3)

    @ti.func
    def f2b(struc_f2b: MyStruct):
        struc_f2b._f2b[0, 0] = 103

    @ti.func
    def f2a(struc_f2a: MyStruct):
        struc_f2a._f2a[0, 0] = 102
        f2b(struc_f2a)

    @ti.func
    def f1(struc_f1: MyStruct):
        struc_f1._f1[0, 0] = 101
        f2a(struc_f1)
        f3(struc_f1)

    @ti.kernel
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
        _f1b: ti.types.NDArray[ti.f32, 2]
        _f1a: ti.types.NDArray[ti.f32, 2]
        _k1: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            _f1b=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1a=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f1(flag: ti.template(), struc_f1: MyStruct):
        if ti.static(flag):
            struc_f1._f1a[0, 0] = 101
        else:
            struc_f1._f1b[0, 0] = 102

    @ti.kernel
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
        _f2b: ti.types.NDArray[ti.f32, 2]
        _f2a: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _k1: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def create_struct():
        my_struct = MyStruct(
            _f2b=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2a=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f2(flag: ti.template(), struc_f2: MyStruct):
        if ti.static(flag):
            struc_f2._f2a[0, 0] = 102
        else:
            struc_f2._f2b[0, 0] = 103

    @ti.func
    def f1(struct_f1: MyStruct):
        struct_f1._f1[0, 0] = 101
        f2(False, struct_f1)
        f2(True, struct_f1)

    @ti.kernel
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _k2: ti.types.NDArray[ti.f32, 2]
        _f1_no_flag: ti.types.NDArray[ti.f32, 2]
        _f1_with_flag: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _k2=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1_no_flag=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1_with_flag=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f1(flag: ti.template(), struct_f1: MyStruct):
        if ti.static(flag):
            struct_f1._f1_with_flag[0, 0] = 102
        else:
            struct_f1._f1_no_flag[0, 0] = 103

    @ti.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 101
        f1(False, struct_k1)

    @ti.kernel
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1_no_flag: ti.types.NDArray[ti.f32, 2]
        _f1_with_flag: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1_no_flag=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1_with_flag=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f1(flag: ti.template(), struct_f1: MyStruct):
        if ti.static(flag):
            struct_f1._f1_with_flag[0, 0] = 101
        else:
            struct_f1._f1_no_flag[0, 0] = 102

    @ti.kernel
    def k1(flag: ti.Template, struct_k1: MyStruct):
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _f2a: ti.types.NDArray[ti.f32, 2]
        _f2b: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2a=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2b=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f2a(struct_f2a: MyStruct):
        struct_f2a._f2a[0, 0] += 3

    @ti.func
    def f2b(struct_f2b: MyStruct):
        struct_f2b._f2b[0, 0] += 5

    @ti.func
    def f1(struct_f1: MyStruct):
        struct_f1._f1[0, 0] = 101
        f2a(struct_f2a=struct_f1)
        f2a(struct_f2a=struct_f1)
        f2b(struct_f2b=struct_f1)

    @ti.kernel
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


@pytest.mark.xfail(reason="cannot use * when calling ti.func")
@test_utils.test()
def test_pruning_func_return_star_to_another() -> None:
    """
    Using the tuple return from one fucntion as the args to
    another
    """

    @ti.func
    def return_params(a: ti.i32):
        return a + 1, a + 5

    @ti.func
    def f2(t: ti.types.NDArray[ti.i32, 1], a: ti.i32, b: ti.i32) -> None:
        t[0] = a
        t[1] = b

    @ti.kernel
    def k1(t: ti.types.NDArray[ti.i32, 1], a: ti.i32) -> None:
        f2(t, *return_params(a))

    t = ti.ndarray(ti.i32, (10,))
    k1(t, 3)
    assert t[0] == 4
    assert t[0] == 8


@pytest.mark.xfail(reason="cannot use * when calling ti.func")
@test_utils.test()
def test_pruning_func_return_star_to_another_two_step() -> None:
    """
    Using the tuple return from one fucntion as the args to
    another
    """

    @ti.func
    def return_params(a: ti.i32):
        return a + 1, a + 5

    @ti.func
    def f2(t: ti.types.NDArray[ti.i32, 1], a: ti.i32, b: ti.i32) -> None:
        t[0] = a
        t[1] = b

    @ti.kernel
    def k1(t: ti.types.NDArray[ti.i32, 1], a: ti.i32) -> None:
        res = return_params(a)
        f2(t, *res)

    t = ti.ndarray(ti.i32, (10,))
    k1(t, 3)
    assert t[0] == 4
    assert t[0] == 8


@test_utils.test()
def test_pruning_func_return_star_to_another_explicit_vars() -> None:
    """
    Using the tuple return from one fucntion as the args to
    another
    """

    @ti.func
    def return_params(a: ti.i32):
        return a + 1, a + 5

    @ti.func
    def f2(t: ti.types.NDArray[ti.i32, 1], a: ti.i32, b: ti.i32) -> None:
        t[0] = a
        t[1] = b

    @ti.kernel
    def k1(t: ti.types.NDArray[ti.i32, 1], a: ti.i32) -> None:
        b, c = return_params(a)
        f2(t, b, c)

    t = ti.ndarray(ti.i32, (10,))
    k1(t, 3)
    assert t[0] == 4
    assert t[1] == 8


@test_utils.test()
def test_pruning_pass_element_of_tensor_of_dataclass() -> None:
    vec3 = ti.types.vector(3, ti.f32)

    @dataclasses.dataclass
    class MyStruct:
        _unused0: ti.types.NDArray[vec3, 2]
        _k1: ti.types.NDArray[vec3, 2]
        _unused0b: ti.types.NDArray[vec3, 2]
        _f1: ti.types.NDArray[vec3, 2]
        _unused1: ti.types.NDArray[vec3, 2]
        _in: ti.types.NDArray[vec3, 2]
        _unused2: ti.types.NDArray[vec3, 2]
        _out: ti.types.NDArray[vec3, 2]
        _unused3: ti.types.NDArray[vec3, 2]

    def make_struct():
        my_struct = MyStruct(
            _unused0=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _k1=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _unused0b=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _f1=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _unused1=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _in=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _unused2=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _out=ti.ndarray(dtype=vec3, shape=(1, 1)),
            _unused3=ti.ndarray(dtype=vec3, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f2(_in: vec3) -> vec3:
        return _in + 5.0

    @ti.func
    def f1(struct_f1: MyStruct):
        struct_f1._f1[0, 0] = 101
        struct_f1._out[0, 0] = f2(struct_f1._in[0, 0])

    @ti.kernel
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _unused1: ti.types.NDArray[ti.f32, 2]
        _unused2: ti.types.NDArray[ti.f32, 2]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused2=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct1, my_struct2

    @ti.func
    def f1(struct1_f1: MyStruct1, struct2_f1: MyStruct2):
        struct1_f1._f1[0, 0] = 102
        struct2_f1._f1[0, 0] = 103

    @ti.kernel
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _unused1: ti.types.NDArray[ti.f32, 2]
        _unused2: ti.types.NDArray[ti.f32, 2]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused2=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct1, my_struct2

    @ti.data_oriented
    class MyDataOriented:
        def __init__(self) -> None: ...

        @ti.func
        def f1(self, struct1_f1: MyStruct1, struct2_f1: MyStruct2):
            struct1_f1._f1[0, 0] = 102
            struct2_f1._f1[0, 0] = 103

    @ti.kernel
    def k1(my_data_oriented: ti.Template, struct1_k1: MyStruct1, struct2_k1: MyStruct2):
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
        _k1: ti.types.NDArray[ti.f32, 1]
        _f1: ti.types.NDArray[ti.f32, 2]
        _unused1: ti.types.NDArray[ti.f32, 4]
        _unused2: ti.types.NDArray[ti.f32, 4]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: ti.types.NDArray[ti.f32, 1]
        _f1: ti.types.NDArray[ti.f32, 3]
        _unused: ti.types.NDArray[ti.f32, 4]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused1=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1, 1)),
            _unused2=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1, 1)),
        )
        return my_struct1, my_struct2

    @ti.data_oriented
    class MyDataOriented:
        def __init__(self) -> None: ...

        @ti.func
        def f1(self, struct1_f1: MyStruct1, struct2_f1: MyStruct2):
            struct1_f1._f1[0, 0] = 102
            struct2_f1._f1[0, 0, 0] = 103

    @ti.kernel
    def k1(my_data_oriented: ti.Template, struct1_k1: MyStruct1, struct2_k1: MyStruct2):
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
        _k1: ti.types.NDArray[ti.f32, 1]
        _f1: ti.types.NDArray[ti.f32, 2]
        _unused1: ti.types.NDArray[ti.f32, 4]
        _unused2: ti.types.NDArray[ti.f32, 4]

    @dataclasses.dataclass
    class MyStruct2:
        _k1: ti.types.NDArray[ti.f32, 1]
        _f1: ti.types.NDArray[ti.f32, 3]
        _unused: ti.types.NDArray[ti.f32, 4]

    def make_structs():
        my_struct1 = MyStruct1(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused1=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1, 1)),
            _unused2=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1, 1)),
        )
        my_struct2 = MyStruct2(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1, 1, 1)),
        )
        return my_struct1, my_struct2

    @ti.data_oriented
    class MyDataOriented:
        def __init__(self) -> None: ...

        @ti.func
        def f1(self, struct1_f1: MyStruct1, struct2_f1: MyStruct2):
            struct1_f1._f1[0, 0] = 102
            struct2_f1._f1[0, 0, 0] = 103

    @ti.kernel
    def k1(my_data_oriented: ti.Template, struct1_k1: MyStruct1, struct2_k1: MyStruct2):
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

    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1], b: ti.i32, c: ti.i32):
        a[0] = b
        a[1] = c

    @ti.kernel
    def k1(a: ti.types.NDArray[ti.i32, 1]) -> None:
        f1(a, *star_args)

    star_args = [3, 5]

    a = ti.ndarray(ti.i32, (10,))
    k1(a)
    assert a[0] == 3
    assert a[1] == 5


@test_utils.test()
def test_pruning_star_args_error_not_at_end_another_arg() -> None:
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1], b: ti.i32, c: ti.i32, d: ti.i32):
        a[0] = b
        a[1] = c

    @ti.kernel
    def k1(a: ti.types.NDArray[ti.i32, 1]) -> None:
        f1(a, *star_args, 3)

    star_args = [3, 5]

    a = ti.ndarray(ti.i32, (10,))
    with pytest.raises(QuadrantsSyntaxError) as e:
        k1(a)
    assert "STARNOTLAST" in e.value.args[0]


@test_utils.test()
def test_pruning_star_args_error_not_at_end_kwargs() -> None:
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1], b: ti.i32, c: ti.i32, d: ti.i32):
        a[0] = b
        a[1] = c

    @ti.kernel
    def k1(a: ti.types.NDArray[ti.i32, 1]) -> None:
        f1(a, *star_args, d=3)

    star_args = [3, 5]

    a = ti.ndarray(ti.i32, (10,))
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _f2: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f1(struct: MyStruct):
        struct._f1[0, 0] = 101

    @ti.func
    def f2(struct: MyStruct):
        struct._f2[0, 0] = 102

    functions = [f1, f2]

    @ti.kernel
    def k1(struct_k1: MyStruct):
        struct_k1._k1[0, 0] = 100
        for fn in ti.static(functions):
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
        _k1: ti.types.NDArray[ti.f32, 2]
        _f1: ti.types.NDArray[ti.f32, 2]
        _f2: ti.types.NDArray[ti.f32, 2]
        _unused: ti.types.NDArray[ti.f32, 2]

    def make_struct():
        my_struct = MyStruct(
            _k1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f1=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _f2=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
            _unused=ti.ndarray(dtype=ti.f32, shape=(1, 1)),
        )
        return my_struct

    @ti.func
    def f1(struct: MyStruct):
        struct._f1[0, 0] = 101

    @ti.func
    def f2(struct: MyStruct):
        struct._f2[0, 0] = 102

    @ti.kernel
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
