import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_order_scalar():
    X = 16
    Y = 8
    Z = 4
    S = 4

    a = qd.field(qd.i32, shape=(X, Y, Z), order="ijk")
    b = qd.field(qd.i32, shape=(X, Y, Z), order="ikj")
    c = qd.field(qd.i32, shape=(X, Y, Z), order="jik")
    d = qd.field(qd.i32, shape=(X, Y, Z), order="jki")
    e = qd.field(qd.i32, shape=(X, Y, Z), order="kij")
    f = qd.field(qd.i32, shape=(X, Y, Z), order="kji")

    @qd.kernel
    def fill():
        for i, j, k in b:
            a[i, j, k] = i * j * k
            b[i, j, k] = i * j * k
            c[i, j, k] = i * j * k
            d[i, j, k] = i * j * k
            e[i, j, k] = i * j * k
            f[i, j, k] = i * j * k

    @qd.kernel
    def get_field_addr(a: qd.template(), i: qd.i32, j: qd.i32, k: qd.i32) -> qd.u64:
        return qd.get_addr(a, [i, j, k])

    fill()

    a_addr = get_field_addr(a, 0, 0, 0)
    b_addr = get_field_addr(b, 0, 0, 0)
    c_addr = get_field_addr(c, 0, 0, 0)
    d_addr = get_field_addr(d, 0, 0, 0)
    e_addr = get_field_addr(e, 0, 0, 0)
    f_addr = get_field_addr(f, 0, 0, 0)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                assert a[i, j, k] == b[i, j, k] == c[i, j, k] == i * j * k
                assert d[i, j, k] == e[i, j, k] == f[i, j, k] == i * j * k
                assert a_addr + (i * (Y * Z) + j * Z + k) * S == get_field_addr(a, i, j, k)
                assert b_addr + (i * (Z * Y) + k * Y + j) * S == get_field_addr(b, i, j, k)
                assert c_addr + (j * (X * Z) + i * Z + k) * S == get_field_addr(c, i, j, k)
                assert d_addr + (j * (Z * X) + k * X + i) * S == get_field_addr(d, i, j, k)
                assert e_addr + (k * (X * Y) + i * Y + j) * S == get_field_addr(e, i, j, k)
                assert f_addr + (k * (Y * X) + j * X + i) * S == get_field_addr(f, i, j, k)


@test_utils.test(arch=get_host_arch_list())
def test_order_vector():
    X = 4
    Y = 2
    Z = 2
    S = 4

    a = qd.Vector.field(Z, qd.i32, shape=(X, Y), order="ij", layout=qd.Layout.AOS)
    b = qd.Vector.field(Z, qd.i32, shape=(X, Y), order="ji", layout=qd.Layout.AOS)
    c = qd.Vector.field(Z, qd.i32, shape=(X, Y), order="ij", layout=qd.Layout.SOA)
    d = qd.Vector.field(Z, qd.i32, shape=(X, Y), order="ji", layout=qd.Layout.SOA)

    @qd.kernel
    def fill():
        for i, j in b:
            a[i, j] = [i, j]
            b[i, j] = [i, j]
            c[i, j] = [i, j]
            d[i, j] = [i, j]

    @qd.kernel
    def get_field_addr(a: qd.template(), i: qd.i32, j: qd.i32) -> qd.u64:
        return qd.get_addr(a, [i, j])

    fill()

    a_addr = get_field_addr(a, 0, 0)
    b_addr = get_field_addr(b, 0, 0)
    c_addr = get_field_addr(c, 0, 0)
    d_addr = get_field_addr(d, 0, 0)
    for i in range(X):
        for j in range(Y):
            assert a[i, j] == b[i, j] == c[i, j] == d[i, j] == [i, j]
            for k in range(Z):
                assert a_addr + (i * (Y * Z) + j * Z + k) * S == get_field_addr(a.get_scalar_field(k), i, j)
                assert b_addr + (j * (X * Z) + i * Z + k) * S == get_field_addr(b.get_scalar_field(k), i, j)
                assert c_addr + (k * (X * Y) + i * Y + j) * S == get_field_addr(c.get_scalar_field(k), i, j)
                assert d_addr + (k * (Y * X) + j * X + i) * S == get_field_addr(d.get_scalar_field(k), i, j)


@test_utils.test(arch=get_host_arch_list())
def test_order_must_throw_scalar():
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="The dimensionality of shape and order must be the same",
    ):
        a = qd.field(dtype=qd.f32, shape=3, order="ij")
    with pytest.raises(qd.QuadrantsCompilationError, match="shape cannot be None when order is set"):
        b = qd.field(dtype=qd.f32, shape=None, order="i")
    with pytest.raises(qd.QuadrantsCompilationError, match="The axes in order must be different"):
        c = qd.field(dtype=qd.f32, shape=(3, 4, 3), order="iji")
    with pytest.raises(qd.QuadrantsCompilationError, match="Invalid axis"):
        d = qd.field(dtype=qd.f32, shape=(3, 4, 3), order="ijl")


@test_utils.test(arch=get_host_arch_list())
def test_order_must_throw_vector():
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="The dimensionality of shape and order must be the same",
    ):
        a = qd.Vector.field(3, dtype=qd.f32, shape=3, order="ij")
    with pytest.raises(qd.QuadrantsCompilationError, match="shape cannot be None when order is set"):
        b = qd.Vector.field(3, dtype=qd.f32, shape=None, order="i")
    with pytest.raises(qd.QuadrantsCompilationError, match="The axes in order must be different"):
        c = qd.Vector.field(3, dtype=qd.f32, shape=(3, 4, 3), order="iii")
    with pytest.raises(qd.QuadrantsCompilationError, match="Invalid axis"):
        d = qd.Vector.field(3, dtype=qd.f32, shape=(3, 4, 3), order="ihj")
