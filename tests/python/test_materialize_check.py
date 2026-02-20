import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_check_field_not_placed():
    a = qd.field(qd.i32)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(RuntimeError, match=r"These field\(s\) are not placed.*"):
        foo()


@test_utils.test()
def test_check_grad_field_not_placed():
    a = qd.field(qd.f32, needs_grad=True)
    qd.root.dense(qd.i, 1).place(a)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_grad_vector_field_not_placed():
    b = qd.Vector.field(3, qd.f32, needs_grad=True)
    qd.root.dense(qd.i, 1).place(b)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_grad_matrix_field_not_placed():
    c = qd.Matrix.field(2, 3, qd.f32, needs_grad=True)
    qd.root.dense(qd.i, 1).place(c)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_grad_struct_field_not_placed():
    d = qd.Struct.field(
        {
            "pos": qd.types.vector(3, float),
            "vel": qd.types.vector(3, float),
            "acc": qd.types.vector(3, float),
            "mass": qd.f32,
        },
        needs_grad=True,
    )
    qd.root.dense(qd.i, 1).place(d)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_field_not_placed():
    a = qd.field(qd.f32, needs_dual=True)
    qd.root.dense(qd.i, 1).place(a)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_vector_field_not_placed():
    b = qd.Vector.field(3, qd.f32, needs_dual=True)
    qd.root.dense(qd.i, 1).place(b)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_matrix_field_not_placed():
    c = qd.Matrix.field(2, 3, qd.f32, needs_dual=True)
    qd.root.dense(qd.i, 1).place(c)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_struct_field_not_placed():
    d = qd.Struct.field(
        {
            "pos": qd.types.vector(3, float),
            "vel": qd.types.vector(3, float),
            "acc": qd.types.vector(3, float),
            "mass": qd.f32,
        },
        needs_dual=True,
    )
    qd.root.dense(qd.i, 1).place(d)

    @qd.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_matrix_field_member_shape():
    a = qd.Matrix.field(2, 2, qd.i32)
    qd.root.dense(qd.i, 10).place(a.get_scalar_field(0, 0))
    qd.root.dense(qd.i, 11).place(a.get_scalar_field(0, 1))
    qd.root.dense(qd.i, 10).place(a.get_scalar_field(1, 0))
    qd.root.dense(qd.i, 11).place(a.get_scalar_field(1, 1))

    @qd.kernel
    def foo():
        pass

    with pytest.raises(RuntimeError, match=r"Members of the following field have different shapes.*"):
        foo()
