import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_literal_u32():
    @qd.kernel
    def pcg_hash(inp: qd.u32) -> qd.u32:
        state: qd.u32 = inp * qd.u32(747796405) + qd.u32(2891336453)
        word: qd.u32 = ((state >> ((state >> qd.u32(28)) + qd.u32(4))) ^ state) * qd.u32(277803737)
        return (word >> qd.u32(22)) ^ word

    assert pcg_hash(12345678) == 119515934
    assert pcg_hash(98765432) == 4244201195


@test_utils.test()
def test_literal_multi_args_error():
    @qd.kernel
    def multi_args_error():
        a = qd.i64(1, 2)

    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match="A primitive type can only decorate a single expression.",
    ):
        multi_args_error()


@test_utils.test()
def test_literal_keywords_error():
    @qd.kernel
    def keywords_error():
        a = qd.f64(1, x=2)

    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match="A primitive type can only decorate a single expression.",
    ):
        keywords_error()


@test_utils.test()
def test_literal_compound_error():
    @qd.kernel
    def expr_error():
        a = qd.Vector([1])
        b = qd.f16(a)

    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match="A primitive type cannot decorate an expression with a compound type.",
    ):
        expr_error()


@test_utils.test()
def test_literal_int_annotation_error():
    @qd.kernel
    def int_annotation_error():
        a = qd.f32(0)

    with pytest.raises(
        qd.QuadrantsTypeError,
        match="Integer literals must be annotated with a integer type. For type casting, use `qd.cast`.",
    ):
        int_annotation_error()


@test_utils.test()
def test_literal_float_annotation_error():
    @qd.kernel
    def float_annotation_error():
        a = qd.i32(0.0)

    with pytest.raises(
        qd.QuadrantsTypeError,
        match="Floating-point literals must be annotated with a floating-point type. For type casting, use `qd.cast`.",
    ):
        float_annotation_error()


@test_utils.test()
def test_literal_exceed_default_ip():
    @qd.kernel
    def func():
        b = 0x80000000

    with pytest.raises(qd.QuadrantsTypeError, match="exceeded the range of default_ip"):
        func()


@test_utils.test()
def test_literal_exceed_specified_dtype():
    @qd.kernel
    def func():
        b = qd.u16(-1)

    with pytest.raises(qd.QuadrantsTypeError, match="exceeded the range of specified dtype"):
        func()
