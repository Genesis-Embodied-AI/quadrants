import pytest
from pytest import approx

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("exponent_bits", [5, 6, 7, 8])
@test_utils.test(require=qd.extension.quant)
def test_shared_exponents(exponent_bits):
    if exponent_bits == 8:
        pytest.skip("quant with exponent bits fails currently. Reason unclear. TODO: fixme")
    qflt1 = qd.types.quant.float(exp=exponent_bits, frac=10, signed=False)
    qflt2 = qd.types.quant.float(exp=exponent_bits, frac=14, signed=False)
    a = qd.field(dtype=qflt1)
    b = qd.field(dtype=qflt2)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a, b, shared_exponent=True)
    qd.root.place(bitpack)

    assert a[()] == 0.0
    assert b[()] == 0.0

    a[()] = 10
    assert a[()] == 10.0
    assert b[()] == 0.0

    a[()] = 0
    assert a[()] == 0.0
    assert b[()] == 0.0

    @qd.kernel
    def foo(x: qd.f32, y: qd.f32):
        a[()] = x
        b[()] = y

    foo(3.2, 0.25)

    assert a[()] == approx(3.2, rel=1e-3)
    assert b[()] == approx(0.25, rel=2e-2)
    a[()] = 0.27
    assert a[()] == approx(0.27, rel=1e-2)
    assert b[()] == approx(0.25, rel=2e-2)
    a[()] = 100
    assert a[()] == approx(100, rel=1e-3)
    assert b[()] == approx(0.25, rel=1e-2)

    b[()] = 0
    assert a[()] == approx(100, rel=1e-3)
    assert b[()] == 0

    foo(0, 0)
    assert a[()] == 0.0
    assert b[()] == 0.0

    # test flush to zero
    foo(1000, 1e-6)
    assert a[()] == 1000.0
    assert b[()] == 0.0

    foo(1000, 1000)
    assert a[()] == 1000.0
    assert b[()] == 1000.0

    foo(1e-30, 1e-30)
    if exponent_bits == 8:
        assert a[()] == approx(1e-30, 1e-3)
        assert b[()] == approx(1e-30, 1e-4)
    else:
        # Insufficient exponent bits: should flush to zero
        assert a[()] == 0
        assert b[()] == 0


@pytest.mark.parametrize("exponent_bits", [5, 6, 7, 8])
@test_utils.test(require=qd.extension.quant)
def test_shared_exponent_add(exponent_bits):
    if exponent_bits == 8:
        pytest.skip("quant with exponent bits fails currently. Reason unclear. TODO: fixme")
    qflt1 = qd.types.quant.float(exp=exponent_bits, frac=10, signed=False)
    qflt2 = qd.types.quant.float(exp=exponent_bits, frac=14, signed=False)
    a = qd.field(dtype=qflt1)
    b = qd.field(dtype=qflt2)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a, b, shared_exponent=True)
    qd.root.place(bitpack)

    @qd.kernel
    def foo(x: qd.f32, y: qd.f32):
        a[()] = x
        b[()] = y

    a[()] = 4
    assert a[()] == 4
    assert b[()] == 0
    b[()] = 3
    assert a[()] == 4
    assert b[()] == 3

    b[()] += 1

    assert a[()] == 4
    assert b[()] == 4

    for i in range(100):
        a[()] += 4
        b[()] += 1
        assert a[()] == 4 + (i + 1) * 4
        assert b[()] == 4 + (i + 1)


@pytest.mark.parametrize("exponent_bits", [5, 6, 7, 8])
@test_utils.test(require=qd.extension.quant)
def test_shared_exponent_borrow(exponent_bits):
    if exponent_bits == 8:
        pytest.skip("quant with exponent bits fails currently. Reason unclear. TODO: fixme")
    qflt1 = qd.types.quant.float(exp=exponent_bits, frac=10, signed=False)
    qflt2 = qd.types.quant.float(exp=exponent_bits, frac=14, signed=False)
    a = qd.field(dtype=qflt1)
    b = qd.field(dtype=qflt2)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a, b, shared_exponent=True)
    qd.root.place(bitpack)

    @qd.kernel
    def foo(x: qd.f32, y: qd.f32):
        a[()] = x
        b[()] = y

    def inc():
        a[()] += 1
        b[()] -= 1

    foo(0, 100)

    for i in range(100):
        assert a[()] == i
        assert b[()] == 100 - i
        inc()


@pytest.mark.parametrize("exponent_bits", [5, 6, 7, 8])
@test_utils.test(require=qd.extension.quant)
def test_shared_exponent_negative(exponent_bits):
    if exponent_bits == 8:
        pytest.skip("quant with exponent bits fails currently. Reason unclear. TODO: fixme")
    qflt1 = qd.types.quant.float(exp=exponent_bits, frac=10, signed=False)
    qflt2 = qd.types.quant.float(exp=exponent_bits, frac=14, signed=True)
    a = qd.field(dtype=qflt1)
    b = qd.field(dtype=qflt2)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a, b, shared_exponent=True)
    qd.root.place(bitpack)

    a[()] = 37
    assert a[()] == 37
    b[()] = -123
    assert b[()] == -123


# TODO: test precision
# TODO: make sure unsigned has one more effective significand bit
# TODO: test shared exponent floats with quant int in a single bit struct
