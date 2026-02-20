import math

from pytest import approx

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.quant_basic)
def test_quant_fixed():
    qfxt = qd.types.quant.fixed(bits=32, max_value=2)
    x = qd.field(dtype=qfxt)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    qd.root.place(bitpack)

    @qd.kernel
    def foo():
        x[None] = 0.7
        print(x[None])
        x[None] = x[None] + 0.4

    foo()
    assert x[None] == approx(1.1)
    x[None] = 0.64
    assert x[None] == approx(0.64)
    x[None] = 0.66
    assert x[None] == approx(0.66)


@test_utils.test(require=qd.extension.quant_basic)
def test_quant_fixed_matrix_rotation():
    qfxt = qd.types.quant.fixed(bits=16, max_value=1.2)

    x = qd.Matrix.field(2, 2, dtype=qfxt)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x.get_scalar_field(0, 0), x.get_scalar_field(0, 1))
    qd.root.place(bitpack)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x.get_scalar_field(1, 0), x.get_scalar_field(1, 1))
    qd.root.place(bitpack)

    x[None] = [[1.0, 0.0], [0.0, 1.0]]

    @qd.kernel
    def rotate_18_degrees():
        angle = math.pi / 10
        x[None] = x[None] @ qd.Matrix([[qd.cos(angle), qd.sin(angle)], [-qd.sin(angle), qd.cos(angle)]])

    for i in range(5):
        rotate_18_degrees()
    assert x[None][0, 0] == approx(0, abs=1e-4)
    assert x[None][0, 1] == approx(1, abs=1e-4)
    assert x[None][1, 0] == approx(-1, abs=1e-4)
    assert x[None][1, 1] == approx(0, abs=1e-4)


@test_utils.test(require=qd.extension.quant_basic)
def test_quant_fixed_implicit_cast():
    qfxt = qd.types.quant.fixed(bits=13, scale=0.1)
    x = qd.field(dtype=qfxt)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    qd.root.place(bitpack)

    @qd.kernel
    def foo():
        x[None] = 10

    foo()
    assert x[None] == approx(10.0)


@test_utils.test(require=qd.extension.quant_basic)
def test_quant_fixed_cache_read_only():
    qfxt = qd.types.quant.fixed(bits=15, scale=0.1)
    x = qd.field(dtype=qfxt)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    qd.root.place(bitpack)

    @qd.kernel
    def test(data: qd.f32):
        qd.cache_read_only(x)
        assert x[None] == data

    x[None] = 0.7
    test(0.7)
    x[None] = 1.2
    test(1.2)
