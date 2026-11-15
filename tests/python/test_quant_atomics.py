from pytest import approx

import quadrants as qd

from tests.helpers import test_utils


# TODO: remove excluding of qd.metal.
@test_utils.test(require=qd.extension.quant_basic, exclude=[qd.metal], debug=True)
def test_quant_int_atomics():
    qi13 = qd.types.quant.int(13, True)
    qi5 = qd.types.quant.int(5, True)
    qu2 = qd.types.quant.int(2, False)

    x = qd.field(dtype=qi13)
    y = qd.field(dtype=qi5)
    z = qd.field(dtype=qu2)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y, z)
    qd.root.place(bitpack)

    x[()] = 3
    y[()] = 2
    z[()] = 0

    @qd.kernel
    def foo():
        for i in range(10):
            x[()] += 4

        for j in range(5):
            y[()] -= 1

        for k in range(3):
            z[()] += 1

    foo()

    assert x[()] == 43
    assert y[()] == -3
    assert z[()] == 3


@test_utils.test(require=[qd.extension.quant_basic, qd.extension.data64], debug=True)
def test_quant_int_atomics_b64():
    qi13 = qd.types.quant.int(13, True)

    x = qd.field(dtype=qi13)

    qd.root.quant_array(qd.i, 4, max_num_bits=64).place(x)

    x[0] = 100
    x[1] = 200
    x[2] = 300

    @qd.kernel
    def foo():
        for i in range(9):
            x[i % 3] += i

    foo()

    assert x[0] == 109
    assert x[1] == 212
    assert x[2] == 315


@test_utils.test(require=qd.extension.quant_basic, debug=True)
def test_quant_fixed_atomics():
    qfxt13 = qd.types.quant.fixed(bits=13, signed=True, scale=0.1)
    qfxt19 = qd.types.quant.fixed(bits=19, signed=False, scale=0.1)

    x = qd.field(dtype=qfxt13)
    y = qd.field(dtype=qfxt19)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y)
    qd.root.place(bitpack)

    @qd.kernel
    def foo():
        x[()] = 0.7
        y[()] = 123.4
        for _ in range(10):
            x[()] -= 0.4
            y[()] += 100.1

    foo()
    assert x[()] == approx(-3.3)
    assert y[()] == approx(1124.4)
