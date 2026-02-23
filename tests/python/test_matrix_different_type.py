import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.quant_basic)
def test_valid():
    qflt = qd.types.quant.float(exp=8, frac=5, signed=True)
    qfxt = qd.types.quant.fixed(bits=10, signed=True, scale=0.1)
    type_list = [[qflt, qfxt], [qflt, qfxt]]
    a = qd.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    b = qd.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    c = qd.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a.get_scalar_field(0, 0), a.get_scalar_field(0, 1))
    qd.root.dense(qd.i, 1).place(bitpack)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a.get_scalar_field(1, 0), a.get_scalar_field(1, 1))
    qd.root.dense(qd.i, 1).place(bitpack)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(b.get_scalar_field(0, 0), b.get_scalar_field(0, 1))
    qd.root.dense(qd.i, 1).place(bitpack)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(b.get_scalar_field(1, 0), b.get_scalar_field(1, 1))
    qd.root.dense(qd.i, 1).place(bitpack)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(c.get_scalar_field(0, 0), c.get_scalar_field(0, 1))
    qd.root.dense(qd.i, 1).place(bitpack)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(c.get_scalar_field(1, 0), c.get_scalar_field(1, 1))
    qd.root.dense(qd.i, 1).place(bitpack)

    @qd.kernel
    def init():
        a[0] = [[1.0, 3.0], [2.0, 1.0]]
        b[0] = [[2.0, 4.0], [-2.0, 1.0]]
        c[0] = a[0] + b[0]

    def verify():
        assert c[0][0, 0] == pytest.approx(3.0)
        assert c[0][0, 1] == pytest.approx(7.0)
        assert c[0][1, 0] == pytest.approx(0.0)
        assert c[0][1, 1] == pytest.approx(2.0)

    init()
    verify()


@test_utils.test(require=qd.extension.quant_basic)
def test_invalid():
    qit = qd.types.quant.int(bits=10, signed=True)
    qfxt = qd.types.quant.fixed(bits=10, signed=True, scale=0.1)
    type_list = [qit, qfxt]
    with pytest.raises(
        RuntimeError,
        match="Member fields of a matrix field must have the same compute type",
    ):
        a = qd.Vector.field(len(type_list), dtype=type_list)
