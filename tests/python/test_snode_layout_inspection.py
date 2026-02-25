import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_primitives():
    x = qd.field(dtype=qd.i16)
    y = qd.field(dtype=qd.f32)
    z = qd.field(dtype=qd.f64)

    p = qd.field(dtype=qd.f32)
    q = qd.field(dtype=qd.f32)
    r = qd.field(dtype=qd.f64)

    n1 = qd.root.dense(qd.i, 32)
    n1.place(x)

    n2 = qd.root.dense(qd.i, 32)
    n2.place(y, z)

    n3 = qd.root.dense(qd.i, 1)
    n3.place(p, q, r)

    assert n1._cell_size_bytes == 2
    assert n2._cell_size_bytes in [12, 16]
    assert n3._cell_size_bytes == 16

    assert n1._offset_bytes_in_parent_cell == 0
    assert n2._offset_bytes_in_parent_cell == 2 * 32
    assert n3._offset_bytes_in_parent_cell in [2 * 32 + 12 * 32, 2 * 32 + 16 * 32]

    assert x.snode._offset_bytes_in_parent_cell == 0
    assert y.snode._offset_bytes_in_parent_cell == 0
    assert z.snode._offset_bytes_in_parent_cell in [4, 8]
    assert p.snode._offset_bytes_in_parent_cell == 0
    assert q.snode._offset_bytes_in_parent_cell == 4
    assert r.snode._offset_bytes_in_parent_cell == 8


@test_utils.test(arch=qd.cpu)
def test_bitpacked_fields():
    x = qd.field(dtype=qd.types.quant.int(16, False))
    y = qd.field(dtype=qd.types.quant.fixed(16, False))
    z = qd.field(dtype=qd.f32)

    n1 = qd.root.dense(qd.i, 32)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    n1.place(bitpack)

    n2 = qd.root.dense(qd.i, 4)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(y)
    n2.place(bitpack)
    n2.place(z)

    assert n1._cell_size_bytes == 4
    assert n2._cell_size_bytes == 8
